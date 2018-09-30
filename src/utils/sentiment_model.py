"""
Model for sentiment analysis.
"""
import random
import sys
import numpy as np
import pickle

import _dynet as dynet
from sklearn.metrics import f1_score, accuracy_score

from progress.bar import Bar

from lib.mnnl import Layer
from lib.mmappers import TRAINER_MAP


def load_model(path_to_model):
    """
    load a model from file; specify the .model file, it assumes the *pickle file in the same location
    """
    myparams = pickle.load(open(path_to_model + ".params.pickle", "rb"))
    model = SentimentModel(myparams["h_layers"],
                           myparams["h_dim"],
                           myparams["vocab_size"],
                           myparams["noise_sigma"])
    model.initialize_graph()
    model.model.populate(path_to_model + '.model')
    print("model loaded: {}".format(path_to_model), file=sys.stderr)
    return model


def save_model(model, path_to_model):
    """
    save a model; dynet only saves the parameters, need to store the rest separately
    """
    modelname = path_to_model + ".model"
    model.model.save(modelname)
    myparams = {"vocab_size": model.vocab_size,
                "h_dim": model.h_dim,
                "h_layers": model.h_layers,
                "noise_sigma": model.noise_sigma
                }
    pickle.dump(myparams, open(path_to_model + ".params.pickle", "wb" ))
    print("model stored: {}".format(modelname), file=sys.stderr)


def activation2func(activation):
    if activation == 'rectify':
        return dynet.rectify
    if activation == 'sigmoid':
        return dynet.logistic
    raise ValueError('%s is not a valid activation.' % activation)


class SentimentModel(object):
    def __init__(self, h_layers, h_dim, vocab_size, noise_sigma=0.1, trainer="adam", clip_threshold=5.0, activation='rectify'):
        self.model = dynet.ParameterCollection()
        self.h_layers = h_layers
        self.h_dim = h_dim
        self.vocab_size = vocab_size
        self.noise_sigma = noise_sigma
        self.activation = activation2func(activation)
        self.layers = []
        self.trainer = TRAINER_MAP[trainer](self.model)
        self.trainer.set_clip_threshold(clip_threshold)

    def initialize_graph(self):
        assert self.h_layers > 0
        for i in range(self.h_layers):
            layer = Layer(self.model, in_dim=self.vocab_size if i == 0 else self.h_dim,
                          output_dim=2 if i == self.h_layers-1 else self.h_dim,
                          activation=dynet.softmax if i == self.h_layers-1 else self.activation)
            self.layers.append(layer)

    def pick_neg_log(self, pred, gold):
        # TODO make this a static function in both classes
        if not isinstance(gold, int) and not isinstance(gold, np.int64):
            # calculate cross-entropy loss against the whole vector
            dy_gold = dynet.inputVector(gold)
            return -dynet.sum_elems(dynet.cmult(dy_gold, dynet.log(pred)))
        return -dynet.log(dynet.pick(pred, gold))

    def fit(self, train_X, train_Y, num_epochs, val_X=None,
            val_Y=None, patience=2, model_path=None, seed=None,
            word_dropout_rate=0.25, trg_vectors=None,
            unsup_weight=1.0, labeled_weight_proportion=1.0):
        """
        train the model
        :param trg_vectors: the prediction targets used for the unsupervised loss
                            in temporal ensembling
        :param unsup_weight: weight for the unsupervised consistency loss
                                    used in temporal ensembling
        """
        if seed:
            print(">>> using seed: ", seed, file=sys.stderr)
            random.seed(seed)  #setting random seed

        assert(train_X.shape[0] == len(train_Y)), \
            '# examples %d != # labels %d.' % (train_X.shape[0], len(train_Y))
        train_data = list(zip(train_X, train_Y))

        print('Starting training for %d epochs...' % num_epochs)
        best_val_f1, epochs_no_improvement = 0., 0
        if val_X is not None and val_Y is not None and model_path is not None:
            print('Using early stopping with patience of %d...' % patience)
        for cur_iter in range(num_epochs):
            bar = Bar('Training epoch %d/%d...' % (cur_iter + 1, num_epochs),
                      max=len(train_data), flush=True)
            total_loss = 0.0

            random_indices = np.arange(len(train_data))
            random.shuffle(random_indices)

            for i, idx in enumerate(random_indices):

                x, y = train_data[idx]
                output = self.predict(x, train=True, dropout_rate=word_dropout_rate)
                # in temporal ensembling, we assign a dummy label of -1 for
                # unlabeled sequences; we skip the supervised loss for these
                loss = dynet.scalarInput(0) if y == -1 else self.pick_neg_log(output, y)

                if trg_vectors is not None:
                    # the consistency loss in temporal ensembling is used for
                    # both supervised and unsupervised input
                    target = trg_vectors[idx]

                    other_loss = dynet.squared_distance(
                        output, dynet.inputVector(target))

                    if y != -1:
                        other_loss *= labeled_weight_proportion
                    loss += other_loss * unsup_weight
                total_loss += loss.value()

                loss.backward()
                self.trainer.update()
                bar.next()

            print(" iter {2} {0:>12}: {1:.2f}".format(
                "total loss", total_loss/len(train_data), cur_iter), file=sys.stderr)

            if val_X is not None and val_Y is not None and model_path is not None:
                # get the best F1 score on the validation set
                val_f1 = self.evaluate(val_X, val_Y)

                if val_f1 > best_val_f1:
                    print('F1 %.4f is better than best val F1 %.4f.' % (val_f1, best_val_f1))
                    best_val_f1 = val_f1
                    epochs_no_improvement = 0
                    save_model(self, model_path)
                else:
                    print('F1 %.4f is worse than best val F1 %.4f.' % (val_f1, best_val_f1))
                    epochs_no_improvement += 1
                if epochs_no_improvement == patience:
                    print('No improvement for %d epochs. Early stopping...' % epochs_no_improvement)
                    break

    def predict(self, feature_vector, train=False, soft_labels=False,
                temperature=None, dropout_rate=None):
        dynet.renew_cg()  # new graph

        feature_vector = feature_vector.toarray()
        feature_vector = np.squeeze(feature_vector, axis=0)

        # self.input = dynet.vecInput(self.vocab_size)
        # self.input.set(feature_vector)
        # TODO this takes too long; can we speed this up somehow?
        input = dynet.inputVector(feature_vector)
        for i in range(self.h_layers-1):
            if train:  # add some noise
                input = dynet.noise(input, self.noise_sigma)
                input = dynet.dropout(input, dropout_rate)
            input = self.layers[i](input)
        output = self.layers[-1](input, soft_labels=soft_labels,
                                 temperature=temperature)
        return output

    def get_predictions(self, test_X, soft_labels=False):
        predictions = []
        for x in test_X:
            o = self.predict(x)
            # print('Prediction:', o.value())
            predictions.append(o.value() if soft_labels
                               else int(np.argmax(o.value())))
        return predictions

    def evaluate(self, test_X, test_Y):
        preds = self.get_predictions(test_X)
        return f1_score(test_Y, preds)


def load_mttri_model(path_to_model):
    """
    load a model from file; specify the .model file, it assumes the *pickle file in the same location
    """
    myparams = pickle.load(open(path_to_model + ".params.pickle", "rb"))
    model = MttriSentimentModel(myparams["h_layers"],
                               myparams["h_dim"],
                               myparams["vocab_size"],
                               noise_sigma=myparams["noise_sigma"],
                               add_hidden=myparams["add_hidden"],
                               activation=myparams["activation"])
    model.initialize_graph()
    model.model.populate(path_to_model + '.model')
    print("model loaded: {}".format(path_to_model), file=sys.stderr)
    return model


def save_mttri_model(model, path_to_model):
    """
    save a model; dynet only saves the parameters, need to store the rest separately
    """
    modelname = path_to_model + ".model"
    model.model.save(modelname)
    myparams = {"vocab_size": model.vocab_size,
                "h_dim": model.h_dim,
                "h_layers": model.h_layers,
                "noise_sigma": model.noise_sigma,
                "add_hidden": model.add_hidden,
                "activation": model.activation_func
                }
    pickle.dump(myparams, open(path_to_model + ".params.pickle", "wb"))
    print("model stored: {}".format(modelname), file=sys.stderr)


class MttriSentimentModel(object):
    def __init__(self, h_layers, h_dim, vocab_size, noise_sigma=0.1, trainer="adam",
                 clip_threshold=5.0, add_hidden=False, learning_rate=0.001,
                 activation='rectify'):
        self.model = dynet.ParameterCollection()
        self.h_layers = h_layers
        self.h_dim = h_dim
        self.vocab_size = vocab_size
        self.noise_sigma = noise_sigma
        if self.noise_sigma > 0.05:
            print('Noise sigma > %.4f. Training might not work.' % noise_sigma)
        self.layers = []
        self.output_layers_dict = {}
        self.trainer = TRAINER_MAP[trainer](self.model, learning_rate)
        self.trainer.set_clip_threshold(clip_threshold)
        self.task_ids = ["F0", "F1", "Ft"]
        self.add_hidden = add_hidden
        self.activation_func = activation
        self.activation = activation2func(activation)

    def add_adversarial_loss(self, num_domains=2):
        # TODO try different hidden dimensions, e.g. half the dimension
        self.adv_layer = Layer(self.model, self.h_dim, num_domains,
                               activation=dynet.softmax,
                               mlp=self.h_dim if self.add_hidden else 0)

    def initialize_graph(self):
        assert self.h_layers > 0
        for i in range(self.h_layers):
            layer = Layer(self.model, in_dim=self.vocab_size if i == 0 else self.h_dim,
                          output_dim=self.h_dim,
                          activation=self.activation)
            self.layers.append(layer)

        self.output_layers_dict["F0"] = Layer(
            self.model, self.h_dim, 2, activation=dynet.softmax,
            mlp=self.h_dim if self.add_hidden else 0)
        self.output_layers_dict["F1"] = Layer(
            self.model, self.h_dim, 2, activation=dynet.softmax,
            mlp=self.h_dim if self.add_hidden else 0)
        self.output_layers_dict["Ft"] = Layer(
            self.model, self.h_dim, 2, activation=dynet.softmax,
            mlp=self.h_dim if self.add_hidden else 0)

    def pick_neg_log(self, pred, gold):
        # TODO make this a static function in both classes
        if not isinstance(gold, int) and not isinstance(gold, np.int64):
            # calculate cross-entropy loss against the whole vector
            dy_gold = dynet.inputVector(gold)
            return -dynet.sum_elems(dynet.cmult(dy_gold, dynet.log(pred)))
        return -dynet.log(dynet.pick(pred, gold))

    def fit(self, train_dict, num_epochs, val_X=None,
            val_Y=None, patience=2, model_path=None, seed=None,
            word_dropout_rate=0.25, trg_vectors=None,
            unsup_weight=1.0, orthogonality_weight=0.0,
            adversarial=False):
        """
        train the model
        :param trg_vectors: the prediction targets used for the unsupervised loss
                            in temporal ensembling
        :param unsup_weight: weight for the unsupervised consistency loss
                                    used in temporal ensembling
        """
        if seed:
            print(">>> using seed: ", seed, file=sys.stderr)
            random.seed(seed)  #setting random seed

        train_data = []
        for task, task_dict in train_dict.items():
            for key in ["X", "Y", "domain"]:
                assert key in task_dict, "Error: %s is not available." % key
            examples, labels, domain_tags = task_dict["X"], task_dict["Y"], \
                                            task_dict["domain"]
            assert examples.shape[0] == len(labels)

            # train data is a list of 4-tuples: (example, label, task_id, domain_id)
            train_data += list(
                zip(examples, labels, [[task] * len(labels)][0], domain_tags))

        print('Starting training for %d epochs...' % num_epochs)
        best_val_f1, epochs_no_improvement = 0., 0

        if val_X is not None and val_Y is not None and model_path is not None:
            print('Using early stopping with patience of %d...' % patience)

        for cur_iter in range(num_epochs):
            bar = Bar('Training epoch %d/%d...' % (cur_iter + 1, num_epochs),
                      max=len(train_data), flush=True)
            total_loss, total_constraint, total_adversarial = 0.0, 0.0, 0.0

            random_indices = np.arange(len(train_data))
            random.shuffle(random_indices)

            for i, idx in enumerate(random_indices):

                x, y, task_id, domain_id = train_data[idx]
                task_ids = [task_id]

                if task_id == 'src':
                    # we train both F0 and F1 on source data
                    task_ids = ['F0', 'F1']
                elif task_id == 'src_all':
                    # we train F0, F1, and Ft on source data for base training
                    task_ids = ['F0', 'F1', 'Ft']

                loss = 0
                outputs, constraint, adv = self.predict(
                    x, task_ids, train=True, dropout_rate=word_dropout_rate,
                    orthogonality_weight=orthogonality_weight,
                    domain_id=domain_id if adversarial else None)

                # in temporal ensembling, we assign a dummy label of -1 for
                # unlabeled sequences; we skip the supervised loss for these
                for output in outputs:
                    loss += dynet.scalarInput(0) if y == -1 else self.pick_neg_log(output, y)

                    if trg_vectors is not None:
                        # the consistency loss in temporal ensembling is used for
                        # both supervised and unsupervised input
                        target = trg_vectors[idx]

                        other_loss = dynet.squared_distance(
                            output, dynet.inputVector(target))
                        loss += other_loss * unsup_weight

                # the orthogonality weight is the same for every prediction,
                # so we can add it in the end
                if orthogonality_weight != 0.0:
                    # add the orthogonality constraint to the loss
                    loss += constraint * orthogonality_weight
                    total_constraint += constraint.value()
                if adversarial:
                    total_adversarial += adv.value()
                    loss += adv

                total_loss += loss.value()
                loss.backward()
                self.trainer.update()
                bar.next()

            print("\niter {}. Total loss: {:.3f}, total penalty: {:.3f}, adv: {:.3f}".format(
                cur_iter, total_loss / len(train_data), total_constraint / len(train_data),
                total_adversarial / len(train_data)), file=sys.stderr)

            if val_X is not None and val_Y is not None and model_path is not None:
                # get the best F1 score on the validation set
                val_f1 = self.evaluate(val_X, val_Y, 'F0')

                if val_f1 > best_val_f1:
                    print('F1 %.4f is better than best val F1 %.4f.' % (val_f1, best_val_f1))
                    best_val_f1 = val_f1
                    epochs_no_improvement = 0
                    save_mttri_model(self, model_path)
                else:
                    print('F1 %.4f is worse than best val F1 %.4f.' % (val_f1, best_val_f1))
                    epochs_no_improvement += 1
                if epochs_no_improvement == patience:
                    print('No improvement for %d epochs. Early stopping...' % epochs_no_improvement)
                    break

    def predict(self, feature_vector, task_ids, train=False, soft_labels=False,
                temperature=None, dropout_rate=0.0, orthogonality_weight=0.0,
                domain_id=None):
        dynet.renew_cg()  # new graph

        feature_vector = feature_vector.toarray()
        feature_vector = np.squeeze(feature_vector, axis=0)

        # self.input = dynet.vecInput(self.vocab_size)
        # self.input.set(feature_vector)
        # TODO this takes too long; can we speed this up somehow?
        input = dynet.inputVector(feature_vector)
        for i in range(self.h_layers):
            if train:  # add some noise
                input = dynet.noise(input, self.noise_sigma)
                input = dynet.dropout(input, dropout_rate)
            input = self.layers[i](input)
        outputs = []
        for task_id in task_ids:
            output = self.output_layers_dict[task_id](input, soft_labels=soft_labels,
                                                      temperature=temperature)
            outputs.append(output)

        constraint, adv_loss = 0, 0
        if orthogonality_weight != 0:
            # put the orthogonality constraint either directly on the
            # output layer or on the hidden layer if it's an MLP
            F0_layer = self.output_layers_dict["F0"]
            F1_layer = self.output_layers_dict["F1"]
            F0_param = F0_layer.W_mlp if self.add_hidden else F0_layer.W
            F1_param = F1_layer.W_mlp if self.add_hidden else F1_layer.W
            F0_W = dynet.parameter(F0_param)
            F1_W = dynet.parameter(F1_param)

            # calculate the matrix product of the task matrix with both others
            matrix_product = dynet.transpose(F0_W) * F1_W

            # take the squared Frobenius norm by squaring
            # every element and then summing them
            squared_frobenius_norm = dynet.sum_elems(
                dynet.square(matrix_product))
            constraint += squared_frobenius_norm
            # print('Constraint with first matrix:', squared_frobenius_norm.value())

        if domain_id is not None:
            # flip the gradient when back-propagating through here
            adv_input = dynet.flip_gradient(input)  # last state
            adv_output = self.adv_layer(adv_input)
            adv_loss = self.pick_neg_log(adv_output, domain_id)
            # print('Adversarial loss:', avg_adv_loss.value())
        return outputs, constraint, adv_loss

    def get_predictions(self, test_X, task_id, soft_labels=False):
        predictions = []
        for x in test_X:
            outputs, _, _ = self.predict(x, [task_id])
            # print('Prediction:', o.value())
            predictions.append(outputs[0].value() if soft_labels
                               else int(np.argmax(outputs[0].value())))
        return predictions

    def evaluate(self, test_X, test_Y, task_id):
        preds = self.get_predictions(test_X, task_id)
        return accuracy_score(test_Y, preds)
