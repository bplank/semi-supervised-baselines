#!/usr/bin/env python3
# coding=utf-8
"""
MT-TRI tagger

:author: Barbara Plank

"""
import random
import sys
import numpy as np
import os
import pickle
import _dynet as dynet
from collections import Counter

from progress.bar import Bar

from lib.mnnl import FFSequencePredictor, Layer, RNNSequencePredictor, BiRNNSequencePredictor
from lib.mmappers import TRAINER_MAP
from tagger.lib.mio import read_conll_file, load_embeddings_file


def load_tagger(path_to_model):
    """
    load a model from file; specify the .model file, it assumes the *pickle file in the same location
    """
    myparams = pickle.load(open(path_to_model + ".params.pickle", "rb"))
    tagger = MttriTagger(myparams["in_dim"],
                        myparams["h_dim"],
                        myparams["c_in_dim"],
                        myparams["h_layers"],
                        activation=myparams["activation"],
                        add_hidden=myparams["add_hidden"],
                        adversarial_domains=myparams["adversarial_domains"])
    tagger.set_indices(myparams["w2i"],myparams["c2i"],myparams["tag2idx"])
    tagger.initialize_graph()
    if myparams["adversarial_domains"]:
        tagger.add_adversarial_loss(myparams["adversarial_domains"])
    tagger.model.populate(path_to_model + '.model')
    print("model loaded: {}".format(path_to_model), file=sys.stderr)
    return tagger


def save_tagger(nntagger, path_to_model):
    """
    save a model; dynet only saves the parameters, need to store the rest separately
    """
    modelname = path_to_model + ".model"
    nntagger.model.save(modelname)
    myparams = {"w2i": nntagger.w2i,
                "c2i": nntagger.c2i,
                "tag2idx": nntagger.tag2idx,
                "activation": nntagger.activation,
                "in_dim": nntagger.in_dim,
                "h_dim": nntagger.h_dim,
                "c_in_dim": nntagger.c_in_dim,
                "h_layers": nntagger.h_layers,
                "adversarial_domains": nntagger.adversarial_domains,
                "add_hidden": nntagger.add_hidden
                }
    pickle.dump(myparams, open(path_to_model + ".params.pickle", "wb" ) )
    print("model stored: {}".format(modelname), file=sys.stderr)


class MttriTagger(object):

    def __init__(self,in_dim,h_dim,c_in_dim,h_layers,embeds_file=None,
                 activation=dynet.tanh, noise_sigma=0.1,
                 word2id=None, add_hidden=False,trainer="adam", clip_threshold=5.0, learning_rate=0.001,
                 adversarial_domains=None):
        self.w2i = {} if word2id is None else word2id  # word to index mapping
        self.c2i = {}  # char to index mapping
        self.tag2idx = {} # tag to tag_id mapping
        self.model = dynet.ParameterCollection()  # init model
        # init trainer
        train_algo = TRAINER_MAP[trainer]
        self.trainer = train_algo(self.model, learning_rate)
        if clip_threshold:
            self.trainer.set_clip_threshold(clip_threshold)
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.c_in_dim = c_in_dim
        self.activation = activation
        self.noise_sigma = noise_sigma
        self.h_layers = h_layers
        self.predictors = {"inner": [], "output_layers_dict": {}, "task_expected_at": {} } # the inner layers and predictors
        self.wembeds = None # lookup: embeddings for words
        self.cembeds = None # lookup: embeddings for characters
        self.embeds_file = embeds_file
        self.char_rnn = None # RNN for character input
        self.task_ids = ["F0", "F1", "Ft"]
        self.add_hidden = add_hidden
        self.adversarial_domains=adversarial_domains

    def add_adversarial_loss(self, num_domains=2):
        if not self.adversarial_domains:  # make sure they are set the latest here
            self.adversarial_domains = num_domains
        self.adv_layer = Layer(self.model, 2*self.h_dim, num_domains,
                               activation=dynet.softmax, mlp=self.h_dim if self.add_hidden else 0)

    def pick_neg_log(self, pred, gold):
        if not isinstance(gold, int):
            # calculate cross-entropy loss against the whole vector
            dy_gold = dynet.inputVector(gold)
            return -dynet.sum_elems(dynet.cmult(dy_gold, dynet.log(pred)))
        return -dynet.log(dynet.pick(pred, gold))

    def set_indices(self, w2i, c2i, tag2idx):
        self.tag2idx= tag2idx
        self.w2i = w2i
        self.c2i = c2i

    def fit(self, train_dict, num_epochs,
            val_X=None, val_Y=None, patience=2, model_path=None, seed=None,
            word_dropout_rate=0.25, trg_vectors=None,
            unsup_weight=1.0, clip_threshold=5.0,
            orthogonality_weight=0.0, adversarial=False, adversarial_weight=1.0, ignore_src_Ft=False):
        """
        train the tagger
        :param trg_vectors: the prediction targets used for the unsupervised loss
                            in temporal ensembling
        :param unsup_weight: weight for the unsupervised consistency loss
                                    used in temporal ensembling
        :param adversarial: note: if we want to use adversarial, we have to
                            call add_adversarial_loss before;
        :param adversarial_weight: 1 by default (do not weigh adv loss)
        :param ignore_src_Ft: if asymm.tri. 2nd stage, do not further train Ft on 'src'
        :param train_dict: a dictionary mapping tasks ("F0", "F1", and "Ft")
                           to a dictionary
                           {"X": list of examples,
                            "Y": list of labels,
                            "domain": list of domain tag (0,1) of example}
        Three tasks are indexed as "F0", "F1" and "Ft"

        Note: if a task 'src' is given than a single model with three heads is trained where
        all data is given to all outputs
        """
        print("read training data")

        widCount = Counter()
        train_data = []
        for task, task_dict in train_dict.items(): #task: eg. "F0"
            for key in ["X", "Y", "domain"]:
                assert key in task_dict, "Error: %s is not available." % key
            examples, labels, domain_tags = task_dict["X"], task_dict["Y"], task_dict["domain"]
            assert len(examples) == len(labels)
            if word_dropout_rate > 0.0:
                # keep track of the counts for word dropout
                for sentence, _ in examples:
                    widCount.update([w for w in sentence])

            # train data is a list of 4-tuples: (example, label, task_id, domain_id)
            train_data += list(zip(examples, labels, [[task]*len(labels)][0], domain_tags))

        # if we use target vectors, keep track of the targets per sentence
        if trg_vectors is not None:
            trg_start_id = 0
            sentence_trg_vectors = []
            for i, (example, y) in enumerate(train_data):
                sentence_trg_vectors.append(trg_vectors[trg_start_id:trg_start_id+len(example[0]), :])
                trg_start_id += len(example[0])
            assert trg_start_id == len(trg_vectors),\
                'Error: Idx {} is not at {}.'.format(trg_start_id, len(trg_vectors))

        print('Starting training for {} epochs...'.format(num_epochs))
        best_val_acc, epochs_no_improvement = 0., 0
        if val_X is not None and val_Y is not None and model_path is not None:
            print('Using early stopping with patience of {}...'.format(patience))

        if seed:
            random.seed(seed)

        for cur_iter in range(num_epochs):
            bar = Bar('Training epoch {}/{}...'.format(cur_iter + 1, num_epochs),
                      max=len(train_data), flush=True)

            random_indices = np.arange(len(train_data))
            random.shuffle(random_indices)

            total_loss, total_tagged, total_constraint, total_adversarial = 0.0, 0.0, 0.0, 0.0
            total_orth_constr = 0 # count how many updates

            # log separate losses
            log_losses = {}
            log_total = {}
            for task_id in self.task_ids:
                log_losses[task_id] = 0.0
                log_total[task_id] = 0


            for i, idx in enumerate(random_indices):
                (word_indices, char_indices), y, task_id, domain_id = train_data[idx]

                if word_dropout_rate > 0.0:
                    word_indices = [self.w2i["_UNK"] if
                                        (random.random() > (widCount.get(w)/(word_dropout_rate+widCount.get(w))))
                                        else w for w in word_indices]

                output, constraint, adv = self.predict(
                    word_indices, char_indices, task_id, train=True,
                    orthogonality_weight=orthogonality_weight,
                    domain_id=domain_id if adversarial else None)

                if task_id not in ['src','trg']:

                    if len(y) == 1 and y[0] == 0:
                        # in temporal ensembling, we assign a dummy label of [0] for
                        # unlabeled sequences; we skip the supervised loss for these
                        loss = dynet.scalarInput(0)
                    else:
                        loss = dynet.esum([self.pick_neg_log(pred,gold) for
                                              pred, gold in zip(output, y)])

                    if trg_vectors is not None:
                        # the consistency loss in temporal ensembling is used for
                        # both supervised and unsupervised input
                        targets = sentence_trg_vectors[idx]
                        assert len(output) == len(targets)
                        other_loss = unsup_weight * dynet.average(
                            [dynet.squared_distance(o, dynet.inputVector(t))
                             for o, t in zip(output, targets)])
                        loss += other_loss

                    if orthogonality_weight != 0.0 and task_id != 'Ft':
                        # add the orthogonality constraint to the loss
                        total_constraint += constraint.value() * orthogonality_weight
                        total_orth_constr += 1
                        loss += constraint * orthogonality_weight

                    if adversarial:
                        total_adversarial += adv.value() * adversarial_weight
                        loss += adv * adversarial_weight

                    total_loss += loss.value() # for output

                    log_losses[task_id] += total_loss
                    total_tagged += len(word_indices)
                    log_total[task_id] += total_tagged


                    loss.backward()
                    self.trainer.update()
                    bar.next()
                else:
                    # bootstrap=False, the output contains list of outputs one for each task
                    assert trg_vectors is None, 'temporal ensembling not implemented for bootstrap=False'
                    loss = dynet.scalarInput(1) #initialize
                    if ignore_src_Ft:
                        output = output[:-1] # ignore last = Ft when further training with 'src'

                    for t_i, output_t in enumerate(output): # get loss for each task
                        loss += dynet.esum([self.pick_neg_log(pred, gold) for
                                           pred, gold in zip(output_t, y)])
                        task_id = self.task_ids[t_i]
                        log_losses[task_id] += total_loss
                        log_total[task_id] += total_tagged

                    if orthogonality_weight != 0.0:
                        # add the orthogonality constraint to the loss
                        total_constraint += constraint.value() * orthogonality_weight
                        total_orth_constr += 1
                        loss += constraint * orthogonality_weight

                    if adversarial:
                        total_adversarial += adv.value() * adversarial_weight
                        loss += adv * adversarial_weight

                    total_loss += loss.value() # for output
                    total_tagged += len(word_indices)

                    loss.backward()
                    self.trainer.update()
                    bar.next()

            if adversarial and orthogonality_weight:
                print("\niter {}. Total loss: {:.3f}, total penalty: {:.3f}, total weighted adv loss: {:.3f}".format(
                    cur_iter, total_loss/total_tagged, total_constraint/total_orth_constr, total_adversarial/total_tagged
                ), file=sys.stderr)
            elif orthogonality_weight:
                print("\niter {}. Total loss: {:.3f}, total penalty: {:.3f}".format(
                    cur_iter, total_loss/total_tagged, total_constraint/total_orth_constr), file=sys.stderr)
            else:
                print("\niter {}. Total loss: {:.3f} ".format(cur_iter, total_loss/total_tagged), file=sys.stderr)

            for task_id in self.task_ids:
                if log_total[task_id] > 0:
                    print("{0}: {1:.3f}".format(task_id, log_losses[task_id]/ log_total[task_id]))

            if val_X is not None and val_Y is not None and model_path is not None:
                # get the best accuracy on the validation set
                val_correct, val_total = self.evaluate(val_X, val_Y)
                val_accuracy = val_correct / val_total

                if val_accuracy > best_val_acc:
                    print('Accuracy {:.4f} is better than best val accuracy {:.4f}.'.format(val_accuracy, best_val_acc))
                    best_val_acc = val_accuracy
                    epochs_no_improvement = 0
                    save_tagger(self, model_path)
                else:
                    print('Accuracy {:.4f} is worse than best val loss {:.4f}.'.format(val_accuracy, best_val_acc))
                    epochs_no_improvement += 1
                if epochs_no_improvement == patience:
                    print('No improvement for {} epochs. Early stopping...'.format(epochs_no_improvement))
                    break

    def initialize_graph(self, num_words=None, num_chars=None):
        """
        build graph and link to parameters

        F2=True: activate second auxiliary output
        Ft=True: activate third auxiliary output

        """
        num_words = num_words if num_words is not None else len(self.w2i)
        num_chars = num_chars if num_chars is not None else len(self.c2i)
        if num_words == 0 or num_chars == 0:
            raise ValueError('Word2id and char2id have to be loaded before '
                             'initializing the graph.')
        print('Initializing the graph...')

        # initialize the word embeddings and the parameters
        self.cembeds = None
        if self.embeds_file:
            print("loading embeddings", file=sys.stderr)
            embeddings, emb_dim = load_embeddings_file(self.embeds_file)
            assert(emb_dim==self.in_dim)
            num_words=len(set(embeddings.keys()).union(set(self.w2i.keys()))) # initialize all with embeddings
            # init model parameters and initialize them
            self.wembeds = self.model.add_lookup_parameters(
                (num_words, self.in_dim),init=dynet.ConstInitializer(0.01))

            if self.c_in_dim > 0:
                self.cembeds = self.model.add_lookup_parameters(
                    (num_chars, self.c_in_dim),init=dynet.ConstInitializer(0.01))
               
            init=0
            l = len(embeddings.keys())
            for word in embeddings.keys():
                # for those words we have already in w2i, update vector, otherwise add to w2i (since we keep data as integers)
                if word in self.w2i:
                    self.wembeds.init_row(self.w2i[word], embeddings[word])
                else:
                    self.w2i[word]=len(self.w2i.keys()) # add new word
                    self.wembeds.init_row(self.w2i[word], embeddings[word])
                init+=1
            print("initialized: {}".format(init), file=sys.stderr)

        else:
            self.wembeds = self.model.add_lookup_parameters(
                (num_words, self.in_dim),init=dynet.ConstInitializer(0.01))
            if self.c_in_dim > 0:
                self.cembeds = self.model.add_lookup_parameters(
                    (num_chars, self.c_in_dim),init=dynet.ConstInitializer(0.01))

        # make it more flexible to add number of layers as specified by parameter
        layers = [] # inner layers

        for layer_num in range(0,self.h_layers):

            if layer_num == 0:
                if self.c_in_dim > 0:
                    f_builder = dynet.CoupledLSTMBuilder(1, self.in_dim+self.c_in_dim*2, self.h_dim, self.model) # in_dim: size of each layer
                    b_builder = dynet.CoupledLSTMBuilder(1, self.in_dim+self.c_in_dim*2, self.h_dim, self.model) 
                else:
                    f_builder = dynet.CoupledLSTMBuilder(1, self.in_dim, self.h_dim, self.model)
                    b_builder = dynet.CoupledLSTMBuilder(1, self.in_dim, self.h_dim, self.model)
                layers.append(BiRNNSequencePredictor(f_builder, b_builder)) #returns forward and backward sequence
            else:
                # add inner layers (if h_layers >1)
                f_builder = dynet.LSTMBuilder(1, self.h_dim, self.h_dim, self.model)
                b_builder = dynet.LSTMBuilder(1, self.h_dim, self.h_dim, self.model)
                layers.append(BiRNNSequencePredictor(f_builder,b_builder))

        # store at which layer to predict task
        task_num_labels= len(self.tag2idx)
        output_layers_dict = {}
        output_layers_dict["F0"] = FFSequencePredictor(Layer(
            self.model, self.h_dim*2, task_num_labels, dynet.softmax,
            mlp=self.h_dim if self.add_hidden else 0))

        # for simplicity always add additional outputs, even if they are then not used
        output_layers_dict["F1"] = FFSequencePredictor(
                Layer(self.model, self.h_dim * 2, task_num_labels, dynet.softmax,
                      mlp=self.h_dim if self.add_hidden else 0))

        output_layers_dict["Ft"] = FFSequencePredictor(
                Layer(self.model, self.h_dim * 2, task_num_labels, dynet.softmax,
                      mlp=self.h_dim if self.add_hidden else 0))
        if self.c_in_dim > 0:
            self.char_rnn = BiRNNSequencePredictor(
                dynet.CoupledLSTMBuilder(
                    1, self.c_in_dim, self.c_in_dim, self.model),
                dynet.CoupledLSTMBuilder(
                    1, self.c_in_dim, self.c_in_dim, self.model))
        else:
            self.char_rnn = None

        self.predictors = dict()
        self.predictors["inner"] = layers
        self.predictors["output_layers_dict"] = output_layers_dict
        self.predictors["task_expected_at"] = self.h_layers

    def get_features(self, words):
        """
        from a list of words, return the word and word char indices
        """
        word_indices = []
        word_char_indices = []
        for word in words:
            if word in self.w2i:
                word_indices.append(self.w2i[word])
            else:
                word_indices.append(self.w2i["_UNK"])

            if self.c_in_dim > 0:
                chars_of_word = [self.c2i["<w>"]]
                for char in word:
                    if char in self.c2i:
                        chars_of_word.append(self.c2i[char])
                    else:
                        chars_of_word.append(self.c2i["_UNK"])
                chars_of_word.append(self.c2i["</w>"])
                word_char_indices.append(chars_of_word)
        return word_indices, word_char_indices

    def __get_instances_from_file(self, file_name):
        """
        helper function to convert input file to lists of lists holding input words|tags
        """
        data = [(words, tags) for (words, tags) in list(read_conll_file(file_name))]
        words = [words for (words, _) in data]
        tags = [tags for (_, tags) in data]
        return words, tags

    def get_data_as_indices(self, file_name):
        """
        X = list of (word_indices, word_char_indices)
        Y = list of tag indices
        """
        words, tags = self.__get_instances_from_file(file_name)
        return self.get_data_as_indices_from_instances(words, tags)

    def get_data_as_indices_from_instances(self, dev_words, dev_tags):
        """
        Extension of get_data_as_indices. Use words and tags rather than a file as input.
        X = list of (word_indices, word_char_indices)
        Y = list of tag indices
        """
        X, Y = [], []
        org_X, org_Y = [], []

        for (words, tags) in zip(dev_words, dev_tags):
            word_indices, word_char_indices = self.get_features(words)
            # if tag does not exist in source domain tags, return as default
            # first idx outside of dictionary
            tag_indices = [self.tag2idx.get(
                tag, len(self.tag2idx)) for tag in tags]
            X.append((word_indices, word_char_indices))
            Y.append(tag_indices)
            org_X.append(words)
            org_Y.append(tags)
        return X, Y  # , org_X, org_Y - for now don't use

    def predict(self, word_indices, char_indices, task_id, train=False,
                soft_labels=False, temperature=None, orthogonality_weight=0.0,
                domain_id=None):
        """
        predict tags for a sentence represented as char+word embeddings
        :param domain_id: Predict adversarial loss if domain id is provided.
        """
        dynet.renew_cg() # new graph

        char_emb = []
        rev_char_emb = []

        wfeatures = [self.wembeds[w] for w in word_indices]

        if self.c_in_dim > 0:
            # get representation for words
            for chars_of_token in char_indices:
                char_feats = [self.cembeds[c] for c in chars_of_token]
                # use last state as word representation
                f_char, b_char = self.char_rnn.predict_sequence(char_feats, char_feats)
                last_state = f_char[-1]
                rev_last_state = b_char[-1]
                char_emb.append(last_state)
                rev_char_emb.append(rev_last_state)

            features = [dynet.concatenate([w,c,rev_c]) for w,c,rev_c in zip(wfeatures,char_emb,rev_char_emb)]
        else:
            features = wfeatures
        
        if train: # only do at training time
            features = [dynet.noise(fe,self.noise_sigma) for fe in features]

        output_expected_at_layer = self.h_layers
        output_expected_at_layer -=1

        # go through layers
        prev = features
        prev_rev = features
        num_layers = self.h_layers
        constraint = 0
        adv_loss = 0
        for i in range(0,num_layers):
            predictor = self.predictors["inner"][i]
            forward_sequence, backward_sequence = predictor.predict_sequence(prev, prev_rev)
            if i > 0 and self.activation:
                # activation between LSTM layers
                forward_sequence = [self.activation(s) for s in forward_sequence]
                backward_sequence = [self.activation(s) for s in backward_sequence]

            if i == output_expected_at_layer:

                concat_layer = [dynet.concatenate([f, b]) for f, b in zip(forward_sequence,reversed(backward_sequence))]
                if train and self.noise_sigma > 0.0:
                    concat_layer = [dynet.noise(fe,self.noise_sigma) for fe in concat_layer]

                if task_id not in ["src","trg"]:
                    output_predictor = self.predictors["output_layers_dict"][task_id]
                    output = output_predictor.predict_sequence(
                        concat_layer, soft_labels=soft_labels,
                        temperature=temperature)
                else:
                    # one src example for all three outputs
                    output = []  # in this case it is a list
                    for t_id in self.task_ids:
                        output_predictor = self.predictors["output_layers_dict"][t_id]
                        output_t = output_predictor.predict_sequence(
                            concat_layer, soft_labels=soft_labels,
                            temperature=temperature)
                        output.append(output_t)

                if orthogonality_weight != 0 and task_id != "Ft":
                    # put the orthogonality constraint either directly on the
                    # output layer or on the hidden layer if it's an MLP
                    # use orthogonality_weight only between F0 and F1
                    builder = self.predictors["output_layers_dict"]["F0"].network_builder
                    task_param = builder.W_mlp if self.add_hidden else builder.W
                    task_W = dynet.parameter(task_param)

                    builder = self.predictors["output_layers_dict"]["F1"].network_builder
                    other_param = builder.W_mlp if self.add_hidden else builder.W
                    other_task_W = dynet.parameter(other_param)

                    # calculate the matrix product of the task matrix with the other
                    matrix_product_1 = dynet.transpose(task_W) * other_task_W

                    # take the squared Frobenius norm by squaring
                    # every element and then summing them
                    squared_frobenius_norm = dynet.sum_elems(dynet.square(matrix_product_1))
                    constraint = squared_frobenius_norm

                    #print('Constraint with first matrix:', squared_frobenius_norm.value())

                if domain_id is not None:
                    # flip the gradient when back-propagating through here
                    adv_input = dynet.flip_gradient(concat_layer[-1]) # last state
                    adv_output = self.adv_layer(adv_input)
                    adv_loss = self.pick_neg_log(adv_output, domain_id)
                    #print('Adversarial loss:', avg_adv_loss.value())

                # output is list if task_id = 'src'
                return output, constraint, adv_loss

            prev = forward_sequence
            prev_rev = backward_sequence

        raise Exception("oops should not be here")
        return None

    def evaluate(self, test_X, test_Y, task_id="F0"):
        """
        compute accuracy on a test file; by default use "F0" as predictor
        """
        correct = 0
        total = 0.0

        for i, ((word_indices, word_char_indices), gold_tag_indices) in enumerate(zip(test_X, test_Y)):

            output, _ , _= self.predict(word_indices, word_char_indices, task_id)
            predicted_tag_indices = [np.argmax(o.value()) for o in output]

            correct += sum([1 for (predicted, gold) in zip(predicted_tag_indices, gold_tag_indices) if predicted == gold])
            total += len(gold_tag_indices)

        return correct, total

    def get_predictions(self, test_X, soft_labels=False, task_id="F0"):
        """
        get flat list of predictions
        """
        predictions = []
        for word_indices, word_char_indices in test_X:
            output, _ ,_ = self.predict(word_indices, word_char_indices, task_id)
            predictions += [o.value() if soft_labels else
                            int(np.argmax(o.value())) for o in output]
        return predictions

    def get_train_data_from_instances(self, train_words, train_tags):
        """
        Extension of get_train_data method. Extracts training data from two arrays of word and label lists.
        transform training data to features (word indices)
        map tags to integers
        :param train_words: a numpy array containing lists of words
        :param train_tags: a numpy array containing lists of corresponding tags
        """
        X = []
        Y = []

        # check if we continue training
        continue_training = False
        if self.w2i and self.tag2idx:
            continue_training = True

        if continue_training:
            print("update existing vocabulary")
            # fetch already existing
            w2i = self.w2i.copy()
            c2i = self.c2i.copy()
            tag2idx = self.tag2idx

            assert w2i["_UNK"] == 0, "No _UNK found!"
        else:
            # word 2 indices and tag 2 indices
            w2i = self.w2i.copy()  # get a copy that refers to a different object
            c2i = {}  # char to index
            tag2idx = {}  # tag2idx

            if len(w2i) > 0:
                assert w2i["_UNK"] == 0
            else:
                w2i["_UNK"] = 0  # unk word / OOV

            c2i["_UNK"] = 0  # unk char
            c2i["<w>"] = 1  # word start
            c2i["</w>"] = 2  # word end index

        num_sentences = 0
        num_tokens = 0
        for instance_idx, (words, tags) in enumerate(zip(train_words, train_tags)):
            instance_word_indices = []  # sequence of word indices
            instance_char_indices = []  # sequence of char indices
            instance_tags_indices = []  # sequence of tag indices

            for i, (word, tag) in enumerate(zip(words, tags)):
                # map words and tags to indices
                if word not in w2i:
                    w2i[word] = len(w2i)
                    instance_word_indices.append(w2i[word])
                else:
                    instance_word_indices.append(w2i[word])

                chars_of_word = [c2i["<w>"]]
                for char in word:
                    if char not in c2i:
                        c2i[char] = len(c2i)
                    chars_of_word.append(c2i[char])
                chars_of_word.append(c2i["</w>"])
                instance_char_indices.append(chars_of_word)

                if tag not in tag2idx:
                    tag2idx[tag] = len(tag2idx)

                instance_tags_indices.append(tag2idx.get(tag))

                num_tokens += 1

            num_sentences += 1

            X.append((instance_word_indices,
                      instance_char_indices))  # list of word indices, for every word list of char indices
            Y.append(instance_tags_indices)

        print("%s sentences %s tokens" % (num_sentences, num_tokens), file=sys.stderr)
        print("%s w features, %s c features " % (len(w2i), len(c2i)), file=sys.stderr)

        assert (len(X) == len(Y))

        # store mappings of words and tags to indices
        self.set_indices(w2i, c2i, tag2idx)

        return X, Y

    def get_train_data(self, train_data):
        """
        transform training data to features (word indices)
        map tags to integers
        """
        train_words, train_tags = self.__get_instances_from_file(train_data)
        return self.get_train_data_from_instances(train_words, train_tags)

    def get_predictions_output(self, test_X, test_labels, output_filename, task_id="F0"):
        """
        get predictions to output to file
        assume test_labels are not indices (as target domain can have tags that are not in source)
        text_X: indices
        test_labels: original labels
        """
        i2w = {self.w2i[w]: w for w in self.w2i.keys()}
        i2t = {self.tag2idx[t]: t for t in self.tag2idx.keys()}

        OUT = open(output_filename, "w")
        for (word_indices, word_char_indices), gold_tags in zip(test_X, test_labels):
            output, _, _ = self.predict(word_indices, word_char_indices, task_id)
            predicted_tag_ids = [int(np.argmax(o.value())) for o in output]

            for word_id, tag_id, gold_tag in zip(word_indices, predicted_tag_ids, gold_tags):
                known_tag_prefix = "{}" if gold_tag in self.tag2idx else "*{}"
                word, pred_tag, gold_tag = i2w[word_id], i2t[tag_id], known_tag_prefix.format(gold_tag)
                OUT.write("{}\t{}\t{}\n".format(word, gold_tag, pred_tag))
            OUT.write("\n")
        OUT.close()
