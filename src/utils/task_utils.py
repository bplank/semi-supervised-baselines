"""
Utility methods that are used for training and evaluation of the different tasks.
"""
import numpy as np
import operator
import os
import json
from tagger.simplebilty import SimpleBiltyTagger, load_tagger, save_tagger
from tagger.mttri import MttriTagger
from tagger.mttri import save_tagger as mttri_save_tagger
from tagger.mttri import load_tagger as mttri_load_tagger
from collections import namedtuple, Counter, defaultdict
from sklearn.metrics import accuracy_score
import scipy.sparse
import scipy.stats
from progress.bar import Bar


from utils import sentiment_model
from utils.data_readers import POS_ID, NEG_ID
from utils import data_utils
from utils.vocab import Vocab

from utils.constants import SENTIMENT, POS, RANDOM, EPOCH_1,\
    PATIENCE_2, SELF_TRAINING, TRI_TRAINING, BASE, TEMPORAL_ENSEMBLING, MTTRI_BASE,\
    MTTRI, TASK_IDS

NUM_EPOCHS = 50
PATIENCE = 2


def task2train_func(task, strategy):
    if task == POS:
        if strategy == SELF_TRAINING:
            return self_training_pos
        if strategy == TRI_TRAINING:
            return tri_training_pos
        if strategy == BASE:
            return train_base_model_pos
        if strategy == MTTRI_BASE:
            return mttri_base
        if strategy == MTTRI:
            return mttri_training_pos
        if strategy == TEMPORAL_ENSEMBLING:
            return temporal_ensembling_pos
    if task == SENTIMENT:
        if strategy == BASE:
            return train_base_model_sentiment
        if strategy == SELF_TRAINING:
            return self_training_sentiment
        if strategy == TRI_TRAINING:
            return tri_training_sentiment
        if strategy == TEMPORAL_ENSEMBLING:
            return temporal_ensembling_sentiment
        if strategy == MTTRI_BASE:
            return mttri_base_sentiment
        if strategy == MTTRI:
            return mttri_training_sentiment
    raise NotImplementedError('%s is not implemented for %s' % (strategy, task))


def get_data_subsets(feature_vals, feature_weights, train_data, train_labels, task, num_train_examples):
    """
    Given the feature values and the feature weights, return the stratified subset of the training data with the highest
    feature scores.
    :param feature_vals: a numpy array of shape (num_train_data, num_features) containing the feature values
    :param feature_weights: a numpy array of shape (num_features, ) containing the weight for each feature
    :param train_data: a sparse numpy array of shape (num_train_data, vocab_size) containing the training data
    :param train_labels: a numpy array of shape (num_train_data) containing the training labels
    :param task: the task; this determines whether we use stratification
    :param num_train_examples: the number of training examples for the respective task
    :return: subsets of the training data and its labels as a tuple of two numpy arrays
    """
    # calculate the scores as the dot product between feature values and weights
    scores = feature_vals.dot(np.transpose(feature_weights))

    # sort the indices by their scores
    sorted_index_score_pairs = sorted(zip(range(len(scores)), scores), key=operator.itemgetter(1), reverse=True)

    # get the top indices
    top_indices, _ = zip(*sorted_index_score_pairs)

    if task == 'sentiment':
        # for sentiment, rather than taking the top n indices, we still want to have a stratified training set
        # so we take the top n/2 positive and top n/2 negative indices
        top_pos_indices = [idx for idx in top_indices if train_labels[idx] == POS_ID][:int(num_train_examples/2)]
        top_neg_indices = [idx for idx in top_indices if train_labels[idx] == NEG_ID][:int(num_train_examples/2)]
        top_indices = top_pos_indices + top_neg_indices
    elif task in ['pos', 'pos_bilstm', 'parsing']:
        # for part-of-speech tagging and parsing, we don't need an exactly stratified training set
        top_indices = list(top_indices[:num_train_examples])
    else:
        raise ValueError('Top index retrieval has not been implemented for task %s.' % task)

    # we get the corresponding subsets of the training data and the labels
    return top_indices # return only indices to allow logging
    #return train_data[top_indices], train_labels[top_indices]


def load_config(config):
    """
    load configuration file from json
    convert to namedtuple so that it can be accessed with config.in_dim etc
    """
    d = json.load(open(config))
    config = namedtuple("options", d.keys())(*d.values())
    return config


def train_base_model_pos(vocab, args, train_data, train_labels, val_data,
                         val_labels, dev_test_data, dev_test_labels,
                         test_data, test_labels, unlabeled_data,
                         unlabeled_labels, run_num):
    """
    Train baseline models
    If args.bootstrap: train 3 bootstrap sample models (used for tri-training)
    if args.save_epoch_1: also store model after first epoch
    """
    # initialize the config
    options = load_config(args.config)
    print(options)

    dev_test_accs = []
    test_accs = []

    iterations = 3 if args.bootstrap else 1
    for i in range(1, iterations + 1):
        tagger = SimpleBiltyTagger(options.in_dim, options.h_dim, options.c_in_dim, options.h_layers,
                                   embeds_file=options.embeds, word2id=vocab.word2id,
                                   trainer="adam", clip_threshold=options.clip_threshold)
        train_X, train_Y = tagger.get_train_data_from_instances(train_data, train_labels)
        # initialize the graph after reading the training data and before reading rest (in case we have embeds)
        tagger.initialize_graph()
        print("len vocab",len(vocab.word2id), len(tagger.w2i))
        if len(vocab.word2id) != len(tagger.w2i):
            print("update vocabulary") # if we have embeds
            vocab_dir = args.model_dir if args.strategy == BASE else args.start_model_dir
            vocab_path = os.path.join(vocab_dir, 'vocab.txt')
            vocab = Vocab(vocab_path, max_vocab_size=args.max_vocab_size)
            vocab.create_from_w2id(tagger.w2i) # update vocabulary

        val_X, val_Y = tagger.get_data_as_indices_from_instances(val_data,
                                                                 val_labels)
        dev_test_X, dev_test_Y = tagger.get_data_as_indices_from_instances(dev_test_data,
                                                                 dev_test_labels)
        test_X, test_Y = tagger.get_data_as_indices_from_instances(test_data,
                                                                   test_labels)

        if args.bootstrap:
            train_X_org, train_Y_org = tagger.get_train_data_from_instances(train_data, train_labels)
            val_X_org, val_Y_org = tagger.get_data_as_indices_from_instances(val_data,
                                                                             val_labels)

            print("Create bootstrap sample... {}/3 ".format(i))
            # get bootstrap sample of train and dev, keep test same
            train_X, train_Y = bootstrap_sample(train_X_org, train_Y_org)
            val_X, val_Y = bootstrap_sample(val_X_org, val_Y_org)

            # naming of models - include bootstrap
            model_epoch_1_file = os.path.join(args.model_dir, EPOCH_1 + "_bootstrap" + str(i) +"_run" + str(run_num))
            model_patience_2_file = os.path.join(args.model_dir,
                                                 PATIENCE_2 + "_bootstrap" + str(i) + "_run" + str(run_num))
            model_patience_N_file = os.path.join(args.model_dir, "patience_{}".format(options.patience) + "_bootstrap" + str(i)
                                                 + "_run" + str(run_num))

        else:
            # naming of models
            model_epoch_1_file = os.path.join(args.model_dir, EPOCH_1 + "_run" + str(run_num))
            model_patience_2_file = os.path.join(args.model_dir, PATIENCE_2 + "_run" + str(run_num))
            model_patience_N_file = os.path.join(args.model_dir, "patience_{}".format(options.patience)+ "_run" + str(run_num))

        # flatten labels
        val_y = [y for y_tags in val_Y for y in y_tags]
        dev_test_y = [y for y_tags in dev_test_Y for y in y_tags]
        test_y = [y for y_tags in test_Y for y in y_tags]

        if args.save_epoch_1:
            # train a model for one epoch and save it for later
            tagger.fit(train_X, train_Y, 1, word_dropout_rate=options.word_dropout_rate, seed=args.seed)
            save_tagger(tagger, model_epoch_1_file)

            val_acc = accuracy_score(val_y, tagger.get_predictions(val_X))
            dev_test_acc = accuracy_score(dev_test_y, tagger.get_predictions(dev_test_X))
            test_acc = accuracy_score(test_y, tagger.get_predictions(test_X))
            print('1 Epoch model. Validation acc: %.4f. Dev test acc: %.4f. Test acc: %4f.'
                  % (val_acc, dev_test_acc, test_acc))

            if args.output_predictions:
                # save base outputs
                tagger.get_predictions_output(dev_test_X, dev_test_labels, os.path.join(args.output_predictions, "dev_test_{}_run{}.out".format(EPOCH_1,run_num)))
                tagger.get_predictions_output(test_X, test_labels, os.path.join(args.output_predictions, "test_{}_run{}.out".format(EPOCH_1,run_num)))

        # continue training it with patience 2
        tagger.fit(train_X, train_Y, 100, val_X, val_Y, patience=2,
                   model_path=model_patience_2_file, word_dropout_rate=options.word_dropout_rate, seed=args.seed)
        del tagger # delete old object
        tagger = load_tagger(model_patience_2_file) # make sure we reload! (otherwise it is not patience 2 model)
        val_acc = accuracy_score(val_y, tagger.get_predictions(val_X))
        dev_test_acc = accuracy_score(dev_test_y,
                                      tagger.get_predictions(dev_test_X))
        test_acc = accuracy_score(test_y, tagger.get_predictions(test_X))

        val_correct, val_total = tagger.evaluate(val_X, val_Y)
        val_accuracy = val_correct / val_total
        print("Dev test acc: {0:.2f}".format(val_accuracy)) # just checking

        if args.output_predictions:
            # save base outputs
            tagger.get_predictions_output(dev_test_X, dev_test_labels, os.path.join(args.output_predictions, "dev_test_{}_run{}.out".format(PATIENCE_2,run_num)))
            tagger.get_predictions_output(test_X, test_labels, os.path.join(args.output_predictions, "test_{}_run{}.out".format(PATIENCE_2,run_num)))

        print('Patience 2 model. Validation acc: %.4f. Dev test acc: %.4f. Test acc: %.4f.'
              % (val_acc, dev_test_acc, test_acc))

        if options.patience > 2:
            # run further
            tagger = load_tagger(model_patience_2_file)
            tagger.fit(train_X, train_Y, 100, val_X, val_Y, patience=options.patience,
                       model_path=model_patience_N_file, word_dropout_rate=options.word_dropout_rate, seed=args.seed)
            tagger = load_tagger(model_patience_N_file)
            val_acc = accuracy_score(val_y, tagger.get_predictions(val_X))
            dev_test_acc = accuracy_score(dev_test_y,
                                          tagger.get_predictions(dev_test_X))
            test_acc = accuracy_score(test_y, tagger.get_predictions(test_X))
            print('Patience %s model. Validation acc: %.4f. Dev test acc: %.4f. Test acc: %4f.'
                  % (options.patience, val_acc, dev_test_acc, test_acc))

        # log the results to the log file
        data_utils.log_self_training(args, dev_test_acc, test_acc, run_num=run_num)
        dev_test_accs.append(dev_test_acc)
        test_accs.append(test_acc)

    return np.mean(dev_test_accs), np.mean(test_accs) # return average (over just 1 or 3 bootstrap samples)

def get_training_dict(args, train_dict):
    """
    get training dict for mttri training
    if bootstrap is not used, initializes a 'src' task (for which later all output nodes are trained)
    if predict='org' also initializes a 'trg' task to which all pseudo-labeled instances will be added, which will be given to all outputs
    """
    if args.size_bootstrap == 0.0:
        print('Training all outputs on source samples... skip bootstrap step, reinitialize train_dict')
        # other train_dicts are not initialized, use F0 as source -> but rename to 'src' to avoid confusion with added
        # pseudo-labels later
        new_train_dict = {'src' : defaultdict(list)}
        new_train_dict['src']['X'] = train_dict['F0']['X']
        new_train_dict['src']['Y'] = train_dict['F0']['Y']
        new_train_dict['src']['domain'] = [0] * len(train_dict['F0']['X'])
        train_dict = new_train_dict
    else:
        # get bootstrap sample of train and dev set
        print('Getting bootstrap samples...')
        len_data = len(train_dict['F0']['X'])
        size_bootstrap_sample = int(len_data*args.size_bootstrap)

        train_dict['F1']['X'], train_dict['F1']['Y'] =\
            bootstrap_sample(train_dict['F0']['X'], train_dict['F0']['Y'], size_bootstrap_sample)
        # we don't use the other validation sets at the moment and do
        # early stopping based on the main validation set
        # val2_X, val2_Y = bootstrap_sample(val1_X, val1_Y)
        # initialize also Ft on source
        train_dict['Ft']['X'], train_dict['Ft']['Y'] =\
                bootstrap_sample(train_dict['F0']['X'], train_dict['F0']['Y'], size_bootstrap_sample)
        # val3_X, val3_Y = bootstrap_sample(val1_X, val1_Y)
        train_dict['F0']['X'], train_dict['F0']['Y'] =\
            bootstrap_sample(train_dict['F0']['X'], train_dict['F0']['Y'], size_bootstrap_sample)
        # val1_X, val1_Y = bootstrap_sample(val1_X, val1_Y)

        # add the domain keys (we don't use adversarial training yet as we only have
        # in-domain data)
        train_dict['F0']['domain'] = train_dict['F1']['domain'] = train_dict['Ft']['domain'] = [0] * size_bootstrap_sample
    return train_dict

def mttri_base(vocab, args, train_data, train_labels, val_data,
              val_labels, dev_test_data, dev_test_labels,
              test_data, test_labels, unlabeled_data, unlabeled_labels, run_num):
    """
    train base model for Multi-task tri-training (mttri)
    - runs training data through shared model, training each target output with src data
    - uses orthogonality weight between M1 and M2 (=F0 and F1)
    - bootstrap: not used
    """
    # initialize the model
    options = load_config(args.config)
    print(options)

    task_identities = TASK_IDS
    if args.orthogonality_weight:
        print("using orthogonality_weight: ", args.orthogonality_weight)
    # a dictionary mapping tasks ("F0", "F1", and "Ft") to a dictionary
    # {"X": list of examples, "Y": list of labels, "domain": list of domain
    # tag (0,1) of example}
    train_dict = {t: defaultdict(list) for t in task_identities}

    tagger = MttriTagger(options.in_dim, options.h_dim,
                        options.c_in_dim, options.h_layers,
                        embeds_file=options.embeds,
                        word2id=vocab.word2id,
                        add_hidden=args.add_hidden,
                        trainer="adam", clip_threshold=options.clip_threshold)

    print("len vocab", len(vocab.word2id))
    train_dict['F0']['X'], train_dict['F0']['Y'] = \
        tagger.get_train_data_from_instances(train_data, train_labels)

    # initialize the graph
    tagger.initialize_graph()

    val_X, val_Y = \
        tagger.get_data_as_indices_from_instances(val_data, val_labels)
    dev_test_X, dev_test_Y =\
        tagger.get_data_as_indices_from_instances(dev_test_data, dev_test_labels)
    test_X, test_Y =\
        tagger.get_data_as_indices_from_instances(test_data, test_labels)

    ## get training data - checks if bootstrap is used (in which case it creates 'src')
    train_dict = get_training_dict(args, train_dict)

    print(train_dict.keys())
    # naming of models
    model_epoch_1_file = os.path.join(
        args.model_dir, EPOCH_1 + "_run" + str(run_num))
    model_patience_2_file = os.path.join(
        args.model_dir, PATIENCE_2 + "_run" + str(run_num))

    # flatten labels
    val_y = [y for y_tags in val_Y for y in y_tags]
    dev_test_y = [y for y_tags in dev_test_Y for y in y_tags]
    test_y = [y for y_tags in test_Y for y in y_tags]

    if args.save_epoch_1:
        # train a model for one epoch and save it for later
        tagger.fit(train_dict, 1,
                   word_dropout_rate=options.word_dropout_rate,
                   orthogonality_weight=args.orthogonality_weight
                   )
        mttri_save_tagger(tagger, model_epoch_1_file)

        val_acc = accuracy_score(val_y, tagger.get_predictions(val_X))
        dev_test_acc = accuracy_score(dev_test_y,
                                      tagger.get_predictions(dev_test_X))
        test_acc = accuracy_score(test_y, tagger.get_predictions(test_X))
        print('1 Epoch model. Validation acc: %.4f. Dev test acc: %.4f. Test acc: %4f.'
              % (val_acc, dev_test_acc, test_acc))
        tagger = mttri_load_tagger(model_epoch_1_file)

    # load the 1 epoch model and continue training it with patience 2
    tagger.fit(train_dict, 100,
               val_X=val_X, val_Y=val_Y,
               patience=2,
               model_path=model_patience_2_file,
               word_dropout_rate=options.word_dropout_rate,
               orthogonality_weight=args.orthogonality_weight)
    tagger = mttri_load_tagger(model_patience_2_file)
    val_acc = accuracy_score(val_y, tagger.get_predictions(val_X))
    dev_test_acc = accuracy_score(dev_test_y,
                                  tagger.get_predictions(dev_test_X))
    test_acc = accuracy_score(test_y, tagger.get_predictions(test_X))
    print('Patience 2 model. Validation acc: %.4f. Dev test acc: %.4f. Test acc: %.4f.'
          % (val_acc, dev_test_acc, test_acc))

    # log the results to the log file
    data_utils.log_self_training(args, dev_test_acc, test_acc, run_num=run_num)

    if args.output_predictions:
        tagger.get_predictions_output(dev_test_X, dev_test_labels, os.path.join(args.output_predictions,
                                                                                "mttri_dev_test_run{}.out".format(run_num)))
        tagger.get_predictions_output(test_X, test_labels, os.path.join(args.output_predictions,
                                                                        "mttri_test_run{}.out".format(run_num)))
    return dev_test_acc, test_acc


def mttri_base_sentiment(vocab, args, train_X, train_Y, val_X,
                        val_Y, dev_test_X, dev_test_Y, test_X, test_Y,
                        unlabeled_X, unlabeled_Y, run_num):
    # initialize the model
    options = load_config(args.config)
    print(options)

    # a dictionary mapping tasks ("F0", "F1", and "Ft") to a dictionary
    # {"X": list of examples, "Y": list of labels, "domain": list of domain
    # tag (0,1) of example}
    train_dict = {t: defaultdict(list) for t in ['src_all']}

    print('Using smaller DANN configuration...')
    assert args.orthogonality_weight > 0, 'An orthogonality weight should be used.'
    model = sentiment_model.MttriSentimentModel(
        1, 50, len(vocab.word2id), noise_sigma=0.0, trainer=options.trainer,
        activation='sigmoid')

    # train_dict['F0']['X'], train_dict['F0']['Y'] = train_X, train_Y
    # train_dict['F1']['X'], train_dict['F1']['Y'] = train_X, train_Y
    # train_dict['Ft']['X'], train_dict['Ft']['Y'] = train_X, train_Y
    train_dict['src_all']['X'], train_dict['src_all']['Y'] = train_X, train_Y

    # get bootstrap sample of train and dev set
    # print('Getting bootstrap samples...')
    # train_dict['F1']['X'], train_dict['F1']['Y'] = \
    #     bootstrap_sample(train_dict['F0']['X'], train_dict['F0']['Y'])
    # we don't use the other validation sets at the moment and do
    # early stopping based on the main validation set
    # val2_X, val2_Y = bootstrap_sample(val1_X, val1_Y)
    # train_dict['Ft']['X'], train_dict['Ft']['Y'] = \
    #     bootstrap_sample(train_dict['F0']['X'], train_dict['F0']['Y'])
    # val3_X, val3_Y = bootstrap_sample(val1_X, val1_Y)
    # train_dict['F0']['X'], train_dict['F0']['Y'] = \
    #     bootstrap_sample(train_dict['F0']['X'], train_dict['F0']['Y'])
    # val1_X, val1_Y = bootstrap_sample(val1_X, val1_Y)

    # add the domain keys (we don't use adversarial training yet as we only have
    # in-domain data)
    # train_dict['F0']['domain'] = train_dict['F1']['domain'] = \
    #     train_dict['Ft']['domain'] = [0] * train_dict['F0']['X'].shape[0]
    train_dict['src_all']['domain'] = [0] * train_dict['src_all']['X'].shape[0]

    # naming of models
    model_epoch_1_file = os.path.join(
        args.model_dir, EPOCH_1 + "_run" + str(run_num))
    model_patience_2_file = os.path.join(
        args.model_dir, PATIENCE_2 + "_run" + str(run_num))

    # initialize the graph
    model.initialize_graph()

    if args.save_epoch_1:
        # train a model for one epoch and save it for later
        model.fit(train_dict, 1,
                  word_dropout_rate=options.word_dropout_rate,
                  orthogonality_weight=args.orthogonality_weight)
        sentiment_model.save_mttri_model(model, model_epoch_1_file)

        val_acc = model.evaluate(val_X, val_Y, 'F0')
        test_acc = model.evaluate(test_X, test_Y, 'F0')
        print('1 Epoch model. Validation acc: %.4f. Test acc: %4f.'
              % (val_acc, test_acc))

    # continue training it with patience 2
    model.fit(train_dict, 100,
              val_X=val_X, val_Y=val_Y,
              patience=options.patience,
              model_path=model_patience_2_file,
              word_dropout_rate=options.word_dropout_rate,
              orthogonality_weight=args.orthogonality_weight,
              seed=args.seed)

    model = sentiment_model.load_mttri_model(model_patience_2_file)

    val_acc = model.evaluate(val_X, val_Y, 'F0')
    test_acc = model.evaluate(test_X, test_Y, 'F0')
    print('Patience %d model. Validation acc: %.4f. Test acc: %4f.'
          % (2, val_acc, test_acc))

    # log the results to the log file
    data_utils.log_self_training(args, val_acc, test_acc, run_num=run_num)
    return val_acc, test_acc


def bootstrap_sample(input_X, input_Y, num_examples=None):
    """
    Generate a bootstrap sample of <X,Y> (sample with replacement)
    :param input_X: input X
    :param input_Y: labels X
    :return: list of same size as input_X but sampled with replacement
    """
    input_X = input_X if scipy.sparse.issparse(input_X) else np.array(input_X)
    X_len = input_X.shape[0]
    assert X_len == len(input_Y)
    input_Y = np.array(input_Y)
    if num_examples is None:
        num_examples = X_len
    indices = np.random.choice(range(0,X_len),num_examples,replace=True) # with replacement
    return input_X[indices], input_Y[indices]


def get_candidates(args, original_unlabeled_X, unlabeled_data=None, feature_values=None, domain_labels=None, return_indices=False, epoch_num=-1):
    """
    Select random subset of data from word indiced data (original_unlabeled_X)
    :param original_unlabeled_X: unlabeled data (word, char_indices) # required for random
    :param unlabeled_data: raw unlabeled data (tokens) # required for --candidates-jensen-shannon
    :param feature_values: required JS scores for --candidates-jensen-shannon
    :param epoch_num: required if candidate_pool_scheduling is active
    """
    selected_candidate_pool_size = args.candidate_pool_size
    if args.candidate_pool_scheduling:
        assert epoch_num >= 0, "epoch num needs to be specified when --candidate-pool-scheduling is active"
        N = args.candidate_pool_size
        selected_candidate_pool_size = int((epoch_num+1) / 10 * N)
        print("using adjusted candidate_pool_size: {}".format(selected_candidate_pool_size))
    selected_indices = np.random.choice(list(range(0, len(original_unlabeled_X))),
                                        min(len(original_unlabeled_X), selected_candidate_pool_size), replace=False)
    unlabeled_X = original_unlabeled_X[selected_indices]
    if not return_indices:
        return list(unlabeled_X)
    else:
        return list(unlabeled_X), selected_indices


def self_training_pos(vocab, args, train_data, train_labels, val_data,
                      val_labels, dev_test_data, dev_test_labels,
                      test_data, test_labels, unlabeled_data,
                      unlabeled_labels, run_num):
    assert (len(unlabeled_labels) == 0)
    options = load_config(args.config)

    # load the model from the starting point
    tagger = load_tagger(os.path.join(args.start_model_dir, args.start + "_run" + str(run_num)))
    train_X, train_Y = tagger.get_data_as_indices_from_instances(train_data,
                                                                 train_labels)
    val_X, val_Y = tagger.get_data_as_indices_from_instances(val_data,
                                                             val_labels)
    dev_test_X, dev_test_Y = tagger.get_data_as_indices_from_instances(
        dev_test_data, dev_test_labels)
    test_X, test_Y = tagger.get_data_as_indices_from_instances(test_data,
                                                               test_labels)

    print("eval start tagger")
    dev_test_corr, dev_test_total = tagger.evaluate(dev_test_X, dev_test_Y)
    test_corr, test_total = tagger.evaluate(test_X, test_Y)
    print("Dev test acc: {0:.4f}. Test acc: {1:.4f}.".format(dev_test_corr/dev_test_total, test_corr/test_total))

    # convert raw unlabeled data to indices
    original_unlabeled_X = np.array([tagger.get_features(words) for words in unlabeled_data])
    # convert original data to np array for slicing
    unlabeled_data = np.array(unlabeled_data)

    if not args.candidate_pool_size:
        unlabeled_X = original_unlabeled_X # use all

    # flatten the true validation and test labels
    dev_test_y = [y for y_tags in dev_test_Y for y in y_tags]
    test_y = [y for y_tags in test_Y for y in y_tags]

    # train only on new items
    if args.online:
        train_X, train_Y = [], []

    # stop once no more confident examples have been added in an iteration
    # if specified, also stop after max_iteration
    num_new_examples = len(train_X)
    num_epochs, val_acc, dev_test_acc, test_acc = 0, 0, 0, 0
    while num_new_examples != 0:
        if num_epochs != 0:
            print('Epoch #%d. Training on %d examples. # unlabeled examples: %d.'
                % (num_epochs, len(train_X), len(unlabeled_X)))
            # train the model on the examples selected from the last epoch
            tagger.fit(train_X, train_Y, 1, word_dropout_rate=options.word_dropout_rate)

            # choose hyperparameters on target dev (dev_test)
            dev_test_acc = accuracy_score(dev_test_y,
                                          tagger.get_predictions(dev_test_X))
            test_acc = accuracy_score(test_y, tagger.get_predictions(test_X))

            if args.output_predictions:
                tagger.get_predictions_output(dev_test_X, dev_test_labels, os.path.join(args.output_predictions,
                                                                                        "selftr_dev_test_{}_run{}.out".format(
                                                                                            num_epochs, run_num)))
                tagger.get_predictions_output(test_X, test_labels, os.path.join(args.output_predictions,
                                                                                "selftr_test_{}_run{}.out".format(num_epochs,
                                                                                                           run_num)))
            data_utils.log_self_training(args, dev_test_acc, test_acc, num_epochs,
                                         num_new_examples, run_num)

        if args.candidate_pool_size != 0:
            unlabeled_X = get_candidates(args, original_unlabeled_X) # random

        num_new_examples = 0
        new_unlabeled_X, confidences, new_unlabeled_preds = [], [], []
        print("Unlabeled pool size: {} instances".format(len(unlabeled_X)))
        for word_indices, word_char_indices in unlabeled_X:
            # perform predictions on the unlabeled examples
            prob_dists = tagger.predict(word_indices, word_char_indices,
                                        soft_labels=args.soft_labels,
                                        temperature=args.temperature)
            # for soft labels with a high temperature, max_prob is lower
            max_probs = [np.max(p.value()) for p in prob_dists]

            # use the argmax or the entire probability distribution as label
            pseudo_labels = [p.value() if args.soft_labels else
                             int(np.argmax(p.value())) for p in prob_dists]
            mean_prob = np.mean(max_probs)

            if args.num_select_examples and args.num_select_examples > 0:
                # collect all unlabeled samples, predictions, and confidences
                new_unlabeled_X.append((word_indices, word_char_indices))
                new_unlabeled_preds.append(pseudo_labels)
                confidences.append(mean_prob)
            elif mean_prob > args.confidence_threshold:
                # add all predictions that are higher than a threshold to the
                # labelled data
                train_X.append((word_indices, word_char_indices))
                train_Y.append(pseudo_labels)
                num_new_examples += 1
            else:
                new_unlabeled_X.append((word_indices, word_char_indices))

        if args.num_select_examples:
            if args.selection == RANDOM:
                # randomly select n examples and their predictions
                indices = np.random.permutation(
                    range(len(new_unlabeled_X)))[:args.num_select_examples]
            else:
                # select examples with most confident predictions
                indices = [i for _, i in sorted(zip(confidences, range(len(
                    new_unlabeled_X))), reverse=True)][:args.num_select_examples]
            for i in indices:
                train_X.append(new_unlabeled_X[i])
                train_Y.append(new_unlabeled_preds[i])
                num_new_examples += 1
            new_unlabeled_X = [x for i, x in enumerate(new_unlabeled_X)
                               if i not in indices]

        print('Added %d/%d pseudo-labeled examples after epoch %d.'
              % (num_new_examples, len(unlabeled_X), num_epochs))
        unlabeled_X = new_unlabeled_X
        num_epochs += 1
        if args.max_iterations and num_epochs > args.max_iterations:
            print('Stopping after max iterations %d...'.format(args.max_iterations))
            if args.save_final_model:
                print("save final model in...", args.model_dir)
                final_model_file = os.path.join(args.model_dir, "selftraining_run{}".format(run_num))
                save_tagger(tagger, final_model_file)

            break
    return dev_test_acc, test_acc


def temporal_ensembling_pos(vocab, args, train_data, train_labels, val_data,
                            val_labels, dev_test_data, dev_test_labels,
                            test_data, test_labels, unlabeled_data,
                            unlabeled_labels, run_num):
    """Temporal ensembling as proposed in Laine & Aila (2017).
    Temporal Ensembling for Semi-Supervised Learning. ICLR 2017."""
    assert len(unlabeled_labels) == 0
    options = load_config(args.config)

    print('Ramp-up length: %d. Ensemble momentum: %.2f. Unsupervised weight: '
          '%.2f. Candidate pool size: %d. Labeled_weight_proportion: %.2f'
          % (args.ramp_up_len, args.ensemble_momentum, args.unsupervised_weight,
             args.candidate_pool_size, args.labeled_weight_proportion))

    # load the model that has been trained for one epoch on the training set
    tagger = load_tagger(os.path.join(args.start_model_dir, '%s_run%d' % (args.start, run_num)))

    train_X, train_Y = tagger.get_data_as_indices_from_instances(train_data,
                                                                 train_labels)
    val_X, val_Y = tagger.get_data_as_indices_from_instances(val_data,
                                                             val_labels)
    dev_test_X, dev_test_Y = tagger.get_data_as_indices_from_instances(
        dev_test_data, dev_test_labels)
    test_X, test_Y = tagger.get_data_as_indices_from_instances(test_data,
                                                               test_labels)
    unlabeled_X = [tagger.get_features(words) for words in unlabeled_data]

    # create dummy unlabeled labels
    sampled_unlabeled_labels = [[0] for _ in range(min(len(unlabeled_X), args.candidate_pool_size))]

    # flatten the true validation and test labels
    val_y = [y for y_tags in val_Y for y in y_tags]
    dev_test_y = [y for y_tags in dev_test_Y for y in y_tags]
    test_y = [y for y_tags in test_Y for y in y_tags]

    # we use the unsupervised objective on both labeled and unlabeled examples
    num_tags = len(tagger.tag2idx)
    # note: these are the # of individual training/unlabeled predictions
    n_train = len([1 for examples in train_X for _ in examples[0]])
    n_unlabeled = len([1 for examples in unlabeled_X for _ in examples[0]])
    num_examples = n_train + n_unlabeled
    ensemble_preds = np.zeros((num_examples, num_tags))
    trg_vectors = np.zeros((num_examples, num_tags))
    val_acc, dev_test_acc, test_acc = 0, 0, 0

    # get the prediction indices of each unlabeled example
    unlabeled_pred_indices_list = []
    current_idx = 0
    for i in range(len(unlabeled_X)):
        unlabeled_pred_indices_list.append(np.array(range(current_idx, current_idx+len(unlabeled_X[i][0]))))
        current_idx += len(unlabeled_X[i][0])

    # default learning rate of Adam is 0.001
    orig_learning_rate = tagger.trainer.learning_rate
    for i in range(1, args.max_iterations):
        # calculate the weight for the unsupervised loss;
        # they use a Gaussian ramp-up curve of e^(-5(1-t)^2) for the first n
        # epochs of training, where t goes linearly from 0-1
        factor = (i-1) / args.ramp_up_len if i-1 < args.ramp_up_len else 1
        ramp_up_weight = np.exp(-5*(1-factor)**2)
        unsup_weight = ramp_up_weight * args.unsupervised_weight
        print("iter: {0}. ramp_up_weight: {1:.4f}. unsup_weight: {2:.4f} ".format(
            i, ramp_up_weight, unsup_weight))

        if args.lr_ramp_up:
            tagger.trainer.learning_rate = orig_learning_rate * ramp_up_weight
            print('Ramping-up lr. New lr: {:.7f}.'.format(tagger.trainer.learning_rate))

        if i > 1:
            # skip the first epoch as the model was already trained for 1 epoch
            # and for simplification as the trg_vectors won't be useful yet
            # at each iteration, we get the target vectors and variance weights
            # for the sampled indices ex_indices
            tagger.fit(train_X + sampled_unlabeled_X, train_Y + sampled_unlabeled_labels, 1,
                       trg_vectors=trg_vectors[ex_indices, :],
                       unsup_weight=unsup_weight,
                       labeled_weight_proportion=args.labeled_weight_proportion, seed=args.seed)

            # evaluate on the validation and test dataset; we do this only
            # for hyperparameter optimization
            # val_acc = accuracy_score(val_y, tagger.get_predictions(val_X))
            dev_test_acc = accuracy_score(dev_test_y,
                                          tagger.get_predictions(dev_test_X))
            test_acc = accuracy_score(test_y, tagger.get_predictions(test_X))
            print("Dev test acc: {0:.4f}. Test acc: {1:.4f}.".format(dev_test_acc, test_acc))
            data_utils.log_self_training(args, dev_test_acc, test_acc, i, 0, run_num)

            if args.output_predictions:
                tagger.get_predictions_output(dev_test_X, dev_test_labels, os.path.join(args.output_predictions,
                                                                                        "te_e_dev_test_{}_{}_run{}.out".format(args.start,
                                                                                            i, run_num)))
                tagger.get_predictions_output(test_X, test_labels, os.path.join(args.output_predictions,
                                                                                "te_e_test_{}_{}_run{}.out".format(
                                                                                    args.start, i, run_num)))
        else:
            # just evaluate starting tagger, do not log
            print("eval start tagger")
            dev_test_acc = accuracy_score(dev_test_y,
                                          tagger.get_predictions(dev_test_X))
            test_acc = accuracy_score(test_y, tagger.get_predictions(test_X))
            print("Dev test acc: {0:.4f}. Test acc: {1:.4f}.".format(dev_test_acc, test_acc))


        # sample unlabeled sentence indices
        sampled_unlabeled_X, unlabeled_sent_indices = get_candidates(args, np.array(unlabeled_X), return_indices=True)

        print('Obtaining predictions on %d labeled and %d unlabeled examples...'
              % (len(train_X), len(sampled_unlabeled_X)))

        # get the predictions for the labeled examples and the selected unlabeled examples
        preds = np.array(tagger.get_predictions((train_X + sampled_unlabeled_X),
                                                soft_labels=True))

        # get the indices of the labeled predictions; we always update these
        ex_indices = list(range(n_train))
        # get the indices that correspond to the unlabeled examples
        for sent_idx in unlabeled_sent_indices:
            # add # of labeled predictions as go in front of unlabeled preds
            sent_ex_indices = list(unlabeled_pred_indices_list[sent_idx] + n_train)
            ex_indices += sent_ex_indices
        assert len(preds) == len(ex_indices), '%d != %d.' % (len(preds), len(ex_indices))

        # update ensemble predictions
        ensemble_preds[ex_indices, :] = args.ensemble_momentum * ensemble_preds[ex_indices, :] + \
                         (1 - args.ensemble_momentum) * preds

        # construct target vectors
        trg_vectors[ex_indices, :] = ensemble_preds[ex_indices, :] / (1 - args.ensemble_momentum ** i)
    return dev_test_acc, test_acc


def tri_training_pos(vocab, args, train_data, train_labels, val_data,
                     val_labels, dev_test_data, dev_test_labels,
                     test_data, test_labels, unlabeled_data,
                     unlabeled_labels, run_num):
    """
    Basic tri-training as in Zhou & Lin, 2005.
    disagreement=True: tri-training with disagreement (Søgaard, 2010)
    Online: continue training on additional batch, do not add original train_X
    """
    # assert len(unlabeled_labels) == 0  # if JS selection given these contain the domain identifiers

    options = load_config(args.config)

    # load the base models from the starting point
    tagger1 = load_tagger(os.path.join(args.start_model_dir, args.start + "_bootstrap1_run" + str(run_num)))
    tagger2 = load_tagger(os.path.join(args.start_model_dir, args.start + "_bootstrap2_run" + str(run_num)))
    tagger3 = load_tagger(os.path.join(args.start_model_dir, args.start + "_bootstrap3_run" + str(run_num)))

    tagger_ensemble = [tagger1, tagger2, tagger3]
    tagger_indices = [tagger_ensemble.index(t) for t in tagger_ensemble] # to access by index later

    train_X_org, train_Y_org = tagger1.get_data_as_indices_from_instances(train_data,
                                                                 train_labels)
    val_X, val_Y = tagger1.get_data_as_indices_from_instances(val_data,
                                                             val_labels)
    dev_test_X, dev_test_Y = tagger1.get_data_as_indices_from_instances(
        dev_test_data, dev_test_labels)
    test_X, test_Y = tagger1.get_data_as_indices_from_instances(test_data,
                                                               test_labels)
    unlabeled_X = [tagger1.get_features(words) for words in unlabeled_data]

    original_unlabeled_X = np.array([tagger1.get_features(words) for words in unlabeled_data])

    if not args.candidate_pool_size:
        unlabeled_X = original_unlabeled_X  # use all
    else:
        # check
        assert len(unlabeled_data) > args.candidate_pool_size, "--candidate-pool-size > len(unlabeled_data)"

    # flatten the true validation and test labels
    # val_y = [y for y_tags in val_Y for y in y_tags]
    dev_test_y = [y for y_tags in dev_test_Y for y in y_tags]
    test_y = [y for y_tags in test_Y for y in y_tags]

    # stop after max_iteration
    avg_num_new_examples = 0
    val_acc, dev_test_acc, test_acc = 0, 0, 0


    print("tri-training online: {}".format(args.online))

    # get base model starting accuracies
    all_base_dev = []
    all_base_dev_test = []
    all_base_test = []
    for i, tagger in zip(tagger_indices, tagger_ensemble):
        # predictions_dev = tagger.get_predictions(val_X) # flattened list
        # all_base_dev.append(predictions_dev)
        predictions_dev_test = tagger.get_predictions(dev_test_X)
        all_base_dev_test.append(predictions_dev_test)
        predictions_test = tagger.get_predictions(test_X)
        all_base_test.append(predictions_test)
        print("## tagger {} base dev test acc: {:.4f} - test acc: {:.4f}".format(i, \
                                                        accuracy_score(dev_test_y, predictions_dev_test),\
                                                        accuracy_score(test_y, predictions_test)))

    print("majority base dev test {}".format(accuracy_score(dev_test_y, get_majority_vote(all_base_dev_test))))
    print("majority base test {}".format(accuracy_score(test_y, get_majority_vote(all_base_test))))

    # notice diff to self-training:
    # "unlabeled examples labeled in (t-1)th round won't be put into the original labeled example set" (Zhou & Li, 2005)
    for num_epochs in range(0, args.max_iterations+1):
        print("num_epochs".format(num_epochs))
        if num_epochs != 0:
            predictions_dev, predictions_dev_test, predictions_test = [], [], []

            for i, tagger in zip(tagger_indices, tagger_ensemble):
                train_X = []
                train_Y = []
                if not args.online:
                    # add original training data
                    train_X = train_X_org.copy()
                    train_Y = train_Y_org.copy()

               # add pseudo-labeled data
                for word_indices, word_char_indices in new_train_X[i]:
                    train_X.append((word_indices, word_char_indices))
                for labels in new_train_preds[i]:
                    train_Y.append(labels)

                print('Epoch #{}. Training on {} examples. # unlabeled examples: {}.'.format(num_epochs, len(train_X), len(unlabeled_X)))
                # train the model on the examples selected from the last epoch
                tagger.fit(train_X, train_Y, 1, word_dropout_rate=options.word_dropout_rate)

                # evaluate on the validation and test dataset (if --prototyping on: evaluate on dev set (dev=source, test=target dev)
                # predictions_dev.append(tagger.get_predictions(val_X))
                predictions_dev_test.append(tagger.get_predictions(dev_test_X))
                predictions_test.append(tagger.get_predictions(test_X))
                
            # majority vote
            print("get majority prediction")
            # dev_majority_pred = get_majority_vote(predictions_dev)
            dev_test_majority_pred = get_majority_vote(predictions_dev_test)
            test_majority_pred = get_majority_vote(predictions_test)

            # val_acc = accuracy_score(val_y, dev_majority_pred)
            dev_test_acc = accuracy_score(dev_test_y, dev_test_majority_pred)
            test_acc = accuracy_score(test_y, test_majority_pred)
            data_utils.log_self_training(args, dev_test_acc, test_acc, num_epochs, avg_num_new_examples/3.0, run_num)

            print('Dev test acc: {:.4f}. Test acc: {:.4f}.'.format(dev_test_acc, test_acc))

            ## get predictions - need to access flat list
            if args.output_predictions:
                output_prefix = "tritr" if not args.disagreement else "disagr"
                store_flat_output_predictions(
                    tagger, dev_test_majority_pred, dev_test_data, dev_test_labels,
                    os.path.join(args.output_predictions, f"{output_prefix}_dev_test_{num_epochs}_run{run_num}.out"))
                store_flat_output_predictions(
                    tagger, test_majority_pred, test_data, test_labels,
                    os.path.join(args.output_predictions, f"{output_prefix}_test_{num_epochs}_run{run_num}.out"))

            if num_epochs == args.max_iterations:
                print("stopping, reached --max-iterations")
                if args.save_final_model:
                    print("save final models in... {}".format(args.model_dir))
                    for i, tagger in zip(tagger_indices, tagger_ensemble):
                        final_model_file = os.path.join(args.model_dir, "tritraining_tagger{}_run{}".format(i, run_num))
                        save_tagger(tagger, final_model_file)
                return dev_test_acc, test_acc

        if args.candidate_pool_size != 0:
            unlabeled_X = get_candidates(args, original_unlabeled_X)

        # reset for next round
        new_train_X = defaultdict(list)  # store new instances for each tagger
        new_train_preds = defaultdict(list)  # labels
        new_train_confidences = defaultdict(list)

        print("Tag unlabeled examples...: {} instances".format(len(unlabeled_X)))
        bar = Bar('Tagging unlabeled examples...',
                  max=len(unlabeled_X), flush=True)
        for word_indices, word_char_indices in unlabeled_X:
            predictions_ensemble = {}  # keep per tagger predictions for example X
            predictions_confidences = {}

            for i, tagger in enumerate(tagger_ensemble):
                # perform predictions on the unlabeled examples
                prob_dists = tagger.predict(word_indices, word_char_indices)

                # use the argmax
                predictions_ensemble[i] = [int(np.argmax(p.value())) for p in prob_dists] # get pseudo labels
                assert len(predictions_ensemble[i]) == len(word_indices), "no label for instance"
                # calculate confidence
                max_probs = [np.max(p.value()) for p in prob_dists]
                confidence = np.mean(max_probs)
                predictions_confidences[i] = confidence

            # if the other two taggers agree, add prediction to third
            for i in range(len(tagger_ensemble)):
                j, k = [x for x in tagger_indices if x != i]

                # if tagger j and k agree, add to train data of i
                # args.confidence_threshold might filter
                if predictions_ensemble[j] == predictions_ensemble[k]:
                    if args.confidence_threshold > 0:
                        max_prob = max(predictions_confidences[j],predictions_confidences[k])
                        if max_prob < args.confidence_threshold: #skip if below threshold
                            continue
                    if not args.disagreement: # original formulation
                        new_train_X[i].append((word_indices, word_char_indices))
                        new_train_preds[i].append(predictions_ensemble[j])
                        new_train_confidences[i].append(predictions_confidences[j])
                    else:
                        # tri-training with disagreement
                        # only add if two agree and third predicted something different
                        if predictions_ensemble[j] != predictions_ensemble[i]:
                            new_train_X[i].append((word_indices, word_char_indices))
                            new_train_preds[i].append(predictions_ensemble[j])
                            new_train_confidences[i].append(predictions_confidences[j])
            bar.next()

        if args.num_select_examples:
            print("restrict to --num-select-exampeles, get most confident predictions")
            for i in range(len(tagger_ensemble)):
                if new_train_confidences[i]:
                    # select examples with most confident predictions
                    indices = [i for _, i in sorted(zip(new_train_confidences[i], range(len(
                        new_train_X[i]))), reverse=True)][:args.num_select_examples]

                    new_train_X[i] = np.array(new_train_X[i])[indices] #selected train X
                    new_train_preds[i] = np.array(new_train_preds[i])[indices] #selected train Y
                    new_train_confidences[i] = [] # no need

        # get some output statistics
        for i in tagger_indices:
            num_new_examples = len(new_train_X[i])
            avg_num_new_examples+= num_new_examples
            print('Added {}/{} pseudo-labeled examples after epoch {}.'.format(num_new_examples, len(unlabeled_X), num_epochs))

    return dev_test_acc, test_acc


def _debug_get_base_starting_accuracies(tagger, val_X, val_Y, dev_test_X, dev_test_y, test_X, test_y):
    all_base_dev = []
    all_base_dev_test = []
    all_base_test = []
    for task_id in TASK_IDS:
        # predictions_dev = tagger.get_predictions(val_X, task_id=task_id)
        # all_base_dev.append(predictions_dev)
        predictions_dev_test = tagger.get_predictions(dev_test_X, task_id=task_id)
        all_base_dev_test.append(predictions_dev_test)
        predictions_test = tagger.get_predictions(test_X, task_id=task_id)
        all_base_test.append(predictions_test)
        print("## tagger {} base dev test acc: {:.4f} - test acc: {:.4f}".format(
            task_id, accuracy_score(dev_test_y, predictions_dev_test),
            accuracy_score(test_y, predictions_test)))
    # print("majority base dev {}".format(
    #     accuracy_score(val_y, get_majority_vote(all_base_dev))))
    print("majority base dev test {}".format(
        accuracy_score(dev_test_y, get_majority_vote(all_base_dev_test))))
    print("majority base test {}".format(
        accuracy_score(test_y, get_majority_vote(all_base_test))))
    return


def mttri_training_pos(vocab, args, train_data, train_labels, val_data,
                      val_labels, dev_test_data, dev_test_labels,
                      test_data, test_labels, unlabeled_data,
                      unlabeled_labels, run_num):
    """
    Adversarial multi-task tri-training

    asymmetric: (Saito et al., 2017)
     - train Ft on agreements from F0 and F1
     - use orthogonality on F0 and F1
     - use only Ft for final predictions (not majority vote) - can be changed via --predict option
     - train F0, F1 and Ft on source data (if bootstrap on a bootstrap sample, otherwise give data to all tasks via common 'src' task)
    """
    assert len(unlabeled_labels) == 0

    options = load_config(args.config)

    # load the base models from the starting point
    tagger = mttri_load_tagger(os.path.join(args.start_model_dir,
                              args.start + "_run" + str(run_num)))

    train_X_org, train_Y_org = tagger.get_data_as_indices_from_instances(
        train_data, train_labels)
    val_X, val_Y = tagger.get_data_as_indices_from_instances(
        val_data, val_labels)
    dev_test_X, dev_test_Y = tagger.get_data_as_indices_from_instances(
        dev_test_data, dev_test_labels)
    test_X, test_Y = tagger.get_data_as_indices_from_instances(
        test_data, test_labels)

    # in each round sample candidate pool
    unlabeled_X = [tagger.get_features(words) for words in unlabeled_data]

    original_unlabeled_X = np.array([tagger.get_features(words) for words in unlabeled_data])

    if not args.candidate_pool_size:
        unlabeled_X = original_unlabeled_X  # use all
    else:
        # check
        assert len(unlabeled_data) > args.candidate_pool_size, "--candidate-pool-size > len(unlabeled_data)"
        if args.candidate_pool_scheduling:
            print("use candidate pool scheduling: k/20*n")

    # flatten the true validation and test labels
    val_y = [y for y_tags in val_Y for y in y_tags]
    dev_test_y = [y for y_tags in dev_test_Y for y in y_tags]
    test_y = [y for y_tags in test_Y for y in y_tags]

    # stop after max_iteration
    avg_num_new_examples = 0
    val_acc, test_acc = 0, 0

    src_domain_tag = 0
    trg_domain_tag = 1

    # a dictionary mapping tasks ("F0", "F1", and "Ft") to a dictionary
    # {"X": list of examples, "Y": list of labels, "domain": list of domain
    # tag (0,1) of example}
    train_dict = {t: defaultdict(list) for t in TASK_IDS}

    if args.size_bootstrap == 0.0:
        tmp_train_dict = {t: defaultdict(list) for t in TASK_IDS}
        # set up 'src' dictionary to later fetch source examples
        tmp_train_dict["F0"]["X"] = train_X_org
        tmp_train_dict["F0"]["Y"] = train_Y_org
        tmp_train_dict["F0"]["domain"] = [src_domain_tag] * len(train_Y_org)
        train_dict_src = get_training_dict(args, tmp_train_dict) 
        # initialize 'src' for asymmetric tri-training, eventually also 'trg'; gets 'src' data, 'F0' will be removed 
    else:
        # get bootstrap samples of source
        train_dict_src = {t: defaultdict(list) for t in TASK_IDS}
        train_dict_src["F0"]["X"] = train_X_org
        train_dict_src["F0"]["Y"] = train_Y_org
        train_dict_src["F0"]["domain"] = [src_domain_tag] * len(train_Y_org)
        train_dict_src = get_training_dict(args, train_dict_src) 


    print("asymmetric tri-training: {}".format(args.asymmetric))
    if args.adversarial:
        print('Adding an adversarial layer to the model with weight {}...'.format(args.adversarial_weight))
        tagger.add_adversarial_loss()
    if args.confidence_threshold:
        print("using confidence threshold: {}".format(args.confidence_threshold))

    # get base model starting accuracies
    _debug_get_base_starting_accuracies(tagger, val_X, val_Y, dev_test_X, dev_test_y, test_X, test_y)

    predictions_dev, predictions_dev_test, predictions_test = [], [], []  # reset

    # notice diff to self-training:
    # "unlabeled examples labeled in (t-1)th round won't be put into the original labeled example set" (Zhou & Li, 2005)
    for num_epochs in range(0, args.max_iterations + 1):
        print("num_epochs", num_epochs)
        if num_epochs > 0:
            # train the model on the examples selected from the last epoch
            print('Epoch #%d. # unlabeled examples: %d.'
                % (num_epochs, len(unlabeled_X)))

            tagger.fit(train_dict, 1,
                       word_dropout_rate=options.word_dropout_rate,
                       orthogonality_weight=args.orthogonality_weight,
                       adversarial=args.adversarial, adversarial_weight=args.adversarial_weight, ignore_src_Ft=True)
#                       adversarial=args.adversarial, adversarial_weight=args.adversarial_weight, ignore_src_Ft=False)

            for task_id in TASK_IDS:
                # evaluate on the validation and test dataset; we do this only
                # for hyperparameter optimization (set on dev)
                # get the predictions of each model
                # predictions_dev.append(tagger.get_predictions(val_X, task_id=task_id))
                predictions_dev_test.append(tagger.get_predictions(dev_test_X, task_id=task_id))
                predictions_test.append(tagger.get_predictions(test_X, task_id=task_id))

            if args.predict == 'majority':
                # majority vote
                print("get majority prediction")
                # dev_majority_pred = get_majority_vote(predictions_dev)
                dev_test_majority_pred = get_majority_vote(predictions_dev_test)
                test_majority_pred = get_majority_vote(predictions_test)


            elif args.predict == 'Ft': # this is Saito et al.'s default
                print("use Ft for prediction")
                # get last = Ft
                dev_test_majority_pred = predictions_dev_test[-1]
                test_majority_pred = predictions_test[-1]

                if args.output_predictions:
                    output_prefix = "asym_"
                    ### store majority prediction just for later inspection
                    store_flat_output_predictions(tagger, get_majority_vote(predictions_dev_test), dev_test_data, dev_test_labels,
                                                  os.path.join(args.output_predictions,
                                                               "{}_dev_test_{}_run{}_majority.out".format(
                                                                   output_prefix, num_epochs, run_num)))
                    store_flat_output_predictions(tagger, get_majority_vote(predictions_test), test_data, test_labels,
                                                  os.path.join(args.output_predictions,
                                                               "{}_test_{}_run{}_majority.out".format(
                                                                   output_prefix, num_epochs, run_num)))

            # val_acc = accuracy_score(val_y, dev_majority_pred)
            dev_test_acc = accuracy_score(dev_test_y, dev_test_majority_pred)
            test_acc = accuracy_score(test_y, test_majority_pred)
            data_utils.log_self_training(args, dev_test_acc, test_acc, num_epochs,
                                         avg_num_new_examples / 3.0, run_num)

            for pred_dev_test, pred_test in zip(predictions_dev_test, predictions_test):
                ## evaluate per-task predictions:
                print(accuracy_score(dev_test_y, pred_dev_test))
                print(accuracy_score(test_y, pred_test))
                print("====")


            print('Dev test acc: %.4f. Test acc: %4f.'
                  % (dev_test_acc, test_acc))

            ## get predictions - need to access flat list
            if args.output_predictions:
                output_prefix = "mttri-d0" if not args.disagreement else "mttri-d1"
                if args.asymmetric:
                    output_prefix = "asym_{}".format(args.predict)
                store_flat_output_predictions(tagger, dev_test_majority_pred, dev_test_data, dev_test_labels,
                                              os.path.join(args.output_predictions,
                                                           "{}_dev_test_{}_run{}.out".format(
                                                               output_prefix, num_epochs, run_num)))
                store_flat_output_predictions(tagger, test_majority_pred, test_data, test_labels,
                                              os.path.join(args.output_predictions,
                                                           "{}_test_{}_run{}.out".format(
                                                               output_prefix, num_epochs, run_num)))

            # reset for next round
            predictions_dev, predictions_dev_test, predictions_test = [], [], []

            if num_epochs == args.max_iterations:
                print("stopping, reached --max-iterations")
                if args.save_final_model:
                    print("save final models in...", args.model_dir)
                    final_model_file = os.path.join(
                        args.model_dir, "mttri_tagger_run{}".format(run_num))
                    mttri_save_tagger(tagger, final_model_file)
                return val_acc, test_acc

        if args.candidate_pool_size != 0:
            unlabeled_X = get_candidates(args, original_unlabeled_X, epoch_num=num_epochs)

        print("Tag unlabeled examples...: {} instances".format(len(unlabeled_X)))
        bar = Bar('Tagging unlabeled examples...',
                  max=len(unlabeled_X), flush=True)
        if args.asymmetric_type == 'org':
            train_dict = defaultdict(list)
            train_dict['trg'] = defaultdict(list) #initialize target (updates all tasks F0/F1/Ft)
            train_dict['trg']["X"] = []
            train_dict['trg']["Y"] = []
            train_dict['trg']["domain"] = []
            train_dict['trg']["confidence"] = []
        else:
            train_dict = {t: defaultdict(list) for t in TASK_IDS}
            for i in TASK_IDS:
                train_dict[i]["confidence"] = []  # separate pseudo-labeled targets for the separate tasks (F0/F1/Ft)
            

        for word_indices, word_char_indices in unlabeled_X:

            predictions_ensemble = {}  # keep per tagger predictions for example X
            predictions_confidences = {}
            for task_id in TASK_IDS:
                # perform predictions on the unlabeled examples
                prob_dists, _, _ = tagger.predict(word_indices, word_char_indices,
                                                  task_id=task_id)
                # use the argmax
                pseudo_labels = [int(np.argmax(p.value())) for p in prob_dists]
                predictions_ensemble[task_id] = pseudo_labels

                # calculate confidence
                max_probs = [np.max(p.value()) for p in prob_dists]
                confidence = np.mean(max_probs)
                predictions_confidences[task_id] = confidence

            # if the other two taggers agree, add prediction to third
            # (or if asymmetric tri-training is used, only add agreements of F0/F1)
            for i in TASK_IDS:
                j, k = [x for x in TASK_IDS if x != i]

                pseudo_labels = predictions_ensemble[j]
                pseudo_labels2 = predictions_ensemble[k]
                if pseudo_labels == pseudo_labels2:
                    max_prob = 0.0
                    # check if we use confidence threshold
                    if args.confidence_threshold > 0:
                        max_prob = max(predictions_confidences[j], predictions_confidences[k])
                        if max_prob < args.confidence_threshold:  # skip if below threshold
                            continue

                    if not args.disagreement:  # original formulation
                        if args.asymmetric and args.asymmetric_type == 'org':
                            if i == 'Ft': # F0 and F1 agree, add to Ft (and also others if 'org')
                                train_dict['trg']["X"].append((word_indices, word_char_indices)) # add to all target nodes in 'org' formulation
                                train_dict['trg']["Y"].append(pseudo_labels)
                                train_dict['trg']["domain"].append(trg_domain_tag)
                                train_dict['trg']["confidence"].append(max_prob)

                                # train_dict[i]["X"].append((word_indices, word_char_indices))
                                # train_dict[i]["Y"].append(pseudo_labels)
                                # train_dict[i]["domain"].append(trg_domain_tag)
                                # train_dict[i]["confidence"].append(max_prob)

                                # # add to both other
                                # train_dict[j]["X"].append((word_indices, word_char_indices))
                                # train_dict[j]["Y"].append(pseudo_labels)
                                # train_dict[j]["domain"].append(trg_domain_tag)
                                # train_dict[j]["confidence"].append(max_prob)

                                # train_dict[k]["X"].append((word_indices, word_char_indices))
                                # train_dict[k]["Y"].append(pseudo_labels)
                                # train_dict[k]["domain"].append(trg_domain_tag)
                                # train_dict[k]["confidence"].append(max_prob)
                        else:
                            # add pair-wise agreements
                            train_dict[i]["X"].append((word_indices, word_char_indices))
                            train_dict[i]["Y"].append(pseudo_labels)
                            train_dict[i]["domain"].append(trg_domain_tag)
                            train_dict[i]["confidence"].append(max_prob)
                    else:
                        # tri-training with disagreement
                        # only add if two agree and third predicted something different
                        pseudo_label3 = predictions_ensemble[i]
                        if pseudo_labels != pseudo_label3:
                            train_dict[i]["X"].append(
                                (word_indices, word_char_indices))
                            train_dict[i]["Y"].append(pseudo_labels)
                            train_dict[i]["domain"].append(trg_domain_tag)
                            train_dict[i]["confidence"].append(max_prob)

            bar.next()

        if args.num_select_examples:
            print("restrict to --num-select-examples, get most confident")
            for task_idx in TASK_IDS:
                if train_dict[task_idx]:
                    # select examples with most confident predictions
                    indices = [i for _, i in sorted(zip(train_dict[task_idx]["confidence"], range(len(
                            train_dict[task_idx]["X"]))), reverse=True)][:args.num_select_examples]

                    train_dict[task_idx]["X"] = np.array(train_dict[task_idx]["X"])[indices]  # selected train X
                    train_dict[task_idx]["Y"] = np.array(train_dict[task_idx]["Y"])[indices]  # selected train Y
                    train_dict[task_idx]["domain"] = np.array(train_dict[task_idx]["domain"])[indices]  # selected train Y
                    train_dict[task_idx]["confidence"] = []  # no need

        print("")
        # get some output statistics
        for task_id in train_dict.keys():
            num_new_examples = len(train_dict[task_id]["X"])
            print('%s - Added %d/%d pseudo-labeled examples after epoch %d.'
                  % (task_id, num_new_examples, len(unlabeled_X), num_epochs))
            avg_num_new_examples += num_new_examples

        print('Adding source training examples...')
        if not args.online:
            # add original training data for all tasks
            if args.bootstrap_src:
                print("resample source: get new bootstrap sample")
                train_dict_src = get_training_dict(args, train_dict_src)  # get new bootstrap sample

            if args.size_bootstrap > 0:
                for task_id in TASK_IDS:
                    for (word_indices, word_char_indices), labels, domain_tag in zip(train_dict_src[task_id]["X"],
                                                                                     train_dict_src[task_id]["Y"],
                                                                                     train_dict_src[task_id]["domain"]):
                        train_dict[task_id]["X"].append((word_indices, word_char_indices))
                        train_dict[task_id]["Y"].append(labels)
                        train_dict[task_id]["domain"].append(domain_tag)
            else:
                print("adding 'src' data for all tasks")
                train_dict["src"] = defaultdict(list)
                train_dict["src"]["X"], train_dict["src"]["Y"], train_dict["src"]["domain"] = [], [], []
                for (word_indices, word_char_indices), labels, domain_tag in zip(train_dict_src["src"]["X"],
                                                                                     train_dict_src["src"]["Y"],
                                                                                     train_dict_src["src"]["domain"]):
                    train_dict["src"]["X"].append((word_indices, word_char_indices))
                    train_dict["src"]["Y"].append(labels)
                    train_dict["src"]["domain"].append(domain_tag)

        else:
            assert not args.adversarial, \
                'Error: Adversarial loss also requires access to source domain.'
        for t in train_dict:
            print(t, "len:", len(train_dict[t]["X"]))

    return dev_test_acc, test_acc


def mttri_training_sentiment(vocab, args, train_X, train_Y, val_X, val_Y,
                            dev_test_X, dev_test_Y, test_X, test_Y, unlabeled_X,
                            unlabeled_Y, run_num):
    """
    Basic tri-training as in Zhou & Lin, 2005.
    disagreement=True: tri-training with disagreement (Søgaard, 2010)
    Online: continue training on additional batch, do not add original train_X
    """
    assert len(unlabeled_Y) == 0

    options = load_config(args.config)

    # load the base models from the starting point
    model = sentiment_model.load_mttri_model(os.path.join(args.start_model_dir,
                              args.start + "_run" + str(run_num)))

    src_domain_tag = 0
    trg_domain_tag = 1

    # a dictionary mapping tasks ("F0", "F1", and "Ft") to a dictionary
    # {"X": list of examples, "Y": list of labels, "domain": list of domain
    # tag (0,1) of example}
    train_dict = {t: defaultdict(list) for t in TASK_IDS + ['src']}
    print("tri-training online: {}".format(args.online))
    print("tri-training adversarial: {}".format(args.adversarial))
    if args.adversarial:
        print('Adding an adversarial layer to the model...')
        model.add_adversarial_loss()
    if args.orthogonality_weight != 0:
        print('Using orthogonality weight of  %.4f...' % args.orthogonality_weight)

    # stop after max_iteration
    avg_num_new_examples = 0
    val_acc, dev_test_acc, test_acc = 0, 0, 0

    # get base model starting accuracies
    all_base_dev = []
    all_base_dev_test = []
    all_base_test = []

    for task_id in TASK_IDS:
        predictions_dev = model.get_predictions(val_X, task_id=task_id) # flattened list
        all_base_dev.append(predictions_dev)
        predictions_dev_test = model.get_predictions(dev_test_X, task_id=task_id)
        all_base_dev_test.append(predictions_dev_test)
        predictions_test = model.get_predictions(test_X, task_id=task_id)
        all_base_test.append(predictions_test)
        print("## model {} base dev test acc: {:.4f} - test acc: {:.4f}".format(
              task_id, accuracy_score(dev_test_Y, predictions_dev_test), accuracy_score(test_Y, predictions_test)))

    print("majority base dev test {}".format(
        accuracy_score(dev_test_Y, get_majority_vote(all_base_dev_test))))
    print("majority base test {}".format(
        accuracy_score(test_Y, get_majority_vote(all_base_test))))
    predictions_dev, predictions_dev_test, predictions_test = [], [], []  # reset

    # notice diff to self-training:
    # "unlabeled examples labeled in (t-1)th round won't be put into the original labeled example set" (Zhou & Li, 2005)
    best_val_acc, best_test_acc, num_epochs_no_improvement = 0, 0, 0
    for num_epochs in range(0, args.max_iterations + 1):
        if num_epochs != 0:
            print('Epoch #{}.'.format(num_epochs))

            # train the model on the examples selected from the last epoch
            model.fit(train_dict, 1,
                      word_dropout_rate=options.word_dropout_rate,
                      orthogonality_weight=args.orthogonality_weight,
                      adversarial=args.adversarial
                      )

            for task_id in TASK_IDS:
                # evaluate on the validation and test dataset (if --prototyping on: evaluate on dev set (dev=source, test=target dev)
                predictions_dev.append(model.get_predictions(val_X, task_id=task_id))
                # predictions_dev_test.append(model.get_predictions(dev_test_X, task_id=task_id))
                predictions_test.append(model.get_predictions(test_X, task_id=task_id))

            # majority vote
            print("get majority prediction")
            dev_majority_pred = get_majority_vote(predictions_dev)
            # dev_test_majority_pred = get_majority_vote(predictions_dev_test)
            test_majority_pred = get_majority_vote(predictions_test)

            val_acc = accuracy_score(val_Y, dev_majority_pred)
            # dev_test_acc = accuracy_score(dev_test_Y, dev_test_majority_pred)
            test_acc = accuracy_score(test_Y, test_majority_pred)
            data_utils.log_self_training(args, dev_test_acc, test_acc,
                                         num_epochs, avg_num_new_examples / 3.0,
                                         run_num)

            print('Validation acc: {:.4f}. Test acc: {:.4f}.'.format(val_acc,
                                                                     test_acc))

            # reset for next round
            predictions_dev, predictions_dev_test, predictions_test = [], [], []
            train_dict = {t: defaultdict(list) for t in TASK_IDS + ['src']}

            if val_acc < best_val_acc:
                print('Val %.4f < best val %.4f.' % (val_acc, best_val_acc))
                num_epochs_no_improvement += 1
            else:
                print('Val %.4f > best val %.4f.' % (val_acc, best_val_acc))
                num_epochs_no_improvement = 0
                best_val_acc = val_acc
                best_test_acc = test_acc
            if num_epochs_no_improvement == 3:
                print('Best validation f1: %.4f. Test f1: %.4f' % (
                    best_val_acc, best_test_acc))
                return best_val_acc, best_test_acc

            if num_epochs == args.max_iterations:
                print("stopping, reached --max-iterations")
                if args.save_final_model:
                    print("save final models in... {}".format(args.model_dir))
                    final_model_file = os.path.join(args.model_dir,
                                                    "mttri_model_run{}".format(run_num))
                    sentiment_model.save_mttri_model(model, final_model_file)
                return val_acc, test_acc

        if not args.online:
            # add original training data
            train_dict['src']['X'] = train_X[:]
            train_dict['src']['Y'] = train_Y[:]
            train_dict['src']['domain'] = [src_domain_tag for _ in range(train_X.shape[0])]
        else:
            assert not args.adversarial, \
                'Error: Adversarial loss also requires access to source domain.'

        bar = Bar('Labeling unlabeled examples with each model...',
                  max=unlabeled_X.shape[0]*len(TASK_IDS), flush=True)

        for i in TASK_IDS:
            new_examples = []
            for example in unlabeled_X:
                j, k = [x for x in TASK_IDS if x != i]
                prob_dists1, _, _ = model.predict(example, task_ids=[j])
                label1 = int(np.argmax(prob_dists1[0].value()))
                max_prob1 = max(prob_dists1[0].value())
                prob_dists2, _, _ = model.predict(example, task_ids=[k])
                label2 = int(np.argmax(prob_dists2[0].value()))
                max_prob2 = max(prob_dists2[0].value())

                # if the other two taggers agree, add prediction to third
                if label1 == label2 and (max_prob1 > args.confidence_threshold or
                         max_prob2 > args.confidence_threshold):
                    if not args.disagreement:  # original formulation
                        new_examples.append(example)
                        train_dict[i]["Y"].append(label1)
                        train_dict[i]["domain"].append(trg_domain_tag)
                    else:
                        # tri-training with disagreement
                        # only add if two agree and third predicted something different
                        prob_dists3, _, _ = model.predict(example, task_ids=[i])
                        label3 = int(np.argmax(prob_dists3[0].value()))
                        if label1 != label3:
                            new_examples.append(example)
                            train_dict[i]["Y"].append(label1)
                            train_dict[i]["domain"].append(trg_domain_tag)
                bar.next()
            if not train_dict[i]["X"]:
                # if Ft is only trained on pseudo-labeled data,
                # train_dict[Ft]["X"] is empty before;
                # also empty when only adding source samples
                train_dict[i]["X"] = scipy.sparse.vstack(new_examples)
            else:
                train_dict[i]["X"] = scipy.sparse.vstack([train_dict[i]["X"]] + new_examples)
        print()

        # get some output statistics
        for task_id in TASK_IDS:
            if args.online or task_id in ['F0', 'F1', 'Ft']:
                num_new_examples = train_dict[task_id]["X"].shape[0]
            else:
                num_new_examples = train_dict[task_id]["X"].shape[0] - train_X.shape[0]
            avg_num_new_examples += num_new_examples
            print('Added {}/{} pseudo-labeled examples after epoch {}.'.format(
                num_new_examples, unlabeled_X.shape[0], num_epochs))
    return dev_test_acc, test_acc


def store_flat_output_predictions(tagger, flat_prediction_list, target_data, target_labels, output_filename):
    """
    for tri-training match back flat prediction list to output targets and store to file
    """
    i=0
    i2t = {tagger.tag2idx[t]: t for t in tagger.tag2idx.keys()}
    OUT = open(output_filename, "w")
    for words, gold_tags in zip(target_data, target_labels):
        for word, gold_tag in zip(words, gold_tags):
            known_tag_prefix = "" if gold_tag in tagger.tag2idx else "*"
            pred_tag = i2t[flat_prediction_list[i]]
            i+=1
            OUT.write("{}\t{}\t{}\n".format(word, known_tag_prefix+gold_tag, pred_tag))
        OUT.write("\n")
    OUT.close()

def get_majority_vote(prediction_list, debug=False):
    """
    dictionary with list of predictions for every classifier (key) for tri-training
    :param predictions_per_classifier
    :return: majority prediction
    """
    predictions1, predictions2, predictions3 = prediction_list
    assert(len(predictions1) ==  len(predictions2))
    assert(len(predictions1) == len(predictions3))

    output_majority_predictions = []

    # prepare for hstack
    predictions_first = [[t] for t in predictions1]
    predictions_second = [[t] for t in predictions2]
    predictions_third = [[t] for t in predictions3]

    all_tags = np.hstack((predictions_first, predictions_second, predictions_third))

    agree_all=0
    agree_two=0
    for tags_per_item in all_tags:
        counter = Counter(tags_per_item)
        most_common = counter.most_common()[0][0] # get most common tags (if all diff: takes first)
        if counter.most_common()[0][1] == 2:
            agree_two+=1
        if counter.most_common()[0][1] == 3:
            agree_all += 1
        output_majority_predictions.append(most_common)
    total = len(predictions1)
    if debug:
        print("Agreement statistics: agree all 3: {0}/{1}, agree two: {2}/{1}".format(agree_all, total, agree_two))
    return output_majority_predictions


def pos_accuracy_score(gold, predicted):
    """
    accuracy for POS (expects gold|predicted as lists of lists)
    """
    tags_correct = np.sum([1 for gold_tags, pred_tags in zip(gold, predicted) for g, p in zip(gold_tags, pred_tags) if g == p])
    tags_total = len([t for g in gold for t in g])  # ravel list
    return tags_correct/float(tags_total)


def train_base_model_sentiment(vocab, args, train_X, train_Y, val_X,
                               val_Y, test_X, test_Y, dev_test_X, dev_test_Y,
                               unlabeled_X, unlabeled_Y, run_num):
    options = load_config(args.config)
    print(options)

    val_f1s = []
    test_f1s = []
    iterations = 3 if args.bootstrap else 1
    train_X_org, train_Y_org = train_X, train_Y
    val_X_org, val_Y_org = val_X, val_Y
    for i in range(1, iterations + 1):
        print('Using small DANN model...')
        model = sentiment_model.SentimentModel(
            1, 50, len(vocab.word2id),noise_sigma=0.0, trainer=options.trainer, activation='sigmoid')

        if args.bootstrap:
            print("Create bootstrap sample... {}/3 ".format(i))
            # get bootstrap sample of train and dev, keep test same
            train_X, train_Y = bootstrap_sample(train_X_org, train_Y_org)
            val_X, val_Y = bootstrap_sample(val_X_org, val_Y_org)

            # naming of models - include bootstrap
            model_epoch_1_file = os.path.join(args.model_dir, EPOCH_1 + "_bootstrap" + str(i) +"_run" + str(run_num))
            model_patience_2_file = os.path.join(args.model_dir,
                                                 PATIENCE_2 + "_bootstrap" + str(i) + "_run" + str(run_num))
            model_patience_N_file = os.path.join(args.model_dir, "patience_{}".format(options.patience) + "_bootstrap" + str(i)
                                                 + "_run" + str(run_num))

        else:
            # naming of models
            model_epoch_1_file = os.path.join(args.model_dir, EPOCH_1 + "_run" + str(run_num))
            model_patience_2_file = os.path.join(args.model_dir, PATIENCE_2 + "_run" + str(run_num))
            model_patience_N_file = os.path.join(args.model_dir, "patience_{}".format(options.patience)+ "_run" + str(run_num))

        # initialize the graph
        model.initialize_graph()

        if args.save_epoch_1:
            # train a model for one epoch and save it for later
            model.fit(train_X, train_Y, 1, word_dropout_rate=options.word_dropout_rate, seed=args.seed)
            sentiment_model.save_model(model, model_epoch_1_file)

            val_f1 = model.evaluate(val_X, val_Y)
            test_f1 = model.evaluate(test_X, test_Y)
            print('1 Epoch model. Validation acc: %.4f. Test acc: %4f.'
                  % (val_f1, test_f1))

        # load the 1 epoch model and continue training it until convergence
        model.fit(train_X, train_Y, 100, val_X, val_Y, patience=options.patience,
                  model_path=model_patience_2_file, word_dropout_rate=options.word_dropout_rate, seed=args.seed)
        model = sentiment_model.load_model(model_patience_2_file)
        val_f1 = model.evaluate(val_X, val_Y)
        test_f1 = model.evaluate(test_X, test_Y)
        print('Patience %d model. Validation F1: %.4f. Test F1: %4f.'
              % (options.patience, val_f1, test_f1))

        # log the results to the log file
        data_utils.log_self_training(args, val_f1, test_f1, run_num=run_num)
        val_f1s.append(val_f1)
        test_f1s.append(test_f1)
    return np.mean(val_f1s), np.mean(test_f1s) # return average (over just 1 or 3 bootstrap samples)


def self_training_sentiment(vocab, args, train_X, train_Y, val_X, val_Y,
                            dev_test_X, dev_test_Y, test_X, test_Y, unlabeled_X,
                            unlabeled_Y, run_num):
    assert len(unlabeled_Y) == 0

    train_Y = train_Y[:]  # pass train_Y by value for multiple runs
    options = load_config(args.config)

    model = sentiment_model.load_model(os.path.join(args.start_model_dir, args.start + "_run" + str(run_num)))

    # train only on new items
    if args.online:
        train_X, train_Y = [], []

    # stop once no more confident examples have been added in an iteration
    # if specified, also stop after max_iteration
    num_new_examples = train_X.shape[0]
    num_epochs, val_f1, test_f1 = 0, 0, 0
    best_val_f1, best_test_f1, num_epochs_no_improvement = 0, 0, 0

    val_f1 = model.evaluate(val_X, val_Y)
    test_f1 = model.evaluate(test_X, test_Y)
    print('Validation F1: %.4f. Test F1: %4f.' % (val_f1, test_f1))
    data_utils.log_self_training(args, val_f1, test_f1, num_epochs,
                                 num_new_examples, run_num)
    while num_new_examples != 0:
        # if num_epochs != 0:
        print('Epoch #%d. Training on %d examples.'
            % (num_epochs, train_X.shape[0]))
        # train the model on the examples selected from the last epoch
        model.fit(train_X, train_Y, 1, word_dropout_rate=options.word_dropout_rate)

        # evaluate on the validation and test dataset; we do this only
        # for hyperparameter optimization
        val_f1 = model.evaluate(val_X, val_Y)
        test_f1 = model.evaluate(test_X, test_Y)
        print('Validation F1: %.4f. Test F1: %4f.' % (val_f1, test_f1))
        data_utils.log_self_training(args, val_f1, test_f1, num_epochs,
                                     num_new_examples, run_num)

        num_new_examples = 0
        new_unlabeled_X, confidences, new_unlabeled_preds = [], [], []
        for x in unlabeled_X:
            # perform predictions on the unlabeled examples
            prob_dist = model.predict(x)
            # for soft labels with a high temperature, max_prob is lower
            max_prob = np.max(prob_dist.value())
            pseudo_label = prob_dist.value()

            if args.num_select_examples and args.num_select_examples > 0:
                # collect all unlabeled samples, predictions, and confidences
                new_unlabeled_X.append(x)
                new_unlabeled_preds.append(pseudo_label)
                confidences.append(max_prob)
            elif max_prob > args.confidence_threshold:
                # add all predictions that are higher than a threshold to the
                # labelled data
                train_X = scipy.sparse.vstack(train_X, x)
                train_Y.append(max_prob)
                num_new_examples += 1
            else:
                new_unlabeled_X.append(x)

        if args.num_select_examples:
            new_examples, new_labels, new_unlabeled_X = selection(
                new_unlabeled_X, new_unlabeled_preds, confidences,
                args.num_select_examples, args.selection == RANDOM)
            train_X = scipy.sparse.vstack([train_X] + new_examples)
            train_Y += new_labels
            num_new_examples += len(new_examples)

        print('Added %d/%d pseudo-labeled examples after epoch %d.'
              % (num_new_examples, len(new_unlabeled_X)+num_new_examples, num_epochs))
        unlabeled_X = new_unlabeled_X
        num_epochs += 1
        if val_f1 < best_val_f1:
            print('Val %.4f < best val %.4f.' % (val_f1, best_val_f1))
            num_epochs_no_improvement += 1
        else:
            print('Val %.4f > best val %.4f.' % (val_f1, best_val_f1))
            num_epochs_no_improvement = 0
            best_val_f1 = val_f1
            best_test_f1 = test_f1
        if num_epochs_no_improvement == 3:
            print('Best validation f1: %.4f. Test f1: %.4f' % (best_val_f1, best_test_f1))
            return best_val_f1, best_test_f1
        if args.max_iterations and num_epochs > args.max_iterations:
            print('Stopping after max iterations %d...' % (
                args.max_iterations))
            if args.save_final_model:
                print("save final model in...", args.model_dir)
                final_model_file = os.path.join(args.model_dir, "selftraining_run{}".format(run_num))
                sentiment_model.save_model(model, final_model_file)
            break
    return best_val_f1, best_test_f1


def tri_training_sentiment(vocab, args, train_X, train_Y, val_X, val_Y,
                            dev_test_X, dev_test_Y, test_X, test_Y, unlabeled_X,
                            unlabeled_Y, run_num):
    """
    Basic tri-training as in Zhou & Lin, 2005.
    disagreement=True: tri-training with disagreement (Søgaard, 2010)
    Online: continue training on additional batch, do not add original train_X
    """
    assert len(unlabeled_Y) == 0

    options = load_config(args.config)

    # load the base models from the starting point
    model1 = sentiment_model.load_model(os.path.join(args.start_model_dir,
                                       args.start + "_bootstrap1_run" + str(
                                           run_num)))
    model2 = sentiment_model.load_model(os.path.join(args.start_model_dir,
                                       args.start + "_bootstrap2_run" + str(
                                           run_num)))
    model3 = sentiment_model.load_model(os.path.join(args.start_model_dir,
                                       args.start + "_bootstrap3_run" + str(
                                           run_num)))

    model_ensemble = [model1, model2, model3]
    model_indices = [model_ensemble.index(t) for t in
                      model_ensemble]  # to access by index later

    original_train_X = train_X[:]
    original_train_Y = train_Y[:]
    original_unlabeled_X = unlabeled_X[:]
    if not args.candidate_pool_size or args.candidate_pool_size > unlabeled_X.shape[0]:
        unlabeled_X = original_unlabeled_X  # use all
    else:
        # check
        assert unlabeled_X.shape[0] > args.candidate_pool_size, "--candidate-pool-size > len(unlabeled_data)"

    # stop after max_iteration
    avg_num_new_examples = 0
    val_acc, dev_test_acc, test_acc = 0, 0, 0

    new_train_X = defaultdict(list)  # store new instances for each tagger
    new_train_preds = defaultdict(list)

    print("tri-training online: {}".format(args.online))

    # get base model starting accuracies
    all_base_dev = []
    all_base_dev_test = []
    all_base_test = []
    for i, model in zip(model_indices, model_ensemble):
        # predictions_dev = tagger.get_predictions(val_X) # flattened list
        # all_base_dev.append(predictions_dev)
        predictions_dev_test = model.get_predictions(dev_test_X)
        all_base_dev_test.append(predictions_dev_test)
        predictions_test = model.get_predictions(test_X)
        all_base_test.append(predictions_test)
        print("## tagger {} base dev test acc: {:.4f} - test acc: {:.4f}".format(
              i, accuracy_score(dev_test_Y, predictions_dev_test), accuracy_score(test_Y, predictions_test)))

    print("majority base dev test {}".format(
        accuracy_score(dev_test_Y, get_majority_vote(all_base_dev_test))))
    print("majority base test {}".format(
        accuracy_score(test_Y, get_majority_vote(all_base_test))))
    predictions_dev, predictions_dev_test, predictions_test = [], [], []  # reset

    # notice diff to self-training:
    # "unlabeled examples labeled in (t-1)th round won't be put into the original labeled example set" (Zhou & Li, 2005)
    num_epochs_no_improvement, best_val_acc, best_test_acc = 0, 0, 0
    for num_epochs in range(0, args.max_iterations + 1):
        if num_epochs != 0:
            for i, model in zip(model_indices, model_ensemble):
                train_X = new_train_X[i]
                train_Y = new_train_preds[i]

                print(
                    'Epoch #{}. Training on {} examples. # unlabeled examples: {}.'.format(
                        num_epochs, train_X.shape[0], unlabeled_X.shape[0]))
                # train the model on the examples selected from the last epoch
                model.fit(train_X, train_Y, 1,
                           word_dropout_rate=options.word_dropout_rate)

                # evaluate on the validation and test dataset (if --prototyping on: evaluate on dev set (dev=source, test=target dev)
                predictions_dev.append(model.get_predictions(val_X))
                # predictions_dev_test.append(model.get_predictions(dev_test_X))
                predictions_test.append(model.get_predictions(test_X))

            # majority vote
            print("get majority prediction")
            dev_majority_pred = get_majority_vote(predictions_dev)
            # dev_test_majority_pred = get_majority_vote(predictions_dev_test)
            test_majority_pred = get_majority_vote(predictions_test)

            val_acc = accuracy_score(val_Y, dev_majority_pred)
            # dev_test_acc = accuracy_score(dev_test_Y, dev_test_majority_pred)
            test_acc = accuracy_score(test_Y, test_majority_pred)
            data_utils.log_self_training(args, dev_test_acc, test_acc,
                                         num_epochs, avg_num_new_examples / 3.0,
                                         run_num)

            print('Validation acc: {:.4f}. Test acc: {:.4f}.'.format(val_acc,
                                                                     test_acc))

            # reset
            predictions_dev, predictions_dev_test, predictions_test = [], [], []

            # reset for next round
            new_train_X = {}
            new_train_preds = defaultdict(list)

            if num_epochs == args.max_iterations:
                print("stopping, reached --max-iterations")
                if args.save_final_model:
                    print("save final models in... {}".format(args.model_dir))
                    for i, model in zip(model_indices, model_ensemble):
                        final_model_file = os.path.join(args.model_dir,
                                                        "tritraining_tagger{}_run{}".format(
                                                            i, run_num))
                        sentiment_model.save_model(model, final_model_file)
                return val_acc, test_acc

        if not args.online:
            # add original training data
            for i in model_indices:
                new_train_X[i] = original_train_X[:]
                new_train_preds[i] += original_train_Y

        if args.candidate_pool_size != 0:
            selected_indices = np.random.choice(
                list(range(0, original_unlabeled_X.shape[0])),
                min(original_unlabeled_X.shape[0], args.candidate_pool_size),
                replace=False)
            unlabeled_X = original_unlabeled_X[selected_indices]
        print(
            "Tag unlabeled examples...: {} instances".format(unlabeled_X.shape[0]))
        predictions_ensemble = {}  # keep per tagger predictions for example X
        for example in unlabeled_X:
            for i, model in enumerate(model_ensemble):
                # perform predictions on the unlabeled examples
                prob_dist = model.predict(example)

                # use the argmax
                predictions_ensemble[i] = int(np.argmax(prob_dist.value())) # get pseudo labels

            # if the other two taggers agree, add prediction to third
            for i in range(len(model_ensemble)):
                j, k = [x for x in model_indices if x != i]
                new_examples = []

                # if tagger j and k agree, add to train data of i
                if predictions_ensemble[j] == predictions_ensemble[k]:
                    if not args.disagreement:  # original formulation
                        new_examples.append(example)
                        new_train_preds[i].append(predictions_ensemble[j])
                    else:
                        # tri-training with disagreement
                        # only add if two agree and third predicted something different
                        if predictions_ensemble[j] != predictions_ensemble[i]:
                            new_examples.append(example)
                            new_train_preds[i].append(predictions_ensemble[j])
                new_train_X[i] = scipy.sparse.vstack([new_train_X[i]] + new_examples)

            # reset
            predictions_ensemble = {}  # keep per tagger predictions for example X

        # get some output statistics
        for i in range(len(model_ensemble)):
            if not args.online:
                num_new_examples = new_train_X[i].shape[0] - original_train_X.shape[0]
            else:
                num_new_examples = new_train_X[i].shape[0]
            avg_num_new_examples += num_new_examples
            print('Added {}/{} pseudo-labeled examples after epoch {}.'.format(
                num_new_examples, unlabeled_X.shape[0], num_epochs))

        if num_epochs != 0:
            # do early stopping on target validation data
            if val_acc > best_val_acc:
                print('Val acc %.4f is better than best val acc %.4f.' % (val_acc, best_val_acc))
                best_val_acc = val_acc
                best_test_acc = test_acc
                num_epochs_no_improvement = 0
            else:
                print('Val acc %.4f is worse than best val acc %.4f.' % (val_acc, best_val_acc))
                num_epochs_no_improvement += 1
            if num_epochs_no_improvement == 2:
                print('No improvement for %d epochs. Early stopping...' % num_epochs_no_improvement)
                return best_val_acc, best_test_acc
    return dev_test_acc, test_acc


def temporal_ensembling_sentiment(vocab, args, train_X, train_Y, val_X, val_Y,
                            dev_test_X, dev_test_Y, test_X, test_Y, unlabeled_X,
                            unlabeled_Y, run_num):
    """Temporal ensembling as proposed in Laine & Aila (2017).
    Temporal Ensembling for Semi-Supervised Learning. ICLR 2017."""
    assert len(unlabeled_Y) == 0
    options = load_config(args.config)

    print('Ramp-up length: %d. Ensemble momentum: %.2f. Unsupervised weight: '
          '%.2f. Candidate pool size: %d.'
          % (args.ramp_up_len, args.ensemble_momentum, args.unsupervised_weight,
             args.candidate_pool_size))

    # load the model that has been trained for one epoch on the training set
    model = sentiment_model.load_model(os.path.join(args.start_model_dir, '%s_run%d' % (args.start, run_num)))

    # create dummy unlabeled labels
    unlabeled_labels = [-1] * unlabeled_X.shape[0]

    # we use the unsupervised objective on both labeled and unlabeled examples
    num_labels = 2
    # note: these are the # of individual training/unlabeled predictions
    n_train = train_X.shape[0]
    n_unlabeled = unlabeled_X.shape[0]
    num_examples = n_train + n_unlabeled
    ensemble_preds = np.zeros((num_examples, num_labels))
    trg_vectors = np.zeros((num_examples, num_labels))
    val_acc, dev_test_acc, test_acc = 0, 0, 0

    # default learning rate of Adam is 0.001
    orig_learning_rate = model.trainer.learning_rate
    for i in range(1, args.max_iterations):
        # calculate the weight for the unsupervised loss;
        # they use a Gaussian ramp-up curve of e^(-5(1-t)^2) for the first n
        # epochs of training, where t goes linearly from 0-1
        factor = (i-1) / args.ramp_up_len if i-1 < args.ramp_up_len else 1
        ramp_up_weight = np.exp(-5*(1-factor)**2)
        unsup_weight = ramp_up_weight * args.unsupervised_weight
        print("iter: {0}. ramp_up_weight: {1:.4f}. unsup_weight: {2:.4f} ".format(
            i, ramp_up_weight, unsup_weight))

        if args.lr_ramp_up:
            model.trainer.learning_rate = orig_learning_rate * ramp_up_weight
            print('Ramping-up lr. New lr: {:.7f}.'.format(model.trainer.learning_rate))

        if i > 1:
            # skip the first epoch as the model was already trained for 1 epoch
            # and for simplification as the trg_vectors won't be useful yet
            # at each iteration, we get the target vectors and variance weights
            # for the sampled indices ex_indices
            model.fit(scipy.sparse.vstack([train_X, unlabeled_X]), train_Y + unlabeled_labels, 1,
                      trg_vectors=trg_vectors,
                      unsup_weight=unsup_weight,
                      labeled_weight_proportion=args.labeled_weight_proportion, seed=args.seed)

            # evaluate on the validation and test dataset; we do this only
            # for hyperparameter optimization
            val_acc = accuracy_score(val_Y, model.get_predictions(val_X))
            test_acc = accuracy_score(test_Y, model.get_predictions(test_X))
            print("Val acc: {0:.4f}. Test acc: {1:.4f}.".format(val_acc, test_acc))
            data_utils.log_self_training(args, val_acc, test_acc, i, 0, run_num)
        else:
            # just evaluate starting tagger, do not log
            print("eval start model")
            test_acc = accuracy_score(test_Y, model.get_predictions(test_X))
            print("Test acc: {0:.4f}.".format(test_acc))

        print('Obtaining predictions on %d labeled and %d unlabeled examples...'
              % (n_train, n_unlabeled))

        # get the predictions for the labeled examples and the selected unlabeled examples
        preds = np.array(model.get_predictions(scipy.sparse.vstack([train_X, unlabeled_X]),
                                                soft_labels=True))

        # update ensemble predictions
        ensemble_preds = args.ensemble_momentum * ensemble_preds + \
                         (1 - args.ensemble_momentum) * preds

        # construct target vectors
        trg_vectors = ensemble_preds / (1 - args.ensemble_momentum ** i)
    return dev_test_acc, test_acc


def selection(unlabeled_X, pseudo_labels, confidence_scores, num_examples,
              random):

    if random:
        # randomly select n examples and their predictions
        indices = np.random.permutation(
            range(len(unlabeled_X)))[:num_examples]
    else:
        # select examples with most confident predictions
        indices = [i for _, i in sorted(zip(confidence_scores, range(len(
            unlabeled_X))), reverse=True)][:num_examples]
    new_examples = [unlabeled_X[i] for i in indices]
    new_labels = [pseudo_labels[i] for i in indices]
    new_unlabeled = [x for i, x in enumerate(unlabeled_X)
                     if i not in indices]
    return new_examples, new_labels, new_unlabeled
