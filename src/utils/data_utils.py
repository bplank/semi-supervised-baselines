#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from datetime import datetime

from utils.constants import SENTIMENT

FORMAT = '%Y-%m-%d-%H%M%S'
# =========== general data helper functions =========


def get_tfidf_data(split2data, vocab, tfidf=True):
    """
    Transform the tokenized documents into a tf-idf matrix for each split.
    :param split2data: mapping of data splits (train, dev, test, unlabeled) to
                       a tuple containing (word_sequences, labels)
    :param vocab: an instance of the Vocabulary class
    :return: a mapping of splits to (tf_idf_matrix, labels) where the matrix is
            of type scipy.sparse.csr.csr_matrix and shape (num_examples,
            vocab_size) and labels is a list of length num_examples
    """
    vectorizer_class = TfidfVectorizer if tfidf else CountVectorizer
    if tfidf:
        print('Using tf-idf weighting on features...')
    vectorizer = vectorizer_class(vocabulary=vocab.word2id, tokenizer=lambda x: x, preprocessor=lambda x: x)
    examples = []
    for split, data in split2data.items():
        if split == 'test':
            continue
        examples += data[0]
    vectorizer.fit(examples)
    for split, data in split2data.items():
        split2data[split] = vectorizer.transform(data[0]), data[1]
    return split2data

def log_selection(args, top_indices, domain_labels):
    """
    log JS selection results
    """
    outfile = "{}.js.log".format(args.log_file)
    with open(outfile, 'a') as f:
        print('Writing results to {}...'.format(outfile))
        f.write('%s\n' %
                str(" ".join(["{}-{}".format(domain,inst_num) for inst_num,domain in zip(top_indices,domain_labels)])))


def log_self_training(args, val_acc, test_acc, epoch=0, num_new_examples=0, run_num=0):
    """
    Log the results of self-training to a file.
    :param args: the arguments used as input to the script
    :param val_acc: the validation accuracy
    :param test_acc: the test accuracy
    :param epoch: the number of the self-training epoch
    :param num_new_examples: the number that of pseudo-labeled examples that
                             have been added for this epoch
    :param run_num: the number of the run
    """
    trg_domain = '%s-%s' % (args.src_domain, args.trg_domain)\
        if args.task == SENTIMENT else args.trg_domain
    with open(args.log_file, 'a') as f:
        print('Writing results to %s...' % args.log_file)
        f.write('%s\t%s\t%s\t%s\t%.4f\t%.4f\t%d\t%d\t%s\t%s\t%s\t%s\n' %
                (datetime.now().strftime(FORMAT), args.task, trg_domain, args.strategy, val_acc,
                 test_acc, epoch, num_new_examples, str(args.max_train),
                 str(args.max_unlabeled), str(run_num), ' '.join(
                     ['%s=%s' % (arg, str(getattr(args, arg))) for
                      arg in vars(args)])))


def log_to_file(args, run_scores):
    """
    Log the results of experiment runs to a file.
    :param args: the arguments used as input to the script
    :param args: the scores obtained in the runs; a list of (val_acc, test_acc)
                 tuples
    """
    trg_domain = '%s-%s' % (args.src_domain, args.trg_domain)\
        if args.task == SENTIMENT else args.trg_domain
    with open(args.log_file + ".final_avg", 'a') as f:
        val_accs, test_accs = zip(*run_scores)
        mean_val_acc, std_val_acc = np.mean(val_accs), np.std(val_accs)
        mean_test_acc, std_test_acc = np.mean(test_accs), np.std(test_accs)

        # target domain \t method \t val_acc \t test_acc \t
        # max_train \t max_unlabelled \t other parameters \t scores
        f.write('%s\t%s\t%s\t%s\t%.4f +-%.4f\t%.4f +-%.4f\t%s\t%s\t%s\t[%s]\t[%s]\n'
                % (datetime.now().strftime(FORMAT), args.task, trg_domain, args.strategy, mean_val_acc,
                   std_val_acc, mean_test_acc, std_test_acc, str(args.max_train),
                   str(args.max_unlabeled),
                   ' '.join(['%s=%s' % (arg, str(getattr(args, arg))) for arg in
                             vars(args)]),
                   ', '.join(['%.4f' % v for v in val_accs]),
                   ', '.join(['%.4f' % v for v in test_accs])))


def read_feature_weights_file(feature_weights_path):
    """
    Reads a manually created file containing the learned feature weights for some task, trg domain, and feature set and
    returns them.
    The file format is this (note that ~ is used as delimiter to avoid clash with other delimiters in the feature sets):
    books~similarity diversity~[0.0, -0.66, -0.66, 0.66, 0.66, -0.66, 0.66, 0.0, 0.0, -0.66, 0.66, 0.66]
    ...
    :param feature_weights_path: the path to the feature weights file
    :return: a generator of tuples (feature_weights_domain, feature_set, feature_weights)
    """
    print('Reading feature weights from %s...' % feature_weights_path)
    with open(feature_weights_path, 'r') as f:
        for line in f:
            feature_weights_domain, feature_set, feature_weights = line.split('~')
            feature_weights = feature_weights.strip('[]\n')
            feature_weights = feature_weights.split(', ')
            feature_weights = [float(f) for f in feature_weights]
            print('Feature weights domain: %s. Feature set: %s. Feature weights: %s' %
                  (feature_weights_domain, feature_set, str(feature_weights)))
            yield feature_weights_domain, feature_set, feature_weights


def read_parsing_evaluation(evaluation_file_path):
    """
    Read the labeled attachment score, unlabeled attachment score, and label accuracy score from a file produced
    by the parsing evaluation perl script. The beginning of the file looks like this:
    Labeled   attachment score: 6995 / 9615 * 100 = 72.75 %
    Unlabeled attachment score: 7472 / 9615 * 100 = 77.71 %
    Label accuracy score:       8038 / 9615 * 100 = 83.60 %
    ...
    :param evaluation_file_path: the path of the evaluation file produced by the perl script
    :return: the labeled attachment score, the unlabeled attachment score, and the label accuracy score
    """
    try:
        with open(evaluation_file_path, 'r') as f:
            lines = f.readlines()
            las = float(lines[0].split('=')[1].strip('% \n'))
            uas = float(lines[1].split('=')[1].strip('% \n'))
            acc = float(lines[2].split('=')[1].strip('% \n'))
    except:
        las = 0.0
        uas = 0.0
        acc = 0.0
    return las, uas, acc
