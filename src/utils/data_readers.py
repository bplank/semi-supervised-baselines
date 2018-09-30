#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility methods for loading and processing review/tagging data.

The readers process the data and return a data2domain dictionary:

    domain2data = {domain: [[], [], None] for domain in domains}

which consists of:

    X, y, X_unlabeled

where

    TOKENS: raw sentences from labeled data
    LABELS: corresponding labels
    UNLABELED: raw sentences from unlabeled data (optional)

"""

TOKENS = 0
LABELS = 1
UNLABELED = 2

import os
import codecs
import numpy as np
import scipy.sparse

from utils.constants import SENTIMENT, POS

#from simpletagger import read_conll_file

#from bist_parser.bmstparser.src.utils import read_conll

### TODO: use names for indices: 0 => X (sentences, 1: labels, 2: unlabeled tokens

# =============== sentiment data reader functions ======

NEG_ID = 0  # the negative sentiment id
POS_ID = 1  # the positive sentiment id


def task2read_data_func(task):
    """Returns the read data method for each task."""
    if task == SENTIMENT:
        return read_processed
    if task in POS:
        return read_tagging_data
    #if task == PARSING:
    #    return read_parsing_data
    raise ValueError('No data reading function available for task %s.' % task)


def read_processed(domain_dir_path, unlabeled=False, max_train=None, max_unlabeled=None):
    """
    Reads the processed files in the processed_acl directory. Outputs the documents as strings of outputs the unigram
    and bigram features and their label. The domains are books, dvd, electronics, and kitchen and the files are named
    positive.review, negative.review, and unlabeled.review.
    :param dir_path: the path to the directory of the given domain
    :return: a dictionary that maps domains to a tuple of (labeled_reviews, labels, unlabeled_reviews)
             labeled_reviews is a list of reviews where each review is a list of (unordered) ngrams
             labels is a numpy array of label ids of shape (num_labels)
             unlabeled_reviews has the same format as labeled_reviews
    """
    def line2features(line):
        """Convert a row in the dataset to a sequence of ngrams."""
        # get the pre-processed features; these are a white-space separated list
        # of unigram/bigram occurrence counts in the document,
        # e.g. "must:1", "still_has:1"
        features = line.split(' ')[:-1]
        label = label2label_id(line.split(' ')[-1].split(':')[1].strip())

        # convert the features to a sequence (note: order does not matter here)
        # we do this to be able to later use the same post-processing for all
        ngram_seq = []
        for feature in features:
            ngram, count = feature.split(':')
            for _ in range(int(count)):
                ngram_seq.append(ngram)
        return ngram_seq, label

    reviews = []
    labels = []
    if unlabeled:
        file_path = os.path.join(domain_dir_path, 'unlabeled.review')
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                ngram_seq, label = line2features(line)
                reviews.append(ngram_seq)
                labels.append(label)
        return reviews, labels

    pos_file_path = os.path.join(domain_dir_path, 'positive.review')
    neg_file_path = os.path.join(domain_dir_path, 'negative.review')
    with open(pos_file_path, encoding='utf-8') as f_pos,\
            open(neg_file_path, encoding='utf-8') as f_neg:
        # we iterate over both files together so that positive and negative
        # examples alternate and we can do a straightforward split later
        for pos_line, neg_line in zip(f_pos, f_neg):
            ngram_seq, _ = line2features(pos_line)
            reviews.append(ngram_seq)
            labels.append(POS_ID)
            ngram_seq, _ = line2features(neg_line)
            reviews.append(ngram_seq)
            labels.append(NEG_ID)
    return reviews, labels


def label2label_id(label):
    if label == 'positive':
        return POS_ID
    elif label == 'negative':
        return NEG_ID
    raise ValueError('%s is not a valid label.' % label)


def get_all_docs(domain_data_pairs, unlabeled=True):
    """
    Return all labeled and undocumented documents of multiple domains.
    :param domain_data_pairs: a list of (domain, (labeled_reviews, labels, unlabeled_reviews)) tuples as obtained by
                              domain2data.items()
    :param unlabeled: whether unlabeled documents should be incorporated
    :return: a list containing the documents from all domains, the corresponding labels, and a list containing the
             domain of each example
    """
    docs, labels, domains = [], [], []
    for domain, (labeled_docs, doc_labels, unlabeled_docs) in domain_data_pairs:
        length_of_docs = 0
        if not scipy.sparse.issparse(labeled_docs):
            # if the labeled documents are not a sparse matrix, i.e. a tf-idf matrix, we can just flatten them into one array
            docs += labeled_docs
            length_of_docs += len(labeled_docs)
            if unlabeled:
                # if specified, we add the unlabeled documents
                docs += unlabeled_docs
                length_of_docs += len(labeled_docs)
        else:
            # if it is a sparse matrix, we just append the docs as a list and then stack the list in the end
            docs.append(labeled_docs)
            length_of_docs += labeled_docs.shape[0]
            if unlabeled and unlabeled_docs is not None:
                docs.append(unlabeled_docs)
                length_of_docs += unlabeled_docs.shape[0]
        labels.append(doc_labels)

        # we just add the corresponding domain for each document so that we can later see where the docs came from
        domains += [domain] * length_of_docs
    if scipy.sparse.issparse(labeled_docs):
        # finally, if the matrix was sparse, we can stack the documents together
        docs = scipy.sparse.vstack(docs)
    return docs, np.hstack(labels), domains


# =============== tagging data functions ======

def read_tagging_data(file_path, unlabeled=False, max_unlabeled=0, max_train=0, dann_setup=False):
    """
    Reads the CoNLL tagging files in the gweb_sancl/pos directory. Outputs the documents as list of lists with
    tokens and lists of corresponding tags. The domains are reviews, answer, emails, newsblogs, weblogs, wsj and
    the corresponding files are called gweb-{domain}-{dev|test}.conll in folder gweb_sancl/pos/{domain}

    :param dir_path: the path to the directory gweb_sancl
    :param unlabeled: read raw data
    :param max_unlabeled: max instances to read
    :param max_train: max instances to read (NB. assumes test/dev is anyway smaller than this)
    :return: words: a list of tokenized sentences
             tags: a list of lists of tags for each word
    """
    print('Reading data from {}...'.format(file_path))
    if unlabeled:
        data = []
        with open(file_path, 'rb') as f:
            for line in f:
                if max_unlabeled and len(data) == max_unlabeled:
                    break
                line = line.decode('utf-8','ignore').strip().split()
                data.append(line)
        print('Loaded... {} unlabeled instances'.format(len(data)))
        return data, []
    if max_train:
        data = list(read_conll_file(file_path))[:max_train]
    else:
        data = list(read_conll_file(file_path))
    words = [word for word, tag in data]
    tags = [tag for word, tag in data]
    print('Loaded... {} instances'.format(len(words)))
    return words, tags

def read_conll_file(file_name):
    """
    read in a file with format:
    word1    tag1
    ...      ...
    wordN    tagN

    Sentences MUST be separated by newlines!

    :param file_name: file to read in
    :return: generator of instances ((list of  words, list of tags) pairs)
    """
    current_words = []
    current_tags = []

    for line in open(file_name, encoding='utf-8'):
        line = line.strip()

        if line:
            word, tag = line.split('\t')
            current_words.append(word)
            current_tags.append(tag)

        else:
            yield (current_words, current_tags)
            current_words = []
            current_tags = []

    # if file does not end in newline (it should...), check whether there is an instance in the buffer
    if current_tags != []:
        yield (current_words, current_tags)

