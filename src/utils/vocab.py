#!/usr/bin python3
# -*- coding: utf-8 -*-

import codecs
import itertools
import operator
from collections import Counter


class Vocab:
    """
    The vocabulary class. Stores the word-to-id mapping.
    """
    def __init__(self, vocab_path, max_vocab_size=0):
        self.max_vocab_size = max_vocab_size
        self.vocab_path = vocab_path
        self.size = 0
        self.word2id = {}
        self.id2word = {}

    def load(self):
        """
        Loads the vocabulary from the vocabulary path.
        """
        assert self.size == 0, 'Vocabulary has already been loaded or built.'
        print('Reading vocabulary from %s...' % self.vocab_path)
        with codecs.open(self.vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.max_vocab_size != 0 and i >= self.max_vocab_size:
                    print('Vocab in file is larger than max vocab size. Only using top %d words.' % self.max_vocab_size)
                    break
                word, idx = line.split('\t')
                self.word2id[word] = int(idx.strip())
        self.size = len(self.word2id)
        self.id2word = {index: word for word, index in self.word2id.items()}
        if self.max_vocab_size != 0:
            assert self.size <= self.max_vocab_size, 'Loaded vocab is of size %d., max vocab size is %d.' \
                                                 % (self.size, self.max_vocab_size)

    def create(self, texts, lowercase=False):
        """
        Creates the vocabulary and stores it at the vocabulary path.
        :param texts: a list of lists of tokens
        :param lowercase: lowercase the input texts
        """
        assert self.size == 0, 'Vocabulary has already been loaded or built.'
        print('Building the vocabulary...')
        if lowercase:
            print('Lower-casing the input texts...')
            texts = [[word.lower() for word in text] for text in texts]

        word_counts = Counter(itertools.chain(*texts))

        if self.max_vocab_size != 0:
            # get the n most common words
            most_common = word_counts.most_common(n=self.max_vocab_size-1)
        else:
            most_common = word_counts.most_common(n=len(word_counts)) # use all

        # construct the word to index mapping
        self.word2id = {'_UNK': 0}
        for i, (word, _) in enumerate(most_common):
            self.word2id[word] = len(self.word2id)
        self.id2word = {index: word for word, index in self.word2id.items()}

        print('Writing vocabulary to %s...' % self.vocab_path)
        with codecs.open(self.vocab_path, 'w', encoding='utf-8') as f:
            for word, index in sorted(self.word2id.items(), key=operator.itemgetter(1)):
                f.write('%s\t%d\n' % (word, index))
        self.size = len(self.word2id)

    def create_from_w2id(self, w2i):
        """
        create from an existing mapping
        """
        self.word2id = {'_UNK': 0}
        for w in w2i:
            self.word2id[w] = w2i[w]
        self.id2word = {index: word for word, index in self.word2id.items()}

        print('Writing vocabulary to %s...' % self.vocab_path)
        with codecs.open(self.vocab_path, 'w', encoding='utf-8') as f:
            for word, index in sorted(self.word2id.items(), key=operator.itemgetter(1)):
                f.write('%s\t%d\n' % (word, index))
        self.size = len(self.word2id)
