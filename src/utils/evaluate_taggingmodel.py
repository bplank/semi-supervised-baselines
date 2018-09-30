#### Takes bilstm model and evaluates it on an eval file
## 
## Usage: ./evaluate_taggingmodel.py PATH_TO_MODEL PATH_TO_EVAL_FILE

from simplebilty import SimpleBiltyTagger
from simplebilty import load_tagger, save_tagger
from lib.mio import read_conll_file
from vocab import Vocab
import os

from collections import namedtuple
import sys
import json
import dynet as dynet

config=sys.argv[1]
model=sys.argv[2]
testfile=sys.argv[3]
vocabfile=os.path.dirname(model)+"/vocab.txt"

d = json.load(open(config))
config = namedtuple("options", d.keys())(*d.values())

vocab = Vocab(vocabfile)

if "embeds" in config:
    tagger = SimpleBiltyTagger(config.in_dim, config.h_dim, config.c_in_dim, config.h_layers,embeds_file=config.embeds,word2id=vocab.word2id,)
else:
    tagger = SimpleBiltyTagger(config.in_dim, config.h_dim, config.c_in_dim, config.h_layers,embeds_file=None,word2id=vocab.word2id)



tagger = load_tagger(model)

test_X, test_Y = tagger.get_data_as_indices(testfile)

correct, total = tagger.evaluate(test_X, test_Y)
print("accuracy", correct/total)

dev_test_labels=[]
for _, tags in read_conll_file(testfile):
    dev_test_labels.append(tags)
tagger.get_predictions_output(test_X, dev_test_labels, "dev.xxx.out")
