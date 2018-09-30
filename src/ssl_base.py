"""
Neural semi-supervised learning
"""

import sys
import argparse
import os
import itertools
import numpy as np
import random
import _dynet as dynet

from utils.constants import TASKS, SENTIMENT, POS, SENTIMENT_DOMAINS, \
    POS_DOMAINS, SELF_TRAINING, CO_TRAINING, CODA, TRI_TRAINING, \
    STRATEGIES, RANDOM, RANKED, BASE, EPOCH_1, \
    PATIENCE_2, TEMPORAL_ENSEMBLING, MTTRI_BASE, MAX_SEED
from utils import data_readers, task_utils, data_utils, similarity_measures
from utils.vocab import Vocab


def main(args):
    
    # print argument values
    print("Info: arguments\n\t" + "\n\t".join(["{}: {}".format(a, v) for a, v in vars(args).items()]), file=sys.stderr)

    # set seed
    if not args.seed:
        seed = random.randint(1, MAX_SEED)
        args.seed = seed
        print("using seed: ", args.seed)

    init_dynet(args)

    assert os.path.exists(args.data)
    if args.task == SENTIMENT:
        assert args.trg_domain in SENTIMENT_DOMAINS, f'Error: {args.trg_domain} is not a sentiment domain.'
        assert args.src_domain is not None, 'Error: A source domain must be specified.'
    elif args.task == POS:
        assert args.trg_domain in POS_DOMAINS, f'Error: {args.trg_domain} is not a POS domain.'

    if args.task == SENTIMENT:
        assert args.max_vocab_size == 5000, f'Error: Max vocab size is not 5000.'

    # create the model and log directories if they do not exist
    for dir_path in [args.model_dir, os.path.dirname(args.log_file)]:
        print("Check if directory exists:", dir_path)
        if not os.path.exists(dir_path):
            print('Creating %s...' % dir_path)
            os.makedirs(dir_path)
    # create predictions folder if it does not exist
    if args.output_predictions:
        if not os.path.exists(args.output_predictions):
            print('Creating output predictions folder: {}'.format(args.output_predictions))
            os.makedirs(args.output_predictions)

    if args.strategy not in [BASE, MTTRI_BASE]:
        # check that pre-trained models exist
        assert args.start_model_dir is not None,\
            'Error: start_model_dir needs to be provided.'
        for suffix in ['.model', '.params.pickle']:
            if args.strategy != TRI_TRAINING: # tri-training w/ disagreement is enabled with --disagreement
                model_file = os.path.join(args.start_model_dir, args.start + "_run"+ str(args.start_run) + suffix)
                assert os.path.exists(model_file),\
                    'Error: %s does not exist.' % model_file
            else:
                # check if 3 exists for tri_training
                model_name = args.start + "_bootstrap3_run" + str(args.start_run) + suffix
                model_file = os.path.join(args.start_model_dir, model_name)
                assert os.path.exists(model_file), \
                    'Error: %s does not exist.' % model_file

    if args.task == POS:
        pos_path = os.path.join(args.data, 'gweb_sancl', 'pos_fine')
        assert os.path.exists(pos_path)
        train_path = os.path.join(pos_path, 'wsj', 'gweb-wsj-train.conll')
        dev_path = os.path.join(pos_path, 'wsj', 'gweb-wsj-dev.conll')
        unlabeled_path = os.path.join(args.data, 'gweb_sancl', 'unlabeled',
                                      'gweb-%s.unlabeled.txt' % args.trg_domain)
        dev_test_path = os.path.join(pos_path, args.trg_domain,
                                     'gweb-%s-dev.conll' % args.trg_domain)
        test_path = os.path.join(pos_path, args.trg_domain,
                                 'gweb-%s-test.conll' % args.trg_domain)
    elif args.task == SENTIMENT:
        sentiment_path = os.path.join(args.data, 'processed_acl')
        train_path = dev_path = os.path.join(sentiment_path, args.src_domain)
        # since there is no target domain test set, we just tune hyperparams
        # on book->dvd
        unlabeled_path = dev_test_path = test_path = os.path.join(sentiment_path, args.trg_domain)
    else:
        raise ValueError()

    # load the data and save it to a pickle file
    split2data = {}
    read_data = data_readers.task2read_data_func(args.task)
    for split, path_ in zip(['train', 'dev', 'dev_test', 'test', 'unlabeled'],
                            [train_path, dev_path, dev_test_path, test_path, unlabeled_path]):
        if split == 'unlabeled':
            data = read_data(path_, unlabeled=True, max_unlabeled=args.max_unlabeled)  # [[instances],[]]
        else:
            data = read_data(path_, unlabeled=False, max_train=args.max_train) # keeps [[instances],[labels]]

        # the DANN paper uses somewhat different splits than the standard, so
        # we create the splits here
        if args.task == SENTIMENT:
            if split == 'train':
                # in the DANN paper, they use all 2000 training examples
                pass
            elif split == 'dev':
                # the DANN paper uses 200 target samples for testing,
                # which are read from the unlabeled file
                continue
            elif split == 'unlabeled':
                # in the DANN paper, we use the content of the unlabeled file
                # for testing
                split = 'test'
                data, data_dev = (data[0][:-200], data[1][:-200]), (data[0][-200:], data[1][-200:])
                # we use 200 labeled samples for validation
                split2data['dev'] = list(data_dev)
            elif split == 'test':
                # in the DANN set-up, we use this data as unlabeled data
                split = 'unlabeled'
                data = data[0], []
            elif split == 'unlabeled':
                data = data[0], []
        elif args.max_unlabeled and split == 'unlabeled':
            print('Restricting # of unlabeled examples to',
                  args.max_unlabeled, file=sys.stderr)
            new_data = data[0][:args.max_unlabeled], data[1][:args.max_unlabeled]
            if len(new_data[0]) < args.max_unlabeled:
                args.max_unlabeled = len(new_data[0]) # set if |unlabeled| < --max-unlabeled
            data = new_data
        elif args.max_train and split == 'train':
            print('Restricting # of labeled training examples to',
                  args.max_train, file=sys.stderr)
            data = data[0][:args.max_train], data[1][:args.max_train]

        split2data[split] = list(data)
        print('# of %s examples: %d.' % (split, len(data[0])))

    vocab_dir = args.model_dir if args.strategy in [BASE, MTTRI_BASE] else args.start_model_dir
    vocab_path = os.path.join(vocab_dir, 'vocab.txt')
    vocab = Vocab(vocab_path, max_vocab_size=args.max_vocab_size)
    if not os.path.exists(vocab_path):
        # build the vocabulary
        assert args.strategy in [BASE, MTTRI_BASE],\
            'Error: Vocabulary should only be created with the base model.'
        vocab.create(split2data['train'][0] + split2data['unlabeled'][0],
                     lowercase=args.lowercase)
    else:
        vocab.load()

    if args.task == SENTIMENT:
        print('Creating binary training data...')
        split2data = data_utils.get_tfidf_data(split2data, vocab,
                                               tfidf=True)
    elif args.task.startswith('pos'):
        print('Using words as training data for POS tagging...')
    elif args.task == 'parsing':
        print('Using CoNLL entries as training data for parsing. Using word forms to extract feature representations...')
        for split, data in split2data.items():
            split2data[split][0] = [[conll_entry.form for conll_entry in
                                     conll_entries] for conll_entries in
                                    data[0]]
    else:
        raise ValueError('Training data retrieval for task %s is not implemented.' % args.task)

    run_scores = []
    train_func = task_utils.task2train_func(args.task, args.strategy)
    for i in range(args.num_runs):
        run_num = i+1
        print('\nRun %d/%d.' % (run_num, args.num_runs))

        val_score, test_score = train_func(
            vocab, args, *itertools.chain.from_iterable(
                [split2data['train'], split2data['dev'], split2data['dev_test'],
                 split2data['test'], split2data['unlabeled']]), run_num
        )
        print('Validation score: %.3f. Test score: %.3f'
              % (val_score, test_score))
        run_scores.append((val_score, test_score))

    if args.num_runs > 1:
        # log the results of multiple runs to a file
        data_utils.log_to_file(args, run_scores)

def init_dynet(args):
    """initialize DyNet"""
    dyparams = dynet.DynetParams()
    # Fetch the command line arguments (optional)
    dyparams.from_args()
    # Set some parameters manualy (see the command line arguments documentation)
    dyparams.set_random_seed(args.seed)
    # Initialize with the given parameters
    dyparams.init()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn a domain similarity metric using Bayesian Optimization.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # DyNet parameters
    parser.add_argument('--dynet-autobatch', type=int, required=True,
                        help='turn on DyNet auto-batching with 1')
    parser.add_argument('--dynet-mem', type=int, required=False,
                         help='the memory used')
    parser.add_argument('--seed', type=int, required=False, help='the (DyNet) seed used throughout. If not given, random integer between 1 and MAX_SEED')

    # task and paths parameters
    parser.add_argument('-d', '--data', required=True,
                        help='the data folder containing the gweb_sancl and '
                             'processed_acl folders')
    parser.add_argument('-m', '--model-dir', required=True,
                        help='the directory where the model should be saved')
    parser.add_argument('--task', choices=TASKS, default='pos',
                        help='the task on which we train')
    parser.add_argument('-s', '--src-domain', default='books',
                        choices=SENTIMENT_DOMAINS,
                        help='for SA, we also require a source domain')
    parser.add_argument('-t', '--trg-domain', required=True,
                        choices=SENTIMENT_DOMAINS + POS_DOMAINS,
                        help='the target domain to which we adapt our model')
    parser.add_argument('--strategy', choices=STRATEGIES, required=True,
                        help='the data selection strategy that should be used')
    parser.add_argument('--config', required=True,
                        help='task-specific json configuration file')
    parser.add_argument('--save-epoch-1', default=False,
                        help='save model after first epoch', action="store_true")
    # data parameters
    parser.add_argument('--max-train', type=int, default=0,
                        help='restrict the # of labeled training examples (0 means use all); '
                             'good for faster prototyping')
    parser.add_argument('--max-unlabeled', type=int, default=0,
                        help='restrict the # of unlabeled examples (0 means use all); good for '
                             'faster prototyping')
    parser.add_argument('-l', '--log-file', required=True,
                        help='the path to which validation and test accuracies '
                             'should be logged')
    parser.add_argument('--output-predictions', default=None,
                        help='the data folder to save predictions to')

    # general SSL parameters
    parser.add_argument('--start', choices=[EPOCH_1, PATIENCE_2],
                        default=PATIENCE_2,
                        help='starting point model')
    parser.add_argument('--start-run', default=1, type=int)
    parser.add_argument('--start-model-dir', help="path to starting point model")
    parser.add_argument('--max-iterations', type=int, default=20,
                        help='the maximum number of iterations for '
                             'self-training, temporal ensembling or tri-training')
    parser.add_argument('--save-final-model', help="save last model", default=False, action="store_true")
    parser.add_argument('--candidate-pool-size', type=int, default=10000,
                        help='the candidate pool size sampled in each round of self-training / tri-training')

    # tri-training parameters
    parser.add_argument('--bootstrap', action='store_true',
                        help='use bootstrap samples for training the base method'
                             '(necessary for tri-training)')
    parser.add_argument('--disagreement', default=False, action="store_true",
                        help='use tri-training with disagreement')

    # MTTRI parameters
    parser.add_argument('--orthogonality-weight', default=0., type=float,
                        help='weight for the orthogonality constraint;'
                             '0 == unused')
    parser.add_argument('--adversarial', action='store_true',
                        help='use a domain-adversarial loss')
    parser.add_argument('--add-hidden', action='store_true',
                        help='add another that produces task-specific '
                             'transformations')
    parser.add_argument('--adversarial-weight', default=1.0, type=float,
                        help='weight for adversarial loss (1.0=no weighting)')
    parser.add_argument('--asymmetric', default=False, action="store_true",
                        help='train third model only on pseudo-labeled examples similar to '
                             'asymmetric tri-training (Saito et al., 2017)')
    parser.add_argument('--predict', default='majority', choices=('Ft','majority'),
                        help='asymmetric tri-training (Saito et al., 2017) uses Ft; use majority instead')
    parser.add_argument('--asymmetric-type', default='pair', choices=('org', 'pair'),
                        help='how to add pseudo-labeled samples:'
                             'org: is as in asymmetric tri-training (Saito et al., 2017), add only agreements of F0 & F1'
                             'pair: add pair-wise agreements')
    parser.add_argument('--candidate-pool-scheduling', default=False, action="store_true",
                       help='use candidate pool size scheduling as in Saito et al., (2017)')
    parser.add_argument('--bootstrap-src', default=False, action="store_true",
                        help='resample src instances in each asymmetric tri-training round')


    # MTTRI base parameters
    parser.add_argument('--size-bootstrap', default=1.0, type=float,
                        help='1.0 = proper bootstrap, if < 1.0 use smaller size for initial bootstrap sample')


    # self-training parameters
    parser.add_argument('--confidence-threshold', type=float, default=0,
                        help='the confidence threshold for self-training/tri-training (0=not used)')
    parser.add_argument('--num-select-examples', type=int, default=0,
                        help='# of examples that should be selected with '
                             'self-training/tri-training; if > 0, selects examples '
                             'rather than using the confidence threshold')
    parser.add_argument('--selection', choices=[RANKED, RANDOM], default=RANKED,
                        help='how examples for self-training should be '
                             'selected; top n examples or random examples')
    parser.add_argument('--soft-labels', action='store_true',
                        help='use soft labels instead of hard labels')
    parser.add_argument('--temperature', type=float, default=1.,
                        help='temperature with which to smooth probability '
                             'distribution; higher temperatures smooth more;'
                             'range of 2.5-8 should work best')

    # temporal ensembling parameters
    parser.add_argument('--ensemble-momentum', default=.6, type=float,
                        help='the momentum for aggregating ensemble predictions')
    parser.add_argument('--ramp-up-len', default=10, type=int,
                        help='the length of the ramp-up period for the weight '
                             'for the unsupervised loss')
    parser.add_argument('--unsupervised-weight', default=1.0, type=float,
                        help='the weight for the unsupervised loss')
    parser.add_argument('--labeled-weight-proportion', default=1.0, type=float,
                        help='the proportion of the unsupervised weight that '
                             'should be assigned to labeled examples')
    parser.add_argument('--lr-ramp-up', action='store_true',
                        help='apply the same ramp-up schedule to the '
                             'learning rate')

    # processing parameters
    parser.add_argument('-v', '--max-vocab-size', default=0, type=int, help='the maximum size of the vocabulary. if 0 use all from train')
    parser.add_argument('--lowercase', action='store_true',
                        help='lower-case words for processing')

    # training parameters
    parser.add_argument('--num-runs', type=int, default=5,
                        help='the number of experiment runs for each domain')
    parser.add_argument('--online', default=False, action="store_true",
                        help='use only newly labeled data')

    args = parser.parse_args()
    main(args)
