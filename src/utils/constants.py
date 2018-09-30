"""
Some constants that are shared across files.
We also define the features here at the moment. This can later also be done in a task-dependent way.
"""

POS = 'pos'
SENTIMENT = 'sentiment'
TASKS = [POS, SENTIMENT]
POS_DOMAINS = ['answers', 'emails', 'newsgroups', 'reviews', 'weblogs', 'wsj']
SENTIMENT_DOMAINS = ['books', 'dvd', 'electronics', 'kitchen']

FEATURE_SETS = ['similarity', 'topic_similarity', 'word_embedding_similarity', 'diversity']
SIMILARITY_FUNCTIONS = ['jensen-shannon', 'renyi', 'cosine', 'euclidean', 'variational', 'bhattacharyya']
DIVERSITY_FEATURES = ['num_word_types', 'type_token_ratio', 'entropy', 'simpsons_index', 'quadratic_entropy', 'renyi_entropy']

BASE = 'base'
MTTRI_BASE = 'mttri_base'
SELF_TRAINING = 'self-training'
CO_TRAINING = 'co-training'
CODA = 'coda'  # (Chen et al., 2011)
TRI_TRAINING = 'tri-training'
MTTRI = 'mttri'
TASK_IDS = ['F0', 'F1', 'Ft']   # inspired by asymmetric tri-training (Saito et al., 2017)
TEMPORAL_ENSEMBLING = 'temporal_ensembling'
STRATEGIES = [BASE, SELF_TRAINING, CO_TRAINING, CODA, TRI_TRAINING,
              TEMPORAL_ENSEMBLING, MTTRI_BASE, MTTRI]

# self-training selection possibilities
RANDOM = 'random'
RANKED = 'ranked'

MAX_SEED=99999999

# model files for loading pre-trained models
EPOCH_1 = 'epoch_1'
PATIENCE_2 = 'patience_2'
