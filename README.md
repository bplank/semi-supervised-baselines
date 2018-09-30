# Strong Baselines for Neural Semi-supervised Learning under Domain Shift

Sebastian Ruder, Barbara Plank (2018). [Strong Baselines for Neural Semi-supervised Learning under Domain Shift](https://arxiv.org/pdf/1804.09530.pdf). _In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, Melbourne, Australia.



## Requirements

Python 3.6, `scipy`, `progress`, `sklearn`, and `gensim`. The packages can be installed with the following commands:
```
pip install scipy
pip install progress
pip install sklearn
pip install gensim
```
Alternatively, you can also install all packages directly with: ```pip install -r requirements.txt```

### DyNet

We use the neural network library [DyNet](http://dynet.readthedocs.io/en/latest/index.html),
which works well with networks that have dynamic structures. 
If you just want to run DyNet on CPU, you can install it with: ```pip install dynet```.
If you want GPU compatibility, follow the instructions [here](https://dynet.readthedocs.io/en/latest/python.html).

## Data

### Part-of-speech tagging

For part-of-speech tagging, Wall Street Journal (WSJ) data is used for training and development
(in `data/gweb_sancl/pos_fine/wsj`). For a specific target domain `TARGET`, unlabeled data from
`data/gweb_sancl/pos_fine/unlabeled/gweb-TARGET-unlabeled.txt` is used and test data is in
`data/gweb_sancl/TARGET`.

### Sentiment analysis

For sentiment analysis, download the processed version of the Multi-Domain Sentiment Dataset (`processed_acl.tar.gz`)from [here](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/).
Extract it to `data/processed_acl`. We use the same splits as in (Ganin et al., 2016; Saito et al., 2017). 
Note that these are different from the "standard" splits used in (Blitzer et al., 2007).

## Examples

We provide examples for running the MT-Tri POS tagger and sentiment model.

### Example for running the Tagger (MT-Tri)

1. Make sure the unlabeled and labeled seed data is in `data/gweb_sancl` (example for `answers` is provided)
2. Download the embeddings from https://www.dropbox.com/s/4easof0ggsbox9y/embeds-acl2018.tar.gz?dl=0 and extract to `embeds/`

Now run the tagger (which by default uses the `pos_glove` setup and the 10% setup):

```
sh run-tagger-mttri.sh
```

### Example for running the sentiment model (MT-Tri)

1. Download the sentiment data and extract it to `data/processed_acl` as described above.

Run the sentiment model (the default setting is Books -> DVD):
```
sh run-sentiment-mttri.sh
```

## Reference

If you make use of the contents of this repository, we appreciate citing the following paper:
```
@InProceedings{P18-1096,
  author =	"Ruder, Sebastian
  	 and Plank, Barbara",
  title =    "Strong Baselines for Neural Semi-Supervised Learning under Domain Shift",
  booktitle = 	     "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year =    "2018",
  publisher =	"Association for Computational Linguistics",
  pages =   "1044--1054",
  location =	"Melbourne, Australia",
  url =    "http://aclweb.org/anthology/P18-1096"
}

```
