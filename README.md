# Topic Context Modell (TCM)

Calculates the surprisal of a word given a context.

![Tests](https://github.com/jnphilipp/tcm/actions/workflows/tests.yml/badge.svg)
[![pypi Version](https://img.shields.io/pypi/v/topic-context-model.svg?logo=pypi&logoColor=white)](https://pypi.org/project/topic-context-model/)

## Requirements

* Python >= 3.10
* scipy
* scikit-learn

## Usage

```
$ python tcm.py -h
usage: tcm [-h] [-V] [-m {lda,lsa}] [--model-file MODEL_FILE] [--data DATA [DATA ...]]
           [--fields FIELDS [FIELDS ...]] [--words WORDS] [--n-components N_COMPONENTS]
           [--doc-topic-prior DOC_TOPIC_PRIOR] [--topic-word-prior TOPIC_WORD_PRIOR]
           [--learning-method LEARNING_METHOD] [--learning-decay LEARNING_DECAY] [--learning-offset LEARNING_OFFSET]
           [--max-iter MAX_ITER] [--batch-size BATCH_SIZE] [--evaluate-every EVALUATE_EVERY] [--perp-tol PERP_TOL]
           [--mean-change-tol MEAN_CHANGE_TOL] [--max-doc-update-iter MAX_DOC_UPDATE_ITER] [--n-jobs N_JOBS]
           [--random-state RANDOM_STATE] [-v] [--log-format LOG_FORMAT] [--log-file LOG_FILE]
           [--log-file-format LOG_FILE_FORMAT]
           {train,surprisal} [{train,surprisal} ...]

positional arguments:
  {train,surprisal}     what to do, train lda/lsa or calculate surprisal.

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  -m {lda,lsa}, --model {lda,lsa}
                        which model to use. (default: lda)
  --model-file MODEL_FILE
                        file to load model from or save to, if path exists tries to load model. (default: lda.jl.z)
  --data DATA [DATA ...]
                        file(s) to load texts from, either txt or csv optionally gzip compressed. (default: None)
  --fields FIELDS [FIELDS ...]
                        field(s) to load texts when using csv data. (default: None)
  --words WORDS         file to load words from and/or save to, either txt or json optionally gzip compressed. (default: words.txt.gz)
  -v, --verbose         verbosity level; multiple times increases the level, the maximum is 3, for debugging. (default: 0)
  --log-format LOG_FORMAT
                        set logging format. (default: %(message)s)
  --log-file LOG_FILE   log output to a file. (default: None)
  --log-file-format LOG_FILE_FORMAT
                        set logging format for log file. (default: [%(levelname)s] %(message)s)

LDA config:
  --n-components N_COMPONENTS
                        number of topics. (default: 10)
  --doc-topic-prior DOC_TOPIC_PRIOR
                        prior of document topic distribution `theta`. If the value is None, defaults to `1 / n_components`. (default: None)
  --topic-word-prior TOPIC_WORD_PRIOR
                        prior of topic word distribution `beta`. If the value is None, defaults to `1 / n_components`. (default: None)
  --learning-method LEARNING_METHOD
                        method used to update `_component`. (default: batch)
  --learning-decay LEARNING_DECAY
                        it is a parameter that control learning rate in the online learning method. The value should be set between (0.5, 1.0] to guarantee asymptotic convergence. When the value is 0.0 and batch_size is `n_samples`, the update method is same as batch learning. In the literature, this is called kappa. (default: 0.7)
  --learning-offset LEARNING_OFFSET
                        a (positive) parameter that downweights early iterations in online learning.  It should be greater than 1.0. In the literature, this is called tau_0. (default: 10.0)
  --max-iter MAX_ITER   the maximum number of passes over the training data (aka epochs). (default: 10)
  --batch-size BATCH_SIZE
                        number of documents to use in each EM iteration. Only used in online learning. (default: 128)
  --evaluate-every EVALUATE_EVERY
                        how often to evaluate perplexity. Set it to 0 or negative number to not evaluate perplexity in training at all. Evaluating perplexity can help you check convergence in training process, but it will also increase total training time. Evaluating perplexity in every iteration might increase training time up to two-fold. (default: -1)
  --perp-tol PERP_TOL   perplexity tolerance in batch learning. Only used when `evaluate_every` is greater than 0. (default: 0.1)
  --mean-change-tol MEAN_CHANGE_TOL
                        stopping tolerance for updating document topic distribution in E-step. (default: 0.001)
  --max-doc-update-iter MAX_DOC_UPDATE_ITER
                        max number of iterations for updating document topic distribution in the E-step. (default: 100)
  --n-jobs N_JOBS       the number of jobs to use in the E-step. `None` means 1. `-1` means using all processors. (default: None)
  --random-state RANDOM_STATE
                        pass an int for reproducible results across multiple function calls. (default: None)
```

## References
* [Max Kölbl, Yuki Kyogoku, J. Nathanael Philipp, Michael Richter, Tariq Yousef: Keyword extraction in German: Information-theory vs. deep learning. ICAART 2020 Special Session NLPinAI, Volume: Vol. 1: 459 - 464](https://doi.org/10.1007/978-3-030-63787-3_5)
* [Max Kölbl, Yuki Kyogoku, J. Nathanael Philipp, Michael Richter, Clemens Rietdorf, and Tariq Yousef: The Semantic Level of Shannon Information: Are Highly Informative Words Good Keywords? A Study on German. Natural Language Processing in Artificial Intelligence - NLPinAI 2020 939 (2021): 139-161.](https://doi.org/10.1007/978-3-030-63787-3_5)
* [Nathanael Philipp, Max Kölbl, Yuki Kyogoku, Tariq Yousef, Michael Richter (2022) One Step Beyond: Keyword Extraction in German Utilising Surprisal from Topic Contexts. In: Arai, K. (eds) Intelligent Computing. SAI 2022. Lecture Notes in Networks and Systems, vol 507. Springer, Cham. doi: 10.1007/978-3-031-10464-0_53](https://doi.org/10.1007/978-3-031-10464-0_53)
