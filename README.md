# Topic Context Modell (TCM)

Calculates the surprisal of a word given a context based on the topics in a text.

![Tests](https://github.com/jnphilipp/tcm/actions/workflows/tests.yml/badge.svg)
[![pypi Version](https://img.shields.io/pypi/v/topic-context-model.svg?logo=pypi&logoColor=white)](https://pypi.org/project/topic-context-model/)

## Requirements

* Python >= 3.10
* scipy
* scikit-learn

## Usage

```python
from tcm import TopicContextModel
from tcm.data import load, save
from tcm.tokenizer import default_tokenizer

data, words, out_of_vocab = load(PATHS, None)

tcm = TopicContextModel.LatentDirichletAllocation(
    words, n_components=num_topics, max_iter=20, n_jobs=-1
)
tcm.fit(data)
tcm.save("./tcm.jl.z")

surprisal_data = tcm.surprisal(data)
save(PATHS, None, surprisal_data, words, out_of_vocab)


tcm = TopicContextModel.load("./tcm.jl.z")
data, words, out_of_vocab = load(PATHS, None, words=tcm.words, tokenizer=default_tokenizer)
surprisal_data = tcm.surprisal(data)
save(PATHS, None, surprisal_data, words, out_of_vocab, tokenizer=default_tokenizer)
```

```
usage: tcm [-h] [-V] [--model-file MODEL_FILE] [--data DATA [DATA ...]] [--fields FIELDS [FIELDS ...]]
           [-t] [--file-as-text] [--surprisal-file-name-part SURPRISAL_FILE_NAME_PART]
           [--exclude-pos-tags EXCLUDE_POS_TAGS [EXCLUDE_POS_TAGS ...]]
           [--conllu-keyname CONLLU_KEYNAME] [-v] [--log-format LOG_FORMAT] [--log-file LOG_FILE]
           [--log-file-format LOG_FILE_FORMAT]
           {train,surprisal} [{train,surprisal} ...] {lda,lsa} ...

positional arguments:
  {train,surprisal}     what to do, train lda/lsa or calculate surprisal.

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  --model-file MODEL_FILE
                        file to load model from or save to, if path exists tries to load model. (default: tcm.jl.z)
  --data DATA [DATA ...]
                        file(s) to load texts from, either txt or csv optionally gzip compressed. (default: None)
  --fields FIELDS [FIELDS ...]
                        field(s) to load texts from, when using csv data. (default: None)
  -t, --tokenize        use the build in tokenizer to tokenize, do not use with already tokenized data. (default: False)
  --file-as-text        treat all texts in a file as a single text. (default: False)
  --surprisal-file-name-part SURPRISAL_FILE_NAME_PART
                        added to the name of input file to when saving surprisal data. (default: -surprisal)
  --exclude-pos-tags EXCLUDE_POS_TAGS [EXCLUDE_POS_TAGS ...]
                        exclude words with these PoS-tags. (default: [])
  --conllu-keyname CONLLU_KEYNAME
                        key name to use when saving to CoNLL-U, in misc. (default: surprisal)
  -v, --verbose         verbosity level; multiple times increases the level, the maximum is 3, for debugging. (default: 0)
  --log-format LOG_FORMAT
                        set logging format. (default: %(message)s)
  --log-file LOG_FILE   log output to a file. (default: None)
  --log-file-format LOG_FILE_FORMAT
                        set logging format for log file. (default: [%(levelname)s] %(message)s)

models:
  {lda,lsa}             which model to use.
    lda                 use LDA as model for TCM.
    lsa                 use LSA as model for TCM.
```

## References
* [Max Kölbl, Yuki Kyogoku, J. Nathanael Philipp, Michael Richter, Tariq Yousef (2020) Keyword extraction in German: Information-theory vs. deep learning. ICAART 2020 Special Session NLPinAI, Volume: Vol. 1: 459 - 464. doi: 10.5220/0009374704590464](https://doi.org/10.5220/0009374704590464)
* [Max Kölbl, Yuki Kyogoku, J. Nathanael Philipp, Michael Richter, Clemens Rietdorf, and Tariq Yousef (2021) The Semantic Level of Shannon Information: Are Highly Informative Words Good Keywords? A Study on German. Natural Language Processing in Artificial Intelligence - NLPinAI 2020 939 (2021): 139-161. doi: 10.1007/978-3-030-63787-3_5](https://doi.org/10.1007/978-3-030-63787-3_5)
* [Nathanael Philipp, Max Kölbl, Yuki Kyogoku, Tariq Yousef, Michael Richter (2022) One Step Beyond: Keyword Extraction in German Utilising Surprisal from Topic Contexts. In: Arai, K. (eds) Intelligent Computing. SAI 2022. Lecture Notes in Networks and Systems, vol 507. Springer, Cham. doi: 10.1007/978-3-031-10464-0_53](https://doi.org/10.1007/978-3-031-10464-0_53)
* [J. Nathanael Philipp, Michael Richter, Erik Daas, and Max Kölbl (2023) Are idioms surprising?. In Proceedings of the 19th Conference on Natural Language Processing (KONVENS 2023), pages 149–154, Ingolstadt, Germany. Association for Computational Lingustics.](https://aclanthology.org/2023.konvens-main.15/)
* [J. Nathanael Philipp, Michael Richter, Tatjana Scheffler, und Roeland van Hout (2024) The Role of Information in Modeling German Intensifiers. In Information Structure and Information Theory, 117–45. Berlin: Language Science Press, 2024. doi: 10.5281/zenodo.13383791](https://doi.org/10.5281/zenodo.13383791)
