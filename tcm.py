#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:
# Copyright (C) 2019-2022 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
#
# TCM.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""Topic Context Model (TCM)."""

import gzip
import joblib
import json
import logging
import math
import numpy as np
import re
import sys

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawTextHelpFormatter,
)
from collections import Counter
from csv import DictReader, DictWriter, Sniffer
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


__author__ = "J. Nathanael Philipp (jnphilipp)"
__copyright__ = "Copyright 2023 J. Nathanael Philipp (jnphilipp)"
__email__ = "nathanael@philipp.land"
__license__ = "GPLv3"
__version__ = "0.1.0"
__github__ = "https://github.com/jnphilipp/tcm"


VERSION = (
    f"%(prog)s v{__version__}\n{__copyright__}\n"
    + "License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>."
    + "\nThis is free software: you are free to change and redistribute it.\n"
    + "There is NO WARRANTY, to the extent permitted by law.\n\n"
    + f"Report bugs to {__github__}/issues."
    + f"\nWritten by {__author__} <{__email__}>"
)


class ArgFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    """Combination of ArgumentDefaultsHelpFormatter and RawTextHelpFormatter."""

    pass


def words_load(path: Path) -> List[str]:
    logging.info(f"Load words from {path}.")
    fopen: Callable
    words: List[str] = []
    if path.name.endswith(".gz"):
        fopen = gzip.open
    elif path.name.endswith((".txt", ".json")):
        fopen = open
    else:
        raise RuntimeError("Unsopported file type.")
    with fopen(path, "rt", encoding="utf8") as f:
        if path.name.endswith((".json", ".json.gz")):
            data = json.loads(f.read())
            if isinstance(data, list):
                words = data
            else:
                words = [k for k, v in sorted(data.items(), key=lambda x: x[1])]
        elif path.name.endswith((".txt", ".txt.gz")):
            for line in f:
                words.append(line.strip())
        else:
            raise RuntimeError("Unsopported file type.")
    return words


def words_save(path: Path, words: List[str]) -> None:
    logging.info(f"Save words to {path}.")
    fopen: Callable
    if path.name.endswith(".gz"):
        fopen = gzip.open
    elif path.name.endswith((".txt", ".json")):
        fopen = open
    else:
        raise RuntimeError("Unsopported file type.")
    with fopen(path, "wt", encoding="utf8") as f:
        if path.name.endswith((".json", ".json.gz")):
            f.write(json.dumps(words, ensure_ascii=False, indent=4))
            f.write("\n")
        elif path.name.endswith((".txt", ".txt.gz")):
            for word in words:
                f.write(word)
                f.write("\n")


def data_load(
    paths: str | Path | List[str | Path],
    fields: Optional[str | List[str]],
    words: Optional[List[str]] = None,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    batch_size: int = 128,
    verbose: int = 10,
) -> Tuple[csr_matrix, List[str]]:
    """Load texts from text or csv files.

    Text files need to have a text per line, for csv files fields names need to be
    given. The text needs to betokenized with `;` for senteces and `,` for words.

    paths: path or list of paths to load texts from
    fields: if files are in csv format, which field to use
    """

    def convert_line(*texts: str) -> Tuple[List[int], List[int]]:
        data = []
        indices = []
        counter: Counter = Counter()
        for text in texts:
            if re.fullmatch(r"[^\s]+\t+[^\t]+", text.strip()):
                idx, text = re.split(r"\t+", text.strip())
            if tokenizer is not None:
                counter.update(tokenizer(text.strip()))
            elif ";" in text and "," in text:
                for s in text.strip().split(";"):
                    counter.update(s.split(","))
            elif "," in text:
                counter.update(text.strip().split(","))
            else:
                raise RuntimeError(f"Unrecognized format for line {text}.")
        for k, v in counter.items():
            idx = vocab.setdefault(k, len(vocab))
            data.append(v)
            indices.append(idx)
        return data, indices

    if isinstance(paths, str) or isinstance(paths, Path):
        paths = [paths]
    if isinstance(fields, str):
        fields = [fields]

    fopen: Callable
    vocab: Dict[str, int] = {}
    for path in paths:
        if isinstance(path, str):
            path = Path(path)
        logging.info(f"Load data from {path}.")
        if path.name.endswith(".gz"):
            fopen = gzip.open
        elif path.name.endswith((".txt", ".csv")):
            fopen = open
        else:
            raise RuntimeError("Unsopported file type.")
        with fopen(path, "rt", encoding="utf8") as f:
            if path.name.endswith((".csv", ".csv.gz")):
                assert fields is not None
                dialect = Sniffer().sniff(f.readline() + f.readline())
                f.seek(0)
                reader = DictReader(f, dialect=dialect)
                with Parallel(
                    n_jobs=joblib.cpu_count(),
                    verbose=verbose,
                    require="sharedmem",
                    batch_size=batch_size,
                ) as parallel:
                    data = []
                    indices = []
                    indptr = [0]
                    for i, j in parallel(
                        [
                            delayed(convert_line)(*(row[field] for field in fields))
                            for row in reader
                        ]
                    ):
                        data += i
                        indices += j
                        indptr.append(len(indices))
            elif path.name.endswith((".txt", ".txt.gz")):
                with Parallel(
                    n_jobs=joblib.cpu_count(),
                    verbose=verbose,
                    require="sharedmem",
                    batch_size=batch_size,
                ) as parallel:
                    data = []
                    indices = []
                    indptr = [0]
                    for i, j in parallel(
                        [delayed(convert_line)(line.strip()) for line in f]
                    ):
                        data += i
                        indices += j
                        indptr.append(len(indices))
            else:
                raise RuntimeError("Unsopported file type.")
    logging.info(f"Loaded {len(indptr) - 1} texts with {len(vocab)} words.")
    return (
        csr_matrix((data, indices, indptr), dtype=np.uint),
        [k for k, v in sorted(vocab.items(), key=lambda x: x[1])],
    )


def surprisal_save(
    paths: str | Path | List[str | Path],
    fields: Optional[str | List[str]],
    surprisal_data: csr_matrix,
    words: List[str],
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> None:
    if isinstance(paths, str) or isinstance(paths, Path):
        paths = [paths]

    rwords = {w: i for i, w in enumerate(words)}

    fopen: Callable
    idx = 0
    for path in paths:
        if isinstance(path, str):
            path = Path(path)
        out_path = path.parent / re.sub(
            r"(.+?)(\.(txt|.csv)(\.gz)?)$", r"\g<1>-surprisal\g<2>", path.name
        )

        if path.name.endswith(".gz"):
            fopen = gzip.open
        elif path.name.endswith((".txt", ".csv")):
            fopen = open
        else:
            raise RuntimeError("Unsopported file type.")
        with fopen(path, "rt", encoding="utf8") as fr:
            with fopen(out_path, "wt", encoding="utf8") as fw:
                logging.info(f"Save surprisal data for {path} in {out_path}.")
                if path.name.endswith((".csv", ".csv.gz")):
                    assert fields is not None
                    dialect = Sniffer().sniff(fr.readline() + fr.readline())
                    fr.seek(0)
                    reader = DictReader(fr, dialect=dialect)
                    writer = DictWriter(
                        fw,
                        fr.fieldnames + [f"{field}-surprisal" for field in fields],
                        dialect=dialect,
                    )
                    for row in reader:
                        doc = surprisal_data.getrow(idx).toarray().squeeze()
                        idx += 1
                        for field in fields:
                            if tokenizer is not None:
                                row[f"{field}-surprisal"] = ",".join(
                                    [
                                        f"{w}|{doc[rwords[w]]}"
                                        for w in tokenizer(row[field].strip())
                                    ]
                                )
                            elif ";" in row[field] and "," in row[field]:
                                row[f"{field}-surprisal"] = ";".join(
                                    [
                                        ",".join(
                                            [
                                                f"{w}|{doc[rwords[w]]}"
                                                for w in s.split(",")
                                            ]
                                        )
                                        for s in row[field].strip().split(";")
                                    ]
                                )
                            elif "," in row[field]:
                                row[f"{field}-surprisal"] = ",".join(
                                    [
                                        f"{w}|{doc[rwords[w]]}"
                                        for w in row[field].strip().split(",")
                                    ]
                                )
                            else:
                                raise RuntimeError(
                                    f"Unrecognized format for {row[field]} in {field}."
                                )
                        writer.writerow(row)
                elif path.name.endswith((".txt", ".txt.gz")):
                    for i, line in enumerate(fr):
                        line_idx: Optional[str]
                        if re.fullmatch(r"[^\s]+\t+[^\t]+", line.strip()):
                            line_idx, line = re.split(r"\t+", line.strip())
                        else:
                            line_idx = None
                            line = line.strip()

                        doc = surprisal_data.getrow(idx).toarray().squeeze()
                        idx += 1
                        if tokenizer is not None:
                            if line_idx is not None:
                                fw.write(f"{line_idx}\t")
                            fw.write(
                                ",".join(
                                    [
                                        f"{w}|{doc[rwords[w]]}"
                                        for w in tokenizer(line.strip())
                                    ]
                                )
                            )
                            fw.write("\n")
                        elif ";" in line and "," in line:
                            if line_idx is not None:
                                fw.write(f"{line_idx}\t")
                            fw.write(
                                ";".join(
                                    [
                                        ",".join(
                                            [
                                                f"{w}|{doc[rwords[w]]}"
                                                for w in s.split(",")
                                            ]
                                        )
                                        for s in line.strip().split(";")
                                    ]
                                )
                            )
                            fw.write("\n")
                        elif "," in line:
                            if line_idx is not None:
                                fw.write(f"{line_idx}\t")
                            fw.write(
                                ",".join(
                                    [
                                        f"{w}|{doc[rwords[w]]}"
                                        for w in line.strip().split(",")
                                    ]
                                )
                            )
                            fw.write("\n")
                        else:
                            raise RuntimeError(
                                f"Unrecognized format for line {line.strip()}."
                            )


def lda_build(
    n_components: int = 10,
    doc_topic_prior: Optional[float] = None,
    topic_word_prior: Optional[float] = None,
    learning_method: str = "batch",
    learning_decay: float = 0.7,
    learning_offset: float = 10.0,
    max_iter: int = 10,
    batch_size: int = 128,
    evaluate_every: int = -1,
    total_samples: int = 1000000,
    perp_tol: float = 0.1,
    mean_change_tol: float = 0.001,
    max_doc_update_iter: int = 100,
    n_jobs: Optional[int] = None,
    verbose: int = 0,
    random_state: Optional[int] = None,
) -> LatentDirichletAllocation:
    """Build Latent Dirichlet Allocation model.

    https://scikit-learn.org/stable/modules/generated/
    sklearn.decomposition.LatentDirichletAllocation.html
    """
    assert learning_method in ["batch", "online"]

    logging.info(f"Build LDA with {n_components} topics.")
    logging.debug(
        f"doc_topic_prior={doc_topic_prior}, topic_word_prior={topic_word_prior}, "
        f"learning_method={learning_method}, learning_decay={learning_decay}, "
        f"learning_offset={learning_offset}, max_iter={max_iter}, "
        f"batch_size={batch_size}, evaluate_every={evaluate_every}, "
        f"total_samples={total_samples}, perp_tol={perp_tol}, "
        f"mean_change_tol={mean_change_tol}, max_doc_update_iter={max_doc_update_iter},"
        f" n_jobs={n_jobs}, verbose={verbose}, random_state={random_state}"
    )
    return LatentDirichletAllocation(
        n_components=n_components,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
        learning_method=learning_method,
        learning_decay=learning_decay,
        learning_offset=learning_offset,
        max_iter=max_iter,
        batch_size=batch_size,
        evaluate_every=evaluate_every,
        total_samples=total_samples,
        perp_tol=perp_tol,
        mean_change_tol=mean_change_tol,
        max_doc_update_iter=max_doc_update_iter,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
    )


def lda_load(path: Path) -> LatentDirichletAllocation:
    logging.info(f"Load LDA from {path}.")
    return joblib.load(path)


def lda_save(lda: LatentDirichletAllocation, path: Path) -> None:
    """Save LDA."""
    logging.info(f"Save LDA to {path}.")
    joblib.dump(lda, path)


def lda_surprisal(
    lda: LatentDirichletAllocation,
    data: csr_matrix,
    verbose: int = 0,
    batch_size: int = 128,
) -> csr_matrix:
    """Calculate LDA based surprisal."""

    def surprisal(doc: csr_matrix) -> Tuple[List[int], List[int]]:
        data = []
        indices = []
        total = doc.sum()
        tdata = lda.transform(doc).squeeze()
        doc = doc.toarray().squeeze()
        for i in np.nonzero(doc)[0]:
            data.append(
                (-1.0 / lda.n_components)
                * sum(
                    [
                        math.log2((doc[i] / total) * topics_words[t, i] * tdata[t])
                        for t in range(lda.n_components)
                    ]
                )
            )
            indices.append(i)
        return data, indices

    logging.info("Calculate LDA surprisal.")
    logging.debug("Calculate topic-word matrix.")
    topics_words = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

    logging.debug("Calculate surprisal.")
    with Parallel(
        n_jobs=joblib.cpu_count(),
        verbose=verbose,
        require="sharedmem",
        batch_size=batch_size,
    ) as parallel:
        surprisal_data = []
        indices = []
        indptr = [0]
        for i, j in parallel(
            [
                delayed(surprisal)(data.getrow(i))
                for i in range(data.shape[0])
            ]
        ):
            surprisal_data += i
            indices += j
            indptr.append(len(indices))

    return csr_matrix((surprisal_data, indices, indptr))


def default_tokenizer(text: str) -> List[str]:
    words = []
    text = re.sub(r"https?://[^\s]+", "URL", text)
    for m in re.finditer(r"\b(\w+(-\w+)+|\w+&\w+|\w+)\b", text):
        words.append(
            "NUM"
            if re.fullmatch(r"\d+", m.group())
            else (m.group() if m.group() == "URL" else m.group().lower())
        )
    words = [
        word
        for i, word in enumerate(words)
        if i == 0 or (word == "NUM" and words[i - 1] != word) or word != "NUM"
    ]
    return words


def lda_train(
    lda: LatentDirichletAllocation,
    data: csr_matrix,
) -> None:
    """Train TCM with LDA."""
    logging.info(f"Train LDA for max {lda.max_iter} iterations.")
    lda.fit(data)


def filter_info(rec: logging.LogRecord) -> bool:
    """Log record filter for info and lower levels.

    Args:
     * rec: LogRecord object
    """
    return rec.levelno <= logging.INFO


if __name__ == "__main__":
    parser = ArgumentParser(prog="tcm", formatter_class=ArgFormatter)
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=VERSION,
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["lda", "lsa"],
        default="lda",
        help="which model to use.",
    )
    parser.add_argument(
        "--model-file",
        type=lambda p: Path(p).absolute(),
        default="lda.jl.z",
        help="file to load model from or save to, if path exists tries to load model.",
    )
    parser.add_argument(
        "action",
        choices=["train", "surprisal"],
        nargs="+",
        help="what to do, train lda/lsa or calculate surprisal.",
    )

    # data
    parser.add_argument(
        "--data",
        nargs="+",
        type=lambda p: Path(p).absolute(),
        help="file(s) to load texts from, either txt or csv optionally gzip "
        + "compressed.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        type=str,
        help="field(s) to load texts when using csv data.",
    )
    parser.add_argument(
        "--words",
        type=lambda p: Path(p).absolute(),
        default="words.txt.gz",
        help="file to load words from and/or save to, either txt or json optionally "
        + "gzip compressed.",
    )

    # lda
    lda_config = parser.add_argument_group("LDA config")
    lda_config.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="number of topics.",
    )
    lda_config.add_argument(
        "--doc-topic-prior",
        type=float,
        default=None,
        help="prior of document topic distribution `theta`. If the value is None, "
        + "defaults to `1 / n_components`.",
    )
    lda_config.add_argument(
        "--topic-word-prior",
        type=float,
        default=None,
        help="prior of topic word distribution `beta`. If the value is None, defaults "
        + "to `1 / n_components`.",
    )
    lda_config.add_argument(
        "--learning-method",
        type=str,
        default="batch",
        help="method used to update `_component`.",
    )
    lda_config.add_argument(
        "--learning-decay",
        type=float,
        default=0.7,
        help="it is a parameter that control learning rate in the online learning "
        + "method. The value should be set between (0.5, 1.0] to guarantee asymptotic "
        + "convergence. When the value is 0.0 and batch_size is `n_samples`, the "
        + "update method is same as batch learning. In the literature, this is called "
        + "kappa.",
    )
    lda_config.add_argument(
        "--learning-offset",
        type=float,
        default=10.0,
        help="a (positive) parameter that downweights early iterations in online "
        + "learning.  It should be greater than 1.0. In the literature, this is called "
        + "tau_0.",
    )
    lda_config.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="the maximum number of passes over the training data (aka epochs).",
    )
    lda_config.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="number of documents to use in each EM iteration. Only used in online "
        + "learning.",
    )
    lda_config.add_argument(
        "--evaluate-every",
        type=int,
        default=-1,
        help="how often to evaluate perplexity. Set it to 0 or negative number to not "
        + "evaluate perplexity in training at all. Evaluating perplexity can help you "
        + "check convergence in training process, but it will also increase total "
        + "training time. Evaluating perplexity in every iteration might increase "
        + "training time up to two-fold.",
    )
    lda_config.add_argument(
        "--perp-tol",
        type=float,
        default=0.1,
        help="perplexity tolerance in batch learning. Only used when `evaluate_every` "
        + "is greater than 0.",
    )
    lda_config.add_argument(
        "--mean-change-tol",
        type=float,
        default=0.001,
        help="stopping tolerance for updating document topic distribution in E-step.",
    )
    lda_config.add_argument(
        "--max-doc-update-iter",
        type=int,
        default=100,
        help="max number of iterations for updating document topic distribution in the "
        + "E-step.",
    )
    lda_config.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="the number of jobs to use in the E-step. `None` means 1. `-1` means "
        + "using all processors.",
    )
    lda_config.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="pass an int for reproducible results across multiple function calls.",
    )

    # logging
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="verbosity level; multiple times increases the level, the maximum is 3, "
        + "for debugging.",
    )
    parser.add_argument(
        "--log-format",
        default="%(message)s",
        help="set logging format.",
    )
    parser.add_argument(
        "--log-file",
        type=lambda p: Path(p).absolute(),
        help="log output to a file.",
    )
    parser.add_argument(
        "--log-file-format",
        default="[%(levelname)s] %(message)s",
        help="set logging format for log file.",
    )
    args = parser.parse_args()

    if args.verbose == 0:
        level = logging.WARNING
        verbosity = 0
    elif args.verbose == 1:
        level = logging.INFO
        verbosity = 1
    elif args.verbose == 2:
        level = logging.INFO
        verbosity = 10
    else:
        level = logging.DEBUG
        verbosity = 100

    handlers: List[logging.Handler] = []
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.addFilter(filter_info)
    handlers.append(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    if "%(levelname)s" not in args.log_format:
        stderr_handler.setFormatter(
            logging.Formatter(f"[%(levelname)s] {args.log_format}")
        )
    handlers.append(stderr_handler)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(level)
        if args.log_file_format:
            file_handler.setFormatter(logging.Formatter(args.log_file_format))
        handlers.append(file_handler)

    logging.basicConfig(
        format=args.log_format,
        level=logging.DEBUG,
        handlers=handlers,
    )

    data, words = data_load(
        args.data,
        args.fields,
        words_load(args.words) if args.words.exists() else None,
        None,
        args.batch_size,
        verbose=verbosity,
    )
    if args.model == "lda":
        if args.model_file.exists():
            lda = lda_load(args.model_file)
        else:
            lda = lda_build(
                total_samples=data.size,
                verbose=verbosity,
                n_components=args.n_components,
                doc_topic_prior=args.doc_topic_prior,
                topic_word_prior=args.topic_word_prior,
                learning_method=args.learning_method,
                learning_decay=args.learning_decay,
                learning_offset=args.learning_offset,
                max_iter=args.max_iter,
                batch_size=args.batch_size,
                evaluate_every=args.evaluate_every,
                perp_tol=args.perp_tol,
                mean_change_tol=args.mean_change_tol,
                max_doc_update_iter=args.max_doc_update_iter,
                n_jobs=args.n_jobs,
                random_state=args.random_state,
            )
        if "train" in args.action:
            lda_train(lda, data)
            lda_save(lda, args.model_file)
            words_save(args.words, words)
        if "surprisal" in args.action:
            surprisal_data = lda_surprisal(lda, data)
            surprisal_save(args.data, args.fields, surprisal_data, words)
