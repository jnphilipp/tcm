# Copyright (C) 2023-2024 J. Nathanael Philipp (jnphilipp) <nathanael@philipp.land>
#
# This file is part of tcm.
#
# tcm is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tcm is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tcm. If not, see <http://www.gnu.org/licenses/>
"""Topic Context Model (TCM) app module."""

import logging
import sys

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawTextHelpFormatter,
)
from pathlib import Path

from . import __app_name__, VERSION
from .data import load, save
from .model import TopicContextModel
from .tokenizer import default_tokenizer


class ArgFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    """Combination of ArgumentDefaultsHelpFormatter and RawTextHelpFormatter."""

    pass


def filter_info(rec: logging.LogRecord) -> bool:
    """Log record filter for info and lower levels.

    Args:
     * rec: LogRecord object
    """
    return rec.levelno <= logging.INFO


def main() -> str | None:
    """Run the command line interface."""
    parser = ArgumentParser(prog=__app_name__, formatter_class=ArgFormatter)
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=VERSION,
    )
    parser.add_argument(
        "--model-file",
        type=lambda p: Path(p).absolute(),
        default="tcm.jl.z",
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
        help="field(s) to load texts from, when using csv data.",
    )
    parser.add_argument(
        "-t",
        "--tokenize",
        action="store_true",
        help="use the build in tokenizer to tokenize, do not use with already "
        + "tokenized data.",
    )
    parser.add_argument(
        "--file-as-text",
        action="store_true",
        help="treat all texts in a file as a single text.",
    )
    parser.add_argument(
        "--surprisal-file-name-part",
        type=str,
        default="-surprisal",
        help="added to the name of input file to when saving surprisal data.",
    )
    parser.add_argument(
        "--exclude-pos-tags",
        nargs="+",
        default=[],
        help="exclude words with these PoS-tags.",
    )
    parser.add_argument(
        "--conllu-keyname",
        type=str,
        default="surprisal",
        help="key name to use when saving to CoNLL-U, in misc.",
    )

    # lda
    subparsers = parser.add_subparsers(
        title="models", help="which model to use.", dest="model"
    )
    lda_parser = subparsers.add_parser("lda", help="use LDA as model for TCM.")
    lda_parser.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="number of topics.",
    )
    lda_parser.add_argument(
        "--doc-topic-prior",
        type=float,
        default=None,
        help="prior of document topic distribution `theta`. If the value is None, "
        + "defaults to `1 / n_components`.",
    )
    lda_parser.add_argument(
        "--topic-word-prior",
        type=float,
        default=None,
        help="prior of topic word distribution `beta`. If the value is None, defaults "
        + "to `1 / n_components`.",
    )
    lda_parser.add_argument(
        "--learning-method",
        type=str,
        default="batch",
        help="method used to update `_component`.",
    )
    lda_parser.add_argument(
        "--learning-decay",
        type=float,
        default=0.7,
        help="it is a parameter that control learning rate in the online learning "
        + "method. The value should be set between (0.5, 1.0] to guarantee asymptotic "
        + "convergence. When the value is 0.0 and batch_size is `n_samples`, the "
        + "update method is same as batch learning. In the literature, this is called "
        + "kappa.",
    )
    lda_parser.add_argument(
        "--learning-offset",
        type=float,
        default=10.0,
        help="a (positive) parameter that downweights early iterations in online "
        + "learning.  It should be greater than 1.0. In the literature, this is called "
        + "tau_0.",
    )
    lda_parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="the maximum number of passes over the training data (aka epochs).",
    )
    lda_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="number of documents to use in each EM iteration. Only used in online "
        + "learning.",
    )
    lda_parser.add_argument(
        "--evaluate-every",
        type=int,
        default=-1,
        help="how often to evaluate perplexity. Set it to 0 or negative number to not "
        + "evaluate perplexity in training at all. Evaluating perplexity can help you "
        + "check convergence in training process, but it will also increase total "
        + "training time. Evaluating perplexity in every iteration might increase "
        + "training time up to two-fold.",
    )
    lda_parser.add_argument(
        "--perp-tol",
        type=float,
        default=0.1,
        help="perplexity tolerance in batch learning. Only used when `evaluate_every` "
        + "is greater than 0.",
    )
    lda_parser.add_argument(
        "--mean-change-tol",
        type=float,
        default=0.001,
        help="stopping tolerance for updating document topic distribution in E-step.",
    )
    lda_parser.add_argument(
        "--max-doc-update-iter",
        type=int,
        default=100,
        help="max number of iterations for updating document topic distribution in the "
        + "E-step.",
    )
    lda_parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="the number of jobs to use in the E-step. `None` means 1. `-1` means "
        + "using all processors.",
    )
    lda_parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="pass an int for reproducible results across multiple function calls.",
    )

    # lsa
    lsa_parser = subparsers.add_parser("lsa", help="use LSA as model for TCM.")
    lsa_parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="desired dimensionality of output data. If algorithm='arpack', must be "
        + "strictly less than the number of features. If algorithm='randomized', must "
        + "be less than or equal to the number of features.",
    )
    lsa_parser.add_argument(
        "--algorithm",
        type=str,
        choices=["arpack", "randomized"],
        default="randomized",
        help="SVD solver to use. Either “arpack” for the ARPACK wrapper in SciPy, or "
        + "'randomized' for the randomized algorithm due to Halko (2009).",
    )
    lsa_parser.add_argument(
        "--n_iter",
        type=int,
        default=5,
        help="number of iterations for randomized SVD solver. Not used by ARPACK.",
    )
    lsa_parser.add_argument(
        "--n-oversamples",
        type=int,
        default=10,
        help="number of oversamples for randomized SVD solver. Not used by ARPACK.",
    )
    lsa_parser.add_argument(
        "--power-iteration-normalizer",
        type=str,
        choices=["auto", "QR", "LU", "none"],
        default="auto",
        help="power iteration normalizer for randomized SVD solver. Not used by "
        + "ARPACK.",
    )
    lsa_parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="used during randomized SVD. Pass an int for reproducible results across "
        + "multiple function calls.",
    )
    lsa_parser.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="tolerance for ARPACK. 0 means machine precision. Ignored by randomized "
        + "SVD solver.",
    )
    lsa_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="used in joblib paralisation.",
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

    handlers: list[logging.Handler] = []
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

    tcm = TopicContextModel.load(args.model_file) if args.model_file.exists() else None
    data, words, out_of_vocab = load(
        args.data,
        args.fields,
        None if tcm is None else tcm.words,
        default_tokenizer if args.tokenize else None,
        args.file_as_text,
        args.batch_size,
        args.exclude_pos_tags,
        verbosity,
    )

    if tcm is None and args.model == "lda":
        tcm = TopicContextModel.LatentDirichletAllocation(
            words,
            args.n_components,
            args.doc_topic_prior,
            args.topic_word_prior,
            args.learning_method,
            args.learning_decay,
            args.learning_offset,
            args.max_iter,
            args.batch_size,
            args.evaluate_every,
            data.size,
            args.perp_tol,
            args.mean_change_tol,
            args.max_doc_update_iter,
            args.n_jobs,
            verbosity,
            args.random_state,
        )
    elif tcm is None and args.model == "lsa":
        tcm = TopicContextModel.LatentSemanticAnalysis(
            words,
            args.n_components,
            args.algorithm,
            args.n_iter,
            args.n_oversamples,
            args.power_iteration_normalizer,
            args.random_state,
            args.tol,
            verbosity,
            args.batch_size,
        )
    elif tcm is None:
        raise RuntimeError("No TCM was created.")

    if "train" in args.action:
        if data.shape[1] == len(words):
            x = data
        else:
            x = data.copy()
            x.resize((data.shape[0], len(words)))
        tcm.fit(x)
        tcm.save(args.model_file)
    if "surprisal" in args.action:
        surprisal_data = tcm.surprisal(data)
        save(
            args.data,
            args.fields,
            surprisal_data,
            words,
            out_of_vocab,
            args.surprisal_file_name_part,
            args.file_as_text,
            default_tokenizer if args.tokenize else None,
            args.exclude_pos_tags,
            args.conllu_keyname,
        )
    return None
