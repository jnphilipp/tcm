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
"""Topic Context Model (TCM) data module."""

import gzip
import joblib
import logging
import numpy as np
import re

from collections import Counter
from conllu import parse_incr, TokenList
from csv import DictReader, DictWriter, Sniffer
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from pathlib import Path
from typing import Callable


def load(
    paths: str | Path | list[str | Path],
    fields: str | list[str] | None,
    words: list[str] | None = None,
    tokenizer: Callable[[str], list[str]] | None = None,
    file_as_text: bool = False,
    batch_size: int = 128,
    exclude_pos_tags: list[str] = [],
    verbose: int = 10,
) -> tuple[csr_matrix, list[str], dict[int, str]]:
    """Load texts from text, csv or CoNLL-U files.

    Text files need to have a text per line, for csv files fields names need to be
    given. The text needs to betokenized with `;` for senteces and `,` for words.

    For CoNLL-U files the lemma will be used, optionally a list of PoS-tags can be given
    that will be exclude, the fields for UPOS and XPOS will be checked.

    paths: path or list of paths to load texts from
    fields: if files are in csv format, which field to use
    words: optional list of words (vocab). If given words not in this list will be
           excluded.
    """

    def convert(*texts: str | TokenList) -> tuple[list[int], list[int]]:
        data = []
        indices = []
        counter: Counter = Counter()
        for text in texts:
            if isinstance(text, str):
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
            elif isinstance(text, TokenList):
                for token in text:
                    if (
                        token["upos"] in exclude_pos_tags
                        or token["xpos"] in exclude_pos_tags
                    ):
                        continue
                    counter.update([token["lemma"]])
        for k, v in counter.items():
            if words is None:
                idx = vocab.setdefault(k, len(vocab))
            elif k not in vocab:
                idx = out_of_vocab.setdefault(k, len(vocab) + len(out_of_vocab))
            else:
                idx = vocab[k]
            data.append(v)
            indices.append(idx)
        return data, indices

    if isinstance(paths, str) or isinstance(paths, Path):
        paths = [paths]
    if isinstance(fields, str):
        fields = [fields]

    data = []
    indices = []
    indptr = [0]

    fopen: Callable
    vocab: dict[str, int] = {} if words is None else {w: i for i, w in enumerate(words)}
    out_of_vocab: dict[str, int] = {}
    for c, path in enumerate(paths):
        if isinstance(path, str):
            path = Path(path)
        logging.info(f"Load data from {path}.")
        if path.name.endswith(".gz"):
            fopen = gzip.open
        elif path.name.endswith((".txt", ".csv", ".conllu")):
            fopen = open
        else:
            raise RuntimeError("Unsopported file type.")
        with fopen(path, "rt", encoding="utf8") as f:
            if path.name.endswith((".conllu", ".conllu.gz")):
                with Parallel(
                    n_jobs=joblib.cpu_count(),
                    verbose=verbose,
                    require="sharedmem",
                    batch_size=batch_size,
                ) as parallel:
                    for i, j in parallel(
                        [delayed(convert)(tokenlist) for tokenlist in parse_incr(f)]
                    ):
                        data += i
                        indices += j
                        indptr.append(len(indices))
            elif path.name.endswith((".csv", ".csv.gz")):
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
                    for i, j in parallel(
                        [
                            delayed(convert)(*(row[field] for field in fields))
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
                    for i, j in parallel(
                        [delayed(convert)(line.strip()) for line in f]
                    ):
                        data += i
                        indices += j
                        indptr.append(len(indices))
            else:
                raise RuntimeError("Unsopported file type.")
        if file_as_text:
            indptr = indptr[: c + 1]
            indptr.append(len(indices))
    logging.info(f"Loaded {len(indptr) - 1} texts with {len(vocab)} words.")
    if len(out_of_vocab) > 0:
        logging.info(f"Found {len(out_of_vocab)} out of vocab words.")
    return (
        csr_matrix(
            (data, indices, indptr),
            shape=(len(indptr) - 1, len(vocab) + len(out_of_vocab)),
            dtype=np.uint,
        ),
        (
            [k for k, v in sorted(vocab.items(), key=lambda x: x[1])]
            if words is None
            else words
        ),
        {v: k for k, v in sorted(out_of_vocab.items(), key=lambda x: x[1])},
    )


def save(
    paths: str | Path | list[str | Path],
    fields: str | list[str] | None,
    surprisal_data: csr_matrix,
    words: list[str],
    out_of_vocab: dict[int, str],
    surprisal_file_name_part: str = "-surprisal",
    file_as_text: bool = False,
    tokenizer: Callable[[str], list[str]] | None = None,
    exclude_pos_tags: list[str] = [],
    conllu_keyname: str = "surprisal",
) -> None:
    """Save surprisal values.

    Args:
     * paths:
     * fields:
     * surprisal_data:
     * words:
     * out_of_vocab:
     * surprisal_file_name_part:
     * file_as_text:
     * tokenizer:
     * exclude_pos_tags:
     * conllu_keyname:
    """
    if isinstance(paths, str) or isinstance(paths, Path):
        paths = [paths]

    rwords = {w: i for i, w in enumerate(words)}
    for k, v in out_of_vocab.items():
        rwords[v] = k

    fopen: Callable
    idx = 0
    for path in paths:
        if isinstance(path, str):
            path = Path(path)
        out_path = path.parent / re.sub(
            r"(.+?)(\.(txt|csv|conllu)(\.gz)?)$",
            rf"\g<1>{surprisal_file_name_part}\g<2>",
            path.name,
        )

        if path.name.endswith(".gz"):
            fopen = gzip.open
        elif path.name.endswith((".txt", ".csv", ".conllu")):
            fopen = open
        else:
            raise RuntimeError("Unsopported file type.")
        with fopen(path, "rt", encoding="utf8") as fr:
            if file_as_text:
                doc = surprisal_data.getrow(idx).toarray().squeeze()
                idx += 1

            with fopen(out_path, "wt", encoding="utf8") as fw:
                logging.info(f"Save surprisal data for {path} in {out_path}.")
                if path.name.endswith((".conllu", ".conllu.gz")):
                    for tokenlist in parse_incr(fr):
                        if not file_as_text:
                            doc = surprisal_data.getrow(idx).toarray().squeeze()
                            idx += 1
                        for token in tokenlist:
                            if (
                                token["upos"] in exclude_pos_tags
                                or token["xpos"] in exclude_pos_tags
                            ):
                                continue
                            if token["lemma"] in rwords:
                                if "misc" in token and token["misc"]:
                                    token["misc"][conllu_keyname] = doc[
                                        rwords[token["lemma"]]
                                    ]
                                else:
                                    token["misc"] = {
                                        conllu_keyname: doc[rwords[token["lemma"]]]
                                    }
                        fw.write(tokenlist.serialize())
                elif path.name.endswith((".csv", ".csv.gz")):
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
                        if not file_as_text:
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
                        line_idx: str | None
                        if re.fullmatch(r"[^\s]+\t+[^\t]+", line.strip()):
                            line_idx, line = re.split(r"\t+", line.strip())
                        else:
                            line_idx = None
                            line = line.strip()

                        if not file_as_text:
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
