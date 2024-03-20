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
"""Topic Context Model (TCM) model module."""

import joblib
import math
import numpy as np

from dataclasses import dataclass
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from pathlib import Path
from typing import TypeVar


@dataclass
class TopicContextModel:
    """Topic Context Model (TCM)."""

    T = TypeVar("T", bound="TopicContextModel")

    model: LatentDirichletAllocation | TruncatedSVD
    words: list[str] | None = None
    verbose: int = 0
    batch_size: int = 128

    @classmethod
    def LatentDirichletAllocation(  # noqa: N802
        cls: type[T],
        words: list[str],
        n_components: int = 10,
        doc_topic_prior: float | None = None,
        topic_word_prior: float | None = None,
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
        n_jobs: int | None = None,
        verbose: int = 0,
        random_state: int | None = None,
    ) -> T:
        """Build Topic Context Model with Latent Dirichlet Allocation.

        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
        """
        assert learning_method in ["batch", "online"]

        return cls(
            LatentDirichletAllocation(
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
            ),
            words=words,
            verbose=verbose,
            batch_size=batch_size,
        )

    @classmethod
    def LatentSemanticAnalysis(  # noqa: N802
        cls: type[T],
        words: list[str],
        n_components: int = 2,
        algorithm: str = "randomized",
        n_iter: int = 5,
        n_oversamples: int = 10,
        power_iteration_normalizer: str = "auto",
        random_state: int | None = None,
        tol: float = 0.0,
        verbose: int = 0,
        batch_size: int = 128,
    ) -> T:
        """Build Topic Context Model with Latent semantic analysis (TruncatedSVD).

        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
        """
        assert algorithm in ["arpack", "randomized"]
        assert power_iteration_normalizer in ["auto", "QR", "LU", "none"]

        return cls(
            TruncatedSVD(
                n_components=n_components,
                algorithm=algorithm,
                n_iter=n_iter,
                n_oversamples=n_oversamples,
                power_iteration_normalizer=power_iteration_normalizer,
                random_state=random_state,
                tol=tol,
            ),
            verbose=verbose,
            batch_size=batch_size,
        )

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:
        """Load a Topic Context Model from a file.

        Args:
         * path: model file to load

        Returns:
         * a Topic Context Model
        """
        model = joblib.load(path)
        if isinstance(model, LatentDirichletAllocation) or isinstance(
            model, TruncatedSVD
        ):
            return cls(model)
        elif isinstance(model, cls):
            return model
        else:
            raise RuntimeError(f"Unkown model loaded from {path}.")

    def fit(self, data: csr_matrix) -> None:
        """Train Topic Context Model in the given data.

        Args:
         * data: sparse document term matrix to train on
        """
        self.model.fit(data)

    def save(self, path: str | Path) -> None:
        """Save this Topic Context Model to a file using `joblib.dump`.

        Args:
         * path: file to save to
        """
        joblib.dump(self, path)

    def surprisal(
        self,
        data: csr_matrix,
        include_frequency: bool = False,
        verbose: int = 0,
        batch_size: int = 128,
    ) -> csr_matrix:
        """Calculate the surprisal."""

        def surprisal(doc: csr_matrix) -> tuple[list[int], list[int]]:
            data = []
            indices = []
            total = doc.sum()

            n_features = topics_words.shape[1]
            if doc.shape[1] != n_features:
                x = doc.copy()
                x.resize((doc.shape[0], n_features))
            else:
                x = doc
            tdata = np.atleast_1d(self.model.transform(x).squeeze())

            doc = doc.toarray().squeeze()
            for i in np.nonzero(doc)[0]:
                if isinstance(self.model, LatentDirichletAllocation):
                    data.append(
                        (-1.0 / self.model.n_components)
                        * sum(
                            [
                                math.log2(
                                    ((doc[i] / total) if include_frequency else 1.0)
                                    * (
                                        topics_words[t, i]
                                        if i < n_features
                                        else topics_words[t].mean()
                                    )
                                    * tdata[t]
                                )
                                for t in range(self.model.n_components)
                            ]
                        )
                    )
                elif isinstance(self.model, TruncatedSVD):
                    data.append(
                        -1.0
                        * math.log2(
                            (doc[i] / total)
                            * (
                                topics_words[0, i]
                                if i < n_features
                                else topics_words[0].mean()
                            )
                            * tdata[0]
                        )
                    )
                indices.append(i)
            return data, indices

        if isinstance(self.model, LatentDirichletAllocation):
            topics_words = (
                self.model.components_
                / self.model.components_.sum(axis=1)[:, np.newaxis]
            )
        else:
            topics_words = self.model.components_

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
                [delayed(surprisal)(data.getrow(i)) for i in range(data.shape[0])]
            ):
                surprisal_data += i
                indices += j
                indptr.append(len(indices))

        return csr_matrix(
            (surprisal_data, indices, indptr), shape=data.shape, dtype=float
        )
