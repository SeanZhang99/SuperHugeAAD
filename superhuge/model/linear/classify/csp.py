import numpy as np
import torch
from .abc import ClassifierABC

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superhuge.model.types import EEG_TYPE, LABEL_TYPE


class CSPClassifier(ClassifierABC):

    def __init__(self, /, num_features: int, **kwargs):
        """
        Initialize the CSPClassifier with the number of features to extract.

        Parameters:
        num_features: int, the number of features (weights / eigenvalues / eigenvectors) to be kept with the largest eigenvalues for each class .
        """
        super().__init__(**kwargs)
        self.num_features: int = num_features
        self._fitted = False

    def estimate_feature(self, eeg: EEG_TYPE, label: LABEL_TYPE) -> torch.Tensor:
        """
        Estimate features from EEG data specific to CSPClassifier.
        """
        # if fitted, skip to step 4. otherwise, start from step 1.
        # step 1: calculate the covariance matrix for each class of data
        # step 2: optimize: w_i = argmax (w^T * R_ci * w) / (w^T * R_c_all * w), kept the first `num_features` eigenvectors
        # w_i is obtained by solving the generalized eigenvalue problem R_ci * w = lambda * R_c_all * w
        # step 3: store the eigenvectors for each class
        # if fitted, start from here
        # step 4: project the data onto span(eigenvectors) of its own label
        # step 5: compute the log-energy of each eigen-channel, averaging the log-energy over time
        # step 6: return the log-energy as features
        if not self._fitted:
            # Step 1: Calculate covariance matrices for each class
            unique_labels = torch.unique(label)
            cov_matrices: dict[str | int, torch.Tensor] = {}
            for lbl in unique_labels:
                class_data = eeg[label == lbl]
                cov_matrices[lbl] = torch.cov(class_data.T)

            # Step 2: Solve the generalized eigenvalue problem
            eig_vectors = {}

            # using numpy's eig to solve GEV problem (pytorch's linalg.eig doesn't support generalized eigenvalue problem)
            for lbl, cov_matrix in cov_matrices.items():
                cov_matrix = cov_matrix.cpu().numpy()
                eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

                # Sort eigenvalues and eigenvectors
                sorted_indices = np.argsort(eig_vals)[::-1]

                # Step 3: Store the eigenvectors for each class
                eig_vectors[lbl] = torch.tensor(
                    eig_vecs[:, sorted_indices[: self.num_features]],
                    dtype=torch.float32,
                )

        feature = torch.zeros(
            eeg.shape[0], self.num_features, dtype=torch.float32, device=eeg.device
        )
        for lbl in unique_labels:
            feat = torch.log(
                torch.mean(torch.square(eeg[label == lbl] @ eig_vectors[lbl]), dim=1)
            )
            feature[label == lbl] = feat

        return feature
