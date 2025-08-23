import numpy as np
import torch
import scipy.signal

from ...types import EEG_TYPE, LABEL_TYPE
from .abc import ClassifierABC


class CSPClassifier(ClassifierABC):

    def __init__(
        self,
        /,
        num_features: int,
        passbands: list[tuple[float, float]] | None = None,
        **kwargs,
    ):
        """
        Initialize the CSPClassifier with the number of features to extract and passbands for filterbank.

        Parameters:
        num_features: int, the number of features (weights / eigenvalues / eigenvectors) to be kept with the largest eigenvalues for each class.
        passbands: list of tuples, each tuple contains (low_freq, high_freq) for a passband.
        fs: float, the sampling frequency.
        """
        super().__init__(**kwargs)
        self.num_features: int = num_features
        self.passbands: list[tuple[float, float]] = (
            passbands if passbands is not None else [(1, 4), (4, 8), (8, 12), (12, 19)]
        )
        self.fs: float = kwargs["fs"]
        self.filters: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        self._fitted = False

        # Design filters and store coefficients
        nyquist = self.fs / 2
        for low_freq, high_freq in self.passbands:
            b, a = scipy.signal.butter(  # type: ignore
                4, [low_freq / nyquist, high_freq / nyquist], btype="bandpass"
            )  # type: ignore
            self.filters[(low_freq, high_freq)] = (b, a)

    def bandpass_filter(
        self, eeg: np.ndarray, low_freq: float, high_freq: float
    ) -> np.ndarray:
        """
        Apply a bandpass filter to EEG data using precomputed filter coefficients.

        Parameters:
        eeg: np.ndarray, the EEG data.
        low_freq: float, the lower bound of the frequency band.
        high_freq: float, the upper bound of the frequency band.

        Returns:
        np.ndarray, the filtered EEG data.
        """
        b, a = self.filters[(low_freq, high_freq)]
        return scipy.signal.filtfilt(b, a, eeg, axis=1)

    def estimate_feature(self, eeg: EEG_TYPE, label: LABEL_TYPE) -> torch.Tensor:
        """
        Estimate features from EEG data specific to CSPClassifier, with filterbank settings.
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
            unique_labels = torch.unique(label)
            self.eig_vectors = {
                pb: {} for pb in self.passbands
            }  # Store eigenvectors for each passband

            for low_freq, high_freq in self.passbands:
                cov_matrices: dict[str | int, np.ndarray] = {}
                for lbl in unique_labels:
                    class_data = eeg[label == lbl].cpu().numpy()
                    # Apply bandpass filter for the current passband
                    filtered_data = self.bandpass_filter(
                        class_data, low_freq, high_freq
                    )
                    # Compute covariance matrix for batched data manually using einsum
                    cov_matrices[int(lbl)] = np.einsum(
                        "bti,btj->ij", filtered_data, filtered_data
                    ) / (filtered_data.shape[0] - 1)
                    cov_matrices[int(lbl)] = (
                        cov_matrices[int(lbl)] + cov_matrices[int(lbl)].T
                    ) / 2  # Ensure symmetry

                # Solve the generalized eigenvalue problem for the current passband
                for lbl, cov_matrix in cov_matrices.items():
                    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

                    # Sort eigenvalues and eigenvectors
                    sorted_indices = np.argsort(eig_vals)[::-1]

                    # Store the eigenvectors for each class and passband
                    self.eig_vectors[(low_freq, high_freq)][int(lbl)] = eig_vecs[
                        :, sorted_indices[: self.num_features]
                    ]
            self._fitted = True  # Mark as fitted

        feature = np.zeros(
            (
                eeg.shape[0],
                self.num_features
                * len(self.passbands)
                * len(self.eig_vectors[self.passbands[0]].keys()),
            ),
            dtype=np.float32,
        )
        for p_idx, (low_freq, high_freq) in enumerate(self.passbands):
            filtered_data = self.bandpass_filter(eeg.cpu().numpy(), low_freq, high_freq)
            for i, lbl in enumerate(self.eig_vectors[(low_freq, high_freq)].keys()):
                feat = np.log(
                    np.mean(
                        np.square(
                            filtered_data
                            @ self.eig_vectors[(low_freq, high_freq)][int(lbl)]
                        ),
                        axis=1,
                    )
                )
                feature[
                    :,
                    (p_idx * len(self.eig_vectors[(low_freq, high_freq)].keys()) + i)
                    * self.num_features : (
                        p_idx * len(self.eig_vectors[(low_freq, high_freq)].keys())
                        + i
                        + 1
                    )
                    * self.num_features,
                ] = feat

        return torch.tensor(feature, dtype=torch.float32, device=eeg.device)
