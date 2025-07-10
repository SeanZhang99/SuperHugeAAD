from .abc import ClassifierABC
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superhuge.model.types import EEG_TYPE, LABEL_TYPE


def logm(matrix):
    eigvals, eigvecs = torch.linalg.eigh(matrix)  # Batched eigen decomposition
    log_eigvals = torch.log(eigvals)  # Logarithm of eigenvalues
    return (
        eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-1, -2)
    )  # Batched reconstruction


def expm(matrix):
    eigvals, eigvecs = torch.linalg.eigh(matrix)  # Batched eigen decomposition
    exp_eigvals = torch.exp(eigvals)  # Exponential of eigenvalues
    return (
        eigvecs @ torch.diag_embed(exp_eigvals) @ eigvecs.transpose(-1, -2)
    )  # Batched reconstruction


class RGCClassifer(ClassifierABC):
    def __init__(self, /, cov_reg_param: float, **kwargs):
        super().__init__(**kwargs)
        self.cov_reg_param = cov_reg_param

    def estimate_feature(self, eeg: "EEG_TYPE", label: "LABEL_TYPE") -> torch.Tensor:
        # if not fitted, start from step 1. otherwise, perform step 1, and then skip to step 3
        # step 1: compute the regularized covariance matrix cov = (sample.cov() + sample.cov().T)/2 + cov_reg_param * torch.eye(...)
        # step 2: compute the riemannian mean of the covariance matrices using log-eucliean approx.: R_rie_mean = exp(mean(log(cov)))
        # matrix exponential and logarithm is defined as exp(A) = V * exp(Lambda) * V^-1, where v Lambda is eigenvalue and eigenvector
        # store this matrix: R_rie_mean^(-1/2). the matrix power is also defined in eigendecomposition context.

        # step 3: for each sample, compute the tangent space mapping of its regularized covariance matrix: T = log(R_rie_mean_inv_sqrt * rgl_cov_mtx * R_rie_mean_inv_sqrt).
        # step 4: form the feature for each sample by extracting the upper triangular part of the tangent space matrix T, and flatten it to a vector.

        # Step 1: Compute the regularized covariance matrices in batch
        cov_matrices = (
            torch.einsum("bti,btj->bij", eeg, eeg) / eeg.shape[1]
        )  # Compute covariance matrices
        trace_scaling = (
            cov_matrices.diagonal(dim1=-2, dim2=-1).sum(-1) / eeg.shape[2]
        )  # Compute trace scaling factor for batched tensors
        cov_matrices += (
            self.cov_reg_param
            * trace_scaling.unsqueeze(-1).unsqueeze(-1)
            * torch.eye(eeg.shape[2], device=eeg.device).unsqueeze(0)
        )  # Regularization

        if not self._fitted:
            # Step 2: Compute the Riemannian mean using log-Euclidean approximation in batch
            log_cov_matrices = logm(cov_matrices)  # Batched logm
            mean_log_cov = torch.mean(log_cov_matrices, dim=0)  # Reduce batch dimension
            rie_mean = expm(mean_log_cov)  # Expm on reduced mean_log_cov

            # Compute R_rie_mean^(-1/2)
            eigvals, eigvecs = torch.linalg.eigh(rie_mean)
            self.rie_mean_inv_sqrt: torch.Tensor = (
                eigvecs
                @ torch.diag_embed(1.0 / torch.sqrt(eigvals))
                @ eigvecs.transpose(-1, -2)
            )

        # Step 3: Compute tangent space mapping for each sample
        tangent_space_matrices = logm(
            self.rie_mean_inv_sqrt @ cov_matrices @ self.rie_mean_inv_sqrt
        )  # Batched logm

        # Step 4: Extract upper triangular part and flatten to vector
        mask = torch.triu(torch.ones_like(tangent_space_matrices[0]), diagonal=1).bool()
        features = tangent_space_matrices[:, mask].flatten(start_dim=1)

        return features
