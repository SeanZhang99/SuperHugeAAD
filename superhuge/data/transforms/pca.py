import numpy as np
from .stats_abc import StatisticalTransform
from sklearn.decomposition import PCA as sk_pca


class PCA(StatisticalTransform):
    def __init__(self, /, *, n_components: int, **kwargs):
        self.n_components = n_components
        super().__init__(**kwargs)
        self._pca = sk_pca(n_components=n_components)
        self._data_tmp = []

    def update(self, x: np.ndarray) -> None:
        self._data_tmp.append(x)

    def fit(self):
        self._pca.fit_transform(np.stack(self._data_tmp, axis=0))
        self._stat: sk_pca = self._pca
        self._fitted = True

    def reset(self):
        super().reset()
        self._pca = sk_pca(n_components=self.n_components)
        self._data_tmp = []

    def __call__(self, x: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
        super().__call__(x)
        x_transformed = self._stat.transform(x)
        return {"x": x_transformed}
