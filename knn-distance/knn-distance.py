import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # Hint 1: datos escalares -> (-1, 1)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    # Hint 2: diferencias por pares con broadcasting -> (n_test, n_train, D)
    diff = X_test[:, None, :] - X_train[None, :, :]

    # Distancia euclidiana -> (n_test, n_train)
    dist = np.sqrt((diff ** 2).sum(axis=2))

    # Índices ordenados de menor a mayor distancia
    # kind="stable" garantiza que en empates se mantenga el orden ascendente de índices
    idx = np.argsort(dist, axis=1, kind="stable")[:, :k]

    # Hint 3: rellenar con -1 si k > tamaño del train
    n_test, n_train = dist.shape
    if k > n_train:
        pad = np.full((n_test, k - n_train), -1, dtype=int)
        idx = np.concatenate([idx.astype(int), pad], axis=1)

    return idx.astype(int)