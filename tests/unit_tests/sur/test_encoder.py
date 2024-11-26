"""
Testcases for encoders:
- Exclude
- Log10
- Normalization

Separate linear-reduction tests:
- PCA
- Karuhnen-Loeve
"""

import numpy as np
from profit.sur.encoders import Encoder


def test_exclude():
    COLUMNS = [2]
    CONFIG = {"class": "Exclude", "columns": COLUMNS, "parameters": {}}
    SIZE = (10, 4)
    n = SIZE[0] * SIZE[1]
    X = np.linspace(0, n - 1, n).reshape(SIZE)

    enc = Encoder["Exclude"](COLUMNS)
    assert enc.repr == CONFIG
    X_enc = enc.encode(X)
    assert np.all(X_enc == X[:, [0, 1, 3]])
    X_dec = enc.decode(X_enc)
    assert np.all(X_dec == X)


def test_log10():
    from profit.sur.encoders import Log10Encoder

    COLUMNS = [2, 3]
    CONFIG = {"class": "Log10", "columns": COLUMNS, "parameters": {}}
    SIZE = (10, 4)
    n = SIZE[0] * SIZE[1]
    X = np.linspace(0, n - 1, n).reshape(SIZE)
    X_log = X.copy()
    X_log[:, COLUMNS] = np.log10(X_log[:, COLUMNS])

    enc = Log10Encoder(COLUMNS)
    assert enc.repr == CONFIG
    X_enc = enc.encode(X)
    assert np.all(X_enc == X_log)
    X_dec = enc.decode(X_enc)
    assert np.allclose(X_dec, X, atol=1e-7)


def test_normalization():
    COLUMNS = [0, 1, 2, 3]
    CONFIG = {"class": "Normalization", "columns": COLUMNS, "parameters": {}}
    SIZE = (10, 4)
    n = SIZE[0] * SIZE[1]
    X = np.linspace(0, n - 1, n).reshape(SIZE)
    norm_box = np.hstack(
        [np.linspace(0, 1, SIZE[0]).reshape(-1, 1) for _ in range(SIZE[1])]
    )

    enc = Encoder["Normalization"](COLUMNS)
    assert enc.repr == CONFIG
    X_enc = enc.encode(X)
    assert np.allclose(X_enc, norm_box)
    X_dec = enc.decode(X_enc)
    assert np.allclose(X_dec, X)

    y = np.sin(X[:, [1]])
    enc = Encoder["Normalization"]([0])
    y_enc = enc.encode(y)
    assert np.allclose(np.max(y_enc), 1)
    assert np.allclose(np.min(y_enc), 0)
    y_dec = enc.decode(y_enc)
    assert np.allclose(y_dec, y)

    hyperparameters = {
        "length_scale": np.array([1.3, 0.2, 2.4]),
        "sigma_f": np.array([0.7]),
        "sigma_n": np.array([0.05]),
    }

    xmean = 1.7
    xstd = 2.8
    xmin = 0.5
    xmax = 3
    xmax_centered = (xmax - xmean) / xstd
    xmin_centered = (xmin - xmean) / xstd
    scaling = (xmax_centered - xmin_centered) * xstd

    enc = Encoder["Normalization"](
        COLUMNS,
        parameters={
            "xmean": xmean,
            "xstd": xstd,
            "xmin": xmin,
            "xmax": xmax,
            "xmin_centered": xmin_centered,
            "xmax_centered": xmax_centered,
        },
    )
    length_scale_dec = enc.decode_hyperparameters(hyperparameters["length_scale"])
    sigma_f_dec = np.sqrt(enc.decode_variance(hyperparameters["sigma_f"] ** 2))
    sigma_n_dec = np.sqrt(enc.decode_variance(hyperparameters["sigma_n"] ** 2))
    assert np.allclose(length_scale_dec, hyperparameters["length_scale"] * scaling)
    assert np.allclose(sigma_f_dec, hyperparameters["sigma_f"] * scaling)
    assert np.allclose(sigma_n_dec, hyperparameters["sigma_n"] * scaling)
