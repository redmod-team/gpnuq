from profit.util.base_class import CustomABC
import numpy as np


class Encoder(CustomABC):
    r"""Base class to handle encoding and decoding of the input and output data before creating the surrogate model.

    The base class itself does nothing. It delegates the encoding process to the childs
    which are called by their registered labels.

    Parameters:
        columns (list[int]): Columns of the data the encoder acts on.
        parameters (dict): Miscellaneous parameters stored during encoding, which are needed for decoding. E.g. the
            scaling factor during normalization.

    Attributes:
        label (str): Label of the encoder class.
    """

    labels = {}

    def __init__(self, columns, parameters=None):
        self.label = self.__class__.get_label()
        self.parameters = (
            {key: np.array(values) for key, values in parameters.items()}
            if parameters
            else {}
        )
        self.columns = columns

    @property
    def repr(self):
        """Easy to handle representation of the encoder for saving and loading.
        Returns:
            list: List of all relevant information to reconstruct the encoder.
                (label, columns, parameters dict)
        """
        parameters_dict = {}
        for key, values in self.parameters.items():
            try:
                parameters_dict[key] = values.tolist()
            except AttributeError:
                parameters_dict[key] = [values]
        return {
            "class": self.label,
            "columns": self.columns,
            "parameters": parameters_dict,
        }

    def encode(self, x):
        """Applies the encoding function on given columns.

        Parameters:
            x (ndarray): Array to which the encoding is applied.

        Returns:
            ndarray: An encoded copy of the array x.
        """
        _x = x.copy()
        _x[:, self.columns] = self.encode_func(_x[:, self.columns])
        return _x

    def decode(self, x):
        """Applies the decoding function on given columns.

        Parameters:
            x (ndarray): Array to which the decoding is applied.

        Returns:
            ndarray: A decoded copy of the array x.
        """
        _x = x.copy()
        _x[:, self.columns] = self.decode_func(_x[:, self.columns])
        return _x

    def decode_hyperparameters(self, value):
        """Decoder for the surrogate hyperparameters, as the direct model uses encoded values.
        As a default, the unchanged value is returned.

        Parameters:
            value (np.array): The (encoded) value of the hyperparameter.
        Returns:
            np.array: Decoded value.
        """
        return value

    def decode_variance(self, variance):
        return variance

    def encode_func(self, x):
        r"""
        Returns:
            ndarray: Function used for decoding the data. E.g. $\log_{10}(x)$.
        """
        return x

    def decode_func(self, x):
        r"""
        Returns:
            ndarray: Inverse transform of the encoding function. For an encoding of $\log_{10}(x)$ this
                would be $10^x$.
        """
        return x


@Encoder.register("Exclude")
class ExcludeEncoder(Encoder):
    """Excludes specific columns from the fit. Afterwards they are inserted at the same position.

    Variables:
        excluded_values (np.array): Slice of the input data which is excluded.
    """

    def encode(self, x):
        self.parameters["excluded_values"] = x[:, self.columns]
        return x[:, [i for i in range(x.shape[-1]) if i not in self.columns]]

    def decode(self, x):
        for idx, col in enumerate(self.columns):
            insert_x = self.parameters["excluded_values"][:, idx]
            x = np.insert(x, col, insert_x, axis=1)
        return x


@Encoder.register("Log10")
class Log10Encoder(Encoder):
    r"""Transforms the specified columns with $log_{10}$. This is done for LogUniform variables by default."""

    def encode_func(self, x):
        return np.log10(x)

    def decode_func(self, x):
        return 10**x


@Encoder.register("Normalization")
class Normalization(Encoder):
    r"""Normalization of the specified columns. Usually this is done for all input and output,
        so the surrogate can fit on a (0, 1)^n cube with zero mean and unit variance.

        $$
        \begin{align}
        x' &= (x - x_{min}) / (x_{max} - x_{min}) \\
        x & = (x_{max} - x_{min}) * x' + x_{min}
        \end{align}
        $$

    Parameters:
        xmax (np.array): Max. value of the data for each column.
        xmin (np.array): Min. value of the data for each column.
        xmean (np.array): Mean value of the data for each column.
        xstd (np.array): Standard deviation of the data for each column.
        xmax_centered (np.array): Max. value of the data after mean and variance standardization.
        xmin_centered (np.array): Min. value of the data after mean and variance standardization.
    """

    def encode(self, x):
        if self.parameters.get("xmean") is None:
            self.parameters["xmean"] = x[:, self.columns].mean(axis=0)
        if self.parameters.get("xstd") is None:
            self.parameters["xstd"] = np.maximum(x[:, self.columns].std(axis=0), 1e-10)
        if self.parameters.get("xmax") is None:
            self.parameters["xmax"] = x[:, self.columns].max(axis=0)
        self.parameters["xmax_centered"] = (
            self.parameters["xmax"] - self.parameters["xmean"]
        ) / self.parameters["xstd"]
        if self.parameters.get("xmin") is None:
            self.parameters["xmin"] = x[:, self.columns].min(axis=0)
        self.parameters["xmin_centered"] = (
            self.parameters["xmin"] - self.parameters["xmean"]
        ) / self.parameters["xstd"]
        return super().encode(x)

    def encode_func(self, x):
        _x = (x - self.parameters["xmean"]) / self.parameters["xstd"]
        _x = (_x - self.parameters["xmin_centered"]) / np.maximum(
            self.parameters["xmax_centered"] - self.parameters["xmin_centered"], 1e-10
        )
        return _x

    def decode_func(self, x):
        _x = (
            x * (self.parameters["xmax_centered"] - self.parameters["xmin_centered"])
            + self.parameters["xmin_centered"]
        )
        _x = _x * self.parameters["xstd"] + self.parameters["xmean"]
        return _x

    def decode_hyperparameters(self, value):
        """Decode surrogate's hyperparameters. Distinguish between length_scale (only input_encoders)
        and sigma_f, sigma_n (only output_encoders) done in profit.sur.gp.gaussian_process.GaussianProcess.

        Parameters:
            value (np.array): The (encoded) value of the hyperparameter.
        Returns:
            np.array: Decoded value.
        """
        _value = value * (
            self.parameters["xmax_centered"] - self.parameters["xmin_centered"]
        )
        _value = _value * self.parameters["xstd"]
        return _value

    def decode_variance(self, variance):
        _variance = (
            variance
            * (self.parameters["xmax_centered"] - self.parameters["xmin_centered"]) ** 2
        )
        _variance = _variance * self.parameters["xstd"] ** 2
        return _variance


@Encoder.register("PCA")
class PCA(Encoder):
    def __init__(self, columns=(), parameters=None):
        if parameters is None:
            parameters = {}
        if "tol" not in parameters:
            parameters["tol"] = 1e-2

        super().__init__(columns, parameters)
        if self.parameters.get("ytrain") is not None:
            self.init_eigvalues(self.parameters["ytrain"])
        else:
            self.ymean = None
            self.dy = None
            self.w = None
            self.Q = None

    def init_eigvalues(self, y):
        from scipy.linalg import eigh

        self.parameters["ytrain"] = y
        self.ymean = np.mean(y, 0)
        self.dy = y - self.ymean
        w, Q = eigh(self.dy.T @ self.dy)
        condi = w > self.parameters["tol"]
        self.w = w[condi]
        self.Q = Q[:, condi]

    def encode(self, y):
        """
        Parameters:
            y: ntest sample vectors of length N.

        Returns:
            Expansion coefficients of y in eigenbasis.
        """
        if "ytrain" not in self.parameters:
            self.init_eigvalues(y)
        return (y - self.ymean) @ self.Q

    def decode(self, z):
        """
        Parameters:
            z: Expansion coefficients of y in eigenbasis.

        Returns:
            Reconstructed ntest sample vectors of length N.
        """
        return self.ymean + (self.Q @ z.T).T

    @property
    def features(self):
        """
        Returns:
            neig feature vectors of length N.
        """
        return self.Q

    def decode_variance(self, variance):
        if variance.shape[-1] == self.Q.shape[-1]:
            # For multi-output data
            var_dec = np.empty((variance.shape[0], self.ymean.shape[0]))
            for row, var_enc in enumerate(variance):
                I = np.eye(var_enc.shape[0], var_enc.shape[0])
                var_dec[row, :] = np.diag(self.Q @ (var_enc * I @ self.Q.T))
            return var_dec
        return np.full_like(variance, np.nan)


@Encoder.register("KarhunenLoeve")
class KarhunenLoeve(PCA):
    def encode(self, y):
        """
        Parameters:
            y: ntest sample vectors of length N.

        Returns:
            Expansion coefficients of y in eigenbasis.
        """
        if "ytrain" not in self.parameters:
            self.init_eigvalues(y)
        ntrain = self.dy.shape[0]
        ntest = y.shape[0]
        b = np.empty((ntrain, ntest))
        for i in range(ntrain):
            b[i, :] = (y - self.ymean) @ self.dy[i]
        yres = np.diag(1.0 / self.w) @ self.Q.T @ b
        return yres.T

    def init_eigvalues(self, y):
        from scipy.linalg import eigh

        self.parameters["ytrain"] = y
        self.ymean = np.mean(y, 0)
        self.dy = y - self.ymean
        w, Q = eigh(self.dy @ self.dy.T)
        condi = w > self.parameters["tol"]
        self.w = w[condi]
        self.Q = Q[:, condi]

    def decode(self, z):
        """
        Parameters:
            z: Expansion coefficients of y in eigenbasis.

        Returns:
            Reconstructed ntest sample vectors of length N.
        """
        return self.ymean + (self.dy.T @ (self.Q @ z.T)).T

    @property
    def features(self):
        return self.dy.T @ self.Q

    def decode_variance(self, variance):
        if variance.shape[-1] == self.Q.shape[-1]:
            # For multi-output data
            var_dec = np.empty((variance.shape[0], self.ymean.shape[0]))
            for row, var_enc in enumerate(variance):
                I = np.eye(var_enc.shape[0], var_enc.shape[0])
                var_dec[row, :] = np.diag(
                    (self.dy.T @ self.Q) @ (var_enc * I @ (self.dy.T @ self.Q).T)
                )
            return var_dec
        return np.full_like(variance, np.nan)
