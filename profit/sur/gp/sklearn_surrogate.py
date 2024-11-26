import numpy as np
from profit.sur import Surrogate
from profit.sur.gp import GaussianProcess
from profit.defaults import fit_gaussian_process as defaults, fit as base_defaults


@Surrogate.register("Sklearn")
class SklearnGPSurrogate(GaussianProcess):
    """Surrogate for https://github.com/scikit-learn/scikit-learn Gaussian process.

    Attributes:
        model (sklearn.gaussian_process.GaussianProcessRegressor): Model object of Sklearn.
    """

    def __init__(self):
        super().__init__()
        self.model = None

    def train(
        self,
        X,
        y,
        kernel=defaults["kernel"],
        hyperparameters=defaults["hyperparameters"],
        fixed_sigma_n=base_defaults["fixed_sigma_n"],
        **kwargs
    ):
        from sklearn.gaussian_process import GaussianProcessRegressor

        self.pre_train(X, y, kernel, hyperparameters, fixed_sigma_n)
        numeric_noise = (
            self.hyperparameters["sigma_n"].item() ** 2 if self.fixed_sigma_n else 1e-5
        )

        # Instantiate the model
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=numeric_noise)

        # Train the model
        self.model.fit(self.Xtrain, self.ytrain)
        self.kernel = self.model.kernel_
        self.post_train()

    def post_train(self):
        self._set_hyperparameters_from_model()
        self.print_hyperparameters("Optimized")
        super().post_train()

    def add_training_data(self, X, y):
        """Add training points to existing data.

        Parameters:
            X (ndarray): Input points to add.
            y (ndarray): Observed output to add.
        """
        self.Xtrain = np.concatenate([self.Xtrain, X], axis=0)
        self.ytrain = np.concatenate([self.ytrain, y], axis=0)

    def set_ytrain(self, ydata):
        """Set the observed training outputs. This is important for active learning.

        Parameters:
            ydata (np.array): Full training output data.
        """
        self.ytrain = np.atleast_2d(ydata.copy())

    def predict(self, Xpred, add_data_variance=True):
        Xpred = self.pre_predict(Xpred)

        ymean, ystd = self.model.predict(Xpred, return_std=True)
        yvar = ystd.reshape(-1, 1) ** 2
        if add_data_variance:
            yvar = yvar + self.hyperparameters["sigma_n"] ** 2
        ymean, yvar = self.decode_predict_data(ymean, yvar)
        return ymean, yvar

    def save_model(self, path):
        """Save the SklGPSurrogate model to a pickle file. All attributes of the surrogate are loaded directly from the
        model.

        Parameters:
            path (str): Path including the file name, where the model should be saved.
        """

        from pickle import dump

        dump(self.model, open(path, "wb"))

    @classmethod
    def load_model(cls, path):
        """Load a saved SklGPSurrogate model from a pickle file and update its attributes.

        Parameters:
            path (str): Path including the file name, from where the model should be loaded.

        Returns:
            profit.sur.gaussian_process.SklearnGPSurrogate: Instantiated surrogate model.
        """

        from pickle import load

        self = cls()
        self.model = load(open(path, "rb"))
        self.Xtrain = self.model.X_train_
        self.ytrain = self.model.y_train_
        self.kernel = self.model.kernel_
        self.ndim = self.Xtrain.shape[-1]
        self.fixed_sigma_n = self.model.alpha != 1e-5
        self.trained = True
        self._set_hyperparameters_from_model()
        self.print_hyperparameters("Loaded")
        return self

    def optimize(self, **opt_kwargs):
        """For hyperparameter optimization the Sklearn base optimization is used.

        Currently, the inverse Hessian can not be retrieved, which limits the active learning effectivity.

        Parameters:
            opt_kwargs: Keyword arguments used directly in the Sklearn base optimization.
        """
        self.encode_training_data()
        self.model.fit(self.Xtrain, self.ytrain, **opt_kwargs)
        self.decode_training_data()
        self._set_hyperparameters_from_model()
        self.print_hyperparameters("Optimized")

    def select_kernel(self, kernel):
        """Get the sklearn.gaussian_process.kernels kernel by matching the given kernel identifier.

        Parameters:
            kernel (str): Kernel string such as 'RBF' or depending on the surrogate also product and sum kernels
                such as 'RBF+Matern52'.

        Returns:
            sklearn.gaussian_process.kernels: Scikit-learn kernel object. Currently, for sum and product kernels,
            the initial hyperparameters are the same for all kernels.
        """

        from re import split
        from sklearn.gaussian_process import kernels as sklearn_kernels

        full_str = split("([+*])", kernel)
        try:
            kernel = []
            for key in full_str:
                kernel += [
                    (
                        key
                        if key in ("+", "*")
                        else getattr(sklearn_kernels, key)(
                            length_scale=self.hyperparameters["length_scale"]
                        )
                    )
                ]
        except AttributeError:
            raise RuntimeError("Kernel {} is not implemented.".format(kernel))

        if len(kernel) == 1:
            kernel = kernel[0]
        else:
            kernel = [str(key) if not isinstance(key, str) else key for key in kernel]
            kernel = eval("".join(kernel))

        # Add scale and noise to kernel
        kernel *= sklearn_kernels.ConstantKernel(
            constant_value=1 / self.hyperparameters["sigma_f"].item() ** 2
        )
        if not self.fixed_sigma_n:
            kernel += sklearn_kernels.WhiteKernel(
                noise_level=self.hyperparameters["sigma_n"].item() ** 2
            )

        return kernel

    def _set_hyperparameters_from_model(self):
        r"""Helper function to set the hyperparameter dict from the model.

        It depends on whether $\sigma_n$ is fixed.
        Currently this is only stable for single kernels and not for Sum and Prod kernels.
        """
        if self.fixed_sigma_n:
            self.hyperparameters["length_scale"] = np.atleast_1d(
                self.model.kernel_.k1.length_scale
            )
            self.hyperparameters["sigma_f"] = np.sqrt(
                np.atleast_1d(1 / self.model.kernel_.k2.constant_value)
            )
            self.hyperparameters["sigma_n"] = np.sqrt(np.atleast_1d(self.model.alpha))
        else:
            self.hyperparameters["length_scale"] = np.atleast_1d(
                self.model.kernel_.k1.k1.length_scale
            )
            self.hyperparameters["sigma_f"] = np.sqrt(
                np.atleast_1d(1 / self.model.kernel_.k1.k2.constant_value)
            )
            self.hyperparameters["sigma_n"] = np.sqrt(
                np.atleast_1d(self.model.kernel_.k2.noise_level)
            )
        self.decode_hyperparameters()


# Draft for Scikit-learn implementation of LinearEmbedding kernel of Garnett (2014)
from sklearn.gaussian_process.kernels import (
    Kernel,
    Hyperparameter,
    StationaryKernelMixin,
    NormalizedKernelMixin,
)


class LinearEmbedding(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(
        self, dims, length_scale=np.array([1.0]), length_scale_bounds=(1e-5, 1e5)
    ):
        self.length_scale = length_scale
        self.dims = dims
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds, len(self.length_scale)
        )

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        R = self.length_scale.reshape(self.dims)
        X1 = X @ R.T
        X2 = X @ R.T if Y is None else Y @ R.T
        dX = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        dX_sq = np.linalg.norm(dX, axis=-1) ** 2
        K = np.exp(-0.5 * dX_sq)
        if eval_gradient:
            K_gradient = np.einsum("ijk,kl", dX**2, R) * K[..., np.newaxis]
            return K, K_gradient
        return K

    def __repr__(self):
        return "{0}(length_scale=[{1}])".format(
            self.__class__.__name__, ", ".join(map("{0:.3g}".format, self.length_scale))
        )
