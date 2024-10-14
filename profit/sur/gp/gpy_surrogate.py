import numpy as np
from profit.sur import Surrogate
from profit.sur.gp import GaussianProcess
from profit.defaults import fit_gaussian_process as defaults, fit as base_defaults


@Surrogate.register("GPy")
class GPySurrogate(GaussianProcess):
    """Surrogate for https://github.com/SheffieldML/GPy.

    Attributes:
        model (GPy.models): Model object of GPy.
    """

    def __init__(self):
        import GPy

        self.GPy = GPy
        super().__init__()
        self.model = None

    def train(
        self,
        X,
        y,
        kernel=defaults["kernel"],
        hyperparameters=defaults["hyperparameters"],
        fixed_sigma_n=base_defaults["fixed_sigma_n"],
    ):
        self.pre_train(X, y, kernel, hyperparameters, fixed_sigma_n)
        self.model = self.GPy.models.GPRegression(
            self.Xtrain,
            self.ytrain,
            self.kernel,
            noise_var=self.hyperparameters["sigma_n"] ** 2,
        )
        self.optimize()
        self.post_train()

    def add_training_data(self, X, y):
        """Adds training points to the existing dataset.

        This is important for Active Learning. The data is added but the hyperparameters are not optimized yet.

        Parameters:
            X (ndarray): Input points to add.
            y (ndarray): Observed output to add.
        """
        self.Xtrain, self.ytrain = np.concatenate(
            [self.Xtrain, X], axis=0
        ), np.concatenate([self.ytrain, y], axis=0)
        self.encode_training_data()
        self.model.set_XY(self.Xtrain, self.ytrain)
        self.decode_training_data()

    def set_ytrain(self, y):
        """Set the observed training outputs. This is important for active learning.

        Parameters:
        y (np.array): Full training output data.
        """
        self.ytrain = np.atleast_2d(y.copy())
        self.encode_training_data()
        self.model.set_XY(self.Xtrain, self.ytrain)
        self.decode_training_data()

    def predict(self, Xpred, add_data_variance=True):
        Xpred = self.pre_predict(Xpred)
        ymean, yvar = self.model.predict(Xpred, include_likelihood=add_data_variance)
        ymean, yvar = self.decode_predict_data(ymean, yvar)
        return ymean, yvar

    def save_model(self, path):
        """Save the model as dict to a .hdf5 file.

        Parameters:
            path (str): Path including the file name, where the model should be saved.
        """

        from profit.util.file_handler import FileHandler

        sur_dict = {attr: getattr(self, attr) for attr in ("Xtrain", "ytrain")}
        sur_dict["model"] = self.model.to_dict()
        sur_dict["input_encoders"] = str([enc.repr for enc in self.input_encoders])
        sur_dict["output_encoders"] = str([enc.repr for enc in self.output_encoders])
        FileHandler.save(path, sur_dict)

    @classmethod
    def load_model(cls, path):
        """Loads a saved model from a .hdf5 file and updates its attributes. In case of a multi-output model, the .pkl
        file is loaded, since .hdf5 is not supported yet.

        Parameters:
            path (str): Path including the file name, from where the model should be loaded.

        Returns:
            GPy.models: Instantiated surrogate model.
        """

        from profit.util.file_handler import FileHandler
        from profit.sur.encoders import Encoder
        from GPy import models
        from numpy import array  # needed for eval of arrays

        self = cls()
        sur_dict = FileHandler.load(path, as_type="dict")
        self.model = models.GPRegression.from_dict(sur_dict["model"])
        self.Xtrain, self.ytrain = sur_dict["Xtrain"], sur_dict["ytrain"]

        for enc in eval(sur_dict["input_encoders"]):
            self.add_input_encoder(
                Encoder[enc["class"]](enc["columns"], enc["parameters"])
            )
        for enc in eval(sur_dict["output_encoders"]):
            self.add_output_encoder(
                Encoder[enc["class"]](enc["columns"], enc["parameters"])
            )

        # Initialize the encoder by encoding and decoding the training data once.
        self.encode_training_data()
        self.ndim = self.Xtrain.shape[-1]
        self.decode_training_data()

        self.kernel = self.model.kern
        self._set_hyperparameters_from_model()
        self.trained = True
        self.print_hyperparameters("Loaded")
        return self

    def select_kernel(self, kernel):
        """Get the GPy.kern kernel by matching the given string kernel identifier.

        Parameters:
            kernel (str): Kernel string such as 'RBF' or depending on the surrogate also product and sum kernels
                such as 'RBF+Matern52'.

        Returns:
            GPy.kern: GPy kernel object. Currently, for sum and product kernels, the initial hyperparameters are the
            same for all kernels.
        """

        try:
            if not any(operator in kernel for operator in ("+", "*")):
                # Single kernel
                return getattr(self.GPy.kern, kernel)(
                    self.ndim,
                    lengthscale=self.hyperparameters.get("length_scale", [1]),
                    variance=self.hyperparameters.get("sigma_f", 1) ** 2,
                    ARD=len(self.hyperparameters.get("length_scale", [1])) > 1,
                )
            else:
                from re import split

                full_str = split("([+*])", kernel)
                kern = []
                for key in full_str:
                    kern += [
                        (
                            key
                            if key in ("+", "*")
                            else "self.GPy.kern.{}({}, lengthscale={}, variance={})".format(
                                key,
                                self.ndim,
                                self.hyperparameters.get("length_scale", [1]),
                                self.hyperparameters.get("sigma_f", 1) ** 2,
                            )
                        )
                    ]
                return eval("".join(kern))
        except AttributeError:
            raise RuntimeError("Kernel {} is not implemented.".format(kernel))

    def optimize(self, return_hess_inv=False, **opt_kwargs):
        """For hyperparameter optimization the GPy base optimization is used.

        Currently, the inverse Hessian can not be retrieved, which limits the active learning effectivity.

        Parameters:
            return_hess_inv (bool): Is not considered currently.
            opt_kwargs: Keyword arguments used directly in the GPy base optimization.
        """

        self.model.optimize(**opt_kwargs)
        self._set_hyperparameters_from_model()
        self.print_hyperparameters("Optimized")

    def _set_hyperparameters_from_model(self):
        r"""Helper function to set the hyperparameter dict from the model.

        It depends on whether it is a single kernel or a combined one.
        """
        if hasattr(self.model.kern, "lengthscale"):
            self.hyperparameters["length_scale"] = self.model.kern.lengthscale.values
            self.hyperparameters["sigma_f"] = np.sqrt(self.model.kern.variance)
            self.hyperparameters["sigma_n"] = np.sqrt(self.model.likelihood.variance)
        elif hasattr(self.model.kern, "parts"):
            for part in self.model.kern.parts:
                for key, value in zip(part.parameter_names(), part.param_array):
                    value = np.atleast_1d(value)
                    if key == "lengthscale":
                        self.hyperparameters["length_scale"] = value
                    elif key == "variance":
                        self.hyperparameters["sigma_f"] = np.sqrt(value)
                    else:
                        self.hyperparameters[key] = value
            noise_var = self.model.likelihood.gaussian_variance(
                self.model.Y_metadata
            ).reshape(-1, 1, order="F")
            self.hyperparameters["sigma_n"] = np.sqrt(np.max(noise_var, axis=0))
        self.decode_hyperparameters()

    def special_hyperparameter_decoding(self, key, value):
        has_ard = (
            any(p.ARD for p in self.model.kern.parts)
            if hasattr(self.model.kern, "parts")
            else self.model.kern.ARD
        )
        if key == "length_scale" and self.ndim > 1 and not has_ard:
            return np.atleast_1d(np.linalg.norm(value))
        elif key in ("sigma_f", "sigma_n") and len(value) > 1:
            return np.atleast_1d(np.max(value))
        return value


@Surrogate.register("CoregionalizedGPy")
class CoregionalizedGPySurrogate(GPySurrogate):
    def pre_train(
        self,
        X,
        y,
        kernel=defaults["kernel"],
        hyperparameters=defaults["hyperparameters"],
        fixed_sigma_n=base_defaults["fixed_sigma_n"],
    ):
        super().pre_train(X, y, kernel, hyperparameters, fixed_sigma_n)
        self.output_ndim = self.ytrain.shape[-1]

    def train(
        self,
        X,
        y,
        kernel=defaults["kernel"],
        hyperparameters=defaults["hyperparameters"],
        fixed_sigma_n=base_defaults["fixed_sigma_n"],
    ):
        self.pre_train(X, y, kernel, hyperparameters, fixed_sigma_n)
        icm = self.GPy.util.multioutput.ICM(
            input_dim=self.ndim, num_outputs=self.output_ndim, kernel=self.kernel
        )
        _X = self.output_ndim * [self.Xtrain]
        _y = [self.ytrain[:, d].reshape(-1, 1) for d in range(self.output_ndim)]
        self.model = self.GPy.models.GPCoregionalizedRegression(_X, _y, kernel=icm)
        self.optimize()
        self.post_train()

    def add_training_data(self, X, y):
        self.Xtrain, self.ytrain = np.concatenate(
            [self.Xtrain, X], axis=0
        ), np.concatenate([self.ytrain, y], axis=0)
        self.encode_training_data()
        new_Xtrain = np.empty(
            (self.Xtrain.shape[0] * self.output_ndim, self.Xtrain.shape[-1] + 1)
        )
        for d in range(self.output_ndim):
            start_idx = self.Xtrain.shape[0] * d
            end_idx = start_idx + self.Xtrain.shape[0]
            new_Xtrain[start_idx:end_idx] = np.hstack(
                [self.Xtrain, np.ones((self.Xtrain.shape[0], 1)) * d]
            )
        new_noise_dict = {"output_index": new_Xtrain[:, -1:].astype(int)}
        self.model.Y_metadata = new_noise_dict
        self.model.set_XY(new_Xtrain, self.ytrain.reshape(-1, 1, order="F"))
        self.decode_training_data()

    def set_ytrain(self, y):
        self.ytrain = np.atleast_2d(y.copy())
        self.encode_training_data()
        self.model.set_Y(self.ytrain.reshape(-1, 1, order="F"))
        self.decode_training_data()

    def predict(self, Xpred, add_data_variance=True):
        Xpred = super().pre_predict(Xpred)
        ymean = np.empty((Xpred.shape[0], self.output_ndim))
        yvar = ymean.copy()
        for d in range(self.output_ndim):
            newX = np.hstack([Xpred, np.ones((Xpred.shape[0], 1)) * d])
            noise_dict = {"output_index": newX[:, -1:].astype(int)}
            ym, yv = self.model.predict(newX, Y_metadata=noise_dict)
            ymean[:, d], yvar[:, d] = ym.flatten(), yv.flatten()
        ymean, yvar = self.decode_predict_data(ymean, yvar)
        return ymean, yvar

    def save_model(self, path):
        # GPy does not support to_dict method for Coregionalization kernel yet.
        from profit.util.file_handler import FileHandler
        from os.path import splitext

        filepath, ending = splitext(path)
        if ending != ".pkl":
            print(
                f"Saving to '{ending}' not implemented yet for {self.__class__.__name__} surrogate."
                " Saving to '.pkl' file instead."
            )
        save_list = [
            self.model,
            self.Xtrain,
            self.ytrain,
            str([enc.repr for enc in self.input_encoders]),
            str([enc.repr for enc in self.output_encoders]),
        ]
        FileHandler.save(filepath + ".pkl", save_list)

    @classmethod
    def load_model(cls, path):
        from profit.util.file_handler import FileHandler
        from profit.sur.encoders import Encoder
        from numpy import array  # needed for eval of arrays

        self = cls()
        try:
            (
                self.model,
                self.Xtrain,
                self.ytrain,
                input_encoder_str,
                output_encoder_str,
            ) = FileHandler.load(path, as_type="raw")
        except (OSError, FileNotFoundError):
            from os.path import splitext

            print("File not found. Try changing file ending to '.pkl'!")
            path = splitext(path)[0] + ".pkl"
            # Load multi-output model from pickle file
            (
                self.model,
                self.Xtrain,
                self.ytrain,
                input_encoder_str,
                output_encoder_str,
            ) = FileHandler.load(path, as_type="raw")

        for enc in eval(input_encoder_str):
            self.add_input_encoder(
                Encoder[enc["class"]](enc["columns"], enc["parameters"])
            )
        for enc in eval(output_encoder_str):
            self.add_output_encoder(
                Encoder[enc["class"]](enc["columns"], enc["parameters"])
            )
        self.output_ndim = int(max(self.model.X[:, -1])) + 1

        # Initialize the encoder by encoding and decoding the training data once.
        self.encode_training_data()
        self.decode_training_data()

        self.kernel = self.model.kern
        self.ndim = self.Xtrain.shape[-1]
        self._set_hyperparameters_from_model()
        self.trained = True
        self.print_hyperparameters("Loaded")
        return self

    def special_hyperparameter_decoding(self, key, value):
        if key == "length_scale" and self.ndim > 1 and not self.model.kern.parts[0].ARD:
            return np.atleast_1d(np.linalg.norm(value))
        elif key in ("sigma_f", "sigma_n") and len(value) > 1:
            return np.atleast_1d(np.max(value))
        elif key in ("W", "kappa"):
            new_value = value
            for enc in self.output_encoders:
                if enc.label == "Normalization":
                    new_value = enc.decode_hyperparameters(value)
            return np.atleast_1d(np.min(new_value))
        return value
