from os import path
import yaml
from collections import OrderedDict
from profit import defaults
from profit.util.base_class import CustomABC
import warnings

VALID_FORMATS = (".yaml", ".py")

"""
yaml has to be configured to represent OrderedDict
see https://stackoverflow.com/questions/16782112/can-pyyaml-dump-dict-items-in-non-alphabetical-order
and https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
"""


def represent_ordereddict(dumper, data):
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode("tag:yaml.org,2002:map", value)


def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))


_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
yaml.add_representer(OrderedDict, represent_ordereddict)
yaml.add_constructor(_mapping_tag, dict_constructor)

""" now yaml is configured to handle OrderedDict input and output """


def load_config_from_py(filename):
    """Load the configuration parameters from a python file into dict."""
    from importlib.util import spec_from_file_location, module_from_spec

    spec = spec_from_file_location("f", filename)
    f = module_from_spec(spec)
    spec.loader.exec_module(f)
    return {
        name: value for name, value in f.__dict__.items() if not name.startswith("_")
    }


class AbstractConfig(CustomABC):
    """General class with methods which are useful for all Config classes."""

    labels = {}
    defaults = None

    def __init__(self, **entries):
        if self.defaults:
            self.set_defaults(getattr(defaults, self.defaults))
        self.update(**entries)

    def update(self, **entries):
        """Updates the attributes with user inputs. A warning is issued if the attribute set by the user is unknown.

        Parameters:
            entries (dict): User input of the config parameters.
        """
        for name, value in entries.items():
            if hasattr(self, name) or name in map(str.lower, self.labels):
                attr = getattr(self, name, None)
                if isinstance(attr, dict):
                    attr.update(value)
                    setattr(self, name, attr)
                else:
                    setattr(self, name, value)
            else:
                message = f"Config parameter '{name}' for {self.__class__.__name__} configuration may be unused."
                warnings.warn(message)
                setattr(self, name, value)

    def process_entries(self, base_config):
        """After the attributes are set, they are formatted and edited to standardize the user inputs.

        Parameters:
            base_config (BaseConfig): In sub configs, the data from the base config is needed.
        """
        pass

    def set_defaults(self, default_dict):
        """Default values are set from a default dictionary, which is usually located
        in the global profit.defaults file.
        """
        for name, value in default_dict.items():
            if name in self.labels and isinstance(value, str):
                value = {"class": value}
            setattr(self, name, value)

    def create_subconfig(self, sub_config_label, **entries):
        """Instances of sub configs are created from a string or a dictionary.

        Parameters:
            sub_config_label (str): Dict key of registered sub config.
            entries (dict): User input parameters.
        """
        if "class" in entries:
            # Load specific sub config or default config, if missing.
            try:
                sub = self.labels[sub_config_label][entries["class"]]()
            except KeyError:
                sub = self.labels[sub_config_label]["default"](**entries)
        else:
            # Load general sub config.
            sub = self.labels[sub_config_label]()

        # Split entries into entries for this config and further sub configs.
        base_entries = {k: v for k, v in entries.items() if k.lower() not in sub.labels}
        sub_entries = {
            k: {"class": v} if isinstance(v, str) else v
            for k, v in entries.items()
            if k.lower() in sub.labels
        }

        # Update defaults with user entries
        sub.update(**base_entries)

        # Create second level sub configs.
        for subsub_label in sub.labels:
            subsub_entries = sub[subsub_label]
            subsub_entries.update(sub_entries.get(subsub_label, {}))
            sub.create_subconfig(subsub_label, **subsub_entries)
        setattr(self, sub_config_label, sub)

    def __getitem__(self, item):
        """Implements the dictionary like get method with brackets.

        Parameters:
            item (str): Label of the attribute to return.

        Returns:
            Attribute or if the attribute is a sub config, a dictionary of the sub config items.
        """
        attr = getattr(self, item)
        if item in self.labels:
            if type(attr) is list:
                return {"list": attr}
            return {key: attr[key] for key, _ in attr.items()}
        return attr

    def items(self):
        """Implements the dictionary like self.items() method.

        Returns:
            list: List of (key, value) tuples of the class attributes.
        """
        return [(key, self[key]) for key in vars(self)]

    def get(self, item, default=None):
        """Implements the dictionary like get method with a default value.

        Parameters:
            item (str): Label of the attribute to return.
            default: Default value, if the attribute is not found.

        Returns:
            Attribute or the default value.
        """
        try:
            return self[item]
        except AttributeError:
            return default


class BaseConfig(AbstractConfig):
    """
    This class and its modular subclasses provide all possible configuration parameters.

    Parts of the Config:
        - base_dir
        - run_dir
        - config_file
        - include
        - ntrain
        - variables
        - files
            - input
            - output
        - run
            - runner
            - interface
            - pre
            - post
        - fit
            - surrogate
            - save / load
            - fixed_sigma_n
        - active_learning
        - ui

    Base configuration for fundamental parameters.

    Parameters:
        base_dir (str): Base directory.
        run_dir (str): Run directory.
        config_path (str): Path to configuration file.
        include (list): Paths to custom files which are loaded in the beginning.
        files (dict): Paths for input and output files.
        ntrain (int): Number of training samples.
        variables (dict): All variables.
        input (dict): Input variables.
        output (dict): Output variables.
        independent (dict): Independent variables, if the result of the simulation is a vector.
    """

    labels = {}

    def __init__(self, base_dir=defaults.base_dir, **entries):
        # Set defaults
        self.base_dir = path.abspath(base_dir)
        self.run_dir = self.base_dir
        self.config_path = path.join(self.base_dir, defaults.config_file)
        self.include = defaults.include
        self.ntrain = defaults.ntrain
        self.variables = defaults.variables.copy()
        self.input = {}
        self.output = {}
        self.independent = {}
        self.files = defaults.files.copy()

        # Split user entries in entries for base_config and for sub_configs
        base_entries = {
            k: v for k, v in entries.items() if k.lower() not in self.labels
        }
        sub_entries = {
            k: {"class": v} if isinstance(v, str) else v
            for k, v in entries.items()
            if k.lower() in self.labels
        }

        self.update(**base_entries)  # Update the attributes with given entries.
        self.load_includes()  # Load external files.

        for sub_config_label in self.labels:
            single_sub_entries = sub_entries.get(sub_config_label, {})
            self.create_subconfig(sub_config_label, **single_sub_entries)

        self.process_entries()  # Postprocess the attributes to standardize different user entries.

    def process_entries(self):
        """Sets absolute paths, creates variables and delegates to the sub configs."""
        from profit.util.variable import Variable, VariableGroup

        # Set absolute paths
        self.files["input"] = path.join(
            self.base_dir, self.files.get("input", defaults.files["input"])
        )
        self.files["output"] = path.join(
            self.base_dir, self.files.get("output", defaults.files["output"])
        )

        # Variable configuration as dict
        self.variable_group = VariableGroup(self.ntrain)
        vars = []
        for k, v in self.variables.items():
            if isinstance(v, (int, float)):
                v = f"Constant({v})"
            if isinstance(v, str):
                vars.append(Variable.create_from_str(k, (self.ntrain, 1), v))
            else:
                vars.append(Variable.create(name=k, size=(self.ntrain, 1), **v))
        self.variable_group.add(vars)

        self.variables = self.variable_group.as_dict
        self.input = {
            k: v
            for k, v in self.variables.items()
            if not any(k in v["kind"].lower() for k in ("output", "independent"))
        }
        self.output = {
            k: v for k, v in self.variables.items() if "output" in v["kind"].lower()
        }
        self.independent = {
            k: v
            for k, v in self.variables.items()
            if "independent" in v["kind"].lower() and v["size"] != (1, 1)
        }

        for sub_config_label in self.labels:
            getattr(self, sub_config_label).process_entries(self)

    @classmethod
    def from_file(cls, filename=defaults.config_file):
        """Creates a configuration class from a .yaml or .py file."""

        if filename.endswith(".yaml"):
            with open(filename) as f:
                entries = yaml.safe_load(f)
        elif filename.endswith(".py"):
            entries = load_config_from_py(filename)
        else:
            raise TypeError(
                f"Not supported file extension .{filename.split('.')[-1]} for config file.\n"
                f"Valid file formats: {VALID_FORMATS}"
            )
        self = cls(base_dir=path.split(filename)[0], **entries)
        self.config_path = path.join(self.base_dir, filename)
        return self

    def load_includes(self):
        from profit.util import load_includes
        import os
        import json

        if isinstance(self.include, str):
            self.include = [self.include]

        self.include = [path.abspath(path.join(self.base_dir, p)) for p in self.include]
        load_includes(self.include)
        os.environ["PROFIT_INCLUDES"] = json.dumps(self.include)


@BaseConfig.register("run")
class RunConfig(AbstractConfig):
    """Run configuration with the following sub classes:
        - runner
            - local
            - slurm
        - interface
            - memmap
            - zeromq
        - pre
            - template
        - post
            - json
            - numpytxt
            - hdf5

    A default sub class which just updates the entries from a user input is also implemented and used if the
    class from the user input is not found.

    Custom config classes can also be registered, e.g. as a custom runner:

    .. code-block:: python

        @RunnerConfig.register("custom")
        class CustomRunner(LocalRunnerConfig):
            def process_entries(self, base_config):
                # do something else than the usual LocalRunnerConfig
                pass

    Default values from the global profit.defaults.py file are loaded.
    """

    labels = {}
    defaults = "run"

    def update(self, **entries):
        """Updates the attributes with user inputs. No warning is issued if the attribute set by the user is unknown.

        Parameters:
            entries (dict): User input of the config parameters.
        """
        for name, value in entries.items():
            if hasattr(self, name) or name in map(str.lower, self.labels):
                attr = getattr(self, name, None)
                if isinstance(attr, dict):
                    attr.update(value)
                    setattr(self, name, attr)
                else:
                    setattr(self, name, value)
            else:
                setattr(self, name, value)


@BaseConfig.register("fit")
class FitConfig(AbstractConfig):
    """Configuration for the surrogate and encoder. Currently, the only sub config is for the GaussianProcess classes."""

    labels = {}
    defaults = "fit"

    def __init__(self, **entries):
        self.set_defaults(defaults.fit)
        if len(entries) != 0:
            warnings.warn(
                f"FitConfig should be initialized with empty entries and not with {entries}"
            )

    def update(self, **entries):
        from profit.sur import Surrogate
        from profit.sur.gp.gaussian_process import GaussianProcess
        from profit.sur.linreg import LinearRegression

        if "surrogate" in entries:
            self.surrogate = entries["surrogate"]

        if issubclass(Surrogate.labels[self.surrogate], GaussianProcess):
            self.set_defaults(defaults.fit_gaussian_process)
        elif issubclass(Surrogate.labels[self.surrogate], LinearRegression):
            self.set_defaults(defaults.fit_linear_regression)
        else:
            raise RuntimeError(f"unknown surrogate {self.surrogate}")

        super().update(**entries)

    def process_entries(self, base_config):
        """Set 'load' and 'save' as well as the encoder."""
        for mode_str in ("save", "load"):
            filepath = getattr(self, mode_str)
            if filepath:
                if self.surrogate not in filepath:
                    filepath = filepath.rsplit(".", 1)
                    filepath = (
                        "".join(filepath[:-1]) + f"_{self.surrogate}." + filepath[-1]
                    )
                setattr(
                    self,
                    mode_str,
                    path.abspath(path.join(base_config.base_dir, filepath)),
                )

        if self.load:
            self.save = False

        # Encoders
        from re import match
        import numpy as np

        # array: which columns belong to which variables
        input_columns = np.array(
            sum(
                (
                    [var.name] * var.size[1]
                    for var in base_config.variable_group.input_list
                ),
                [],
            )
        )
        output_columns = np.array(
            sum(
                (
                    [var.name] * var.size[1]
                    for var in base_config.variable_group.output_list
                ),
                [],
            )
        )

        for config in self.encoder:
            # handle shorthand notation, e.g. Name(a,b) -> {class: Name, variables: [a, b]}
            if isinstance(config, str):
                try:
                    name, var_spec = match(r"(\w+)\((.*)\)", config).groups()
                except AttributeError as ex:
                    raise ValueError(
                        f"unable to parse encoder shortcut <{config}>"
                    ) from ex
                var_spec = [
                    v.strip().lower() for v in var_spec.split(",")
                ]  # variable specification
            elif isinstance(config, dict):
                name = config["class"]
                var_spec = [v.strip().lower() for v in config["variables"]]
            else:
                raise ValueError(f"unable to parse encoder <{config}>")

            # ToDo: check if var_spec is valid -> warn otherwise

            # select input columns based on variables or kinds
            if any(s in var_spec for s in ["all", "in", "input", "inputs"]):
                input_vars = base_config.variable_group.input_list
                input_select = np.arange(input_columns.size)
            else:
                input_vars = [
                    var
                    for var in base_config.variable_group.input_list
                    if var.name.lower() in var_spec or var.kind.lower() in var_spec
                ]
                if input_vars:
                    input_select = np.hstack(
                        [
                            np.arange(input_columns.size)[input_columns == var.name]
                            for var in input_vars
                        ]
                    )
                else:
                    input_select = None

            # select output columns based on variable names or kinds
            if any(s in var_spec for s in ["all", "out", "output", "outputs"]):
                output_vars = base_config.variable_group.output_list
                output_select = np.arange(output_columns.size)
            else:
                output_vars = [
                    var
                    for var in base_config.variable_group.output_list
                    if var.name.lower() in var_spec or var.kind.lower() in var_spec
                ]
                if output_vars:
                    output_select = np.hstack(
                        [
                            np.arange(output_columns.size)[output_columns == var.name]
                            for var in output_vars
                        ]
                    )
                else:
                    output_select = None

            # handle special cases
            if name == "Exclude":
                # remove excluded columns from column lists
                input_columns = np.array(
                    [c for c in input_columns if c not in (v.name for v in input_vars)]
                )
                output_columns = np.array(
                    [
                        c
                        for c in output_columns
                        if c not in (v.name for v in output_vars)
                    ]
                )
            elif name in ["PCA", "KarhunenLoeve"]:
                # ToDo: can't handle dimensionality reduction yet
                if config is not self.encoder[-1]:
                    raise NotImplementedError(
                        "reduced dimensions cannot be encoded further"
                    )

            # add processed config to _input_encoders & _output_encoders
            for encoders, select in [
                (self._input_encoders, input_select),
                (self._output_encoders, output_select),
            ]:
                if select is not None:
                    encoders.append(
                        {
                            "class": name,
                            "columns": select,
                            "parameters": (
                                {k: float(v) for k, v in config.get("parameters", {})}
                                if not isinstance(config, str)
                                else {}
                            ),
                        }
                    )


@BaseConfig.register("active_learning")
class ALConfig(AbstractConfig):
    """Active learning configuration."""

    labels = {}
    defaults = "active_learning"

    def process_entries(self, base_config):
        for key in self.labels:
            getattr(self, key.lower()).process_entries(base_config)


@ALConfig.register("algorithm")
class AlgorithmALConfig(AbstractConfig):
    labels = {}
    defaults = None


@AlgorithmALConfig.register("simple")
class SimpleALConfig(AlgorithmALConfig):
    labels = {}
    defaults = "al_algorithm_simple"

    def process_entries(self, base_config):
        if self.save:
            self.save = base_config["fit"]["save"]
        for sub_config_label in self.labels:
            getattr(self, sub_config_label).process_entries(base_config)


@AlgorithmALConfig.register("mcmc")
class McmcConfig(AlgorithmALConfig):
    labels = {}
    defaults = "al_algorithm_mcmc"

    def process_entries(self, base_config):
        self.save = path.abspath(path.join(base_config.base_dir, self.save))
        self.reference_data = path.abspath(
            path.join(base_config.base_dir, self.reference_data)
        )


@SimpleALConfig.register("acquisition_function")
class AcquisitionFunctionConfig(AbstractConfig):
    """Acquisition function configuration."""

    labels = {}
    defaults = None

    def process_entries(self, base_config):
        for k, v in self.items():
            if isinstance(v, str):
                try:
                    setattr(self, k, float(v))
                except ValueError:
                    pass


@AcquisitionFunctionConfig.register("simple_exploration")
class SimpleExplorationConfig(AcquisitionFunctionConfig):
    labels = {}
    defaults = "al_acquisition_function_simple_exploration"


@AcquisitionFunctionConfig.register("exploration_with_distance_penalty")
class ExplorationWithDistancePenaltyConfig(AcquisitionFunctionConfig):
    labels = {}
    defaults = "al_acquisition_function_exploration_with_distance_penalty"


@AcquisitionFunctionConfig.register("weighted_exploration")
class WeightedExplorationConfig(AcquisitionFunctionConfig):
    labels = {}
    defaults = "al_acquisition_function_weighted_exploration"


@AcquisitionFunctionConfig.register("probability_of_improvement")
class ProbabilityOfImprovementConfig(AcquisitionFunctionConfig):
    labels = {}
    defaults = "al_acquisition_function_probability_of_improvement"


@AcquisitionFunctionConfig.register("expected_improvement")
class ExpectedImprovementConfig(AcquisitionFunctionConfig):
    labels = {}
    defaults = "al_acquisition_function_expected_improvement"


@AcquisitionFunctionConfig.register("expected_improvement_2")
class ExpectedImprovement2Config(AcquisitionFunctionConfig):
    labels = {}
    defaults = "al_acquisition_function_expected_improvement_2"


@AcquisitionFunctionConfig.register("alternating_exploration")
class AlternatingExplorationConfig(AcquisitionFunctionConfig):
    labels = {}
    defaults = "al_acquisition_function_alternating_exploration"


@BaseConfig.register("ui")
class UIConfig(AbstractConfig):
    """Configuration for the Graphical User Interface."""

    labels = {}
    defaults = "ui"


@AcquisitionFunctionConfig.register("default")
class DefaultConfig(AbstractConfig):
    """Default config for all run sub configs which just updates the attributes with user entries."""

    labels = {}
    defaults = None

    def __init__(self, **entries):
        name = entries.get("class", self.__class__.__name__)
        warnings.warn(f"Using default config for '{name}'.")
        self.update(**entries)

    def update(self, **entries):
        for name, value in entries.items():
            if hasattr(self, name) or name in map(str.lower, self.labels):
                attr = getattr(self, name, None)
                if isinstance(attr, dict):
                    attr.update(value)
                    setattr(self, name, attr)
                else:
                    setattr(self, name, value)
            else:
                setattr(self, name, value)
