from os import path
import yaml
from collections import OrderedDict
from profit import defaults
from profit.util.base_class import CustomABC
import warnings

VALID_FORMATS = ('.yaml', '.py')

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

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)


def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))


_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
yaml.add_representer(OrderedDict, represent_ordereddict)
yaml.add_constructor(_mapping_tag, dict_constructor)

""" now yaml is configured to handle OrderedDict input and output """


def load_config_from_py(filename):
    """ Load the configuration parameters from a python file into dict. """
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('f', filename)
    f = module_from_spec(spec)
    spec.loader.exec_module(f)
    return {name: value for name, value in f.__dict__.items() if not name.startswith('_')}


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
                value = {'class': value}
            setattr(self, name, value)

    def create_subconfig(self, sub_config_label, **entries):
        """Instances of sub configs are created from a string or a dictionary.

        Parameters:
            sub_config_label (str): Dict key of registered sub config.
            entries (dict): User input parameters.
        """
        if 'class' in entries:
            # Load specific sub config or default config, if missing.
            try:
                sub = self.labels[sub_config_label][entries['class']]()
            except KeyError:
                sub = self.labels[sub_config_label]['default'](**entries)
        else:
            # Load general sub config.
            sub = self.labels[sub_config_label]()

        # Split entries into entries for this config and further sub configs.
        base_entries = {k: v for k, v in entries.items() if k.lower() not in sub.labels}
        sub_entries = {k: {'class': v} if isinstance(v, str) else v
                       for k, v in entries.items() if k.lower() in sub.labels}

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
        base_entries = {k: v for k, v in entries.items() if k.lower() not in self.labels}
        sub_entries = {k: {'class': v} if isinstance(v, str) else v
                       for k, v in entries.items() if k.lower() in self.labels}

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
        self.files['input'] = path.join(self.base_dir, self.files.get('input', defaults.files['input']))
        self.files['output'] = path.join(self.base_dir, self.files.get('output', defaults.files['output']))

        # Variable configuration as dict
        variables = VariableGroup(self.ntrain)
        vars = []
        for k, v in self.variables.items():
            if isinstance(v, (int, float)):
                v = f'Constant({v})'
            if isinstance(v, str):
                vars.append(Variable.create_from_str(k, (self.ntrain, 1), v))
            else:
                vars.append(Variable.create(name=k, size=(self.ntrain,1), **v))
        variables.add(vars)

        self.variables = variables.as_dict
        self.input = {k: v for k, v in self.variables.items()
                      if not any(k in v['kind'].lower() for k in ('output', 'independent'))}
        self.output = {k: v for k, v in self.variables.items()
                       if 'output' in v['kind'].lower()}
        self.independent = {k: v for k, v in self.variables.items()
                            if 'independent' in v['kind'].lower() and v['size'] != (1, 1)}

        for sub_config_label in self.labels:
            getattr(self, sub_config_label).process_entries(self)

    @classmethod
    def from_file(cls, filename=defaults.config_file):
        """Creates a configuration class from a .yaml or .py file."""

        if filename.endswith('.yaml'):
            with open(filename) as f:
                entries = yaml.safe_load(f)
        elif filename.endswith('.py'):
            entries = load_config_from_py(filename)
        else:
            raise TypeError(f"Not supported file extension .{filename.split('.')[-1]} for config file.\n"
                            f"Valid file formats: {VALID_FORMATS}")
        self = cls(base_dir=path.split(filename)[0], **entries)
        self.config_path = path.join(self.base_dir, filename)
        return self

    def load_includes(self):
        from profit.util import load_includes

        if isinstance(self.include, str):
            self.include = [self.include]

        self.include = [path.abspath(path.join(self.base_dir, p)) for p in self.include]
        load_includes(self.include)


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

    def process_entries(self, base_config):
        """Set paths and process entries of sub configs."""

        if not path.isabs(self.log_path):
            self.log_path = path.abspath(path.join(base_config.base_dir, self.log_path))

        for key in self.labels:
            getattr(self, key.lower()).process_entries(base_config)


@RunConfig.register("runner")
class RunnerConfig(AbstractConfig):
    """Base Runner config."""
    labels = {}
    defaults = None


@RunnerConfig.register("local")
class LocalRunnerConfig(RunnerConfig):
    """
    Example:
        .. code-block:: yaml

            class: local
            parallel: all   # maximum number of simultaneous runs (for spawn array)
            sleep: 0        # number of seconds to sleep while polling
            fork: true      # whether to spawn the worker via forking instead of a subprocess (via a shell)
    """
    labels = {}
    defaults = "run_runner_local"

    def process_entries(self, base_config):
        """Converts `parallel: all` to number of available cpus"""
        from os import sched_getaffinity
        if self.parallel == 'all':
            self.parallel = len(sched_getaffinity(0))


@RunnerConfig.register("slurm")
class SlurmRunnerConfig(RunnerConfig):
    """
    Example:
        .. code-block:: yaml

               class: slurm
               parallel: null      # maximum number of simultaneous runs (for spawn array)
               sleep: 0            # number of seconds to sleep while (internally) polling
               poll: 60            # number of seconds between external polls (to catch failed runs), use with care!
               path: slurm.bash    # the path to the generated batch script (relative to the base directory)
               custom: false       # whether a custom batch script is already provided at 'path'
               prefix: srun        # prefix for the command
               OpenMP: false       # whether to set OMP_NUM_THREADS and OMP_PLACES
               cpus: 1             # number of cpus (including hardware threads) to use (may specify 'all')
               options:            # (long) options to be passed to slurm: e.g. time, mem-per-cpu, account, constraint
                   job-name: profit
    """
    labels = {}
    defaults = "run_runner_slurm"

    def process_entries(self, base_config):
        """Converts paths to absolute and check type of 'cpus'"""
        # Convert path to absolute path
        if not path.isabs(self.path):
            self.path = path.abspath(path.join(base_config.base_dir, self.path))
        # Check type of 'cpus'
        if (type(self.cpus) is not int or self.cpus < 1) and self.cpus != 'all':
            raise ValueError(f'config option "cpus" may only be a positive integer or "all" and not {self.cpus}')


@RunConfig.register("interface")
class InterfaceConfig(AbstractConfig):
    """Base runner interface config."""
    labels = {}
    defaults = None


@InterfaceConfig.register("memmap")
class MemmapInterfaceConfig(InterfaceConfig):
    """
    Example:
        .. code-block:: yaml

            class: memmap
            path: interface.npy     # path to memory mapped interface file, relative to base directory
    """
    labels = {}
    defaults = "run_interface_memmap"

    def process_entries(self, base_config):
        """Converts 'path' to absolute."""
        if not path.isabs(self.path):
            self.path = path.abspath(path.join(base_config.base_dir, self.path))


@InterfaceConfig.register("zeromq")
class ZeroMQInterfaceConfig(InterfaceConfig):
    """
    Example:
        .. code-block:: yaml

            class: zeromq
            transport: tcp      # transport system used by zeromq
            port: 9000          # port for the interface
            address: null       # override bind address used by zeromq
            connect: null       # override connect address used by zeromq
            timeout: 2500       # zeromq polling timeout, in ms
            retries: 3          # number of zeromq connection retries
            retry-sleep: 1      # sleep between retries, in s
    """
    labels = {}
    defaults = "run_interface_zeromq"


@RunConfig.register("pre")
class PreConfig(AbstractConfig):
    """Base config for preprocessors."""
    labels = {}
    defaults = None


@PreConfig.register("template")
class TemplatePreConfig(PreConfig):
    """
    Example:
        .. code-block:: yaml

            class: template
            path: template      # directory to copy from, relative to base directory
            param_files: null   # files in template which contain placeholders for variables, null means all files
                                # can be a filename or a list of filenames
    """
    labels = {}
    defaults = "run_pre_template"

    def process_entries(self, base_config):
        """Convert 'path' to absolute and set 'param_files'."""
        if not path.isabs(self.path):
            self.path = path.abspath(path.join(base_config.base_dir, self.path))

        if isinstance(self.param_files, str):
            self.param_files = [self.param_files]


@RunConfig.register("post")
class PostConfig(AbstractConfig):
    """Base class for postprocessor configs."""
    labels = {}
    defaults = None


@PostConfig.register("json")
class JsonPostConfig(PostConfig):
    """
    Example:
        .. code-block:: yaml

            class: json
            path: stdout    # file to read from, relative to the run directory
    """
    labels = {}
    defaults = "run_post_json"


@PostConfig.register("numpytxt")
class NumpytxtPostConfig(PostConfig):
    """
    Example:
        .. code-block:: yaml

            class: numpytxt
            path: stdout    # file to read from, relative to the run directory
            names: "f g"    # list or string of output variables in order, default read from config/variables
            options:        # options which are passed on to numpy.genfromtxt() (fname & dtype are used internally)
                deletechars: ""
    """
    labels = {}
    defaults = "run_post_numpytxt"

    def process_entries(self, base_config):
        """Sets the included names of variables. The Keyword 'all' includes all variables."""
        if isinstance(self.names, str):
            self.names = list(base_config.output.keys()) if self.names == 'all' else self.names.split()


@PostConfig.register("hdf5")
class HDF5PostConfig(PostConfig):
    """
    Example:
        .. code-block:: yaml

            class: hdf5
            path: output.hdf5   # file to read from, relative to the run directory
    """
    labels = {}
    defaults = "run_post_hdf5"


@BaseConfig.register("fit")
class FitConfig(AbstractConfig):
    """Configuration for the surrogate and encoder. Currently, the only sub config is for the GaussianProcess classes."""
    labels = {}
    defaults = "fit"

    def __init__(self, **entries):
        from profit.sur import Surrogate
        from profit.sur.gp.gaussian_process import GaussianProcess
        self.set_defaults(defaults.fit)

        if issubclass(Surrogate.labels[self.surrogate], GaussianProcess):
            self.set_defaults(defaults.fit_gaussian_process)

        self.update(**entries)

    def process_entries(self, base_config):
        """Set 'load' and 'save' as well as the encoder."""
        for mode_str in ('save', 'load'):
            filepath = getattr(self, mode_str)
            if filepath:
                if self.surrogate not in filepath:
                    filepath = filepath.rsplit('.', 1)
                    filepath = ''.join(filepath[:-1]) + f'_{self.surrogate}.' + filepath[-1]
                setattr(self, mode_str, path.abspath(path.join(base_config.base_dir, filepath)))

        if self.load:
            self.save = False

        for in_out in ('input_encoders', 'output_encoders'):
            variables = base_config[in_out.split('_')[0]]
            ncols = sum([v['size'][-1] for v in variables.values()])
            all_cols = list(range(ncols))
            for enc in getattr(self, in_out):

                # Set columns
                cols = enc['columns']

                if isinstance(cols, str):
                    col_str = cols.lower()
                    variables_distr = [v.get('distr', v['kind']).lower() for v in variables.values()]

                    if col_str == 'all':
                        enc['columns'] = all_cols
                    elif col_str in variables_distr:
                        enc['columns'] = [idx for idx, d in enumerate(variables_distr) if d == col_str]
                    else:
                        enc['columns'] = []

                # Set parameters
                if 'parameters' not in enc:
                    enc['parameters'] = {}
                for k, v in enc['parameters'].items():
                    try:
                        enc['parameters'][k] = float(v)
                    except ValueError:
                        pass

        # Delete excluded columns from other encoders
        for in_out_encoders in (self.input_encoders, self.output_encoders):
            for n_enc, enc in enumerate(in_out_encoders):
                if enc['class'].lower() == 'exclude':
                    cols1 = enc['columns']
                    for enc2 in in_out_encoders[n_enc+1:]:
                        cols2 = enc2['columns']
                        removed_cols = 0
                        for idx1, col1 in enumerate(cols1):
                            col_rm = col1 - removed_cols
                            if col_rm in cols2:
                                # Remove excluded column from other encoder and reindex subsequent columns
                                idx2 = cols2.index(col_rm)
                                cols2.pop(idx2)
                                cols2[idx2:] = [c - 1 for c in cols2[idx2:]]
                                removed_cols += 1
                            else:
                                # Reindex subsequent columns
                                idx2 = [c > col1 for c in cols2]
                                cols2[:] = [c-1 if cond else c for c, cond in zip(cols2, idx2)]


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
            self.save = base_config['fit']['save']
        for sub_config_label in self.labels:
            getattr(self, sub_config_label).process_entries(base_config)


@AlgorithmALConfig.register("mcmc")
class McmcConfig(AlgorithmALConfig):
    labels = {}
    defaults = "al_algorithm_mcmc"

    def process_entries(self, base_config):
        self.save = path.abspath(path.join(base_config.base_dir, self.save))
        self.reference_data = path.abspath(path.join(base_config.base_dir, self.reference_data))


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


@RunnerConfig.register("default")
@InterfaceConfig.register("default")
@PreConfig.register("default")
@PostConfig.register("default")
@AcquisitionFunctionConfig.register("default")
class DefaultConfig(AbstractConfig):
    """Default config for all run sub configs which just updates the attributes with user entries."""
    labels = {}
    defaults = None

    def __init__(self, **entries):
        name = entries.get('class', self.__class__.__name__)
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
