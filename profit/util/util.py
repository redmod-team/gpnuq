"""Utility functions.

This file contains miscellaneous useful functions.
"""

from os import path
from typing import Union
from collections.abc import MutableMapping, Mapping
import numpy as np


def safe_path(arg, default, valid_extensions=(".yaml", ".py")):
    if path.isfile(arg):
        if arg.endswith(valid_extensions):
            return path.abspath(arg)
        else:
            raise TypeError(
                "Unsupported file extension. \n"
                "Valid file extensions: {}".format(arg, valid_extensions)
            )
    elif path.isdir(arg):
        return path.join(arg, default)
    else:
        raise FileNotFoundError("Directory or file ({}) not found.".format(arg))


def quasirand(ndim=1, npoint=1):
    from .halton import halton

    return halton(npoint, ndim)


def check_ndim(arr):
    return arr if arr.ndim > 1 else arr.reshape(-1, 1)


class SafeDict(dict):
    def __init__(self, obj, pre="{", post="}"):
        self.pre = pre
        self.post = post
        super().__init__(obj)

    @classmethod
    def from_params(cls, params, **kwargs):
        return cls(params2map(params), **kwargs)

    def __missing__(self, key):
        return self.pre + key + self.post


def params2map(params: Union[None, MutableMapping, np.ndarray, np.void]):
    if params is None:
        return {}
    if isinstance(params, MutableMapping):
        return params
    try:
        return {key: params[key] for key in params.dtype.names}
    except AttributeError:
        pass
    raise TypeError("params are not a Mapping")


def load_includes(paths):
    """load python modules from the specified paths"""
    import os
    import sys
    from importlib.util import spec_from_file_location, module_from_spec
    import logging

    for path in paths:
        name = f"profit_include_{os.path.basename(path).split('.')[0]}"
        if name in sys.modules:  # do not reload modules
            continue
        try:
            spec = spec_from_file_location(name, path)
        except FileNotFoundError:
            logging.getLogger(__name__).error(f"could not find {path} to include")
            continue
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


def flatten_struct(struct_array: np.ndarray):
    # per default vector entries are spread across several columns
    if not struct_array.size:
        return np.array([[]])
    return np.vstack(
        [
            np.hstack([row[key].flatten() for key in struct_array.dtype.names])
            for row in struct_array
        ]
    )
