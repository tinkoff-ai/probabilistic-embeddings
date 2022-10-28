r"""Tools for configuration using default config.

All configurable classes must have :meth:`get_default_config` static method
which returns dictionary of default values. Than you can use
:func:`prepare_config` function to construct actual config. Actual config
can be ``None``, ``dict`` or ``str`` containing path to the file.

**Example**::

    from collections import OrderedDict
    from mdn_metric.config import prepare_config

    class Configurable:
        @staticmethod
        def get_default_config():
            return OrderedDict([
                ("arg1", 10),
                ("arg2", None)
            ])

        def __init__(self, *args, config=None):
            config = prepare_config(self, config)
            self.arg1 = config["arg1"]
            self.arg2 = config["arg2"]

    obj = Configurable(config={"arg1": 5})
    print(obj.arg1)  # 5
    print(obj.arg2)  # None

Config files use YAML syntax. The special key `_type` can be used in configs to specify
target class. If types are provided, they are checked during initialization.

**Example**::

    system:
        subsystem:
            _type: SubsystemClass
            arg1: [5.0, 2.0]

Config can contain hyperparameters for optimization in WandB format like:

**Example**::

    system:
        subsystem:
            arg1: [5.0, 2.0]
            _hopt:
              arg2:
                min: 1
                max: 5

If _hopt dictionary contains some values instead of dictionaries,
these values will used in config as parameters when needed.

"""

from collections import OrderedDict, defaultdict
from .io import read_yaml, write_yaml


CONFIG_TYPE = "_type"
CONFIG_HOPT = "_hopt"


class ConfigError(Exception):
    """Exception class for errors in config."""
    pass


def read_config(filename):
    if filename is None:
        return {}
    return read_yaml(filename)


def write_config(config, filename):
    write_yaml(config, filename)


def get_config(config):
    """Load config from file if string is provided. Return empty dictionary if input is None."""
    if config is None:
        return {}
    if isinstance(config, str):
        config = read_config(config)
    if not isinstance(config, (dict, OrderedDict)):
        raise ConfigError("Config dictionary expected, got {}".format(type(config)))
    return config.copy()


def prepare_config(cls_or_default, config=None):
    """Set defaults and check fields.

    Config is a dictionary of values. Method creates new config using
    default class config. Result config keys are the same as default config keys.

    Args:
        cls_or_default: Class with get_default_config method or default config dictionary.
        config: User-provided config.

    Returns:
        Config dictionary with defaults set.
    """
    if isinstance(cls_or_default, dict):
        default_config = cls_or_default
        cls_name = None
    else:
        default_config = cls_or_default.get_default_config()
        cls_name = type(cls_or_default).__name__

    config = get_config(config)

    # Extract optional values from _hopt.
    hopts = config.pop(CONFIG_HOPT, {})
    optional_values = {k: v for k, v in hopts.items() if not isinstance(v, (dict, OrderedDict))}

    # Check type.
    if CONFIG_TYPE in config:
        if (cls_name is not None) and (cls_name != config[CONFIG_TYPE]):
            raise ConfigError("Type mismatch: expected {}, got {}".format(
                config[CONFIG_TYPE], cls_name))
        del config[CONFIG_TYPE]

    # Merge configs.
    for key in config:
        if key not in default_config:
            raise ConfigError("Unknown parameter {}".format(key))
    new_config = OrderedDict()
    for key, value in default_config.items():
        new_config[key] = config.get(key, value)
    for key, value in optional_values.items():
        if key not in default_config:
            continue
        new_config[key] = value
    return new_config


def as_flat_config(config, separator="."):
    """Convert nested config to flat config."""
    if isinstance(config, str):
        config = read_config(config)
    if isinstance(config, (tuple, list)):
        config = OrderedDict([(str(i), v) for i, v in enumerate(config)])
    if not isinstance(config, (dict, OrderedDict)):
        raise TypeError("Expected dictionary, got {}.".format(type(config)))
    hopt = OrderedDict()
    flat = OrderedDict()
    for k, v in config.items():
        if k == CONFIG_HOPT:
            for hk, hv in v.items():
                hopt[hk] = hv
        elif isinstance(v, (dict, OrderedDict, tuple, list)):
            for sk, sv in as_flat_config(v).items():
                if sk == CONFIG_HOPT:
                    for hk, hv in sv.items():
                        hopt[k + separator + hk] = hv
                else:
                    flat[k + separator + sk] = sv
        else:
            flat[k] = v
    if hopt:
        flat[CONFIG_HOPT] = hopt
    return flat


def _is_index(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def as_nested_config(flat_config, separator="."):
    """Convert flat config to nested config."""
    flat_config = get_config(flat_config)
    by_prefix = defaultdict(OrderedDict)
    nested = OrderedDict()
    for k, v in flat_config.items():
        if k == CONFIG_HOPT:
            for hopt_k, hopt_v in v.items():
                if separator in hopt_k:
                    prefix, sk = hopt_k.split(separator, 1)
                    if CONFIG_HOPT not in by_prefix[prefix]:
                        by_prefix[prefix][CONFIG_HOPT] = {}
                    by_prefix[prefix][CONFIG_HOPT][sk] = hopt_v
                else:
                    if CONFIG_HOPT not in nested:
                        nested[CONFIG_HOPT] = {}
                    nested[CONFIG_HOPT][hopt_k] = hopt_v
        elif separator in k:
            prefix, sk = k.split(separator, 1)
            by_prefix[prefix][sk] = v
        else:
            nested[k] = v
    for k, v in by_prefix.items():
        nested[k] = as_nested_config(v)
    # Some hopts can be precise values. Remove unnecessary hopts if they are equal to config values.
    if CONFIG_HOPT in nested:
        for k, v in nested.items():
            if isinstance(v, (dict, OrderedDict, tuple, list)):
                continue
            if (k in nested[CONFIG_HOPT]) and (nested[CONFIG_HOPT][k] == v):
                del nested[CONFIG_HOPT][k]
    # Convert to list if necessary.
    is_index = [_is_index(k) for k in nested if k != CONFIG_HOPT]
    if nested and any(is_index):
        if nested.pop(CONFIG_HOPT, None):
            raise NotImplementedError("Can't use hopts for list values.")
        if not all(is_index):
            raise ConfigError("Can't mix dict and list configs: some keys are indices and some are strings.")
        length = max(map(int, nested.keys())) + 1
        nested_list = [None] * length
        for k, v in nested.items():
            nested_list[int(k)] = v
        return nested_list
    else:
        return nested


def update_config(config, patch):
    """Merge patch into config recursively."""
    if patch is None:
        return config
    config = get_config(config)
    flat = as_flat_config(config)
    flat_patch = as_flat_config(patch)
    hopt = flat.pop(CONFIG_HOPT, {})
    hopt.update(flat_patch.pop(CONFIG_HOPT, {}))
    flat.update(flat_patch)
    if hopt:
        flat[CONFIG_HOPT] = hopt
    return as_nested_config(flat)


def has_hopts(config):
    return CONFIG_HOPT in as_flat_config(config)


def remove_hopts(config):
    flat = as_flat_config(config)
    flat.pop(CONFIG_HOPT, None)
    return as_nested_config(flat)
