from collections import OrderedDict, defaultdict
from pathlib import PurePath

import numpy as np
import yaml


class Dumper(yaml.CSafeDumper):
    pass


def represent_list(self, data):
    is_flat = True
    for v in data:
        if not isinstance(v, (int, float, str, np.integer, np.floating)):
            is_flat = False
            break
    return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=is_flat)


Dumper.add_representer(list, represent_list)
Dumper.add_representer(defaultdict, Dumper.represent_dict)
Dumper.add_representer(OrderedDict, Dumper.represent_dict)
Dumper.add_representer(np.int32, Dumper.represent_int)
Dumper.add_representer(np.int64, Dumper.represent_int)
Dumper.add_representer(np.float32, Dumper.represent_float)
Dumper.add_representer(np.float64, Dumper.represent_float)


def read_yaml(src):
    """Read yaml from file or stream."""
    if isinstance(src, (str, PurePath)):
        with open(str(src)) as fp:
            return yaml.load(fp, Loader=yaml.CLoader)
    else:
        return yaml.load(src, Loader=yaml.CLoader)


def write_yaml(obj, dst):
    """Dump yaml to file or stream."""
    if isinstance(dst, (str, PurePath)):
        with open(str(dst), "w") as fp:
            yaml.dump(obj, fp, Dumper=Dumper, sort_keys=False)
    else:
        yaml.dump(obj, dst, Dumper=Dumper, sort_keys=False)
