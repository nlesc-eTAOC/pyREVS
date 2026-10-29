"""A set of utility functions for pyREVS."""

import ast
import importlib.util
import inspect
import logging
import re
import sys
import textwrap
from abc import ABCMeta
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt

_logger = logging.getLogger(__name__)


def is_windows_os() -> bool:
    """Indicates Windows platform."""
    system = sys.platform.lower()
    return system.startswith("win")


def is_mac_os() -> bool:
    """Indicates MacOS platform."""
    system = sys.platform.lower()
    return system.startswith("dar")


def setup_logger(loglevel: str, logfile: str | None = None) -> None:
    """Setup the logger parameters.

    Args:
        loglevel: logging level
        logfile: optional logging file
    """
    # Set logging level
    log_level_str = loglevel.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Set formatter
    log_format = "[%(levelname)s] %(asctime)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Set root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove all existing handlers to prevent duplication
    # Definitely a brute-force appraoch
    root_logger.handlers.clear()

    # Query log file
    log_file = logfile

    # Set console handler: warning+ if logfile provided
    # full log otherwise
    target_console_level = logging.WARNING if log_file else log_level
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(target_console_level)
    root_logger.addHandler(console_handler)

    # Add file handler to root logger
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger("").addHandler(file_handler)


def get_min_scored(maxes: npt.NDArray[Any], nworkers: int) -> tuple[list[int], npt.NDArray[Any]]:
    """Get the nworker lower scored trajectories or more if equal score.

    Args:
        maxes: array of maximas across all trajectories
        nworkers: number of workers

    Returns:
        list of indices of the nworker lower scored trajectories
        array of minimas
    """
    ordered_tlist = np.argsort(maxes)
    is_same_min = False
    min_idx_list: list[int] = []
    for idx in ordered_tlist:
        if len(min_idx_list) > 0:
            is_same_min = maxes[idx] == maxes[min_idx_list[-1]]
        if len(min_idx_list) < nworkers or is_same_min:
            min_idx_list.append(int(idx))

    min_vals = maxes[min_idx_list]
    return min_idx_list, min_vals


def moving_avg(arr_in: npt.NDArray[Any], window_l: int) -> npt.NDArray[Any]:
    """Return the moving average of a 1D numpy array.

    Args:
        arr_in: 1D numpy array
        window_l: length of the moving average window

    Returns:
        1D numpy array
    """
    arr_out = np.zeros(arr_in.shape[0])
    for i in range(len(arr_in)):
        lbnd = max(i - int(np.ceil(window_l / 2)), 0)
        hbnd = min(i + int(np.floor(window_l / 2)), len(arr_in) - 1)
        if lbnd == 0:
            hbnd = window_l
        if hbnd == len(arr_in) - 1:
            lbnd = len(arr_in) - window_l - 1
        arr_out[i] = np.mean(arr_in[lbnd:hbnd])
    return arr_out


def get_module_local_import(module_name: str) -> list[str]:
    """Helper function getting local imported mods list.

    When pickling the forward model code, the model itself can import from
    several other local files. We also want to pickle those by value so let's get
    the list.

    Args:
        module_name: a module name we want the locally imported modules

    Returns:
        A list of local modules names imported within the provide module
    """
    # Check that module exists
    if module_name not in sys.modules:
        err_msg = f"Attempting to extract sub import from {module_name} missing from currently loaded modules"
        _logger.error(err_msg)
        raise ValueError(err_msg)

    # Check access to the module file
    if hasattr(sys.modules[module_name], "__file__") and Path(str(sys.modules[module_name].__file__)).exists():
        mfile = Path(str(sys.modules[module_name].__file__))
    else:
        err_msg = f"Attempting to locate sub import file from {module_name}, but file is missing or undefined"
        _logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    # Parse the module file
    # for imports
    with mfile.open("r") as f:
        file_raw = f.read()

    file_ast = ast.parse(file_raw)
    all_modules = []

    for node in ast.walk(file_ast):
        # Append "import X" type
        if isinstance(node, ast.Import):
            all_modules.extend([x.name for x in node.names])
        # Append "from X import Y" type
        if isinstance(node, ast.ImportFrom) and node.module:
            all_modules.append(node.module)

    # Return only those whose file is in the current folder
    # or from the 'examples' folder of pyREVS
    return [
        m
        for m in all_modules
        if (
            hasattr(sys.modules[m], "__file__")
            and (
                Path(str(sys.modules[m].__file__)).parent == Path().absolute()
                or any((p.name == "examples") for p in Path(str(sys.modules[m].__file__)).parents)
            )
        )
    ]


def import_forward_model(file: str, abc_cls: ABCMeta) -> type[ABCMeta]:
    """Import forward model class from file and return it.

    Args:
        file: python module file to look into
        abc_cls: parent class of which we are looking for a subclass

    Return:
        A subclass of abc_cls

    Raises:
        RuntimeError: if not subclass found or multiple ones
    """
    module = Path(file).stem
    spec = importlib.util.spec_from_file_location(module, file)
    if spec is None:
        err_msg = f"Could not get spec from Python's module in {file}"
        _logger.error(err_msg)
        raise RuntimeError(err_msg)
    if spec.loader is None:
        err_msg = f"{spec.name} module is missing loader !"
        _logger.error(err_msg)
        raise RuntimeError(err_msg)

    mod = importlib.util.module_from_spec(spec)

    # Need to add the module path so that pyREVS's worker
    # can find the class.
    sys.path.append(Path(file).parent.as_posix())

    # Append module to system modules
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    fmodel_class = None
    for obj in vars(mod).values():
        if inspect.isclass(obj) and issubclass(obj, abc_cls) and obj is not abc_cls:
            if fmodel_class is not None and fmodel_class is not obj:
                err_msg = (
                    f"pyREVS can only define one {abc_cls.__name__} subclass: "
                    f"both {fmodel_class.__name__} and {obj.__name__} found in {file}!"
                )
                _logger.error(err_msg)
                raise RuntimeError(err_msg)
            fmodel_class = obj

    if fmodel_class is None:
        err_msg = f"pyREVS could not locate subclass of {abc_cls.__name__} in {module}"
        _logger.error(err_msg)
        raise RuntimeError(err_msg)

    return fmodel_class


def clean_signature(sig: inspect.Signature) -> str:
    """Converts a signature to a string and removes TypeVar tildes.

    Args:
        sig: the method signature

    Returns:
        str
    """
    sig_str = str(sig)
    # This regex looks for a tilde followed by a word (the TypeVar name)
    # and replaces it with just the name.
    return re.sub(r"~(\w+)", r"\1", sig_str)


def generate_subclass(abc_cls: ABCMeta, class_name: str, file_path: str, include_optional: bool = False) -> None:
    """Generate a subclass skeleton.

    Implementing all abstract methods from `abc_cls`, written to `file_path`.
    The function is overall not tied to any particular ABC except in handling
    types, where types specific to pyREVS are imported.

    Args:
        abc_cls: an ABC
        class_name: the new subclass name
        file_path: where to write the subclass
        include_optional: whether to include optional (non final) functions
    """
    # Identify abstract methods
    abstract_methods = {
        name: value for name, value in abc_cls.__dict__.items() if getattr(value, "__isabstractmethod__", False)
    }

    # Add optional methods
    if include_optional:
        nonfinal_methods = {
            name: value
            for name, value in abc_cls.__dict__.items()
            if (
                inspect.isfunction(value)
                and not getattr(value, "__final__", False)
                and not getattr(value, "__isabstractmethod__", False)
            )
        }

    # Build import line
    module_name = abc_cls.__module__
    abc_name = abc_cls.__name__
    import_lines = [
        f"from {module_name} import {abc_name}\n",
        "import typing\n",
        "from typing import Any\n",
        "from typing import TypeVar\n",
    ]
    if include_optional:
        import_lines.append("from pyrevs.snapshot import Snapshot\n")
    import_lines.append("\n\n")

    # Append lines for type hints
    for typevar in abc_cls.__dict__["__parameters__"]:
        stripped_typevar = typevar.__name__
        import_lines.append(f'{stripped_typevar} = TypeVar("{stripped_typevar}")\n')
    import_lines.append("\n\n")

    # Build class header
    lines = [*import_lines, f"class {class_name}({abc_name}):\n"]
    lines.append('    """TODO: add class docstring."""\n')

    if not abstract_methods:
        lines.append("    pass\n")
    else:
        # Generate each required method with preserved signature
        for name, func in abstract_methods.items():
            sig = inspect.signature(func)
            clean_sig = clean_signature(sig)
            doc = inspect.getdoc(func)

            lines.append(f"    def {name}{clean_sig}:\n")
            if doc:
                # Indent docstring correctly
                doc_clean = textwrap.indent('"""' + doc + '\n"""', " " * 8)
                lines.append(f"{doc_clean}\n")
            else:
                lines.append('        """TODO: implement method."""\n')

            lines.append("        # Implement concrete method body\n\n")

        if include_optional:
            for func in nonfinal_methods.values():
                src = inspect.getsource(func)
                lines.append(f"{src}\n\n")

    # Add the name class method
    lines.append("    @classmethod\n")
    lines.append("    def name(cls) -> str:\n")
    lines.append('        """Return a the model name."""\n')
    lines.append(f'        return "{class_name}"\n')

    # Write to file
    Path(file_path).write_text("".join(lines))
