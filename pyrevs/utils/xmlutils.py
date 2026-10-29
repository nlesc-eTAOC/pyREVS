"""A set of utility for XML serializing."""

import ast
import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any
import numpy as np

_logger = logging.getLogger(__name__)


@contextmanager
def oneliner_ndarray() -> Generator[Any, None, None]:
    """Force ndarray print on a single line temporarily."""
    oldoptions = np.get_printoptions()
    np.set_printoptions(threshold=np.iinfo(np.int32).max, linewidth=np.iinfo(np.int32).max, precision=16)
    yield
    np.set_printoptions(**oldoptions)


class XMLUtilsError(Exception):
    """Exception class for the xmlutils."""


def manual_cast_snapshot(elem: ET.Element) -> Any:
    """Manually cast XML snapshot state."""
    if not elem.text:
        return elem.tag, None

    return elem.tag, manual_cast_str(elem.attrib["state_type"], elem.text)


def manual_cast_snapshot_noise(elem: ET.Element) -> Any:
    """Manually cast XML snapshot noise."""
    return elem.tag, manual_cast_str(elem.attrib["noise_type"], elem.attrib["noise"])


def manual_cast(elem: ET.Element) -> Any:
    """Manually cast XML elements reads."""
    if not elem.text:
        error_msg = f"Unable to parse XML element {elem.tag} since it is empty"
        _logger.error(error_msg)
        raise XMLUtilsError(error_msg)

    return elem.tag, manual_cast_str(elem.attrib["type"], elem.text)


# Plain old data type cast in map
POD_cast_dict: dict[str, Callable[..., Any]] = {
    "int": int,
    "float": float,
    "float32": float,
    "float64": np.float64,
    "complex": complex,
    "str": str,
    "str_": str,
    "dict": ast.literal_eval,
    "tuple": ast.literal_eval,
    "bool": lambda elem_text: bool(elem_text == "True"),
    "bool_": lambda elem_text: bool(elem_text == "True"),
}


def manual_cast_str(type_str: str, elem_text: str) -> Any:
    """Manually cast from strings."""
    try:
        casted_elem = POD_cast_dict[type_str](elem_text)
    except KeyError as e:
        if type_str == "ndarray[float]":
            stripped_text = elem_text.replace("[", "").replace("]", "").replace("  ", " ")
            casted_elem = np.fromstring(stripped_text, dtype=float, sep=" ")
        elif type_str == "ndarray[int]":
            stripped_text = elem_text.replace("[", "").replace("]", "").replace("  ", " ")
            casted_elem = np.fromstring(stripped_text, dtype=int, sep=" ")
        elif type_str == "ndarray[bool]":
            stripped_text = elem_text.replace("[ ", " ").replace("]", "").replace("  ", " ")
            npofstr = np.array(list(stripped_text.lstrip().split(" ")), dtype=object)
            casted_elem = npofstr == "True"
        elif type_str == "datetime":
            casted_elem = datetime.strptime(elem_text, "%Y-%m-%d %H:%M:%S.%f%z")
        elif type_str == "None":
            casted_elem = None
        else:
            err_msg = f"Type {type_str} not handled by manual_cast !"
            _logger.exception(err_msg)
            raise XMLUtilsError(err_msg) from e
    return casted_elem


def dict_to_xml(tag: str, d: dict[Any, Any]) -> ET.Element:
    """Return an Element from a dictionary.

    Args:
        tag: a root tag
        d: a dictionary
    """
    elem = ET.Element(tag)
    for key, val in d.items():
        # Append an Element
        child = ET.Element(key)
        child.attrib["type"] = get_val_type(val)
        child.text = str(val)
        elem.append(child)

    return elem


def xml_to_dict(elem: ET.Element | None) -> dict[Any, Any]:
    """Return an dictionary an Element.

    Args:
        elem: an etree element

    Return:
        a dictionary containing the element entries
    """
    if elem is None:
        error_msg = "Unable to parse XML element to dict since 'None' was passed"
        _logger.error(error_msg)
        raise XMLUtilsError(error_msg)

    d = {}
    for child in elem:
        tag, entry = manual_cast(child)
        d[tag] = entry

    return d


def get_val_type(val: Any) -> str:
    """Return the type of val.

    Args:
        val: a value

    Return:
        val type
    """
    base_type = type(val).__name__
    if base_type == "ndarray":
        if val.dtype == "float64":
            base_type = base_type + "[float]"
        elif val.dtype == "int64":
            base_type = base_type + "[int]"
        elif val.dtype == "bool":
            base_type = base_type + "[bool]"
        return base_type

    return base_type


def new_element(key: str, val: Any) -> ET.Element:
    """Return an Element from two args.

    Args:
        key: the element key
        val: the element value

    Return:
        an ElementTree element
    """
    elem = ET.Element(key)
    elem.attrib["type"] = get_val_type(val)
    with oneliner_ndarray():
        elem.text = str(val)

    return elem


def make_xml_snapshot(idx: int, time: float, score: float, noise: Any, state: Any) -> ET.Element:
    """Return a snapshot in XML element format.

    Args:
        idx: snapshot index
        time: the time stamp
        score: the snapshot score function
        noise: the stochastic noise
        state: the associated state
    """
    elem = ET.Element(f"Snap_{idx:07d}")
    elem.attrib["time"] = str(time)
    elem.attrib["score"] = str(score)
    elem.attrib["noise_type"] = get_val_type(noise)
    with oneliner_ndarray():
        elem.attrib["noise"] = str(noise)
    if state is None:
        elem.attrib["state_type"] = "None"
    else:
        elem.attrib["state_type"] = get_val_type(state)
        with oneliner_ndarray():
            elem.text = str(state)

    return elem


def read_xml_snapshot(snap: ET.Element) -> tuple[float, float, Any, Any]:
    """Return snapshot data from an XML snapshot element.

    Args:
        snap: an XML snapshot element
    """
    time = float(snap.attrib["time"])
    score = float(snap.attrib["score"])
    _, noise = manual_cast_snapshot_noise(snap)
    _, state = manual_cast_snapshot(snap)

    return time, score, noise, state
