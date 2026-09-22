"""Safe SVG template instantiation for the IR compiler.

Supported contract:
* `data-slot="name"` on text nodes: replace displayed text from `slots[name]`.
* `data-anchor="name"` on groups or invisible shapes: expose a connector point.
* every copied template has all internal SVG ids prefixed by `instance_id`.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from xml.etree import ElementTree as ET

SVG = "http://www.w3.org/2000/svg"
NS = f"{{{SVG}}}"
ET.register_namespace("", SVG)


def _prefix_ids(root: ET.Element, instance_id: str) -> None:
    ids = {node.attrib["id"]: f"{instance_id}__{node.attrib['id']}" for node in root.iter() if "id" in node.attrib}
    for node in root.iter():
        if node.attrib.get("id") in ids:
            node.attrib["id"] = ids[node.attrib["id"]]
        for key, value in list(node.attrib.items()):
            for old, new in ids.items():
                value = value.replace(f"url(#{old})", f"url(#{new})").replace(f"#{old}", f"#{new}")
            node.attrib[key] = value


def instantiate(template: Path, instance_id: str, slots: dict[str, str], x: float, y: float, width: float, height: float) -> tuple[str, dict[str, tuple[float, float]]]:
    """Copy one SVG template, fill slots, prefix ids and return an embedded SVG."""
    root = ET.parse(template).getroot()
    if root.tag != NS + "svg":
        raise ValueError(f"template is not SVG: {template}")
    _prefix_ids(root, instance_id)
    anchors: dict[str, tuple[float, float]] = {}
    for node in root.iter():
        slot = node.attrib.get("data-slot")
        if slot:
            if slot not in slots:
                raise ValueError(f"template {template.name} requires missing slot {slot!r}")
            node.text = str(slots[slot])
        anchor = node.attrib.get("data-anchor")
        if anchor:
            if "cx" in node.attrib and "cy" in node.attrib:
                anchors[anchor] = (float(node.attrib["cx"]), float(node.attrib["cy"]))
            elif "x" in node.attrib and "y" in node.attrib:
                anchors[anchor] = (float(node.attrib["x"]), float(node.attrib["y"]))
            else:
                raise ValueError(f"anchor {anchor!r} in {template.name} needs x/y or cx/cy")
    view_box = root.attrib.get("viewBox")
    if not view_box:
        raise ValueError(f"template {template.name} must declare viewBox")
    fragment = "".join(ET.tostring(copy.deepcopy(child), encoding="unicode") for child in root)
    embedded = f'<svg x="{x}" y="{y}" width="{width}" height="{height}" viewBox="{view_box}">{fragment}</svg>'
    return embedded, anchors


def viewbox_size(template: Path) -> tuple[float, float]:
    """Return the local coordinate dimensions declared by one SVG template."""
    root = ET.parse(template).getroot()
    values = root.attrib.get("viewBox", "").split()
    if len(values) != 4:
        raise ValueError(f"template {template.name} must declare a four-value viewBox")
    return float(values[2]), float(values[3])
