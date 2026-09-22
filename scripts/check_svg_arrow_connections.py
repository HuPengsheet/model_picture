#!/usr/bin/env python3
"""Check that SVG arrows ending near addition circles do not overrun them.

Convention used by the architecture SVG templates:
  * addition nodes are <circle class="sum" ...>
  * arrows are <path class="flow" ...> or <path class="residual" ...>
  * ordinary model boxes are <rect class="module" ...>, "router", or
    "attention"; vertically stacked boxes need visible whitespace between them
  * paths use absolute M, H, V, or L commands

An arrow endpoint that is close to a sum circle must sit just outside its
radius. This allows an arrowhead to touch the circle outline while rejecting
endpoints drawn inside the circle.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from xml.etree import ElementTree


SVG_NS = "http://www.w3.org/2000/svg"
NS = f"{{{SVG_NS}}}"
TOKEN = re.compile(r"([MLHVZmlhvz])|(-?(?:\d+\.\d+|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)")
CSS_RULE = re.compile(r"\.([A-Za-z_-][\w-]*)\s*\{([^}]*)\}")
FONT_SIZE = re.compile(r"(?:font-size\s*:\s*|font\s*:[^;}]*?)(\d+(?:\.\d+)?)px")


def classes(element: ElementTree.Element) -> set[str]:
    return set(element.attrib.get("class", "").split())


def embedded_svg_nodes(root: ElementTree.Element) -> set[int]:
    """Return descendants of nested SVGs, which use their own coordinate system.

    Component templates can be embedded as child ``<svg>`` elements in a full
    diagram. They are checked before composition; this checker must not compare
    their local coordinates with the parent diagram's coordinates.
    """
    ignored = set()
    for node in root.iter(NS + "svg"):
        if node is root:
            continue
        ignored.update(id(descendant) for descendant in node.iter())
    return ignored


def endpoint(path_data: str) -> tuple[float, float]:
    """Return the final point of a path containing absolute M/H/V/L commands."""
    tokens = [(command, number) for command, number in TOKEN.findall(path_data)]
    x = y = None
    command = None
    index = 0
    while index < len(tokens):
        token_command, _ = tokens[index]
        if token_command:
            command = token_command
            index += 1
            if command.upper() == "Z":
                continue
        if command is None or index >= len(tokens):
            continue
        if command.upper() in ("M", "L"):
            x_token, y_token = tokens[index], tokens[index + 1]
            if x_token[1] == "" or y_token[1] == "":
                raise ValueError(f"Malformed {command} command in {path_data!r}")
            x, y = float(x_token[1]), float(y_token[1])
            index += 2
        elif command.upper() == "H":
            if tokens[index][1] == "":
                raise ValueError(f"Malformed H command in {path_data!r}")
            x = float(tokens[index][1])
            index += 1
        elif command.upper() == "V":
            if tokens[index][1] == "":
                raise ValueError(f"Malformed V command in {path_data!r}")
            y = float(tokens[index][1])
            index += 1
        else:
            raise ValueError(f"Unsupported command {command!r} in {path_data!r}")
    if x is None or y is None:
        raise ValueError(f"No endpoint in {path_data!r}")
    return x, y


def frame_rectangles(root: ElementTree.Element) -> list[tuple[str, float, float, float, float]]:
    """Return diagram boxes subject to the vertical-spacing rule.

    Outer containers, colored blocks and dotted panels are deliberately
    excluded: they are expected to contain other rectangles.
    """
    frame_classes = {"module", "router", "attention"}
    ignored = embedded_svg_nodes(root)
    frames = []
    for node in root.iter(NS + "rect"):
        if id(node) in ignored:
            continue
        matched = classes(node) & frame_classes
        if not matched:
            continue
        frames.append((
            "/".join(sorted(matched)),
            float(node.attrib["x"]),
            float(node.attrib["y"]),
            float(node.attrib["width"]),
            float(node.attrib["height"]),
        ))
    return frames


def spacing_failures(
    frames: list[tuple[str, float, float, float, float]], min_gap: float
) -> list[str]:
    """Find vertically adjacent, horizontally overlapping frames that are too close."""
    failures = []
    for upper_index, upper in enumerate(frames):
        upper_class, upper_x, upper_y, upper_width, upper_height = upper
        upper_bottom = upper_y + upper_height
        for lower_index, lower in enumerate(frames):
            if upper_index == lower_index:
                continue
            lower_class, lower_x, lower_y, lower_width, _ = lower
            if lower_y < upper_bottom:
                continue
            overlap = min(upper_x + upper_width, lower_x + lower_width) - max(upper_x, lower_x)
            # A minor edge overlap often comes from a side callout. Require
            # substantial horizontal alignment before treating boxes as stacked.
            if overlap < 0.25 * min(upper_width, lower_width):
                continue
            gap = lower_y - upper_bottom
            if gap < min_gap:
                failures.append(
                    f"rect {upper_index + 1} ({upper_class} at x={upper_x:g}, y={upper_y:g}) "
                    f"and rect {lower_index + 1} ({lower_class} at x={lower_x:g}, y={lower_y:g}) "
                    f"have only {gap:g}px vertical gap (minimum {min_gap:g}px)"
                )
    return failures


def font_sizes(root: ElementTree.Element) -> dict[str, float]:
    """Extract pixel font sizes from the compact CSS used by the SVG templates."""
    css = "\n".join(node.text or "" for node in root.iter(NS + "style"))
    sizes = {}
    for name, body in CSS_RULE.findall(css):
        match = FONT_SIZE.search(body)
        if match:
            sizes[name] = float(match.group(1))
    return sizes


def text_boxes(root: ElementTree.Element) -> list[tuple[str, float, float, float, float]]:
    """Estimate bounding boxes for SVG text elements.

    SVG has no glyph-bound API without a rendering engine. This deliberately
    errs slightly wide (0.56 em per character), so visibly crowded labels are
    reported before handoff.
    """
    sizes = font_sizes(root)
    ignored = embedded_svg_nodes(root)
    boxes = []
    for node in root.iter(NS + "text"):
        if id(node) in ignored:
            continue
        content = "".join(node.itertext()).strip()
        if not content or "x" not in node.attrib or "y" not in node.attrib:
            continue
        size = next((sizes[name] for name in classes(node) if name in sizes), 16.0)
        x, baseline = float(node.attrib["x"]), float(node.attrib["y"])
        width = max(size * 0.56 * len(content), size * 0.35)
        anchor = node.attrib.get("text-anchor", "start")
        if anchor == "middle":
            left = x - width / 2
        elif anchor == "end":
            left = x - width
        else:
            left = x
        boxes.append((content, left, baseline - size * 0.78, width, size))
    return boxes


def text_overlap_failures(boxes: list[tuple[str, float, float, float, float]]) -> list[str]:
    failures = []
    for index, (text_a, x_a, y_a, width_a, height_a) in enumerate(boxes):
        for other_index in range(index + 1, len(boxes)):
            text_b, x_b, y_b, width_b, height_b = boxes[other_index]
            horizontal = min(x_a + width_a, x_b + width_b) - max(x_a, x_b)
            vertical = min(y_a + height_a, y_b + height_b) - max(y_a, y_b)
            if horizontal > 0 and vertical > 0:
                failures.append(
                    f"text {index + 1} ({text_a!r}) overlaps text {other_index + 1} ({text_b!r}) "
                    f"by approximately {horizontal:.1f}px × {vertical:.1f}px"
                )
    return failures


def text_spacing_failures(
    boxes: list[tuple[str, float, float, float, float]], min_gap: float
) -> list[str]:
    """Reject vertically stacked labels that look crowded before they overlap."""
    failures = []
    for index, (text_a, x_a, y_a, width_a, height_a) in enumerate(boxes):
        for other_index in range(index + 1, len(boxes)):
            text_b, x_b, y_b, width_b, height_b = boxes[other_index]
            horizontal = min(x_a + width_a, x_b + width_b) - max(x_a, x_b)
            if horizontal < 0.6 * min(width_a, width_b):
                continue
            if y_a <= y_b:
                upper_index, upper_text, upper_bottom = index, text_a, y_a + height_a
                lower_index, lower_text, lower_top = other_index, text_b, y_b
            else:
                upper_index, upper_text, upper_bottom = other_index, text_b, y_b + height_b
                lower_index, lower_text, lower_top = index, text_a, y_a
            gap = lower_top - upper_bottom
            if 0 <= gap < min_gap:
                failures.append(
                    f"text {upper_index + 1} ({upper_text!r}) and text {lower_index + 1} "
                    f"({lower_text!r}) have only {gap:.1f}px vertical gap "
                    f"(minimum {min_gap:g}px)"
                )
    return failures


def path_segments(path_data: str) -> list[tuple[float, float, float, float]]:
    """Return straight segments from the absolute M/H/V/L paths used in templates."""
    tokens = [(command, number) for command, number in TOKEN.findall(path_data)]
    x = y = None
    command = None
    index = 0
    segments = []
    while index < len(tokens):
        token_command, _ = tokens[index]
        if token_command:
            command = token_command.upper()
            index += 1
        if command is None or index >= len(tokens):
            continue
        old_x, old_y = x, y
        if command in ("M", "L"):
            x, y = float(tokens[index][1]), float(tokens[index + 1][1])
            index += 2
        elif command == "H":
            x = float(tokens[index][1])
            index += 1
        elif command == "V":
            y = float(tokens[index][1])
            index += 1
        elif command == "Z":
            break
        else:
            raise ValueError(f"Unsupported command {command!r} in {path_data!r}")
        if command != "M" and old_x is not None and old_y is not None:
            segments.append((old_x, old_y, x, y))
    return segments


def segment_length(segment: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = segment
    return math.hypot(x2 - x1, y2 - y1)


def arrow_path_failures(
    root: ElementTree.Element, min_length: float, join_tolerance: float
) -> list[str]:
    """Reject tiny arrows and consecutive arrowheads on one connector."""
    arrows = []
    failures = []
    for index, node in enumerate(root.iter(NS + "path"), start=1):
        if not (classes(node) & {"flow", "residual"}):
            continue
        segments = path_segments(node.attrib["d"])
        if not segments:
            continue
        length = sum(segment_length(segment) for segment in segments)
        if length < min_length:
            failures.append(
                f"path {index}: arrow line is only {length:.1f}px long "
                f"(minimum {min_length:g}px)"
            )
        first = segments[0]
        last = segments[-1]
        arrows.append((index, first[0], first[1], last[2], last[3], last))

    for index, x1, y1, x2, y2, last in arrows:
        for other_index, ox1, oy1, ox2, oy2, _ in arrows:
            if index == other_index:
                continue
            if math.hypot(x2 - ox1, y2 - oy1) > join_tolerance:
                continue
            # Two arrow paths meeting at an endpoint create two arrowheads.
            if segment_length(last) == 0:
                continue
            failures.append(
                f"paths {index} and {other_index}: consecutive arrowheads meet at "
                f"({x2:g}, {y2:g})"
            )
            break
    return failures


def segment_intersects_rect(
    x1: float, y1: float, x2: float, y2: float,
    left: float, top: float, width: float, height: float,
) -> bool:
    """Liang-Barsky segment clipping: true when a segment touches a rectangle."""
    right, bottom = left + width, top + height
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - left, right - x1, y1 - top, bottom - y1)
    lower, upper = 0.0, 1.0
    for numerator, denominator in zip(q, p):
        if denominator == 0:
            if numerator < 0:
                return False
            continue
        value = numerator / denominator
        if denominator < 0:
            if value > upper:
                return False
            lower = max(lower, value)
        else:
            if value < lower:
                return False
            upper = min(upper, value)
    return lower <= upper


def text_path_overlap_failures(
    root: ElementTree.Element, boxes: list[tuple[str, float, float, float, float]]
) -> list[str]:
    """Flag diagram connectors that run through estimated text bounds."""
    connector_classes = {"flow", "residual", "merge", "wire"}
    ignored = embedded_svg_nodes(root)
    failures = []
    for path_index, node in enumerate(root.iter(NS + "path"), start=1):
        if id(node) in ignored:
            continue
        if not (classes(node) & connector_classes):
            continue
        for segment in path_segments(node.attrib["d"]):
            for text_index, (content, x, y, width, height) in enumerate(boxes, start=1):
                # Three pixels accommodates the template connector stroke width.
                if segment_intersects_rect(*segment, x - 3, y - 3, width + 6, height + 6):
                    failures.append(
                        f"path {path_index} intersects text {text_index} ({content!r})"
                    )
    return failures


def text_boundary_overlap_failures(
    root: ElementTree.Element, boxes: list[tuple[str, float, float, float, float]]
) -> list[str]:
    """Flag text touching the drawn perimeter of visible architecture rectangles."""
    visible_classes = {"outer", "block", "module", "router", "attention", "moe", "panel"}
    ignored = embedded_svg_nodes(root)
    frames = [
        node for node in root.iter(NS + "rect")
        if id(node) not in ignored and classes(node) & visible_classes
    ]
    failures = []
    for text_index, (content, x, y, width, height) in enumerate(boxes, start=1):
        for frame_index, frame in enumerate(frames, start=1):
            left, top = float(frame.attrib["x"]), float(frame.attrib["y"])
            frame_width, frame_height = float(frame.attrib["width"]), float(frame.attrib["height"])
            # A 4px band matches the standard frame stroke in these SVGs.
            boundary_bands = (
                (left - 2, top - 2, frame_width + 4, 6),
                (left - 2, top + frame_height - 4, frame_width + 4, 6),
                (left - 2, top - 2, 6, frame_height + 4),
                (left + frame_width - 4, top - 2, 6, frame_height + 4),
            )
            for band in boundary_bands:
                if segment_intersects_rect(
                    x, y, x + width, y + height, *band
                ) or segment_intersects_rect(
                    x, y + height, x + width, y, *band
                ):
                    failures.append(
                        f"text {text_index} ({content!r}) touches rectangle {frame_index} boundary"
                    )
                    break
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path, help="SVG file to inspect")
    parser.add_argument("--min-clearance", type=float, default=0.5,
                        help="minimum endpoint clearance outside a circle radius (default: 0.5px)")
    parser.add_argument("--max-clearance", type=float, default=12.0,
                        help="maximum endpoint clearance to still count as targeting a circle (default: 12px)")
    parser.add_argument("--min-rectangle-gap", type=float, default=32.0,
                        help="minimum vertical gap for horizontally aligned model boxes (default: 32px)")
    parser.add_argument("--min-text-gap", type=float, default=6.0,
                        help="minimum vertical gap for vertically stacked labels (default: 6px)")
    parser.add_argument("--min-arrow-length", type=float, default=24.0,
                        help="minimum length of a flow/residual arrow line (default: 24px)")
    parser.add_argument("--arrow-join-tolerance", type=float, default=0.5,
                        help="endpoint tolerance for detecting consecutive arrowheads (default: 0.5px)")
    args = parser.parse_args()

    root = ElementTree.parse(args.svg).getroot()
    ignored = embedded_svg_nodes(root)
    circles = [
        (float(node.attrib["cx"]), float(node.attrib["cy"]), float(node.attrib["r"]))
        for node in root.iter(NS + "circle")
        if id(node) not in ignored and "sum" in classes(node)
    ]
    arrows = [
        node for node in root.iter(NS + "path")
        if id(node) not in ignored and {"flow", "residual"} & classes(node)
    ]
    failures: list[str] = []
    checked = 0
    for number, arrow in enumerate(arrows, start=1):
        x, y = endpoint(arrow.attrib["d"])
        if not circles:
            continue
        cx, cy, radius = min(circles, key=lambda item: math.hypot(x - item[0], y - item[1]))
        distance = math.hypot(x - cx, y - cy)
        # Ignore arrows that clearly terminate at ordinary modules, not sum nodes.
        if distance > radius + args.max_clearance:
            continue
        checked += 1
        clearance = distance - radius
        if clearance < args.min_clearance:
            failures.append(
                f"path {number}: endpoint ({x:g}, {y:g}) is {abs(clearance):.2f}px "
                f"inside/touching sum circle ({cx:g}, {cy:g}, r={radius:g})"
            )
    failures.extend(spacing_failures(frame_rectangles(root), args.min_rectangle_gap))
    failures.extend(arrow_path_failures(root, args.min_arrow_length, args.arrow_join_tolerance))
    text_layout = text_boxes(root)
    failures.extend(text_overlap_failures(text_layout))
    failures.extend(text_spacing_failures(text_layout, args.min_text_gap))
    failures.extend(text_path_overlap_failures(root, text_layout))
    failures.extend(text_boundary_overlap_failures(root, text_layout))

    if failures:
        print(f"FAIL {args.svg}: {len(failures)} layout issue(s)", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(
        f"PASS {args.svg}: checked {checked} arrow endpoint(s) near {len(circles)} sum circle(s); "
        f"checked {len(frame_rectangles(root))} model box(es) with minimum vertical gap {args.min_rectangle_gap:g}px; "
        f"checked {len(text_layout)} text label(s) for overlap/collisions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
