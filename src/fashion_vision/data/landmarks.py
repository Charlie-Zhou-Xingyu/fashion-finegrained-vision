"""
DeepFashion2 landmark parsing utilities.

DeepFashion2 annotations may contain landmark information for each clothing
item. The raw landmark format is usually a flat list:

    [x1, y1, v1, x2, y2, v2, ...]

where:
    x, y: landmark coordinates
    v: visibility/existence flag

Common interpretation:
    0: absent / not labeled
    1: present but occluded
    2: visible
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def parse_flat_landmarks(
    raw_landmarks: Optional[Sequence[Any]],
) -> List[Dict[str, Any]]:
    """
    Parse DeepFashion2 flat landmark list.

    Args:
        raw_landmarks: Flat landmark list in the form
            [x1, y1, v1, x2, y2, v2, ...].

    Returns:
        Structured landmark list:
            [
                {
                    "index": 1,
                    "x": 123.0,
                    "y": 45.0,
                    "visibility": 2,
                    "present": true,
                    "visible": true,
                    "occluded": false,
                    "absent": false
                },
                ...
            ]
    """
    if raw_landmarks is None:
        return []

    if not isinstance(raw_landmarks, (list, tuple)):
        return []

    if len(raw_landmarks) == 0:
        return []

    if len(raw_landmarks) % 3 != 0:
        raise ValueError(
            f"Invalid landmark length: {len(raw_landmarks)}. "
            "Expected a multiple of 3."
        )

    landmarks: List[Dict[str, Any]] = []

    for offset in range(0, len(raw_landmarks), 3):
        x = float(raw_landmarks[offset])
        y = float(raw_landmarks[offset + 1])
        visibility = int(raw_landmarks[offset + 2])

        has_valid_xy = bool(x > 0 and y > 0)
        present = bool(visibility > 0 and has_valid_xy)
        visible = bool(visibility == 2 and has_valid_xy)
        occluded = bool(visibility == 1 and has_valid_xy)
        absent = bool(visibility == 0 or not has_valid_xy)

        landmarks.append(
            {
                "index": int(offset // 3 + 1),
                "x": x,
                "y": y,
                "visibility": visibility,
                "present": present,
                "visible": visible,
                "occluded": occluded,
                "absent": absent,
            }
        )

    return landmarks


def get_present_landmarks(
    landmarks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Get present landmarks.

    Present means visibility > 0 and coordinates are valid.

    Args:
        landmarks: Structured landmark list.

    Returns:
        Present landmark list.
    """
    return [point for point in landmarks if bool(point.get("present", False))]


def get_visible_landmarks(
    landmarks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Get strictly visible landmarks.

    Visible means visibility == 2 and coordinates are valid.

    Args:
        landmarks: Structured landmark list.

    Returns:
        Visible landmark list.
    """
    return [point for point in landmarks if bool(point.get("visible", False))]


def get_occluded_landmarks(
    landmarks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Get occluded but present landmarks.

    Args:
        landmarks: Structured landmark list.

    Returns:
        Occluded landmark list.
    """
    return [point for point in landmarks if bool(point.get("occluded", False))]


def landmarks_to_flat_xy(
    landmarks: List[Dict[str, Any]],
    visible_only: bool = True,
) -> List[List[float]]:
    """
    Convert structured landmarks to xy point list.

    Args:
        landmarks: Structured landmark list.
        visible_only: Whether to keep strictly visible points only.

    Returns:
        List of xy points.
    """
    points = []

    for point in landmarks:
        if visible_only:
            if not point.get("visible", False):
                continue
        else:
            if not point.get("present", False):
                continue

        points.append([float(point["x"]), float(point["y"])])

    return points


def count_visible_landmarks(
    landmarks: List[Dict[str, Any]],
) -> int:
    """
    Count strictly visible landmarks.

    Args:
        landmarks: Structured landmark list.

    Returns:
        Number of visible landmarks.
    """
    return len(get_visible_landmarks(landmarks))


def count_present_landmarks(
    landmarks: List[Dict[str, Any]],
) -> int:
    """
    Count present landmarks.

    Args:
        landmarks: Structured landmark list.

    Returns:
        Number of present landmarks.
    """
    return len(get_present_landmarks(landmarks))


def sanitize_landmarks_for_json(
    landmarks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Make landmarks JSON serializable.

    Args:
        landmarks: Structured landmark list.

    Returns:
        JSON-serializable landmark list.
    """
    sanitized = []

    for point in landmarks:
        sanitized.append(
            {
                "index": int(point.get("index", 0)),
                "x": float(point.get("x", 0.0)),
                "y": float(point.get("y", 0.0)),
                "visibility": int(point.get("visibility", 0)),
                "present": bool(point.get("present", False)),
                "visible": bool(point.get("visible", False)),
                "occluded": bool(point.get("occluded", False)),
                "absent": bool(point.get("absent", False)),
            }
        )

    return sanitized
