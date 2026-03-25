from __future__ import annotations

import os
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
GT_CANDIDATE_KEYS = ("plate_text", "gt_text", "text", "ocr_text", "label")
LAYOUT_CANDIDATE_KEYS = ("plate_layout", "layout")


def _clean_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def extract_gt_text(annotation: dict[str, Any]) -> str | None:
    """Return the safest available GT text.

    Current competition files expose `plate_text`; the fallback keys are kept
    conservative and documented so future schema drift is easy to inspect.
    """

    for key in GT_CANDIDATE_KEYS:
        text = _clean_text(annotation.get(key))
        if text:
            return text
    # TODO: confirm the final competition schema before widening this lookup.
    return None


def extract_layout(annotation: dict[str, Any]) -> str | None:
    for key in LAYOUT_CANDIDATE_KEYS:
        text = _clean_text(annotation.get(key))
        if text:
            return text
    # TODO: confirm whether any future release renames the layout key.
    return None


def normalize_corner_key(name: str) -> str:
    """Normalize a corner key with at most one image extension suffix."""

    current = name
    while True:
        root, ext = os.path.splitext(current)
        if ext.lower() not in IMAGE_SUFFIXES:
            return current
        if os.path.splitext(root)[1].lower() == ext.lower():
            current = root
            continue
        return current


def extract_annotation_notes(
    annotation: dict[str, Any], *, expected_image_names: list[str] | None = None
) -> list[str]:
    notes: list[str] = []
    corners = annotation.get("corners")
    if not isinstance(corners, dict) or not corners:
        return notes

    corner_keys = [str(key) for key in corners.keys()]
    if any(key.lower().endswith((".png.png", ".jpg.jpg", ".jpeg.jpeg")) for key in corner_keys):
        notes.append("corners_keys_have_duplicated_image_suffix")

    if expected_image_names is not None:
        normalized_keys = sorted(normalize_corner_key(key) for key in corner_keys)
        normalized_expected = sorted(expected_image_names)
        if normalized_keys != normalized_expected:
            notes.append("corners_keys_do_not_match_frame_names")

    return notes

