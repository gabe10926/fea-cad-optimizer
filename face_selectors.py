"""Geometric face selectors shared by every CAD backend.

Selectors are re-resolved on each regenerated design, so they keep pointing at
the same physical face when dimensions change (unlike raw face numbers).

Supported selector strings (case-insensitive):

    xmin, xmax, ymin, ymax, zmin, zmax   every face lying on that bounding-box plane
    left, right                          aliases for xmin, xmax
    nearest:X,Y,Z                        the face whose centroid is closest to a point (mm)
    Face3 / Face3,Face7                  explicit face numbers (not stable across designs)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

AXES = {"x": 0, "y": 1, "z": 2}
ALIASES = {"left": "xmin", "right": "xmax", "support": "xmin", "fixed": "xmin", "load": "xmax"}


@dataclass(frozen=True)
class FaceInfo:
    index: int  # 1-based face number
    center: tuple[float, float, float]
    bbox: tuple[float, float, float, float, float, float]  # xmin, ymin, zmin, xmax, ymax, zmax
    area: float


def model_bbox(faces: list[FaceInfo]) -> tuple[float, ...]:
    return (
        *(min(face.bbox[axis] for face in faces) for axis in range(3)),
        *(max(face.bbox[axis + 3] for face in faces) for axis in range(3)),
    )


def _tolerance(bbox) -> float:
    diagonal = math.dist(bbox[:3], bbox[3:])
    return max(diagonal * 1e-6, 1e-7)


def _on_plane(face: FaceInfo, name: str, bbox, tolerance: float) -> bool:
    axis = AXES[name[0]]
    if name.endswith("min"):
        return face.bbox[axis + 3] <= bbox[axis] + tolerance
    return face.bbox[axis] >= bbox[axis + 3] - tolerance


def resolve(selector: str, faces: list[FaceInfo]) -> list[int]:
    """Return the 1-based face numbers matched by ``selector``."""
    if not faces:
        raise ValueError("The model has no faces to select from.")
    value = str(selector).strip().lower()
    value = ALIASES.get(value, value)
    bbox = model_bbox(faces)
    tolerance = _tolerance(bbox)

    if value in {f"{axis}{end}" for axis in AXES for end in ("min", "max")}:
        matched = [face.index for face in faces if _on_plane(face, value, bbox, tolerance)]
    elif value.startswith("nearest:"):
        try:
            point = [float(item) for item in value[len("nearest:"):].split(",")]
        except ValueError:
            point = []
        if len(point) != 3:
            raise ValueError(f"Selector {selector!r} must look like nearest:X,Y,Z")
        matched = [min(faces, key=lambda face: math.dist(face.center, point)).index]
    elif value.startswith("face"):
        indices = {face.index for face in faces}
        matched = []
        for item in value.split(","):
            item = item.strip()
            if not item.startswith("face") or not item[4:].isdigit():
                raise ValueError(f"Invalid face selector: {item!r}")
            number = int(item[4:])
            if number not in indices:
                raise ValueError(f"Face{number} does not exist; the model has {len(faces)} faces.")
            matched.append(number)
    else:
        raise ValueError(
            f"Unknown face selector {selector!r}. Use xmin/xmax/ymin/ymax/zmin/zmax, "
            "left/right, nearest:X,Y,Z, or FaceN."
        )
    if not matched:
        raise ValueError(f"Selector {selector!r} matched no faces.")
    return matched


def describe(faces: list[FaceInfo]) -> list[dict]:
    """Serializable face table, including which extreme planes each face lies on."""
    bbox = model_bbox(faces)
    tolerance = _tolerance(bbox)
    planes = [f"{axis}{end}" for axis in AXES for end in ("min", "max")]
    return [
        {
            "face": f"Face{face.index}",
            "center": [round(value, 4) for value in face.center],
            "area": round(face.area, 4),
            "on": [name for name in planes if _on_plane(face, name, bbox, tolerance)],
        }
        for face in faces
    ]
