"""Write CalculiX input decks and consistent nodal loads from a tetrahedral mesh.

Units inside the deck: mm, N, MPa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Gmsh numbers the last two mid-edge nodes of a 10-node tet the other way round
# from CalculiX/Abaqus: Gmsh 8 = edge 2-3, 9 = edge 1-3; CalculiX 9 = edge 1-4, 10 = edge 3-4.
GMSH_TO_CCX_TET10 = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
ELEMENT_TYPES = {4: "C3D4", 10: "C3D10"}


def _chunks(values, size=8):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def triangle_geometry(corner_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Areas (K,) and unit normals (K,3) of triangles given corner coordinates (K,3,3)."""
    cross = np.cross(corner_coords[:, 1] - corner_coords[:, 0], corner_coords[:, 2] - corner_coords[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    return 0.5 * twice_area, cross / np.maximum(twice_area, 1e-300)[:, None]


def outward_normals(triangles: np.ndarray, tets: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Unit normals of boundary triangles pointing out of the solid.

    Each boundary triangle is a face of exactly one tet; the normal is flipped
    to point away from that tet's fourth corner.
    """
    corners = triangles[:, :3]
    scale = int(max(corners.max(), tets[:, :4].max())) + 1
    encode = lambda a: (np.sort(a, axis=1).astype(np.int64) @ np.array([scale * scale, scale, 1], dtype=np.int64))
    wanted = encode(corners)
    opposite = np.zeros(len(corners), dtype=np.int64)
    found = np.zeros(len(corners), dtype=bool)
    order = np.argsort(wanted)
    for face, other in (((0, 1, 2), 3), ((0, 1, 3), 2), ((0, 2, 3), 1), ((1, 2, 3), 0)):
        keys = encode(tets[:, face])
        hits = np.isin(keys, wanted)
        if not hits.any():
            continue
        slots = order[np.searchsorted(wanted, keys[hits], sorter=order)]
        opposite[slots] = tets[hits, other]
        found[slots] = True
    if not found.all():
        raise RuntimeError("Could not match load triangles to volume elements.")
    tri_xyz = coords[corners]
    _, normals = triangle_geometry(tri_xyz)
    inward = coords[opposite] - tri_xyz[:, 0]
    flip = np.einsum("ij,ij->i", normals, inward) > 0
    normals[flip] *= -1.0
    return normals


def consistent_loads(triangles: np.ndarray, tractions: np.ndarray, coords: np.ndarray) -> dict[int, np.ndarray]:
    """Nodal forces (N) equivalent to a constant traction (MPa) on each triangle.

    3-node triangles give each corner A/3. 6-node triangles give corners 0 and
    each mid-edge node A/3, which is exact for a quadratic element under
    uniform traction.
    """
    areas, _ = triangle_geometry(coords[triangles[:, :3]])
    loaded = triangles[:, 3:6] if triangles.shape[1] == 6 else triangles[:, :3]
    loads: dict[int, np.ndarray] = {}
    for nodes, area, traction in zip(loaded, areas, tractions):
        share = traction * area / 3.0
        for node in nodes:
            node = int(node)
            loads[node] = loads.get(node, 0.0) + share
    return loads


def write_inp(path, node_tags, coords, elements, element_tags, fixed_nodes,
              nodal_forces, youngs_modulus_mpa, poisson_ratio) -> Path:
    """Write a linear static deck with one isotropic material."""
    nodes_per_element = elements.shape[1]
    if nodes_per_element not in ELEMENT_TYPES:
        raise ValueError(f"Unsupported element with {nodes_per_element} nodes.")
    if nodes_per_element == 10:
        elements = elements[:, GMSH_TO_CCX_TET10]
    path = Path(path)
    with path.open("w") as out:
        out.write("*HEADING\nfea-cad-optimizer\n*NODE, NSET=NALL\n")
        for tag, (x, y, z) in zip(node_tags, coords):
            out.write(f"{int(tag)}, {x:.10g}, {y:.10g}, {z:.10g}\n")
        out.write(f"*ELEMENT, TYPE={ELEMENT_TYPES[nodes_per_element]}, ELSET=EALL\n")
        for tag, row in zip(element_tags, elements):
            out.write(f"{int(tag)}, " + ", ".join(str(int(n)) for n in row) + "\n")
        out.write("*NSET, NSET=FIXED\n")
        for chunk in _chunks(sorted(int(n) for n in fixed_nodes)):
            out.write(", ".join(map(str, chunk)) + "\n")
        out.write(
            "*MATERIAL, NAME=MAT\n*ELASTIC\n"
            f"{youngs_modulus_mpa:.10g}, {poisson_ratio:.10g}\n"
            "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT\n"
            "*STEP\n*STATIC\n*BOUNDARY\nFIXED, 1, 3\n*CLOAD\n"
        )
        for node in sorted(nodal_forces):
            for dof, value in enumerate(nodal_forces[node], 1):
                if value != 0.0:
                    out.write(f"{node}, {dof}, {value:.10g}\n")
        out.write("*NODE FILE\nU\n*EL FILE\nS\n*END STEP\n")
    return path
