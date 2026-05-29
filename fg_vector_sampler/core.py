from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

ATOMIC_MASSES = {
    "H": 1.00784,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "P": 30.973761998,
    "F": 18.998403163,
    "Cl": 35.45,
}

VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "Cl": 1.75,
}


def normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Return a unit vector. If the norm is tiny, return fallback or x-axis."""
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        if fallback is not None:
            return normalize(fallback)
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


@jit(nopython=False, cache=True) if HAS_NUMBA else lambda f: f
def _fast_rotation_matrix(x: float, y: float, z: float, c: float, s: float, C: float) -> np.ndarray:
    """Numba-accelerated Rodrigues rotation matrix computation."""
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)


def orthonormal_basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors perpendicular to axis."""
    axis = normalize(axis)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = normalize(np.cross(axis, trial))
    e2 = normalize(np.cross(axis, e1))
    return e1, e2


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix around an arbitrary axis."""
    axis = normalize(axis)
    x, y, z = axis[0], axis[1], axis[2]
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return _fast_rotation_matrix(x, y, z, c, s, C)


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return R such that R @ normalize(a) ~= normalize(b)."""
    a = normalize(a)
    b = normalize(b)
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        # 180-degree rotation around any perpendicular axis.
        e1, _ = orthonormal_basis_from_axis(a)
        return rotation_about_axis(e1, math.pi)
    axis = normalize(cross)
    angle = math.acos(dot)
    return rotation_about_axis(axis, angle)


def random_small_rotation(rng: np.random.Generator, max_angle_deg: float) -> np.ndarray:
    """Small random angular jitter."""
    if max_angle_deg <= 0:
        return np.eye(3)
    axis = normalize(rng.normal(size=3))
    angle = math.radians(float(rng.uniform(-max_angle_deg, max_angle_deg)))
    return rotation_about_axis(axis, angle)


@dataclass(frozen=True)
class Atom:
    element: str
    coord: np.ndarray
    mass: float | None = None
    vdw_radius: float | None = None
    molecule_id: int = 0

    def __post_init__(self):
        object.__setattr__(self, "coord", np.asarray(self.coord, dtype=float))
        if self.mass is None:
            object.__setattr__(self, "mass", ATOMIC_MASSES.get(self.element, 12.0))
        if self.vdw_radius is None:
            object.__setattr__(self, "vdw_radius", VDW_RADII.get(self.element, 1.7))

    def transformed(self, R: np.ndarray, t: np.ndarray, molecule_id: int | None = None) -> "Atom":
        return Atom(
            element=self.element,
            coord=R @ self.coord + t,
            mass=self.mass,
            vdw_radius=self.vdw_radius,
            molecule_id=self.molecule_id if molecule_id is None else molecule_id,
        )


@dataclass
class Feature:
    feature_id: str
    molecule_id: int
    type: str
    local_position: np.ndarray
    local_direction: np.ndarray | None = None
    normal: np.ndarray | None = None
    atom_indices: tuple[int, ...] = field(default_factory=tuple)
    weight: float = 1.0
    global_position: np.ndarray | None = None
    global_direction: np.ndarray | None = None

    def __post_init__(self):
        self.local_position = np.asarray(self.local_position, dtype=float)
        if self.local_direction is not None:
            self.local_direction = normalize(np.asarray(self.local_direction, dtype=float))
        if self.normal is not None:
            self.normal = normalize(np.asarray(self.normal, dtype=float))

    @property
    def outward_local(self) -> np.ndarray:
        return normalize(self.local_position, self.local_direction)

    @property
    def position(self) -> np.ndarray:
        if self.global_position is None:
            return self.local_position
        return self.global_position

    @property
    def outward_global(self) -> np.ndarray:
        if self.global_direction is not None:
            return normalize(self.global_direction)
        return normalize(self.position)

    def transformed(self, R: np.ndarray, t: np.ndarray, molecule_id: int | None = None) -> "Feature":
        mid = self.molecule_id if molecule_id is None else molecule_id
        return Feature(
            feature_id=self.feature_id,
            molecule_id=mid,
            type=self.type,
            local_position=self.local_position.copy(),
            local_direction=None if self.local_direction is None else self.local_direction.copy(),
            normal=None if self.normal is None else self.normal.copy(),
            atom_indices=self.atom_indices,
            weight=self.weight,
            global_position=R @ self.local_position + t,
            global_direction=R @ self.outward_local,
        )


@dataclass
class Monomer:
    name: str
    atoms: list[Atom]
    features: list[Feature] = field(default_factory=list)
    molecule_id: int = 0

    def center_of_mass(self) -> np.ndarray:
        masses = np.array([a.mass for a in self.atoms], dtype=float)
        coords = np.vstack([a.coord for a in self.atoms])
        return np.sum(coords * masses[:, None], axis=0) / np.sum(masses)

    def shifted_to_com(self, molecule_id: int | None = None) -> "Monomer":
        """Return a copy whose atomic coordinates are in the monomer COM frame."""
        mid = self.molecule_id if molecule_id is None else molecule_id
        com = self.center_of_mass()
        atoms = [Atom(a.element, a.coord - com, a.mass, a.vdw_radius, mid) for a in self.atoms]
        features: list[Feature] = []
        for f in self.features:
            nf = Feature(
                feature_id=f.feature_id,
                molecule_id=mid,
                type=f.type,
                local_position=f.local_position - com if f.global_position is None else f.local_position,
                local_direction=f.local_direction,
                normal=f.normal,
                atom_indices=f.atom_indices,
                weight=f.weight,
            )
            features.append(nf)
        return Monomer(self.name, atoms, features, mid)

    def transformed(self, R: np.ndarray, t: np.ndarray, molecule_id: int | None = None) -> "Monomer":
        mid = self.molecule_id if molecule_id is None else molecule_id
        atoms = [a.transformed(R, t, mid) for a in self.atoms]
        features = [f.transformed(R, t, mid) for f in self.features]
        return Monomer(self.name, atoms, features, mid)


@dataclass(frozen=True)
class ContactTemplate:
    type_a: str
    type_b: str
    contact_label: str
    d_min: float
    d_max: float
    d_ideal: float
    priority: float = 1.0
    angle_min_deg: float | None = None
    symmetric: bool = True

    def matches(self, a: str, b: str) -> bool:
        if self.type_a == a and self.type_b == b:
            return True
        return self.symmetric and self.type_a == b and self.type_b == a

    def distance_score(self, d: float) -> float:
        sigma = max((self.d_max - self.d_min) / 4.0, 1e-6)
        return float(math.exp(-((d - self.d_ideal) ** 2) / (2.0 * sigma * sigma)))


@dataclass
class Contact:
    feature_a: Feature
    feature_b: Feature
    contact_label: str
    distance: float
    angle_deg: float | None = None
    score: float = 1.0

    def signature(self) -> tuple[str, str, str]:
        pair = tuple(sorted((self.feature_a.type, self.feature_b.type)))
        return (pair[0], pair[1], self.contact_label)


def center_of_mass_atoms(atoms: list[Atom]) -> np.ndarray:
    masses = np.array([a.mass for a in atoms], dtype=float)
    coords = np.vstack([a.coord for a in atoms])
    return np.sum(coords * masses[:, None], axis=0) / np.sum(masses)


def radius_of_gyration(atoms: list[Atom]) -> float:
    com = center_of_mass_atoms(atoms)
    masses = np.array([a.mass for a in atoms], dtype=float)
    coords = np.vstack([a.coord for a in atoms])
    return float(np.sqrt(np.sum(masses * np.sum((coords - com) ** 2, axis=1)) / np.sum(masses)))


def distance_histogram(points: np.ndarray, bins: int = 16, max_distance: float = 20.0) -> tuple[int, ...]:
    if len(points) < 2:
        return tuple([0] * bins)
    dists = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dists.append(float(np.linalg.norm(points[i] - points[j])))
    hist, _ = np.histogram(dists, bins=bins, range=(0.0, max_distance))
    return tuple(int(x) for x in hist)

