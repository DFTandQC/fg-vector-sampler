from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from .core import (
    Atom,
    Contact,
    ContactTemplate,
    Feature,
    Monomer,
    normalize,
    orthonormal_basis_from_axis,
    rotation_about_axis,
    rotation_matrix_from_vectors,
    random_small_rotation,
    radius_of_gyration,
    distance_histogram,
)
from .priors import FunctionalGroupPrior
from .molecule_lib import write_xyz, write_json


@dataclass
class SamplerConfig:
    distance_bins: tuple[float, ...] = (0.15, 0.50, 0.85)  # interpolation in template range
    twist_degrees: tuple[float, ...] = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
    lateral_offsets: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.45, 0.0),
        (-0.45, 0.0),
        (0.0, 0.45),
        (0.0, -0.45),
    )
    jitter_deg: float = 8.0
    lambda_hard: float = 0.68
    lambda_soft: float = 0.82
    max_soft_overlap: float = 12.0
    rg_min: float = 0.0
    rg_max: float = 50.0
    max_shape_anisotropy: float | None = 0.40
    max_orientation_order: float | None = 0.95
    min_contacts: int = 1
    max_attempts_per_cluster: int = 2000
    max_candidates: int = 100
    beam_width: int = 30
    max_per_mode: int = 4
    random_seed: int = 7
    w_contact: float = 2.0
    w_multicontact: float = 1.2
    w_saturation: float = 0.8
    w_novelty: float = 1.5
    w_compact: float = 0.2
    w_shape: float = 10.0
    w_orientation: float = 8.0
    w_clash: float = 0.5
    w_bridge: float = 1.0
    w_sa_saturation: float = 0.9
    w_carboxyl_multisite: float = 0.8


@dataclass
class PoseMetadata:
    primary_existing_feature: str
    primary_new_feature: str
    primary_contact_label: str
    distance: float
    distance_bin: str
    twist_deg: float
    offset: tuple[float, float]
    R: np.ndarray
    t: np.ndarray


@dataclass
class ClusterCandidate:
    monomers: list[Monomer]
    contacts: list[Contact] = field(default_factory=list)
    molecule_graph_edges: list[tuple[int, int]] = field(default_factory=list)
    mode_label: str = ""
    score: float = 0.0
    rg: float = 0.0
    shape_anisotropy: float = 0.0
    orientation_order: float = 0.0
    clash_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def atoms(self) -> list[Atom]:
        return [a for m in self.monomers for a in m.atoms]

    @property
    def features(self) -> list[Feature]:
        return [f for m in self.monomers for f in m.features]

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_contacts(self) -> int:
        return len(self.contacts)

    @property
    def contact_types(self) -> list[str]:
        return sorted({c.contact_label for c in self.contacts})

    def molecule_ids(self) -> list[int]:
        return sorted({m.molecule_id for m in self.monomers})

    def feature_by_id(self, feature_id: str, molecule_id: int) -> Feature | None:
        for f in self.features:
            if f.feature_id == feature_id and f.molecule_id == molecule_id:
                return f
        return None


@dataclass
class CoverageTracker:
    counts: dict[str, int] = field(default_factory=dict)
    valid_counts: dict[str, int] = field(default_factory=dict)
    recent_unique: list[int] = field(default_factory=list)

    def count(self, key: str) -> int:
        return self.counts.get(key, 0)

    def update_attempt(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def update_valid(self, key: str) -> None:
        self.valid_counts[key] = self.valid_counts.get(key, 0) + 1

    def sampling_weight(self, key: str, priority: float) -> float:
        return priority / (self.counts.get(key, 0) + 1.0)

    def is_saturated(self, window: int = 500, min_new_modes: int = 5) -> bool:
        if len(self.recent_unique) < window:
            return False
        return sum(self.recent_unique[-window:]) < min_new_modes


def _hash_tuple(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


class ClusterSampler:
    """Energy-free COM-anchored functional-group vector cluster sampler."""

    def __init__(self, prior: FunctionalGroupPrior | None = None, config: SamplerConfig | None = None):
        self.prior = prior or FunctionalGroupPrior.default()
        self.config = config or SamplerConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.coverage = CoverageTracker()
        self.seen_modes: dict[str, int] = {}
        self.kept_signatures: set[str] = set()

    def initial_candidate(self, monomer: Monomer, molecule_id: int = 0) -> ClusterCandidate:
        m = monomer.shifted_to_com(molecule_id=molecule_id).transformed(np.eye(3), np.zeros(3), molecule_id)
        return ClusterCandidate([m], metadata={"seed": monomer.name})

    def compatible_pairs(self, cluster: ClusterCandidate, monomer: Monomer) -> list[tuple[Feature, Feature, ContactTemplate]]:
        pairs: list[tuple[Feature, Feature, ContactTemplate]] = []
        for a in cluster.features:
            for b in monomer.features:
                template = self.prior.find_template(a.type, b.type)
                if template is not None:
                    pairs.append((a, b, template))
        return pairs

    def choose_primary_pair(self, pairs: list[tuple[Feature, Feature, ContactTemplate]]) -> tuple[Feature, Feature, ContactTemplate]:
        keys = [f"pair:{a.type}:{b.type}:{t.contact_label}" for a, b, t in pairs]
        weights = np.array(
            [self.coverage.sampling_weight(k, t.priority * a.weight * b.weight) for (a, b, t), k in zip(pairs, keys)],
            dtype=float,
        )
        weights = np.where(np.isfinite(weights), weights, 0.0)
        total = float(weights.sum())
        if total <= 0.0:
            weights = np.ones(len(pairs), dtype=float) / len(pairs)
        else:
            weights = weights / total
        idx = int(self.rng.choice(len(pairs), p=weights))
        self.coverage.update_attempt(keys[idx])
        return pairs[idx]

    def generate_pose(
        self,
        existing_feature: Feature,
        new_feature: Feature,
        template: ContactTemplate,
        distance_fraction: float,
        twist_deg: float,
        offset_xy: tuple[float, float],
        new_molecule_id: int,
    ) -> PoseMetadata:
        axis = normalize(existing_feature.outward_global)
        b_axis = normalize(new_feature.outward_local)
        R_align = rotation_matrix_from_vectors(b_axis, -axis)
        R_twist = rotation_about_axis(axis, math.radians(twist_deg))
        R_jitter = random_small_rotation(self.rng, self.config.jitter_deg)
        R = R_jitter @ R_twist @ R_align

        d = template.d_min + distance_fraction * (template.d_max - template.d_min)
        e1, e2 = orthonormal_basis_from_axis(axis)
        dx, dy = offset_xy
        delta_perp = dx * e1 + dy * e2
        t = existing_feature.position + d * axis + delta_perp - R @ new_feature.local_position
        return PoseMetadata(
            primary_existing_feature=f"{existing_feature.molecule_id}:{existing_feature.feature_id}:{existing_feature.type}",
            primary_new_feature=f"{new_molecule_id}:{new_feature.feature_id}:{new_feature.type}",
            primary_contact_label=template.contact_label,
            distance=float(d),
            distance_bin=f"d{distance_fraction:.2f}",
            twist_deg=float(twist_deg),
            offset=(float(dx), float(dy)),
            R=R,
            t=t,
        )

    def place_monomer(
        self,
        cluster: ClusterCandidate,
        monomer: Monomer,
        new_molecule_id: int,
        child_limit: int | None = None,
        attempt_limit: int | None = None,
    ) -> list[ClusterCandidate]:
        local = monomer.shifted_to_com(molecule_id=new_molecule_id)
        pairs = self.compatible_pairs(cluster, local)
        if not pairs:
            return []
        children: list[ClusterCandidate] = []
        local_signatures: set[str] = set()
        # Try a balanced combination of adaptive pair choices and systematic pose bins.
        max_pair_trials = min(len(pairs), 60)
        attempts = 0
        max_attempts = max(
            1,
            int(self.config.max_attempts_per_cluster if attempt_limit is None else attempt_limit),
        )
        if child_limit is None:
            child_limit = max(self.config.max_candidates, self.config.beam_width) * 2
        child_limit = max(1, int(child_limit))
        for _ in range(max_pair_trials):
            if attempts >= max_attempts or len(children) >= child_limit:
                break
            a, b, template = self.choose_primary_pair(pairs)
            for df in self.config.distance_bins:
                if attempts >= max_attempts or len(children) >= child_limit:
                    break
                for twist in self.config.twist_degrees:
                    if attempts >= max_attempts or len(children) >= child_limit:
                        break
                    for offset in self.config.lateral_offsets:
                        if attempts >= max_attempts or len(children) >= child_limit:
                            break
                        pose = self.generate_pose(a, b, template, df, twist, offset, new_molecule_id)
                        placed = local.transformed(pose.R, pose.t, molecule_id=new_molecule_id)
                        cand = ClusterCandidate(
                            monomers=[*cluster.monomers, placed],
                            metadata={**cluster.metadata, "last_pose": pose.__dict__},
                        )
                        attempts += 1
                        valid = self.evaluate_candidate(cand, pose)
                        if valid is not None:
                            # Early deduplication by local signature to avoid duplicate candidates
                            try:
                                sig = self._dedup_signature(valid)
                            except Exception:
                                sig = None
                            if sig is None or sig not in local_signatures:
                                if sig is not None:
                                    local_signatures.add(sig)
                                children.append(valid)
        return children

    def evaluate_candidate(self, cand: ClusterCandidate, pose: PoseMetadata | None = None) -> ClusterCandidate | None:
        atoms = cand.atoms
        clash_score = self.soft_overlap_score(atoms)
        if clash_score is None:
            return None
        if clash_score > self.config.max_soft_overlap:
            return None
        rg = radius_of_gyration(atoms)
        if not (self.config.rg_min <= rg <= self.config.rg_max):
            return None
        shape_anisotropy = self.com_shape_anisotropy(cand.monomers)
        orientation_order = self.molecular_orientation_order(cand.monomers)
        if (
            self.config.max_shape_anisotropy is not None
            and len(cand.monomers) >= 4
            and shape_anisotropy > self.config.max_shape_anisotropy
        ):
            return None
        if (
            self.config.max_orientation_order is not None
            and len(cand.monomers) >= 4
            and orientation_order > self.config.max_orientation_order
        ):
            return None
        contacts = self.detect_contacts(cand.features)
        if len(contacts) < self.config.min_contacts:
            return None
        mol_edges = self.molecule_contact_edges(contacts)
        if not self.is_molecular_graph_connected(cand.molecule_ids(), mol_edges):
            return None
        cand.contacts = contacts
        cand.molecule_graph_edges = mol_edges
        cand.rg = rg
        cand.shape_anisotropy = shape_anisotropy
        cand.orientation_order = orientation_order
        cand.clash_score = clash_score
        cand.mode_label = self.mode_label(cand, pose)
        cand.score = self.score_candidate(cand)
        self.coverage.update_valid(cand.mode_label)
        return cand

    @staticmethod
    def _normalized_gyration_anisotropy(points: np.ndarray, weights: np.ndarray | None = None) -> float:
        if len(points) < 3:
            return 0.0
        coords = np.asarray(points, dtype=float)
        if weights is None:
            weights = np.ones(len(coords), dtype=float)
        weights = np.asarray(weights, dtype=float)
        total = float(weights.sum())
        if total <= 0.0:
            return 0.0
        center = np.sum(coords * weights[:, None], axis=0) / total
        shifted = coords - center
        tensor = (shifted * weights[:, None]).T @ shifted / total
        eigvals = np.linalg.eigvalsh(tensor)
        eigvals = np.clip(eigvals, 0.0, None)
        trace = float(eigvals.sum())
        if trace < 1e-12:
            return 0.0
        mean = trace / 3.0
        return float(1.5 * np.sum((eigvals - mean) ** 2) / (trace ** 2))

    def com_shape_anisotropy(self, monomers: list[Monomer]) -> float:
        """Return 0 for isotropic COM arrangements and larger values for flat/linear clusters."""
        if len(monomers) < 3:
            return 0.0
        centers = []
        masses = []
        for monomer in monomers:
            atoms = monomer.atoms
            atom_masses = np.array([a.mass for a in atoms], dtype=float)
            coords = np.vstack([a.coord for a in atoms])
            total_mass = float(atom_masses.sum())
            centers.append(np.sum(coords * atom_masses[:, None], axis=0) / total_mass)
            masses.append(total_mass)
        return self._normalized_gyration_anisotropy(np.vstack(centers), np.array(masses, dtype=float))

    def molecular_orientation_order(self, monomers: list[Monomer]) -> float:
        """Nematic-like order parameter for monomer plane normals; 1 means over-parallel."""
        if len(monomers) < 3:
            return 0.0
        normals = []
        for monomer in monomers:
            atoms = monomer.atoms
            if len(atoms) < 3:
                continue
            weights = np.array([a.mass for a in atoms], dtype=float)
            coords = np.vstack([a.coord for a in atoms])
            center = np.sum(coords * weights[:, None], axis=0) / float(weights.sum())
            shifted = coords - center
            tensor = (shifted * weights[:, None]).T @ shifted / float(weights.sum())
            eigvals, eigvecs = np.linalg.eigh(tensor)
            normal = normalize(eigvecs[:, int(np.argmin(eigvals))])
            normals.append(normal)
        if len(normals) < 3:
            return 0.0
        q = np.zeros((3, 3), dtype=float)
        for normal in normals:
            q += 1.5 * np.outer(normal, normal) - 0.5 * np.eye(3)
        q /= len(normals)
        return float(max(0.0, np.max(np.linalg.eigvalsh(q))))

    def soft_overlap_score(self, atoms: list[Atom]) -> float | None:
        if len(atoms) < 2:
            return 0.0
        coords = np.vstack([a.coord for a in atoms])
        mol_ids = np.array([a.molecule_id for a in atoms])
        radii = np.array([a.vdw_radius for a in atoms], dtype=float)
        tree = cKDTree(coords)
        max_cut = self.config.lambda_soft * 2.0 * float(np.max(radii))
        pairs = tree.query_pairs(r=max_cut)
        score = 0.0
        for i, j in pairs:
            if mol_ids[i] == mol_ids[j]:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            hard = self.config.lambda_hard * (radii[i] + radii[j])
            if d < hard:
                return None
            soft = self.config.lambda_soft * (radii[i] + radii[j])
            score += max(0.0, soft - d) ** 2
        return float(score)

    def detect_contacts(self, features: list[Feature]) -> list[Contact]:
        contacts: list[Contact] = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                a, b = features[i], features[j]
                if a.molecule_id == b.molecule_id:
                    continue
                template = self.prior.find_template(a.type, b.type)
                if template is None:
                    continue
                d = float(np.linalg.norm(a.position - b.position))
                if not (template.d_min <= d <= template.d_max):
                    continue
                angle_deg = None
                if template.angle_min_deg is not None and a.global_direction is not None and b.global_direction is not None:
                    # Generic directionality: 180 degrees is ideal when outward vectors oppose.
                    cosang = float(np.clip(np.dot(normalize(a.global_direction), -normalize(b.global_direction)), -1.0, 1.0))
                    deviation_deg = math.degrees(math.acos(cosang))
                    angle_deg = 180.0 - deviation_deg
                    if angle_deg < template.angle_min_deg:
                        continue
                score = template.distance_score(d) * template.angle_score(angle_deg)
                contacts.append(Contact(a, b, template.contact_label, d, angle_deg, score))
        return contacts

    def molecule_contact_edges(self, contacts: list[Contact]) -> list[tuple[int, int]]:
        edges = set()
        for c in contacts:
            i, j = c.feature_a.molecule_id, c.feature_b.molecule_id
            if i != j:
                edges.add(tuple(sorted((i, j))))
        return sorted(edges)

    def is_molecular_graph_connected(self, molecule_ids: list[int], edges: list[tuple[int, int]]) -> bool:
        if len(molecule_ids) <= 1:
            return True
        g = nx.Graph()
        g.add_nodes_from(molecule_ids)
        g.add_edges_from(edges)
        return nx.is_connected(g)

    def contact_graph_signature(self, cand: ClusterCandidate) -> tuple[Any, ...]:
        edges = []
        for c in cand.contacts:
            a = (c.feature_a.molecule_id, c.feature_a.type)
            b = (c.feature_b.molecule_id, c.feature_b.type)
            pair = tuple(sorted((a, b)))
            edges.append((pair[0], pair[1], c.contact_label))
        return tuple(sorted(edges))

    def mode_label(self, cand: ClusterCandidate, pose: PoseMetadata | None) -> str:
        graph_sig = self.contact_graph_signature(cand)
        if pose is None:
            raw = ("seed", graph_sig)
        else:
            raw = (
                pose.primary_contact_label,
                pose.primary_existing_feature,
                pose.primary_new_feature,
                pose.distance_bin,
                round(pose.twist_deg, 3),
                tuple(round(x, 3) for x in pose.offset),
                graph_sig,
            )
        return _hash_tuple(raw)

    def multicontact_score(self, cand: ClusterCandidate) -> float:
        by_molecule: dict[int, int] = {}
        for c in cand.contacts:
            by_molecule[c.feature_a.molecule_id] = by_molecule.get(c.feature_a.molecule_id, 0) + 1
            by_molecule[c.feature_b.molecule_id] = by_molecule.get(c.feature_b.molecule_id, 0) + 1
        return float(sum(math.log1p(n) for n in by_molecule.values()))

    def saturation_score(self, cand: ClusterCandidate) -> float:
        # Fraction of features that are involved in at least one contact, averaged by molecule.
        involved: dict[int, set[str]] = {}
        total: dict[int, int] = {}
        for m in cand.monomers:
            total[m.molecule_id] = len(m.features)
        for c in cand.contacts:
            involved.setdefault(c.feature_a.molecule_id, set()).add(c.feature_a.feature_id)
            involved.setdefault(c.feature_b.molecule_id, set()).add(c.feature_b.feature_id)
        vals = []
        for mid, n in total.items():
            if n:
                vals.append(len(involved.get(mid, set())) / n)
        return float(np.mean(vals)) if vals else 0.0

    def contact_score(self, cand: ClusterCandidate) -> float:
        return float(sum(c.score for c in cand.contacts))

    def compactness_score(self, cand: ClusterCandidate) -> float:
        return 1.0 / (1.0 + cand.rg)

    def novelty_score(self, cand: ClusterCandidate) -> float:
        return 1.0 if cand.mode_label not in self.seen_modes else 0.0

    @staticmethod
    def _is_sulfuric_feature(feature_type: str) -> bool:
        return feature_type in {"SA_OH", "SA_O", "SA_OH_acceptor"}

    @staticmethod
    def _is_base_feature(feature_type: str) -> bool:
        return feature_type in {"nitrogen_base", "amine_N", "ammonia_base", "methylamine_base", "dimethylamine_base", "trimethylamine_base"}

    @staticmethod
    def _is_carboxyl_feature(feature_type: str) -> bool:
        return feature_type.startswith("carboxyl") or feature_type == "carboxylic_acid"

    @staticmethod
    def _is_organic_feature(feature_type: str) -> bool:
        organic_types = {
            "carboxylic_acid",
            "carboxyl_OH",
            "carboxyl_CeqO",
            "carbonyl_O",
            "ester_O",
            "ester_region",
            "alkyl_region",
            "alcohol_OH",
            "hydroperoxide_OOH",
            "aldehyde_O",
            "acid_anhydride_O",
        }
        return feature_type in organic_types or feature_type.startswith("carboxyl")

    def molecule_roles(self, cand: ClusterCandidate) -> dict[int, set[str]]:
        roles: dict[int, set[str]] = {}
        for feature in cand.features:
            mid_roles = roles.setdefault(feature.molecule_id, set())
            if self._is_sulfuric_feature(feature.type):
                mid_roles.add("sulfuric")
            if self._is_base_feature(feature.type):
                mid_roles.add("base")
            if self._is_organic_feature(feature.type):
                mid_roles.add("organic")
        return roles

    def bridge_score(self, cand: ClusterCandidate) -> float:
        """Reward OOM-like molecules that bridge sulfuric acid and a base."""
        roles = self.molecule_roles(cand)
        if not any("sulfuric" in role for role in roles.values()) or not any("base" in role for role in roles.values()):
            return 0.0

        contacts_by_molecule: dict[int, set[str]] = {}
        for contact in cand.contacts:
            a_mid = contact.feature_a.molecule_id
            b_mid = contact.feature_b.molecule_id
            for mid, other_mid in ((a_mid, b_mid), (b_mid, a_mid)):
                other_roles = roles.get(other_mid, set())
                if "sulfuric" in other_roles:
                    contacts_by_molecule.setdefault(mid, set()).add("sulfuric")
                if "base" in other_roles:
                    contacts_by_molecule.setdefault(mid, set()).add("base")

        score = 0.0
        for mid, touched_roles in contacts_by_molecule.items():
            if "organic" in roles.get(mid, set()) and {"sulfuric", "base"} <= touched_roles:
                score += 1.0
        return score

    def sulfuric_saturation_score(self, cand: ClusterCandidate) -> float:
        """Reward sulfuric acid OH/O features that participate in contacts."""
        total_by_molecule: dict[int, set[str]] = {}
        involved_by_molecule: dict[int, set[str]] = {}
        for feature in cand.features:
            if self._is_sulfuric_feature(feature.type):
                total_by_molecule.setdefault(feature.molecule_id, set()).add(feature.feature_id)
        for contact in cand.contacts:
            for feature in (contact.feature_a, contact.feature_b):
                if self._is_sulfuric_feature(feature.type):
                    involved_by_molecule.setdefault(feature.molecule_id, set()).add(feature.feature_id)
        fractions = []
        for mid, feature_ids in total_by_molecule.items():
            if feature_ids:
                fractions.append(len(involved_by_molecule.get(mid, set())) / len(feature_ids))
        return float(np.mean(fractions)) if fractions else 0.0

    def carboxyl_multisite_score(self, cand: ClusterCandidate) -> float:
        """Reward multiple carboxyl-related sites on the same molecule being used."""
        involved: dict[int, set[str]] = {}
        for contact in cand.contacts:
            for feature in (contact.feature_a, contact.feature_b):
                if self._is_carboxyl_feature(feature.type):
                    involved.setdefault(feature.molecule_id, set()).add(feature.feature_id)
        return float(sum(max(0, len(feature_ids) - 1) for feature_ids in involved.values()))

    def score_candidate(self, cand: ClusterCandidate) -> float:
        cfg = self.config
        return (
            cfg.w_contact * self.contact_score(cand)
            + cfg.w_multicontact * self.multicontact_score(cand)
            + cfg.w_saturation * self.saturation_score(cand)
            + cfg.w_novelty * self.novelty_score(cand)
            + cfg.w_compact * self.compactness_score(cand)
            - cfg.w_shape * cand.shape_anisotropy
            - cfg.w_orientation * cand.orientation_order
            + cfg.w_bridge * self.bridge_score(cand)
            + cfg.w_sa_saturation * self.sulfuric_saturation_score(cand)
            + cfg.w_carboxyl_multisite * self.carboxyl_multisite_score(cand)
            - cfg.w_clash * cand.clash_score
        )

    def diverse_select(
        self,
        candidates: list[ClusterCandidate],
        k: int,
        record_selection: bool = True,
    ) -> list[ClusterCandidate]:
        candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        selected: list[ClusterCandidate] = []
        per_mode: dict[str, int] = {}
        local_signatures: set[str] = set()
        for cand in candidates:
            if per_mode.get(cand.mode_label, 0) >= self.config.max_per_mode:
                continue
            sig = self._dedup_signature(cand)
            if sig in local_signatures:
                continue
            was_novel = cand.mode_label not in self.seen_modes
            selected.append(cand)
            local_signatures.add(sig)
            per_mode[cand.mode_label] = per_mode.get(cand.mode_label, 0) + 1
            if record_selection:
                self.seen_modes[cand.mode_label] = self.seen_modes.get(cand.mode_label, 0) + 1
                self.coverage.recent_unique.append(1 if was_novel else 0)
            if len(selected) >= k:
                break
        return selected

    def _dedup_signature(self, cand: ClusterCandidate) -> str:
        if cand.features:
            fpos = np.vstack([f.position for f in cand.features])
            hist = distance_histogram(fpos, bins=12, max_distance=25.0)
        else:
            hist = ()
        raw = (
            cand.mode_label,
            self.contact_graph_signature(cand),
            hist,
            round(cand.rg, 1),
        )
        return _hash_tuple(raw)

    def sample_multimer(self, monomers: list[Monomer]) -> list[ClusterCandidate]:
        if not monomers:
            return []
        # Ensure unique molecule ids and COM-shifted internal coordinates.
        prepared = []
        for idx, m in enumerate(monomers):
            prepared.append(Monomer(m.name, m.atoms, m.features, molecule_id=idx).shifted_to_com(molecule_id=idx))
        beam = [self.initial_candidate(prepared[0], molecule_id=0)]
        for idx, monomer in enumerate(prepared[1:], start=1):
            new_beam: list[ClusterCandidate] = []
            prune_threshold = max(self.config.beam_width * 4, self.config.max_candidates * 2)
            retained_after_prune = max(self.config.beam_width * 2, self.config.max_candidates)
            for partial in beam:
                branch_child_limit = max(8, math.ceil(prune_threshold / len(beam)))
                branch_attempt_limit = min(
                    self.config.max_attempts_per_cluster,
                    max(100, branch_child_limit * 20),
                )
                children = self.place_monomer(
                    partial,
                    monomer,
                    new_molecule_id=idx,
                    child_limit=branch_child_limit,
                    attempt_limit=branch_attempt_limit,
                )
                new_beam.extend(children)
                if len(new_beam) >= prune_threshold:
                    new_beam = self.diverse_select(
                        new_beam,
                        retained_after_prune,
                        record_selection=False,
                    )
            if not new_beam:
                break
            beam = self.diverse_select(new_beam, self.config.beam_width)
        return self.diverse_select(beam, self.config.max_candidates)

    def export(self, candidates: list[ClusterCandidate], output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = []
        for i, cand in enumerate(candidates):
            cid = f"cand_{i:04d}"
            write_xyz(output_dir / f"{cid}.xyz", cand.atoms, comment=f"score={cand.score:.4f} mode={cand.mode_label}")
            metadata = {
                "candidate_id": cid,
                "score": cand.score,
                "mode_label": cand.mode_label,
                "rg": cand.rg,
                "shape_anisotropy": cand.shape_anisotropy,
                "orientation_order": cand.orientation_order,
                "clash_score": cand.clash_score,
                "molecule_graph_edges": cand.molecule_graph_edges,
                "contacts": [
                    {
                        "feature_a": f"{c.feature_a.molecule_id}:{c.feature_a.feature_id}:{c.feature_a.type}",
                        "feature_b": f"{c.feature_b.molecule_id}:{c.feature_b.feature_id}:{c.feature_b.type}",
                        "label": c.contact_label,
                        "distance": c.distance,
                        "angle_deg": c.angle_deg,
                        "score": c.score,
                    }
                    for c in cand.contacts
                ],
                "last_pose": cand.metadata.get("last_pose"),
            }
            write_json(output_dir / f"{cid}.json", metadata)
            summary.append({
                "candidate_id": cid,
                "score": cand.score,
                "mode_label": cand.mode_label,
                "rg": cand.rg,
                "shape_anisotropy": cand.shape_anisotropy,
                "orientation_order": cand.orientation_order,
                "clash_score": cand.clash_score,
                "n_contacts": len(cand.contacts),
            })
        write_json(output_dir / "coverage_report.json", {
            "attempt_counts": self.coverage.counts,
            "valid_counts": self.coverage.valid_counts,
            "n_unique_modes": len(self.seen_modes),
        })
        # Simple CSV without pandas.
        csv_lines = ["candidate_id,score,mode_label,rg,shape_anisotropy,orientation_order,clash_score,n_contacts"]
        for row in summary:
            csv_lines.append(
                f"{row['candidate_id']},{row['score']:.8f},{row['mode_label']},{row['rg']:.8f},"
                f"{row['shape_anisotropy']:.8f},{row['orientation_order']:.8f},"
                f"{row['clash_score']:.8f},{row['n_contacts']}"
            )
        (output_dir / "summary.csv").write_text("\n".join(csv_lines) + "\n")
