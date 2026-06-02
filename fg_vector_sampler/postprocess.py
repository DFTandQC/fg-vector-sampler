"""
Post-processing module for cluster sampling results.

Provides filtering, RMSD-based deduplication, contact-based classification,
and statistical analysis of sampled clusters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING
import numpy as np

from .core import Atom
from .molecule_lib import format_source_conformer_comment, write_xyz, write_json

if TYPE_CHECKING:
    from .sampler import ClusterCandidate


def compute_rmsd(atoms1: list[Atom], atoms2: list[Atom]) -> float | None:
    """Compute RMSD between two structures.

    Returns None if atom counts differ.
    """
    if len(atoms1) != len(atoms2):
        return None

    coords1 = np.array([a.coord for a in atoms1])
    coords2 = np.array([a.coord for a in atoms2])

    # Center on COM
    coords1 -= coords1.mean(axis=0)
    coords2 -= coords2.mean(axis=0)

    # Simple RMSD without rotation optimization
    rmsd = float(np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1))))
    return rmsd


def deduplicate_rmsd(
    candidates: list[ClusterCandidate],
    rmsd_threshold: float = 0.5,
) -> list[ClusterCandidate]:
    """Remove candidates based on RMSD threshold.

    Keeps the first occurrence and removes similar ones.
    """
    if not candidates:
        return []

    kept: list[ClusterCandidate] = [candidates[0]]

    for cand in candidates[1:]:
        is_unique = True
        for kept_cand in kept:
            rmsd = compute_rmsd(cand.atoms, kept_cand.atoms)
            if rmsd is not None and rmsd < rmsd_threshold:
                is_unique = False
                break
        if is_unique:
            kept.append(cand)

    return kept


def filter_by_score(
    candidates: list[ClusterCandidate],
    min_score: float | None = None,
    max_score: float | None = None,
) -> list[ClusterCandidate]:
    """Filter candidates by score range."""
    result = candidates

    if min_score is not None:
        result = [c for c in result if c.score >= min_score]

    if max_score is not None:
        result = [c for c in result if c.score <= max_score]

    return result


def filter_by_contacts(
    candidates: list[ClusterCandidate],
    min_contacts: int | None = None,
    max_contacts: int | None = None,
) -> list[ClusterCandidate]:
    """Filter candidates by number of inter-molecular contacts."""
    result = candidates

    if min_contacts is not None:
        result = [c for c in result if len(c.contacts) >= min_contacts]

    if max_contacts is not None:
        result = [c for c in result if len(c.contacts) <= max_contacts]

    return result


def filter_by_rg(
    candidates: list[ClusterCandidate],
    min_rg: float | None = None,
    max_rg: float | None = None,
) -> list[ClusterCandidate]:
    """Filter candidates by radius of gyration."""
    result = candidates

    if min_rg is not None:
        result = [c for c in result if c.rg >= min_rg]

    if max_rg is not None:
        result = [c for c in result if c.rg <= max_rg]

    return result


def keep_best_by_contacts(candidates: list[ClusterCandidate], n: int) -> list[ClusterCandidate]:
    """Keep only the N candidates with the most contacts."""
    if n <= 0 or len(candidates) == 0:
        return candidates

    sorted_cands = sorted(candidates, key=lambda c: len(c.contacts), reverse=True)
    return sorted_cands[:n]


def classify_by_contact_types(
    candidates: list[ClusterCandidate],
) -> dict[str, list[ClusterCandidate]]:
    """Classify candidates by their dominant contact types.

    Returns a dict mapping contact type signature to list of candidates.
    """
    by_type: dict[str, list[ClusterCandidate]] = {}

    for cand in candidates:
        if not cand.contacts:
            key = "no_contacts"
        else:
            contact_types = sorted(set(c.contact_label for c in cand.contacts))
            key = "|".join(contact_types)

        if key not in by_type:
            by_type[key] = []
        by_type[key].append(cand)

    return by_type


def compute_statistics(candidates: list[ClusterCandidate]) -> dict[str, Any]:
    """Compute aggregate statistics over all candidates."""
    if not candidates:
        return {
            "n_candidates": 0,
            "score": {},
            "rg": {},
            "n_contacts": {},
            "clash_score": {},
            "shape_anisotropy": {},
            "orientation_order": {},
        }

    scores = [c.score for c in candidates]
    rgs = [c.rg for c in candidates]
    contacts = [len(c.contacts) for c in candidates]
    clash_scores = [c.clash_score for c in candidates]
    shape_anisotropies = [getattr(c, "shape_anisotropy", 0.0) for c in candidates]
    orientation_orders = [getattr(c, "orientation_order", 0.0) for c in candidates]

    return {
        "n_candidates": len(candidates),
        "score": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        },
        "rg": {
            "mean": float(np.mean(rgs)),
            "std": float(np.std(rgs)),
            "min": float(np.min(rgs)),
            "max": float(np.max(rgs)),
        },
        "n_contacts": {
            "mean": float(np.mean(contacts)),
            "std": float(np.std(contacts)),
            "min": int(np.min(contacts)),
            "max": int(np.max(contacts)),
        },
        "clash_score": {
            "mean": float(np.mean(clash_scores)),
            "std": float(np.std(clash_scores)),
            "min": float(np.min(clash_scores)),
            "max": float(np.max(clash_scores)),
        },
        "shape_anisotropy": {
            "mean": float(np.mean(shape_anisotropies)),
            "std": float(np.std(shape_anisotropies)),
            "min": float(np.min(shape_anisotropies)),
            "max": float(np.max(shape_anisotropies)),
        },
        "orientation_order": {
            "mean": float(np.mean(orientation_orders)),
            "std": float(np.std(orientation_orders)),
            "min": float(np.min(orientation_orders)),
            "max": float(np.max(orientation_orders)),
        },
    }


def export_with_classification(
    candidates: list[ClusterCandidate],
    output_dir: Path | str,
    by_type: bool = True,
    by_score: bool = False,
) -> dict[str, Any]:
    """Export candidates, optionally organizing by contact type or score range.

    Returns a summary dict with export statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "total_candidates": len(candidates),
        "export_organization": [],
        "statistics": compute_statistics(candidates),
    }

    def export_comment(cand: ClusterCandidate) -> str:
        source_comment = format_source_conformer_comment(cand.metadata.get("source_conformers", []))
        return " ".join(fragment for fragment in (f"score={cand.score:.4f}", source_comment) if fragment)

    def export_metadata(cand: ClusterCandidate) -> dict[str, Any]:
        return {
            "source_conformers": cand.metadata.get("source_conformers", []),
            "last_pose": cand.metadata.get("last_pose"),
        }

    if by_type:
        by_type_dict = classify_by_contact_types(candidates)
        for type_key, cands in sorted(by_type_dict.items()):
            subdir = output_dir / f"type_{type_key.replace('|', '_')}"
            subdir.mkdir(parents=True, exist_ok=True)

            for i, cand in enumerate(cands):
                cid = f"{type_key}_{i:04d}"
                write_xyz(subdir / f"{cid}.xyz", cand.atoms, comment=export_comment(cand))
                metadata = {
                    "id": cid,
                    "type": type_key,
                    "score": cand.score,
                    "rg": cand.rg,
                    "shape_anisotropy": getattr(cand, "shape_anisotropy", 0.0),
                    "orientation_order": getattr(cand, "orientation_order", 0.0),
                    "n_contacts": len(cand.contacts),
                    "contacts": [
                        {
                            "label": c.contact_label,
                            "distance": c.distance,
                        }
                        for c in cand.contacts
                    ],
                    **export_metadata(cand),
                }
                write_json(subdir / f"{cid}.json", metadata)

            summary["export_organization"].append(
                {
                    "type": type_key,
                    "count": len(cands),
                    "directory": str(subdir),
                }
            )
    else:
        for i, cand in enumerate(candidates):
            cid = f"cand_{i:04d}"
            write_xyz(output_dir / f"{cid}.xyz", cand.atoms, comment=export_comment(cand))
            metadata = {
                "id": cid,
                "score": cand.score,
                "rg": cand.rg,
                "shape_anisotropy": getattr(cand, "shape_anisotropy", 0.0),
                "orientation_order": getattr(cand, "orientation_order", 0.0),
                "n_contacts": len(cand.contacts),
                **export_metadata(cand),
            }
            write_json(output_dir / f"{cid}.json", metadata)

    write_json(output_dir / "export_summary.json", summary)
    return summary
