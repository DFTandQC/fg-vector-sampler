"""
Statistical analysis and reporting for sampling results.

Generates detailed summaries of contact patterns, structural diversity,
and mode coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING
from collections import Counter
import numpy as np

if TYPE_CHECKING:
    from .sampler import ClusterCandidate


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def analyze_contact_network(candidates: list[ClusterCandidate]) -> dict[str, Any]:
    """Analyze inter-molecular contact patterns across all candidates."""
    all_contacts = []
    contact_types = Counter()
    contact_distances = []
    contact_pairs = Counter()
    
    for cand in candidates:
        for contact in cand.contacts:
            all_contacts.append(contact)
            contact_types[contact.contact_label] += 1
            contact_distances.append(contact.distance)
            pair = tuple(sorted((contact.feature_a.type, contact.feature_b.type)))
            contact_pairs[pair] += 1
    
    if not all_contacts:
        return {
            "total_contacts": 0,
            "contact_types": {},
            "contact_pairs": {},
            "distance_stats": {},
        }
    
    distances = np.array(contact_distances)
    return {
        "total_contacts": len(all_contacts),
        "contact_types": dict(contact_types.most_common()),
        "contact_pairs": dict(contact_pairs.most_common()),
        "distance_stats": {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
            "median": float(np.median(distances)),
        },
    }


def analyze_structural_diversity(candidates: list[ClusterCandidate]) -> dict[str, Any]:
    """Analyze structural diversity metrics."""
    if not candidates:
        return {
            "n_candidates": 0,
            "rg_mean": 0.0,
            "rg_std": 0.0,
            "rg_min": 0.0,
            "rg_max": 0.0,
            "score_mean": 0.0,
            "score_std": 0.0,
            "score_min": 0.0,
            "score_max": 0.0,
            "rg_distribution": {},
            "score_distribution": {},
            "shape_anisotropy_mean": 0.0,
            "orientation_order_mean": 0.0,
            "mode_count": 0,
        }
    
    rgs = np.array([c.rg for c in candidates])
    scores = np.array([c.score for c in candidates])
    shape_anisotropies = np.array([getattr(c, "shape_anisotropy", 0.0) for c in candidates])
    orientation_orders = np.array([getattr(c, "orientation_order", 0.0) for c in candidates])
    unique_modes = len(set(c.mode_label for c in candidates))
    
    # Binned distributions
    rg_bins = np.linspace(rgs.min(), rgs.max(), 11)
    rg_hist, _ = np.histogram(rgs, bins=rg_bins)
    
    score_bins = np.linspace(scores.min(), scores.max(), 11)
    score_hist, _ = np.histogram(scores, bins=score_bins)
    
    return {
        "n_candidates": len(candidates),
        "rg_mean": float(np.mean(rgs)),
        "rg_std": float(np.std(rgs)),
        "rg_min": float(np.min(rgs)),
        "rg_max": float(np.max(rgs)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "shape_anisotropy_mean": float(np.mean(shape_anisotropies)),
        "shape_anisotropy_std": float(np.std(shape_anisotropies)),
        "shape_anisotropy_min": float(np.min(shape_anisotropies)),
        "shape_anisotropy_max": float(np.max(shape_anisotropies)),
        "orientation_order_mean": float(np.mean(orientation_orders)),
        "orientation_order_std": float(np.std(orientation_orders)),
        "orientation_order_min": float(np.min(orientation_orders)),
        "orientation_order_max": float(np.max(orientation_orders)),
        "rg_distribution": {
            "mean": float(np.mean(rgs)),
            "std": float(np.std(rgs)),
            "min": float(np.min(rgs)),
            "max": float(np.max(rgs)),
            "bins": [float(b) for b in rg_bins],
            "histogram": [int(h) for h in rg_hist],
        },
        "score_distribution": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "bins": [float(b) for b in score_bins],
            "histogram": [int(h) for h in score_hist],
        },
        "unique_modes": unique_modes,
    }


def analyze_coverage(candidates: list[ClusterCandidate]) -> dict[str, Any]:
    """Analyze contact mode coverage and saturation."""
    mode_counts = Counter(c.mode_label for c in candidates)
    contact_type_signatures = Counter()
    
    for cand in candidates:
        contact_types = tuple(sorted(set(c.contact_label for c in cand.contacts)))
        contact_type_signatures[contact_types] += 1
    
    return {
        "total_modes": len(mode_counts),
        "mode_sizes": {
            "min": min(mode_counts.values()) if mode_counts else 0,
            "max": max(mode_counts.values()) if mode_counts else 0,
            "mean": float(np.mean(list(mode_counts.values()))) if mode_counts else 0,
        },
        "most_common_modes": dict(mode_counts.most_common(10)),
        "contact_type_signatures": dict(contact_type_signatures.most_common(10)),
    }


def generate_report(
    candidates: list[ClusterCandidate],
    output_dir: Path | str,
    title: str = "Cluster Sampling Report",
) -> None:
    """Generate a comprehensive statistical report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "title": title,
        "summary": {
            "total_candidates": len(candidates),
            "score_range": (
                (float(min(c.score for c in candidates)), float(max(c.score for c in candidates)))
                if candidates
                else (0, 0)
            ),
            "rg_range": (
                (float(min(c.rg for c in candidates)), float(max(c.rg for c in candidates)))
                if candidates
                else (0, 0)
            ),
            "shape_anisotropy_range": (
                (
                    float(min(getattr(c, "shape_anisotropy", 0.0) for c in candidates)),
                    float(max(getattr(c, "shape_anisotropy", 0.0) for c in candidates)),
                )
                if candidates
                else (0, 0)
            ),
            "orientation_order_range": (
                (
                    float(min(getattr(c, "orientation_order", 0.0) for c in candidates)),
                    float(max(getattr(c, "orientation_order", 0.0) for c in candidates)),
                )
                if candidates
                else (0, 0)
            ),
        },
        "contact_analysis": analyze_contact_network(candidates),
        "structural_diversity": analyze_structural_diversity(candidates),
        "coverage_analysis": analyze_coverage(candidates),
    }
    
    report_path = output_dir / "statistical_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")


def write_summary_csv(candidates: list[ClusterCandidate], output_path: Path | str) -> None:
    """Write a simple CSV summary of all candidates."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "id,score,rg,shape_anisotropy,orientation_order,n_contacts,clash_score,mode_label",
    ]
    for i, cand in enumerate(candidates):
        lines.append(
            f"{i:04d},{cand.score:.8f},{cand.rg:.8f},"
            f"{getattr(cand, 'shape_anisotropy', 0.0):.8f},"
            f"{getattr(cand, 'orientation_order', 0.0):.8f},"
            f"{len(cand.contacts)},{cand.clash_score:.8f},{cand.mode_label}"
        )
    
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
