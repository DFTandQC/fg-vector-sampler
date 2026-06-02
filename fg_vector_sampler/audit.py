from __future__ import annotations

from collections import Counter
from typing import Any

from .core import Monomer
from .priors import FunctionalGroupPrior


def summarize_template_coverage(
    monomers: list[Monomer],
    prior: FunctionalGroupPrior | None = None,
) -> dict[str, Any]:
    """Summarize which contact templates can be exercised by the loaded feature library."""
    prior = prior or FunctionalGroupPrior.default()
    feature_counts = Counter(feature.type for monomer in monomers for feature in monomer.features)
    template_rows = []
    for template in prior.templates:
        missing_types = sorted(
            feature_type
            for feature_type in {template.type_a, template.type_b}
            if feature_counts.get(feature_type, 0) == 0
        )
        template_rows.append(
            {
                "label": template.contact_label,
                "type_a": template.type_a,
                "type_b": template.type_b,
                "available": not missing_types,
                "missing_types": missing_types,
            }
        )

    available_templates = [row for row in template_rows if row["available"]]
    unavailable_templates = [row for row in template_rows if not row["available"]]
    return {
        "n_monomers": len(monomers),
        "feature_counts": dict(sorted(feature_counts.items())),
        "n_available_templates": len(available_templates),
        "n_unavailable_templates": len(unavailable_templates),
        "available_templates": available_templates,
        "unavailable_templates": unavailable_templates,
    }
