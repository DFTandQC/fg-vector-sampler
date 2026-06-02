from __future__ import annotations

from dataclasses import dataclass, field
from .core import ContactTemplate


@dataclass
class FunctionalGroupPrior:
    """Functional-group contact templates and chemical-prior weights.

    The default library is intentionally broad and can be edited for PT/SA/HNO3/SO2.
    Priorities are not energies; they guide geometry-only pre-sampling.
    """

    templates: list[ContactTemplate] = field(default_factory=list)

    @classmethod
    def default(cls) -> "FunctionalGroupPrior":
        t = [
            # High-priority acid/base and sulfuric-acid motifs from OOM cluster studies.
            ContactTemplate(
                "nitrogen_base", "SA_OH", "SA_base_core", 1.35, 2.35, 1.75, 1.25, 125.0, True
            ),
            ContactTemplate(
                "nitrogen_base",
                "acidic_H",
                "acid_base_contact",
                1.40,
                2.45,
                1.80,
                1.05,
                120.0,
                True,
            ),
            ContactTemplate(
                "carboxylic_acid",
                "nitrogen_base",
                "carboxyl_base",
                1.40,
                2.55,
                1.80,
                1.25,
                120.0,
                True,
            ),
            ContactTemplate(
                "carboxylic_acid", "SA_OH", "carboxyl_SA", 1.40, 2.60, 1.85, 1.15, 120.0, True
            ),
            ContactTemplate(
                "carboxylic_acid",
                "SA_O",
                "carboxyl_to_sulfate_O",
                1.45,
                2.70,
                1.90,
                1.10,
                115.0,
                True,
            ),
            ContactTemplate(
                "carboxylic_acid",
                "carboxylic_acid",
                "acid_acid_dimer",
                1.45,
                2.55,
                1.80,
                1.15,
                120.0,
                True,
            ),
            # Explicit sulfuric and nitric acid donor/acceptor contacts.
            ContactTemplate(
                "SA_OH", "carbonyl_O", "SA_to_carbonyl", 1.45, 2.45, 1.80, 1.05, 120.0, True
            ),
            ContactTemplate("SA_OH", "ester_O", "SA_to_ester", 1.50, 2.65, 1.95, 0.95, 110.0, True),
            ContactTemplate(
                "SA_OH", "oxygen_acceptor", "SA_to_O", 1.50, 2.75, 2.00, 0.90, 110.0, True
            ),
            ContactTemplate(
                "acidic_H", "SA_O", "acid_to_sulfate_O", 1.45, 2.65, 1.90, 1.00, 115.0, True
            ),
            ContactTemplate(
                "HNO3_OH", "carbonyl_O", "HNO3_to_carbonyl", 1.50, 2.60, 1.95, 0.85, 110.0, True
            ),
            ContactTemplate(
                "HNO3_OH", "ester_O", "HNO3_to_ester", 1.60, 2.75, 2.05, 0.75, 105.0, True
            ),
            ContactTemplate(
                "HNO3_OH", "SA_O", "HNO3_to_sulfate_O", 1.50, 2.70, 1.95, 0.85, 110.0, True
            ),
            # Generic organic donor/acceptor contacts.
            ContactTemplate(
                "acidic_H", "carbonyl_O", "strong_Hbond", 1.45, 2.45, 1.85, 1.00, 120.0, True
            ),
            ContactTemplate(
                "acidic_H", "ester_O", "Hbond_to_ester", 1.55, 2.65, 1.95, 0.85, 110.0, True
            ),
            ContactTemplate(
                "acidic_H", "oxygen_acceptor", "Hbond_to_O", 1.55, 2.75, 2.00, 0.80, 110.0, True
            ),
            ContactTemplate(
                "hydrogen_donor",
                "oxygen_acceptor",
                "weak_Hbond",
                1.85,
                3.10,
                2.35,
                0.45,
                100.0,
                True,
            ),
            # Weaker proximity templates preserve diversity without dominating selection.
            ContactTemplate(
                "sulfur_center", "ester_region", "SO2_polar_region", 2.8, 4.5, 3.5, 0.55, None, True
            ),
            ContactTemplate(
                "SO2_O", "ester_region", "SO2_ester_region", 2.4, 4.2, 3.2, 0.55, None, True
            ),
            ContactTemplate(
                "ester_region",
                "ester_region",
                "ester_region_contact",
                2.8,
                5.0,
                3.7,
                0.45,
                None,
                True,
            ),
            ContactTemplate(
                "carbonyl_O",
                "carbonyl_O",
                "carbonyl_carbonyl_polar",
                2.8,
                4.5,
                3.6,
                0.25,
                None,
                True,
            ),
            ContactTemplate(
                "alkyl_region",
                "alkyl_region",
                "hydrophobic_contact",
                3.3,
                5.2,
                4.1,
                0.35,
                None,
                True,
            ),
            ContactTemplate(
                "oxygen_acceptor",
                "oxygen_acceptor",
                "O_O_polar_proximity",
                2.8,
                4.6,
                3.7,
                0.20,
                None,
                True,
            ),
        ]
        return cls(t)

    def find_template(self, type_a: str, type_b: str) -> ContactTemplate | None:
        best: ContactTemplate | None = None
        for template in self.templates:
            if template.matches(type_a, type_b):
                if best is None or template.priority > best.priority:
                    best = template
        return best

    def compatible(self, type_a: str, type_b: str) -> bool:
        return self.find_template(type_a, type_b) is not None

    def pair_weight(self, type_a: str, type_b: str) -> float:
        template = self.find_template(type_a, type_b)
        if template is None:
            return 0.0
        return template.priority
