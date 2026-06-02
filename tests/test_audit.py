import unittest

import numpy as np

from fg_vector_sampler.audit import summarize_template_coverage
from fg_vector_sampler.core import ContactTemplate, Feature, Monomer
from fg_vector_sampler.priors import FunctionalGroupPrior


class TemplateCoverageTests(unittest.TestCase):
    @staticmethod
    def monomer_with_feature(name, molecule_id, feature_type):
        return Monomer(
            name=name,
            atoms=[],
            features=[
                Feature(
                    feature_id=f"{name}_{feature_type}",
                    molecule_id=molecule_id,
                    type=feature_type,
                    local_position=np.zeros(3),
                )
            ],
            molecule_id=molecule_id,
        )

    def test_reports_available_and_missing_template_types(self):
        prior = FunctionalGroupPrior(
            [
                ContactTemplate("donor", "acceptor", "hydrogen_bond", 1.5, 2.5, 2.0),
                ContactTemplate("carboxyl", "acceptor", "missing_carboxyl", 1.5, 2.5, 2.0),
            ]
        )
        monomers = [
            self.monomer_with_feature("donor", 0, "donor"),
            self.monomer_with_feature("acceptor", 1, "acceptor"),
        ]

        report = summarize_template_coverage(monomers, prior)

        self.assertEqual(report["n_available_templates"], 1)
        self.assertEqual(report["n_unavailable_templates"], 1)
        self.assertEqual(report["unavailable_templates"][0]["missing_types"], ["carboxyl"])


if __name__ == "__main__":
    unittest.main()
