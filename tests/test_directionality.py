import unittest

import numpy as np

from fg_vector_sampler.core import ContactTemplate, Feature
from fg_vector_sampler.priors import FunctionalGroupPrior
from fg_vector_sampler.sampler import ClusterSampler


class FeatureDirectionTests(unittest.TestCase):
    def test_explicit_local_direction_takes_priority_over_radial_fallback(self):
        feature = Feature(
            feature_id="acceptor",
            molecule_id=0,
            type="acceptor",
            local_position=np.array([1.0, 0.0, 0.0]),
            local_direction=np.array([0.0, 1.0, 0.0]),
        )

        np.testing.assert_allclose(feature.outward_local, np.array([0.0, 1.0, 0.0]))


class ContactDirectionTests(unittest.TestCase):
    def setUp(self):
        template = ContactTemplate(
            "donor",
            "acceptor",
            "directional_contact",
            1.5,
            2.5,
            2.0,
            angle_min_deg=120.0,
        )
        self.sampler = ClusterSampler(prior=FunctionalGroupPrior([template]))

    @staticmethod
    def feature(feature_id, molecule_id, feature_type, position, direction):
        return Feature(
            feature_id=feature_id,
            molecule_id=molecule_id,
            type=feature_type,
            local_position=np.asarray(position, dtype=float),
            global_position=np.asarray(position, dtype=float),
            global_direction=np.asarray(direction, dtype=float),
        )

    def test_opposing_outward_vectors_are_accepted(self):
        donor = self.feature("donor", 0, "donor", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        acceptor = self.feature("acceptor", 1, "acceptor", [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0])

        contacts = self.sampler.detect_contacts([donor, acceptor])

        self.assertEqual(len(contacts), 1)
        self.assertAlmostEqual(contacts[0].angle_deg, 180.0)

    def test_parallel_outward_vectors_are_rejected(self):
        donor = self.feature("donor", 0, "donor", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        acceptor = self.feature("acceptor", 1, "acceptor", [2.0, 0.0, 0.0], [1.0, 0.0, 0.0])

        contacts = self.sampler.detect_contacts([donor, acceptor])

        self.assertEqual(contacts, [])


if __name__ == "__main__":
    unittest.main()
