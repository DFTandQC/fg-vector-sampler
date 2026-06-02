import unittest

import numpy as np

from fg_vector_sampler.core import Atom, ContactTemplate, Feature, Monomer
from fg_vector_sampler.features import atom_centered_features
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

    def test_carbonyl_oxygen_points_away_from_bonded_carbon(self):
        monomer = Monomer(
            "carbonyl",
            [
                Atom("C", [0.0, 0.0, 0.0]),
                Atom("O", [1.2, 0.0, 0.0]),
                Atom("C", [0.0, 4.0, 0.0]),
            ],
        )

        featured = atom_centered_features(monomer)
        carbonyl = next(feature for feature in featured.features if feature.type == "carbonyl_O")

        np.testing.assert_allclose(carbonyl.local_direction, np.array([1.0, 0.0, 0.0]))

    def test_sulfuric_oxygen_points_away_from_sulfur(self):
        monomer = Monomer(
            "cisSA",
            [
                Atom("S", [0.0, 0.0, 0.0]),
                Atom("O", [0.0, 1.4, 0.0]),
                Atom("O", [3.0, 0.0, 0.0]),
            ],
        )

        featured = atom_centered_features(monomer)
        sulfate_oxygen = next(feature for feature in featured.features if feature.type == "SA_O")

        np.testing.assert_allclose(sulfate_oxygen.local_direction, np.array([0.0, 1.0, 0.0]))


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
