import unittest
from pathlib import Path

import fg_vector_sampler
from fg_vector_sampler.cli import main as cli_main
from fg_vector_sampler.sampler import SamplerConfig


class PackageMetadataTests(unittest.TestCase):
    def test_top_level_exports_are_available(self):
        self.assertTrue(hasattr(fg_vector_sampler, "ClusterSampler"))
        self.assertTrue(hasattr(fg_vector_sampler, "FunctionalGroupPrior"))
        self.assertTrue(hasattr(fg_vector_sampler, "SamplerConfig"))

    def test_sampler_config_defaults_are_stable(self):
        config = SamplerConfig()
        self.assertEqual(config.max_candidates, 100)
        self.assertEqual(config.beam_width, 30)
        self.assertEqual(config.random_seed, 7)

    def test_cli_entrypoint_is_callable(self):
        self.assertTrue(callable(cli_main))

    def test_repository_documents_exist(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "CITATION.cff").exists())
        self.assertTrue((root / "CHANGELOG.md").exists())
        self.assertTrue((root / "docs" / "index.rst").exists())


if __name__ == "__main__":
    unittest.main()
