import json
import tempfile
import unittest
from pathlib import Path

from fg_vector_sampler.core import Atom
from fg_vector_sampler.postprocess import export_with_classification
from fg_vector_sampler.sampler import ClusterCandidate
from run_sampling import collect_exported_xyz_files


class ConformerProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.source_conformers = [
            {
                "molecule_id": 0,
                "molecule": "PT",
                "copy_index": 0,
                "selected_file": "monomer/PT_conf_003.xyz",
                "pool_size": 10,
            },
            {
                "molecule_id": 1,
                "molecule": "H2SO4",
                "copy_index": 0,
                "selected_file": "monomer/opt-cisSA-B97-3c.xyz",
                "pool_size": 1,
            },
        ]

    def test_candidate_json_and_xyz_comment_include_source_conformers(self):
        candidate = ClusterCandidate(
            monomers=[],
            score=1.25,
            metadata={"source_conformers": self.source_conformers},
        )
        candidate_atoms = [Atom("H", [0.0, 0.0, 0.0])]
        candidate.monomers = [type("MonomerStub", (), {"atoms": candidate_atoms, "features": []})()]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            export_with_classification([candidate], output_dir, by_type=False)

            metadata = json.loads((output_dir / "cand_0000.json").read_text(encoding="utf-8"))
            xyz_lines = (output_dir / "cand_0000.xyz").read_text(encoding="utf-8").splitlines()

        self.assertEqual(metadata["source_conformers"], self.source_conformers)
        self.assertIn("sources=PT_conf_003.xyz,opt-cisSA-B97-3c.xyz", xyz_lines[1])

    def test_flat_xyz_collection_writes_source_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            job_dir = output_dir / "job_01"
            job_dir.mkdir()
            (job_dir / "cand_0000.xyz").write_text("0\nexample\n", encoding="utf-8")
            (job_dir / "cand_0000.json").write_text(
                json.dumps({"source_conformers": self.source_conformers}),
                encoding="utf-8",
            )

            collected_dir = collect_exported_xyz_files(output_dir)
            manifest = json.loads((collected_dir / "source_conformers.json").read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["structures"]["job_01__cand_0000.xyz"],
            self.source_conformers,
        )


if __name__ == "__main__":
    unittest.main()
