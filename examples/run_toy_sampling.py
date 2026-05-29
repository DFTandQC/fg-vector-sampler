"""Compatibility shim: toy example utilities.

Use `examples/example_workflow.py` for runnable demonstrations.
"""
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fg_vector_sampler.core import Atom, Monomer, Feature
from fg_vector_sampler.sampler import ClusterSampler, SamplerConfig
from fg_vector_sampler.features import atom_centered_features
from fg_vector_sampler.molecule_lib import read_xyz

__all__ = ["make_toy_pt", "make_toy_sa", "make_toy_hno3"]


def load_monomers_from_folder(folder: Path, n_monomers: int | None = None):
    """Load monomers from folder by file order."""
    xyz_files = sorted(folder.glob("*.xyz"))
    if n_monomers is not None:
        xyz_files = xyz_files[:n_monomers]
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {folder}")

    monomers = []
    for molecule_id, xyz_path in enumerate(xyz_files):
        monomer = read_xyz(xyz_path, name=xyz_path.stem, molecule_id=molecule_id)
        monomer = monomer.shifted_to_com(molecule_id=molecule_id)
        monomer = atom_centered_features(monomer)
        monomers.append(monomer)
    return monomers


def make_toy_pt(molecule_id: int = 0) -> Monomer:
    """Very small pseudo-PT toy molecule with four ester-like polar arms and alkyl regions."""
    atoms = [Atom("C", np.array([0.0, 0.0, 0.0]), molecule_id=molecule_id)]
    # Four polar O atoms around a central C, plus outer C as alkyl region proxies.
    positions = [
        np.array([2.2, 0.0, 0.0]),
        np.array([-2.2, 0.0, 0.0]),
        np.array([0.0, 2.2, 0.0]),
        np.array([0.0, -2.2, 0.0]),
    ]
    for p in positions:
        atoms.append(Atom("O", p, molecule_id=molecule_id))
        atoms.append(Atom("C", p + 1.3 * p / np.linalg.norm(p), molecule_id=molecule_id))
    features = []
    for idx, p in enumerate(positions):
        features.append(Feature(f"carbonyl_O_{idx}", molecule_id, "carbonyl_O", p, p, atom_indices=(1 + 2*idx,), weight=1.0))
        features.append(Feature(f"ester_region_{idx}", molecule_id, "ester_region", p * 1.05, p, atom_indices=(1 + 2*idx, 2 + 2*idx), weight=0.8))
        features.append(Feature(f"alkyl_{idx}", molecule_id, "alkyl_region", p * 1.65, p, atom_indices=(2 + 2*idx,), weight=0.5))
    return Monomer("toy_PT", atoms, features, molecule_id).shifted_to_com(molecule_id)


def make_toy_sa(molecule_id: int = 1) -> Monomer:
    atoms = [
        Atom("S", np.array([0.0, 0.0, 0.0]), molecule_id=molecule_id),
        Atom("O", np.array([1.2, 0.0, 0.0]), molecule_id=molecule_id),
        Atom("O", np.array([-1.2, 0.0, 0.0]), molecule_id=molecule_id),
        Atom("O", np.array([0.0, 1.2, 0.0]), molecule_id=molecule_id),
        Atom("H", np.array([0.0, 2.1, 0.0]), molecule_id=molecule_id),
        Atom("O", np.array([0.0, -1.2, 0.0]), molecule_id=molecule_id),
        Atom("H", np.array([0.0, -2.1, 0.0]), molecule_id=molecule_id),
    ]
    features = [
        Feature("SA_OH_1", molecule_id, "SA_OH", np.array([0.0, 2.1, 0.0]), np.array([0.0, 1.0, 0.0]), atom_indices=(4,), weight=1.0),
        Feature("SA_OH_2", molecule_id, "SA_OH", np.array([0.0, -2.1, 0.0]), np.array([0.0, -1.0, 0.0]), atom_indices=(6,), weight=1.0),
        Feature("SO_acceptor_1", molecule_id, "oxygen_acceptor", np.array([1.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), atom_indices=(1,), weight=0.8),
        Feature("SO_acceptor_2", molecule_id, "oxygen_acceptor", np.array([-1.2, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), atom_indices=(2,), weight=0.8),
    ]
    return Monomer("toy_SA", atoms, features, molecule_id).shifted_to_com(molecule_id)


def make_toy_hno3(molecule_id: int = 2) -> Monomer:
    atoms = [
        Atom("N", np.array([0.0, 0.0, 0.0]), molecule_id=molecule_id),
        Atom("O", np.array([1.1, 0.0, 0.0]), molecule_id=molecule_id),
        Atom("O", np.array([-0.6, 1.0, 0.0]), molecule_id=molecule_id),
        Atom("O", np.array([-0.6, -1.0, 0.0]), molecule_id=molecule_id),
        Atom("H", np.array([-1.2, -1.7, 0.0]), molecule_id=molecule_id),
    ]
    features = [
        Feature("HNO3_OH", molecule_id, "HNO3_OH", np.array([-1.2, -1.7, 0.0]), np.array([-0.6, -0.7, 0.0]), atom_indices=(4,), weight=0.9),
        Feature("NO_acceptor", molecule_id, "oxygen_acceptor", np.array([1.1, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), atom_indices=(1,), weight=0.6),
    ]
    return Monomer("toy_HNO3", atoms, features, molecule_id).shifted_to_com(molecule_id)


if __name__ == "__main__":
    monomer_dir = ROOT / "monomer"
    monomers = load_monomers_from_folder(monomer_dir)
    if not monomers:
        monomers = [make_toy_pt(0), make_toy_sa(1), make_toy_hno3(2)]

    config = SamplerConfig(
        max_candidates=30,
        beam_width=30,
        rg_max=22.0,
        lambda_hard=0.55,
        lambda_soft=0.75,
        max_soft_overlap=40.0,
        random_seed=12,
    )
    sampler = ClusterSampler(config=config)
    candidates = sampler.sample_multimer(monomers)
    out = ROOT / "outputs" / "monomer"
    sampler.export(candidates, out)
    print(f"Generated {len(candidates)} candidates from monomer inputs")
    print(f"Output directory: {out}")
