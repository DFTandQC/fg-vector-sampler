#!/usr/bin/env python3
"""
Example: Complete FG-Vector Sampling Workflow
Demonstrates all major features: basic sampling, filtering, classification, and analysis.
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fg_vector_sampler import ClusterSampler, SamplerConfig
from fg_vector_sampler.features import atom_centered_features
from fg_vector_sampler.molecule_lib import read_xyz, write_xyz
import numpy as np
from fg_vector_sampler.core import Atom, Monomer, Feature
from fg_vector_sampler.postprocess import (
    deduplicate_rmsd,
    filter_by_contacts,
    export_with_classification,
)
from fg_vector_sampler.analysis import (
    analyze_contact_network,
    analyze_structural_diversity,
    generate_report,
)


def create_synthetic_water_dimer():
    """Create synthetic water molecules for testing without external files."""
    from fg_vector_sampler.core import Atom, Monomer
    
    # Water 1 (centered at origin)
    atoms_w1 = [
        Atom(element="O", coord=(0.0, 0.0, 0.0)),
        Atom(element="H", coord=(0.96, 0.0, 0.0)),
        Atom(element="H", coord=(-0.24, 0.93, 0.0)),
    ]
    water1 = Monomer(name="water_0", atoms=atoms_w1, molecule_id=0)
    water1 = water1.shifted_to_com(molecule_id=0)
    water1 = atom_centered_features(water1)
    
    # Water 2 (offset)
    atoms_w2 = [
        Atom(element="O", coord=(5.0, 0.0, 0.0)),
        Atom(element="H", coord=(5.96, 0.0, 0.0)),
        Atom(element="H", coord=(4.76, 0.93, 0.0)),
    ]
    water2 = Monomer(name="water_1", atoms=atoms_w2, molecule_id=1)
    water2 = water2.shifted_to_com(molecule_id=1)
    water2 = atom_centered_features(water2)
    
    return [water1, water2]


# --- Toy molecule generators (merged from examples/run_toy_sampling.py) ---
def make_toy_pt(molecule_id: int = 0) -> Monomer:
    atoms = [Atom("C", np.array([0.0, 0.0, 0.0]), molecule_id=molecule_id)]
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


def example_basic_sampling():
    """Example 1: Basic sampling without filtering."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Dimer Sampling")
    print("="*70)
    
    monomers = create_synthetic_water_dimer()
    print(f"✓ Loaded {len(monomers)} monomers")
    
    config = SamplerConfig(
        max_candidates=10,
        beam_width=20,
        rg_max=15.0,
        random_seed=42,
    )
    print(f"✓ Config: max_candidates={config.max_candidates}, beam_width={config.beam_width}")
    
    sampler = ClusterSampler(config=config)
    candidates = sampler.sample_multimer(monomers)
    print(f"✓ Generated {len(candidates)} candidates")
    
    if candidates:
        top = candidates[0]
        print(f"  - Top structure: {top.n_atoms} atoms, {top.n_contacts} contacts")
        print(f"  - Rg: {top.rg:.2f} Å")


def example_with_filtering():
    """Example 2: Sampling with RMSD deduplication and contact filtering."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Sampling with Post-Processing Filters")
    print("="*70)
    
    monomers = create_synthetic_water_dimer()
    print(f"✓ Loaded {len(monomers)} monomers")
    
    config = SamplerConfig(
        max_candidates=15,
        beam_width=25,
        rg_max=15.0,
        random_seed=42,
    )
    sampler = ClusterSampler(config=config)
    candidates = sampler.sample_multimer(monomers)
    print(f"✓ Generated {len(candidates)} initial candidates")
    
    # RMSD deduplication
    candidates_dedup = deduplicate_rmsd(candidates, rmsd_threshold=0.5)
    print(f"✓ After RMSD dedup (threshold=0.5Å): {len(candidates_dedup)} candidates")
    
    # Contact filtering
    candidates_filtered = filter_by_contacts(candidates_dedup, min_contacts=1)
    print(f"✓ After contact filter (min=1): {len(candidates_filtered)} candidates")
    
    if candidates_filtered:
        top = candidates_filtered[0]
        print(f"  - Top structure: {top.n_atoms} atoms, {top.n_contacts} contacts")


def example_classification():
    """Example 3: Export with contact-type classification."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Classification by Contact Type")
    print("="*70)
    
    monomers = create_synthetic_water_dimer()
    
    config = SamplerConfig(
        max_candidates=8,
        beam_width=20,
        random_seed=42,
    )
    sampler = ClusterSampler(config=config)
    candidates = sampler.sample_multimer(monomers)
    
    # Classify by primary contact type
    from collections import defaultdict
    by_type = defaultdict(list)
    
    for cand in candidates:
        if cand.contact_types:
            primary_type = cand.contact_types[0]
        else:
            primary_type = "no_contacts"
        by_type[primary_type].append(cand)
    
    print(f"✓ Classified {len(candidates)} candidates by contact type:")
    for contact_type, cands in sorted(by_type.items()):
        print(f"  - {contact_type}: {len(cands)} structures")


def example_statistical_analysis():
    """Example 4: Statistical analysis of ensemble."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Statistical Analysis")
    print("="*70)
    
    monomers = create_synthetic_water_dimer()
    
    config = SamplerConfig(
        max_candidates=20,
        beam_width=30,
        random_seed=42,
    )
    sampler = ClusterSampler(config=config)
    candidates = sampler.sample_multimer(monomers)
    print(f"✓ Generated {len(candidates)} candidates")
    
    # Analyze
    contact_analysis = analyze_contact_network(candidates)
    diversity_analysis = analyze_structural_diversity(candidates)
    
    print("\n[Contact Network Analysis]")
    print(f"  Total contacts: {contact_analysis['total_contacts']}")
    print(f"  Average contacts per structure: {contact_analysis['total_contacts'] / len(candidates):.1f}")
    
    print("\n[Structural Diversity]")
    print(f"  Rg: {diversity_analysis['rg_mean']:.2f} ± {diversity_analysis['rg_std']:.2f} Å")
    print(f"       range: [{diversity_analysis['rg_min']:.2f}, {diversity_analysis['rg_max']:.2f}] Å")
    print(f"  vdW score: {diversity_analysis['score_mean']:.3f} ± {diversity_analysis['score_std']:.3f}")


def example_full_workflow():
    """Example 5: Complete workflow from scratch."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Full Workflow (Sampling → Filtering → Analysis)")
    print("="*70)
    
    # Setup
    monomers = create_synthetic_water_dimer()
    output_dir = Path("outputs") / "example_workflow"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Input: {len(monomers)} monomers")
    print(f"✓ Output: {output_dir}")
    
    # Sampling
    config = SamplerConfig(
        max_candidates=25,
        beam_width=30,
        random_seed=42,
    )
    sampler = ClusterSampler(config=config)
    candidates = sampler.sample_multimer(monomers)
    print(f"✓ Step 1 - Sampling: {len(candidates)} candidates")
    
    # Post-processing
    candidates = deduplicate_rmsd(candidates, rmsd_threshold=0.5)
    print(f"✓ Step 2 - RMSD dedup: {len(candidates)} candidates")
    
    candidates = filter_by_contacts(candidates, min_contacts=1)
    print(f"✓ Step 3 - Contact filter: {len(candidates)} candidates")
    
    # Export
    export_with_classification(candidates, output_dir, by_type=True)
    print(f"✓ Step 4 - Export: files written to {output_dir}")
    
    # Analysis
    contact_analysis = analyze_contact_network(candidates)
    diversity_analysis = analyze_structural_diversity(candidates)

    generate_report(candidates, output_dir, title="Example Workflow Report")

    report_path = output_dir / "statistical_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"✓ Step 5 - Analysis: report written to {report_path}")
    
    print(f"\n[Summary]")
    print(f"  Total candidates: {len(candidates)}")
    print(f"  Mean Rg: {diversity_analysis['rg_mean']:.2f} Å")
    print(f"  Mean contacts: {contact_analysis['total_contacts'] / len(candidates):.1f}")


def main():
    print("\n" + "=" * 70)
    print("FG-Vector Sampler - Example Workflow Demonstrations")
    print("=" * 70)
    
    try:
        example_basic_sampling()
        example_with_filtering()
        example_classification()
        example_statistical_analysis()
        example_full_workflow()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Place your optimized monomers in monomer/")
        print("  2. Run: python run_sampling.py --list-monomers")
        print("  3. Run: python run_sampling.py --use \"MOL1,MOL2\" --counts \"MOL1=2,MOL2=1\"")
        print("\nFor more information, see USAGE_GUIDE.md")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
