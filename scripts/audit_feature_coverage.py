from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fg_vector_sampler.audit import summarize_template_coverage
from fg_vector_sampler.features import atom_centered_features
from fg_vector_sampler.molecule_lib import read_xyz


def load_featured_monomers(input_dir: Path):
    monomers = []
    for molecule_id, xyz_path in enumerate(sorted(input_dir.glob("*.xyz"))):
        monomer = read_xyz(xyz_path, name=xyz_path.stem, molecule_id=molecule_id)
        monomer = monomer.shifted_to_com(molecule_id=molecule_id)
        monomers.append(atom_centered_features(monomer))
    return monomers


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated feature types against contact templates")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "monomer")
    parser.add_argument("--json", action="store_true", help="Print the complete report as JSON")
    args = parser.parse_args()

    monomers = load_featured_monomers(args.input_dir)
    if not monomers:
        raise SystemExit(f"No .xyz files found in {args.input_dir}")

    report = summarize_template_coverage(monomers)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Monomers: {report['n_monomers']}")
    print("\nGenerated feature types:")
    for feature_type, count in report["feature_counts"].items():
        print(f"  {feature_type}: {count}")

    print(f"\nAvailable templates ({report['n_available_templates']}):")
    for row in report["available_templates"]:
        print(f"  {row['label']}: {row['type_a']} + {row['type_b']}")

    print(f"\nUnavailable templates ({report['n_unavailable_templates']}):")
    for row in report["unavailable_templates"]:
        missing = ", ".join(row["missing_types"])
        print(f"  {row['label']}: missing {missing}")


if __name__ == "__main__":
    main()
