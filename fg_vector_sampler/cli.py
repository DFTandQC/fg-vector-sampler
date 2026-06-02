from __future__ import annotations

import argparse
from pathlib import Path
from .molecule_lib import read_xyz
from .features import atom_centered_features
from .sampler import ClusterSampler, SamplerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy-free FG-vector molecular cluster sampler")
    parser.add_argument("--xyz", nargs="+", required=True, help="Input XYZ monomer files")
    parser.add_argument("--names", nargs="*", default=None, help="Optional monomer names")
    parser.add_argument("--output", default="outputs/sampling", help="Output directory")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    names = args.names or [Path(p).stem for p in args.xyz]
    monomers = []
    for i, path in enumerate(args.xyz):
        m = read_xyz(path, name=names[i] if i < len(names) else Path(path).stem, molecule_id=i)
        m = m.shifted_to_com(molecule_id=i)
        m = atom_centered_features(m)
        monomers.append(m)

    config = SamplerConfig(
        max_candidates=args.max_candidates, beam_width=args.beam_width, random_seed=args.seed
    )
    sampler = ClusterSampler(config=config)
    candidates = sampler.sample_multimer(monomers)
    sampler.export(candidates, args.output)
    print(f"Generated {len(candidates)} candidates in {args.output}")


if __name__ == "__main__":
    main()
