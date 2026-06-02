from __future__ import annotations

import concurrent.futures
import argparse
import json
import random
import sys
import shutil
from pathlib import Path
from typing import Any
import multiprocessing as mp
import time
import os

# Make script runnable from source tree without installation.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fg_vector_sampler import ClusterSampler, SamplerConfig
from fg_vector_sampler.core import Atom, Feature, Monomer
from fg_vector_sampler.features import atom_centered_features
from fg_vector_sampler.molecule_lib import json_safe, read_xyz, write_json
from fg_vector_sampler.molecule_lib import build_cluster_config, list_available_monomers
from fg_vector_sampler.postprocess import (
    deduplicate_rmsd,
    filter_by_contacts,
    filter_by_rg,
    keep_best_by_contacts,
    export_with_classification,
)
from fg_vector_sampler.analysis import (
    analyze_contact_network,
    analyze_structural_diversity,
    analyze_coverage,
    generate_report,
    write_summary_csv,
)


DEFAULT_CONFIG_PATH = ROOT / "config.json"
_MONOMER_TEMPLATE_CACHE: dict[str, Monomer] = {}


def get_cpu_count() -> dict[str, int]:
    """Detect available CPU resources."""
    try:
        physical_count = mp.cpu_count()
    except NotImplementedError:
        physical_count = 1
    
    # Try to get CPU affinity if available (Linux/Unix only)
    available_count = physical_count
    try:
        if hasattr(os, 'sched_getaffinity'):
            available_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    
    return {
        "physical": physical_count,
        "available": available_count,
    }


def _clone_monomer_template(template: Monomer, molecule_id: int, name: str | None = None) -> Monomer:
    """Create a lightweight copy of a cached monomer template with a new molecule id."""
    cloned_name = name or template.name
    atoms = [Atom(a.element, a.coord.copy(), a.mass, a.vdw_radius, molecule_id) for a in template.atoms]
    features: list[Feature] = []
    for feature in template.features:
        feature_id = feature.feature_id.replace(template.name, cloned_name, 1)
        features.append(
            Feature(
                feature_id=feature_id,
                molecule_id=molecule_id,
                type=feature.type,
                local_position=feature.local_position.copy(),
                local_direction=None if feature.local_direction is None else feature.local_direction.copy(),
                normal=None if feature.normal is None else feature.normal.copy(),
                atom_indices=feature.atom_indices,
                weight=feature.weight,
                global_position=None if feature.global_position is None else feature.global_position.copy(),
                global_direction=None if feature.global_direction is None else feature.global_direction.copy(),
            )
        )
    return Monomer(cloned_name, atoms, features, molecule_id)


def _get_monomer_template(path: Path, molecule_id: int, name: str | None = None) -> Monomer:
    """Load and cache a monomer template, then clone it for the requested molecule id."""
    key = str(path.resolve())
    template = _MONOMER_TEMPLATE_CACHE.get(key)
    if template is None:
        loaded = read_xyz(path, name=path.stem, molecule_id=0)
        loaded = loaded.shifted_to_com(molecule_id=0)
        template = atom_centered_features(loaded)
        _MONOMER_TEMPLATE_CACHE[key] = template
    return _clone_monomer_template(template, molecule_id=molecule_id, name=name or path.stem)


def load_runtime_config(config_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not config_path.exists():
        return {}, {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {}) or {}
    presets = data.get("_presets", {}) or {}
    return defaults, presets


def load_monomers_from_molecule_specs(
    molecule_specs: list[Any],
    rng: random.Random,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Load monomer objects from MoleculeSpec list, sampling conformer pools."""
    all_monomers = []
    selected_conformers: list[dict[str, Any]] = []
    for spec in molecule_specs:
        if not spec.enabled:
            continue
        conformer_pool = list(getattr(spec, "conformer_pool", None) or [])
        if not conformer_pool:
            conformer_pool = [spec.file]
        conformer_pool = [str(path) for path in conformer_pool if path]
        if not conformer_pool:
            raise ValueError(f"No conformers available for molecule '{spec.name}'")

        for count_id in range(spec.count):
            mol_id = len(all_monomers)
            selected_path = Path(rng.choice(conformer_pool))
            conformer_name = selected_path.stem
            monomer_name = f"{spec.name}_{count_id}_{conformer_name}"
            monomer = _get_monomer_template(selected_path, molecule_id=mol_id, name=monomer_name)
            all_monomers.append(monomer)
            selected_conformers.append(
                {
                    "molecule_id": mol_id,
                    "molecule": spec.name,
                    "copy_index": count_id,
                    "selected_file": str(selected_path),
                    "pool_size": len(conformer_pool),
                }
            )
    return all_monomers, selected_conformers


def load_monomers_from_folder(folder: Path, n_monomers: int | None = None):
    """Legacy function: load monomers from folder by file order."""
    xyz_files = sorted(folder.glob("*.xyz"))
    if n_monomers is not None:
        xyz_files = xyz_files[:n_monomers]
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {folder}")

    monomers = []
    for molecule_id, xyz_path in enumerate(xyz_files):
        monomer = _get_monomer_template(xyz_path, molecule_id=molecule_id, name=xyz_path.stem)
        monomers.append(monomer)
    return monomers


def describe_folder_monomers(folder: Path, n_monomers: int | None = None) -> list[dict[str, Any]]:
    """Describe the fixed XYZ inputs used by legacy folder mode."""
    xyz_files = sorted(folder.glob("*.xyz"))
    if n_monomers is not None:
        xyz_files = xyz_files[:n_monomers]
    return [
        {
            "molecule_id": molecule_id,
            "molecule": xyz_path.stem,
            "copy_index": 0,
            "selected_file": str(xyz_path),
            "pool_size": 1,
        }
        for molecule_id, xyz_path in enumerate(xyz_files)
    ]


def collect_exported_xyz_files(
    output_dir: Path,
    folder_name: str = "all_xyz",
) -> Path:
    """Collect only the final exported xyz files into a single flat subfolder."""
    collected_dir = output_dir / folder_name
    collected_dir.mkdir(parents=True, exist_ok=True)

    source_xyz_paths = [p for p in output_dir.rglob("*.xyz") if collected_dir not in p.parents]

    provenance_manifest: dict[str, Any] = {}
    for xyz_path in source_xyz_paths:
        if xyz_path.parent == collected_dir:
            continue
        relative_parent = xyz_path.parent.relative_to(output_dir)
        safe_prefix = str(relative_parent).replace("\\", "__").replace("/", "__")
        target_name = f"{safe_prefix}__{xyz_path.name}" if safe_prefix != "." else xyz_path.name
        shutil.copy2(xyz_path, collected_dir / target_name)
        metadata_path = xyz_path.with_suffix(".json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            provenance_manifest[target_name] = metadata.get("source_conformers", [])

    write_json(
        collected_dir / "source_conformers.json",
        {"structures": provenance_manifest},
    )

    return collected_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FG-vector sampling: energy-free molecular cluster pre-sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sampling.py                           # Simple run with defaults
  python run_sampling.py --preset dimer --n-monomers 2        # Dimer sampling
  python run_sampling.py --use "PT,H2SO4,HNO3" --counts "PT=2,H2SO4=1,HNO3=1"
  python run_sampling.py --parallel-jobs 4 --preset balanced  # Parallel jobs (4 jobs)
  python run_sampling.py --parallel-jobs auto                  # Auto-detect CPU cores
  python run_sampling.py --list-monomers                       # List available molecules
        """
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="JSON config file with defaults and presets")
    parser.add_argument("--preset", default=None, help="Preset name from config.json (e.g., balanced, compact, loose, dimer)")
    parser.add_argument("--list-monomers", action="store_true", help="List all available monomers and exit")
    parser.add_argument("--use", type=str, default=None, help="Comma-separated molecule names (e.g., 'PT,H2SO4,HNO3')")
    parser.add_argument("--counts", type=str, default=None, help="Molecule counts (e.g., 'PT=2,H2SO4=1')")
    parser.add_argument("--input-dir", default=None, help="Folder containing input .xyz files")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--n-monomers", type=int, default=None, help="Use only first N monomer files (for simple mode)")
    parser.add_argument("--max-candidates", type=int, default=None, help="Maximum candidates to keep")
    parser.add_argument("--beam-width", type=int, default=None, help="Beam width during sampling")
    parser.add_argument("--max-attempts-per-cluster", type=int, default=None, help="Maximum placement attempts per cluster")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument(
        "--parallel-jobs",
        type=str,
        default=None,
        help="Run multiple independent jobs in parallel (use 'auto' to auto-detect CPU cores, or an integer)"
    )
    parser.add_argument("--rg-max", type=float, default=None, help="Maximum radius of gyration")
    parser.add_argument("--max-shape-anisotropy", type=float, default=None, help="Maximum COM shape anisotropy for >=4 molecule clusters")
    parser.add_argument("--max-orientation-order", type=float, default=None, help="Maximum molecular orientation order for >=4 molecule clusters")
    parser.add_argument("--w-shape", type=float, default=None, help="Penalty weight for non-spherical COM arrangements")
    parser.add_argument("--w-orientation", type=float, default=None, help="Penalty weight for over-parallel monomer orientations")
    parser.add_argument("--lambda-hard", type=float, default=None, help="Hard clash scaling factor")
    parser.add_argument("--lambda-soft", type=float, default=None, help="Soft overlap scaling factor")
    parser.add_argument("--max-soft-overlap", type=float, default=None, help="Maximum tolerated soft overlap score")
    parser.add_argument("--min-contacts", type=int, default=None, help="Minimum number of contacts required")
    parser.add_argument("--jitter-deg", type=float, default=None, help="Angular jitter in degrees")
    parser.add_argument("--enable-filtering", action="store_true", help="Enable post-sampling filtering")
    parser.add_argument("--rmsd-threshold", type=float, default=None, help="RMSD threshold for deduplication")
    parser.add_argument("--min-contacts-filter", type=int, default=None, help="Minimum contacts for filtering")
    parser.add_argument("--max-rg-filter", type=float, default=None, help="Maximum Rg for filtering")
    parser.add_argument("--keep-best-n", type=int, default=None, help="Keep only the N best candidates by contact count")
    parser.add_argument("--classify-by-type", action="store_true", help="Classify output by contact type")
    parser.add_argument("--xyz-folder-name", type=str, default="all_xyz", help="Folder name used to collect all exported xyz files")
    parser.add_argument("--show-cpu-info", action="store_true", help="Show available CPU information and exit")
    return parser


def parse_counts_argument(counts_str: str) -> dict[str, int]:
    """Parse counts string like 'PT=2,H2SO4=1' into a dict."""
    if not counts_str:
        return {}
    result = {}
    for pair in counts_str.split(","):
        mol, count = pair.strip().split("=")
        result[mol.strip()] = int(count.strip())
    return result


def parse_parallel_jobs(parallel_jobs_arg: str | int | None) -> int | None:
    """Parse parallel jobs argument: 'auto' or integer."""
    if parallel_jobs_arg is None:
        return None
    if isinstance(parallel_jobs_arg, int):
        return parallel_jobs_arg
    if isinstance(parallel_jobs_arg, str):
        if parallel_jobs_arg.lower() == "auto":
            cpu_info = get_cpu_count()
            detected = cpu_info["available"]
            print(f"\n[CPU] Detected {detected} available CPU cores (physical: {cpu_info['physical']})")
            return detected
        else:
            return int(parallel_jobs_arg)
    return None


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    defaults, presets = load_runtime_config(config_path)
    merged: dict[str, Any] = {
        "input_dir": str(ROOT / "monomer"),
        "output_dir": str(ROOT / "results" / "cluster"),
        "n_monomers": None,
        "max_candidates": 30,
        "beam_width": 30,
        "max_attempts_per_cluster": 2000,
        "seed": 12,
        "parallel_jobs": 1,
        "rg_max": 22.0,
        "max_shape_anisotropy": 0.40,
        "max_orientation_order": 0.95,
        "w_shape": 10.0,
        "w_orientation": 8.0,
        "lambda_hard": 0.55,
        "lambda_soft": 0.75,
        "max_soft_overlap": 40.0,
        "min_contacts": 1,
        "jitter_deg": 8.0,
        "enable_filtering": False,
        "rmsd_threshold": None,
        "min_contacts_filter": None,
        "max_rg_filter": None,
        "keep_best_n": None,
        "classify_by_contact_type": False,
        "xyz_folder_name": "all_xyz",
    }
    merged.update(defaults)

    preset_name = args.preset or merged.get("preset")
    if preset_name:
        preset = presets.get(preset_name)
        if preset is None:
            available = ", ".join(sorted(presets)) if presets else "(none)"
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
        merged.update(preset)

    cli_overrides = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "n_monomers": args.n_monomers,
        "max_candidates": args.max_candidates,
        "beam_width": args.beam_width,
        "max_attempts_per_cluster": args.max_attempts_per_cluster,
        "seed": args.seed,
        "parallel_jobs": parse_parallel_jobs(args.parallel_jobs),
        "rg_max": args.rg_max,
        "max_shape_anisotropy": args.max_shape_anisotropy,
        "max_orientation_order": args.max_orientation_order,
        "w_shape": args.w_shape,
        "w_orientation": args.w_orientation,
        "lambda_hard": args.lambda_hard,
        "lambda_soft": args.lambda_soft,
        "max_soft_overlap": args.max_soft_overlap,
        "min_contacts": args.min_contacts,
        "jitter_deg": args.jitter_deg,
        "enable_filtering": args.enable_filtering if args.enable_filtering else None,
        "rmsd_threshold": args.rmsd_threshold,
        "min_contacts_filter": args.min_contacts_filter,
        "max_rg_filter": args.max_rg_filter,
        "keep_best_n": args.keep_best_n,
        "classify_by_contact_type": args.classify_by_type if args.classify_by_type else None,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value

    merged["config_path"] = str(config_path)
    merged["preset"] = preset_name
    return merged


def ensure_beam_width_not_below_candidates(settings: dict[str, Any], requested_beam_width: int | None) -> dict[str, Any]:
    """If beam_width was not explicitly set, keep it at least as large as max_candidates."""
    if requested_beam_width is None:
        max_candidates = settings.get("max_candidates")
        beam_width = settings.get("beam_width")
        if isinstance(max_candidates, int) and isinstance(beam_width, int) and beam_width < max_candidates:
            settings["beam_width"] = max_candidates
    return settings


def run_single_job(job_id: int, settings: dict[str, Any], shared_templates: dict | None = None) -> dict[str, Any]:
    """Run a single sampling job. If shared_templates provided, use it instead of module cache."""
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(settings["seed"]) + job_id - 1
    conformer_rng = random.Random(seed)
    selected_conformers: list[dict[str, Any]] = []
    
    # Load monomers
    if settings.get("molecules"):
        monomers, selected_conformers = load_monomers_from_molecule_specs(
            settings["molecules"],
            rng=conformer_rng,
        )
    else:
        input_dir = Path(settings["input_dir"])
        n_monomers = settings.get("n_monomers")
        monomers = load_monomers_from_folder(input_dir, n_monomers=n_monomers)
        selected_conformers = describe_folder_monomers(input_dir, n_monomers=n_monomers)
    if selected_conformers:
        write_json(output_dir / "selected_conformers.json", {"selected_conformers": selected_conformers})
    
    # Sampling
    config = SamplerConfig(
        max_candidates=int(settings["max_candidates"]),
        beam_width=int(settings["beam_width"]),
        max_attempts_per_cluster=int(settings["max_attempts_per_cluster"]),
        random_seed=seed,
        rg_max=float(settings["rg_max"]),
        max_shape_anisotropy=(
            None
            if settings.get("max_shape_anisotropy") is None
            else float(settings["max_shape_anisotropy"])
        ),
        max_orientation_order=(
            None
            if settings.get("max_orientation_order") is None
            else float(settings["max_orientation_order"])
        ),
        w_shape=float(settings.get("w_shape", 10.0)),
        w_orientation=float(settings.get("w_orientation", 8.0)),
        lambda_hard=float(settings["lambda_hard"]),
        lambda_soft=float(settings["lambda_soft"]),
        max_soft_overlap=float(settings["max_soft_overlap"]),
        min_contacts=int(settings["min_contacts"]),
        jitter_deg=float(settings["jitter_deg"]),
    )
    sampler = ClusterSampler(config=config)
    candidates = sampler.sample_multimer(monomers)
    for candidate in candidates:
        candidate.metadata["source_conformers"] = selected_conformers
    
    # Post-processing
    if settings.get("enable_filtering"):
        if settings.get("rmsd_threshold"):
            candidates = deduplicate_rmsd(candidates, rmsd_threshold=float(settings["rmsd_threshold"]))
        if settings.get("min_contacts_filter"):
            candidates = filter_by_contacts(candidates, min_contacts=int(settings["min_contacts_filter"]))
        if settings.get("max_rg_filter"):
            candidates = filter_by_rg(candidates, max_rg=float(settings["max_rg_filter"]))
        if settings.get("keep_best_n"):
            candidates = keep_best_by_contacts(candidates, n=int(settings["keep_best_n"]))
    
    # Export
    export_with_classification(
        candidates,
        output_dir,
        by_type=settings.get("classify_by_contact_type", False),
    )
    
    # Statistical analysis
    contact_analysis = analyze_contact_network(candidates)
    diversity_analysis = analyze_structural_diversity(candidates)
    coverage_analysis = analyze_coverage(candidates)

    # Write the standard statistical report expected by analysis.py
    generate_report(candidates, output_dir)
    
    summary_csv_path = output_dir / "analysis_summary.csv"
    write_summary_csv(candidates, summary_csv_path)
    
    return {
        "job_id": job_id,
        "seed": seed,
        "output_dir": str(output_dir),
        "n_candidates": len(candidates),
        "n_monomers": len(monomers),
        "selected_conformers": selected_conformers,
        "contact_analysis": contact_analysis,
    }


def run_parallel_jobs_optimized(
    parallel_jobs: int,
    base_output: Path,
    xyz_folder_name: str,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run parallel jobs with optimized monomer loading."""
    # Pre-load all unique monomer templates in main process
    preload_start = time.time()
    input_dir = Path(settings["input_dir"])
    all_xyz_files = sorted(input_dir.glob("*.xyz"))
    
    print(f"\n[PRELOAD] Loading {len(all_xyz_files)} monomer templates in main process...")
    for xyz_path in all_xyz_files:
        key = str(xyz_path.resolve())
        if key not in _MONOMER_TEMPLATE_CACHE:
            template = read_xyz(xyz_path, name=xyz_path.stem, molecule_id=0)
            template = template.shifted_to_com(molecule_id=0)
            template = atom_centered_features(template)
            _MONOMER_TEMPLATE_CACHE[key] = template
    preload_time = time.time() - preload_start
    print(f"  Preload complete: {len(_MONOMER_TEMPLATE_CACHE)} templates in {preload_time:.2f}s")
    
    base_output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for job_id in range(1, parallel_jobs + 1):
        job_output = base_output / f"job_{job_id:02d}"
        job_settings = {**settings, "output_dir": str(job_output)}
        jobs.append((job_id, job_settings))

    print(f"\n[PARALLEL] Running {parallel_jobs} jobs with pre-loaded templates...")
    results: list[dict[str, Any]] = []
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
        future_map = {
            executor.submit(run_single_job, job_id, job_settings): job_id
            for job_id, job_settings in jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            elapsed = time.time() - start_time
            print(
                f"  [job {result['job_id']:02d}] {result['n_candidates']} candidates @ {elapsed:.1f}s"
            )

    # Collect all xyz files from all jobs to base_output/all_xyz/
    print(f"\n[COLLECTING] Gathering xyz files from all jobs...")
    global_collected_dir = collect_exported_xyz_files(
        base_output,
        folder_name=xyz_folder_name,
    )

    summary_path = base_output / "parallel_summary.json"
    summary_path.write_text(
        json.dumps(json_safe(sorted(results, key=lambda x: x["job_id"])), indent=2),
        encoding="utf-8",
    )
    total_candidates = sum(r["n_candidates"] for r in results)
    total_time = time.time() - start_time
    print(f"\n[COMPLETE] {total_candidates} candidates from {parallel_jobs} jobs in {total_time:.1f}s")
    print(f"Summary: {summary_path}")
    print(f"Collected xyz: {global_collected_dir}")
    
    return results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    # Handle --show-cpu-info
    if args.show_cpu_info:
        cpu_info = get_cpu_count()
        print("\n[CPU INFORMATION]")
        print(f"  Physical CPU cores: {cpu_info['physical']}")
        print(f"  Available CPU cores: {cpu_info['available']}")
        print(f"\nRecommended parallel jobs: {cpu_info['available']}")
        print(f"  Example: python run_sampling.py --parallel-jobs {cpu_info['available']} ...")
        return
    
    # Handle --list-monomers
    if args.list_monomers:
        input_dir = args.input_dir or "monomer"
        available = list_available_monomers(input_dir=input_dir)
        if available:
            print("\n[AVAILABLE MONOMERS]")
            for name in available:
                print(f"  {name}")
        else:
            print(f"No monomers found in {input_dir}")
        return
    
    # Resolve molecule configuration
    if args.use:
        molecule_names = [m.strip() for m in args.use.split(",")]
        counts_dict = {}
        if args.counts:
            counts_dict = parse_counts_argument(args.counts)
        try:
            cluster_config = build_cluster_config(
                molecules=molecule_names,
                counts=counts_dict,
                input_dir=args.input_dir or "monomer",
                **{
                    k: v
                    for k, v in vars(args).items()
                    if v is not None
                    and k not in ["config", "preset", "use", "counts", "list_monomers", "input_dir"]
                }
            )
            settings = {
                "molecules": cluster_config.molecules,
                "input_dir": cluster_config.input_dir,
                "output_dir": cluster_config.output_dir,
                "n_monomers": None,
                "max_candidates": int(args.max_candidates)
                if args.max_candidates is not None
                else cluster_config.n_candidates,
                "beam_width": int(args.beam_width)
                if args.beam_width is not None
                else cluster_config.beam_width,
                "max_attempts_per_cluster": int(args.max_attempts_per_cluster)
                if args.max_attempts_per_cluster is not None
                else 2000,
                "seed": cluster_config.seed,
                "parallel_jobs": parse_parallel_jobs(args.parallel_jobs) or cluster_config.parallel_jobs,
                "rg_max": cluster_config.rg_max,
                "max_shape_anisotropy": getattr(cluster_config, "max_shape_anisotropy", 0.40),
                "max_orientation_order": getattr(cluster_config, "max_orientation_order", 0.95),
                "w_shape": getattr(cluster_config, "w_shape", 10.0),
                "w_orientation": getattr(cluster_config, "w_orientation", 8.0),
                "lambda_hard": cluster_config.lambda_hard,
                "lambda_soft": cluster_config.lambda_soft,
                "max_soft_overlap": cluster_config.max_soft_overlap,
                "min_contacts": cluster_config.min_contacts,
                "jitter_deg": cluster_config.jitter_deg,
                "enable_filtering": cluster_config.enable_filtering,
                "rmsd_threshold": cluster_config.rmsd_threshold,
                "min_contacts_filter": cluster_config.min_contacts_filter,
                "max_rg_filter": cluster_config.max_rg_filter,
                "keep_best_n": cluster_config.keep_best_n,
                "classify_by_contact_type": cluster_config.classify_by_contact_type,
                "xyz_folder_name": args.xyz_folder_name,
            }
            settings = ensure_beam_width_not_below_candidates(settings, args.beam_width)
        except ValueError as e:
            print(f"[ERROR] {e}")
            return
    else:
        settings = resolve_settings(args)
        settings = ensure_beam_width_not_below_candidates(settings, args.beam_width)

    parallel_jobs = int(settings.get("parallel_jobs", 1))
    base_output = Path(settings["output_dir"])
    xyz_folder_name = str(settings.get("xyz_folder_name", "all_xyz"))

    if parallel_jobs <= 1:
        result = run_single_job(1, settings)
        # Collect xyz files to base_output/all_xyz/
        collected_dir = collect_exported_xyz_files(
            Path(result["output_dir"]),
            folder_name=xyz_folder_name,
        )
        print(f"\n[COMPLETE] Generated {result['n_candidates']} candidates")
        print(f"Output: {result['output_dir']}")
        print(f"Collected xyz: {collected_dir}")
        return

    # Use optimized parallel execution
    run_parallel_jobs_optimized(parallel_jobs, base_output, xyz_folder_name, settings)


if __name__ == "__main__":
    main()
