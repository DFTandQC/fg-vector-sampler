"""
Molecule library and multi-molecule cluster configuration system.

Supports flexible specification of molecular components with counts,
automatic monomer file resolution, and preset configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import json
import re
import numpy as np
from .core import Atom, Monomer


@dataclass
class MoleculeSpec:
    """Specification for a molecular component in a cluster."""

    name: str  # Name of molecule type (e.g., "PT", "H2SO4", "HNO3")
    file: Optional[str] = None  # Path to monomer XYZ file
    conformer_pool: list[str] = field(
        default_factory=list
    )  # Alternative conformers for this molecule type
    count: int = 1  # Number of this molecule in each cluster
    enabled: bool = True  # Whether this molecule participates in sampling


@dataclass
class ClusterConfigSpec:
    """Configuration for multi-molecule cluster generation."""

    input_dir: str = "monomer"
    output_dir: str = "results/cluster"
    molecules: list[MoleculeSpec] = field(default_factory=list)
    n_candidates: int = 30
    beam_width: int = 30
    seed: int = 12
    parallel_jobs: int = 1
    rg_max: float = 22.0
    max_shape_anisotropy: Optional[float] = 0.40
    max_orientation_order: Optional[float] = 0.95
    w_shape: float = 10.0
    w_orientation: float = 8.0
    lambda_hard: float = 0.55
    lambda_soft: float = 0.75
    max_soft_overlap: float = 40.0
    min_contacts: int = 1
    jitter_deg: float = 8.0

    # Post-processing
    enable_filtering: bool = False
    rmsd_threshold: Optional[float] = None
    min_contacts_filter: Optional[int] = None
    max_rg_filter: Optional[float] = None
    keep_best_n: Optional[int] = None
    classify_by_contact_type: bool = False


def _normalize_molecule_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def _clean_monomer_stem(stem: str) -> str:
    stem = re.sub(r"^(opt_|opt-|optimized_|optimized-)", "", stem, flags=re.IGNORECASE)
    return re.sub(
        r"(-B97-3c|-B3LYP|-PBE|_B97-3c|_B3LYP|_PBE).*$",
        "",
        stem,
        flags=re.IGNORECASE,
    )


def _matches_molecule_name(stem: str, name: str) -> bool:
    cleaned = _clean_monomer_stem(stem).lower()
    normalized_cleaned = _normalize_molecule_name(cleaned)
    normalized_name = _normalize_molecule_name(name)
    if normalized_cleaned == normalized_name:
        return True
    # Treat PT_conf_001.xyz / PT-001.xyz / PT-conformer-a.xyz as PT conformers.
    return any(
        cleaned.startswith(f"{name.lower()}{sep}") for sep in ("-", "_", ".", "conf", "conformer")
    )


def resolve_monomer_files(
    name: str, input_dir: str = "monomer", file_hint: Optional[str] = None
) -> list[str]:
    """
    Resolve all conformer files for a molecule name.

    Supported layouts:
    1. Explicit file_hint pointing to one .xyz file.
    2. Explicit file_hint or input_dir/name pointing to a folder of .xyz files.
    3. Flat input_dir files named like PT.xyz, opt-PT-B97-3c.xyz, PT_conf_001.xyz.

    Returns absolute paths sorted for reproducible seeded sampling.
    """
    if file_hint:
        hinted = Path(file_hint)
        if hinted.is_file():
            return [str(hinted.absolute())]
        if hinted.is_dir():
            files = sorted(hinted.glob("*.xyz"))
            if files:
                return [str(path.absolute()) for path in files]

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    folder_candidates = []
    direct_folder = input_path / name
    if direct_folder.is_dir():
        folder_candidates.extend(sorted(direct_folder.glob("*.xyz")))
    for child in sorted(p for p in input_path.iterdir() if p.is_dir()):
        if _normalize_molecule_name(child.name) == _normalize_molecule_name(name):
            folder_candidates.extend(sorted(child.glob("*.xyz")))
    if folder_candidates:
        unique = sorted({path.resolve() for path in folder_candidates})
        return [str(path) for path in unique]

    xyz_files = sorted(input_path.glob("*.xyz"))
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {input_dir}")

    candidates = []
    for xyz_file in xyz_files:
        if not _matches_molecule_name(xyz_file.stem, name):
            continue
        filename_lower = xyz_file.stem.lower()
        cleaned = _clean_monomer_stem(xyz_file.stem)
        exact = int(_normalize_molecule_name(cleaned) == _normalize_molecule_name(name))
        optimized = int("opt" in filename_lower or "optimized" in filename_lower)
        candidates.append((exact, optimized, xyz_file))

    if not candidates:
        available = [f.stem for f in xyz_files]
        raise FileNotFoundError(
            f"No monomer file found for '{name}' in {input_dir}\n"
            f"Available files: {', '.join(available)}"
        )

    candidates.sort(key=lambda x: (-x[0], -x[1], str(x[2]).lower()))
    return [str(path.absolute()) for _, _, path in candidates]


def resolve_monomer_file(
    name: str, input_dir: str = "monomer", file_hint: Optional[str] = None
) -> str:
    """
    Resolve the actual monomer file path.

    Tries:
    1. Explicit file_hint if it exists
    2. Scan input_dir for matching files

    Raises FileNotFoundError if no match found.
    """
    return resolve_monomer_files(name, input_dir=input_dir, file_hint=file_hint)[0]


def list_available_monomers(input_dir: str = "monomer") -> list[str]:
    """List all available monomer names in the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        return []

    xyz_files = sorted(input_path.glob("*.xyz"))
    names = []
    for child in sorted(p for p in input_path.iterdir() if p.is_dir()):
        if any(child.glob("*.xyz")):
            names.append(f"{child.name}/ ({len(list(child.glob('*.xyz')))} conformers)")

    for xyz_file in xyz_files:
        filename = xyz_file.stem
        # Try to clean up filename
        filename_clean = _clean_monomer_stem(filename)
        names.append(filename_clean)

    return names


def load_config_from_json(config_path: Path | str) -> dict[str, Any]:
    """Load runtime configuration from JSON file."""
    config_path = Path(config_path)
    if not config_path.exists():
        return {}

    return json.loads(config_path.read_text(encoding="utf-8"))


# --- IO helpers (merged from fg_vector_sampler.io) ---
def read_xyz(path: str | Path, name: str | None = None, molecule_id: int = 0) -> Monomer:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    n = int(lines[0].strip())
    atoms: list[Atom] = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        if len(parts) < 4:
            continue
        element = parts[0]
        coord = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        atoms.append(Atom(element, coord, molecule_id=molecule_id))
    return Monomer(name or path.stem, atoms, [], molecule_id=molecule_id)


def write_xyz(path: str | Path, atoms: list[Atom], comment: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(atoms)), comment]
    for atom in atoms:
        x, y, z = atom.coord
        lines.append(f"{atom.element:2s} {x:14.8f} {y:14.8f} {z:14.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_source_conformer_comment(source_conformers: list[dict[str, Any]]) -> str:
    """Format a compact XYZ comment fragment for the monomer conformers used."""
    filenames = [
        Path(str(source["selected_file"])).name
        for source in source_conformers
        if source.get("selected_file")
    ]
    return f"sources={','.join(filenames)}" if filenames else ""


def json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalar types
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes, dict, list, tuple)):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    return obj


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def build_cluster_config(
    molecules: list[str] | None = None,
    counts: dict[str, int] | None = None,
    preset: str | None = None,
    **kwargs: Any,
) -> ClusterConfigSpec:
    """Build a cluster configuration from flexible specification.

    Args:
        molecules: List of molecule names, e.g., ["PT", "H2SO4"]
        counts: Dict mapping molecule name to count, e.g., {"PT": 2, "H2SO4": 1}
        preset: Preset name from config.json
        **kwargs: Override any field in ClusterConfigSpec

    Returns:
        Configured ClusterConfigSpec instance
    """
    input_dir = kwargs.pop("input_dir", "monomer")

    # Load defaults and presets
    config_path = Path("config.json")
    if config_path.exists():
        config_data = load_config_from_json(config_path)
        defaults = config_data.get("defaults", {})
    else:
        defaults = {}

    # Avoid passing input_dir twice when it is provided explicitly above.
    defaults = {k: v for k, v in defaults.items() if k != "input_dir"}

    # Create base config from defaults
    config = ClusterConfigSpec(
        input_dir=input_dir,
        **{k: v for k, v in defaults.items() if k in ClusterConfigSpec.__dataclass_fields__},
    )

    # Apply preset if specified
    if preset:
        config_path = Path("config.json")
        if config_path.exists():
            config_data = load_config_from_json(config_path)
            presets = config_data.get("_presets", {})
            if preset in presets:
                preset_data = presets[preset]
                for k, v in preset_data.items():
                    if k in ClusterConfigSpec.__dataclass_fields__:
                        setattr(config, k, v)

    # Apply CLI overrides
    for k, v in kwargs.items():
        if v is not None and k in ClusterConfigSpec.__dataclass_fields__:
            setattr(config, k, v)

    # Build molecule list
    mol_specs = []
    if molecules:
        for mol_name in molecules:
            mol_count = (counts or {}).get(mol_name, 1)
            try:
                conformer_pool = resolve_monomer_files(mol_name, input_dir=input_dir)
                mol_specs.append(
                    MoleculeSpec(
                        name=mol_name,
                        file=conformer_pool[0],
                        conformer_pool=conformer_pool,
                        count=mol_count,
                    )
                )
            except FileNotFoundError as e:
                raise ValueError(f"Failed to resolve molecule '{mol_name}': {e}")

    config.molecules = mol_specs
    return config
