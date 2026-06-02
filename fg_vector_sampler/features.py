from __future__ import annotations

import numpy as np
from .core import Feature, Monomer, normalize


def _nearest_atom_indices(
    monomer: Monomer, atom_index: int, element: str, max_distance: float
) -> list[int]:
    atom = monomer.atoms[atom_index]
    matches: list[tuple[float, int]] = []
    for idx, other in enumerate(monomer.atoms):
        if idx == atom_index or other.element != element:
            continue
        distance = float(np.linalg.norm(atom.coord - other.coord))
        if distance <= max_distance:
            matches.append((distance, idx))
    return [idx for _, idx in sorted(matches)]


def _nearest_atom_index(
    monomer: Monomer, atom_index: int, element: str, max_distance: float
) -> int | None:
    matches = _nearest_atom_indices(monomer, atom_index, element, max_distance)
    return matches[0] if matches else None


def _direction_away_from_atom(
    monomer: Monomer, atom_index: int, neighbor_index: int | None
) -> np.ndarray:
    atom = monomer.atoms[atom_index]
    if neighbor_index is None:
        return normalize(atom.coord)
    return normalize(atom.coord - monomer.atoms[neighbor_index].coord)


def atom_centered_features(monomer: Monomer) -> Monomer:
    """Create atom-centered functional-group features from XYZ geometry.

    This remains a lightweight fallback, but it recognizes common atmospheric
    cluster motifs used by the sampler: sulfuric/nitric acid OH donors, sulfate
    and nitrate acceptors, carbonyl/ester oxygens, and generic polar atoms.
    """
    features: list[Feature] = []
    name = monomer.name.lower()
    is_sa = any(token in name for token in ("h2so4", "sulfuric", "cissa", "transsa", "sa_"))
    is_hno3 = any(token in name for token in ("hno3", "nitric"))
    is_so2 = "so2" in name

    for idx, atom in enumerate(monomer.atoms):
        element = atom.element
        if element == "O":
            bonded_h = _nearest_atom_indices(monomer, idx, "H", 1.25)
            nearest_c = _nearest_atom_index(monomer, idx, "C", 1.65)
            nearest_n = _nearest_atom_index(monomer, idx, "N", 1.70)
            nearest_s = _nearest_atom_index(monomer, idx, "S", 1.90)

            if is_sa:
                ftype = "SA_OH_acceptor" if bonded_h else "SA_O"
                weight = 0.7 if bonded_h else 1.0
                direction = _direction_away_from_atom(monomer, idx, nearest_s)
            elif is_hno3:
                ftype = "HNO3_OH_acceptor" if bonded_h else "HNO3_O"
                weight = 0.6 if bonded_h else 0.8
                direction = _direction_away_from_atom(monomer, idx, nearest_n)
            elif is_so2:
                ftype = "SO2_O"
                weight = 0.7
                direction = _direction_away_from_atom(monomer, idx, nearest_s)
            elif nearest_c is not None:
                c_distance = float(np.linalg.norm(atom.coord - monomer.atoms[nearest_c].coord))
                ftype = "carbonyl_O" if c_distance <= 1.30 else "ester_O"
                weight = 1.0 if ftype == "carbonyl_O" else 0.85
                direction = _direction_away_from_atom(monomer, idx, nearest_c)
            else:
                ftype = "oxygen_acceptor"
                weight = 0.8
                direction = normalize(atom.coord)

            features.append(
                Feature(
                    feature_id=f"{monomer.name}_{idx}_{ftype}",
                    molecule_id=monomer.molecule_id,
                    type=ftype,
                    local_position=atom.coord.copy(),
                    local_direction=direction,
                    atom_indices=(idx,),
                    weight=weight,
                )
            )
            if ftype != "oxygen_acceptor":
                features.append(
                    Feature(
                        feature_id=f"{monomer.name}_{idx}_oxygen_acceptor",
                        molecule_id=monomer.molecule_id,
                        type="oxygen_acceptor",
                        local_position=atom.coord.copy(),
                        local_direction=direction,
                        atom_indices=(idx,),
                        weight=0.45,
                    )
                )
            if ftype in {"carbonyl_O", "ester_O"}:
                features.append(
                    Feature(
                        feature_id=f"{monomer.name}_{idx}_ester_region",
                        molecule_id=monomer.molecule_id,
                        type="ester_region",
                        local_position=atom.coord.copy(),
                        local_direction=direction,
                        atom_indices=(idx,),
                        weight=0.45,
                    )
                )
        elif element == "N":
            if is_hno3 or "no2" in name or "no" in name:
                ftype = "nitrate_center" if is_hno3 else "nitrogen_oxide_center"
                weight = 0.2
            else:
                ftype = "nitrogen_base"
                weight = 0.9
            features.append(
                Feature(
                    feature_id=f"{monomer.name}_{idx}_{ftype}",
                    molecule_id=monomer.molecule_id,
                    type=ftype,
                    local_position=atom.coord.copy(),
                    local_direction=normalize(atom.coord),
                    atom_indices=(idx,),
                    weight=weight,
                )
            )
        elif element == "S":
            ftype = "sulfur_center"
            features.append(
                Feature(
                    feature_id=f"{monomer.name}_{idx}_{ftype}",
                    molecule_id=monomer.molecule_id,
                    type=ftype,
                    local_position=atom.coord.copy(),
                    local_direction=normalize(atom.coord),
                    atom_indices=(idx,),
                    weight=0.6,
                )
            )
        elif element == "H":
            nearest_o = _nearest_atom_index(monomer, idx, "O", 1.25)
            nearest_n = _nearest_atom_index(monomer, idx, "N", 1.25)
            nearest_s = _nearest_atom_index(monomer, idx, "S", 1.55)
            donor_parent = (
                nearest_o
                if nearest_o is not None
                else nearest_n if nearest_n is not None else nearest_s
            )
            if donor_parent is None:
                continue
            direction = normalize(atom.coord - monomer.atoms[donor_parent].coord)
            if is_sa and nearest_o is not None:
                ftype = "SA_OH"
                weight = 1.15
            elif is_hno3 and nearest_o is not None:
                ftype = "HNO3_OH"
                weight = 0.95
            elif nearest_o is not None:
                ftype = "acidic_H"
                weight = 0.85
            else:
                ftype = "hydrogen_donor"
                weight = 0.45
            features.append(
                Feature(
                    feature_id=f"{monomer.name}_{idx}_{ftype}",
                    molecule_id=monomer.molecule_id,
                    type=ftype,
                    local_position=atom.coord.copy(),
                    local_direction=direction,
                    atom_indices=(idx,),
                    weight=weight,
                )
            )
            if ftype != "hydrogen_donor":
                features.append(
                    Feature(
                        feature_id=f"{monomer.name}_{idx}_hydrogen_donor",
                        molecule_id=monomer.molecule_id,
                        type="hydrogen_donor",
                        local_position=atom.coord.copy(),
                        local_direction=direction,
                        atom_indices=(idx,),
                        weight=0.25,
                    )
                )
    return Monomer(monomer.name, monomer.atoms, features, monomer.molecule_id)


def feature_from_atom_indices(
    monomer: Monomer,
    feature_id: str,
    feature_type: str,
    atom_indices: list[int] | tuple[int, ...],
    direction_indices: tuple[int, int] | None = None,
    weight: float = 1.0,
) -> Feature:
    """Build a feature center from selected atoms.

    direction_indices=(i,j) creates a local direction r_j - r_i.
    """
    coords = np.vstack([monomer.atoms[i].coord for i in atom_indices])
    center = coords.mean(axis=0)
    direction = None
    if direction_indices is not None:
        i, j = direction_indices
        direction = normalize(monomer.atoms[j].coord - monomer.atoms[i].coord)
    return Feature(
        feature_id=feature_id,
        molecule_id=monomer.molecule_id,
        type=feature_type,
        local_position=center,
        local_direction=direction,
        atom_indices=tuple(atom_indices),
        weight=weight,
    )


def attach_features(monomer: Monomer, features: list[Feature]) -> Monomer:
    return Monomer(monomer.name, monomer.atoms, features, monomer.molecule_id)
