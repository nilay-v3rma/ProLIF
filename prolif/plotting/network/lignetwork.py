"""
Plot a Ligand Interaction Network --- :mod:`prolif.plotting.network`
====================================================================

.. versionadded:: 0.3.2

.. versionchanged:: 2.0.0
    Replaced ``LigNetwork.from_ifp`` with ``LigNetwork.from_fingerprint`` which works
    without requiring a dataframe with atom indices.

.. autoclass:: LigNetwork
   :members:

"""

import json
import operator
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TextIO, Union, cast
from uuid import uuid4

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor

from prolif.exceptions import RunRequiredError
from prolif.ifp import IFP
from prolif.plotting.utils import grouped_interaction_colors, metadata_iterator
from prolif.residue import ResidueId
from prolif.utils import requires

try:
    from IPython.display import Javascript, display
except ModuleNotFoundError:
    pass
else:
    warnings.filterwarnings(
        "ignore",
        "Consider using IPython.display.IFrame instead",  # pragma: no cover
    )

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from prolif.fingerprint import Fingerprint
    from prolif.ifp import IFP


class LigNetwork:
    """Creates a ligand interaction diagram

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with a 4-level index (ligand, protein, interaction, atoms)
        and ``weight`` and ``distance`` columns for values
    lig_mol : rdkit.Chem.rdChem.Mol
        Ligand molecule
    use_coordinates : bool
        If ``True``, uses the coordinates of the molecule directly, otherwise generates
        2D coordinates from scratch. See also ``flatten_coordinates``.
    flatten_coordinates : bool
        If this is ``True`` and ``use_coordinates=True``, generates 2D coordinates that
        are constrained to fit the 3D conformation of the ligand as best as possible.
    kekulize : bool
        Kekulize the ligand
    molsize : int
        Multiply the coordinates by this number to create a bigger and
        more readable depiction
    rotation : float
        Rotate the structure on the XY plane
    carbon : float
        Size of the carbon atom dots on the depiction. Use `0` to hide the
        carbon dots

    Attributes
    ----------
    COLORS : dict
        Dictionnary of colors used in the diagram. Subdivided in several
        dictionaries:

        - "interactions": mapping between interactions types and colors
        - "atoms": mapping between atom symbol and colors
        - "residues": mapping between residues types and colors

    RESIDUE_TYPES : dict
        Mapping between residue names (3 letter code) and types. The types
        are then used to define how each residue should be colored.

    Notes
    -----
    You can customize the diagram by tweaking :attr:`LigNetwork.COLORS` and
    :attr:`LigNetwork.RESIDUE_TYPES` by adding or modifying the
    dictionaries inplace.

    .. versionchanged:: 2.0.0
        Replaced ``LigNetwork.from_ifp`` with ``LigNetwork.from_fingerprint`` which
        works without requiring a dataframe with atom indices. Replaced ``match3D``
        parameter with ``use_coordinates`` and ``flatten_coordinates`` to give users
        more control and allow them to provide their own 2D coordinates. Added support
        for displaying peptides as the "ligand". Changed the default color for
        VanDerWaals.

    .. versionchanged:: 2.1.0
        Added the ``show_interaction_data`` argument and exposed the ``fontsize`` in
        ``display``.
    """

    COLORS: ClassVar = {
        "interactions": {**grouped_interaction_colors},
        "atoms": {
            "C": "black",
            "N": "blue",
            "O": "red",
            "S": "#dece1b",
            "P": "orange",
            "F": "lime",
            "Cl": "lime",
            "Br": "lime",
            "I": "lime",
        },
        "residues": {
            "Aliphatic": "#59e382",
            "Aromatic": "#b559e3",
            "Acidic": "#e35959",
            "Basic": "#5979e3",
            "Polar": "#59bee3",
            "Sulfur": "#e3ce59",
            "Water": "#323aa8",
        },
    }
    RESIDUE_TYPES: ClassVar = {
        "ALA": "Aliphatic",
        "GLY": "Aliphatic",
        "ILE": "Aliphatic",
        "LEU": "Aliphatic",
        "PRO": "Aliphatic",
        "VAL": "Aliphatic",
        "PHE": "Aromatic",
        "TRP": "Aromatic",
        "TYR": "Aromatic",
        "ASP": "Acidic",
        "GLU": "Acidic",
        "ARG": "Basic",
        "HIS": "Basic",
        "HID": "Basic",
        "HIE": "Basic",
        "HIP": "Basic",
        "HSD": "Basic",
        "HSE": "Basic",
        "HSP": "Basic",
        "LYS": "Basic",
        "SER": "Polar",
        "THR": "Polar",
        "ASN": "Polar",
        "GLN": "Polar",
        "CYS": "Sulfur",
        "CYM": "Sulfur",
        "CYX": "Sulfur",
        "MET": "Sulfur",
        "WAT": "Water",
        "SOL": "Water",
        "H2O": "Water",
        "HOH": "Water",
        "OH2": "Water",
        "HHO": "Water",
        "OHH": "Water",
        "TIP": "Water",
        "T3P": "Water",
        "T4P": "Water",
        "T5P": "Water",
        "TIP2": "Water",
        "TIP3": "Water",
        "TIP4": "Water",
    }
    _FONTCOLORS: ClassVar = {
        "Water": "white",
    }
    _LIG_PI_INTERACTIONS: ClassVar = [
        "EdgeToFace",
        "FaceToFace",
        "PiStacking",
        "PiCation",
    ]
    _BRIDGED_INTERACTIONS: ClassVar[dict[str, str]] = {"WaterBridge": "water_residues"}
    _DISPLAYED_ATOM: ClassVar = {  # index 0 in indices tuple by default
        "HBDonor": 1,
        "XBDonor": 1,
    }

    _JS_FILE = Path(__file__).parent / "network.js"
    _HTML_FILE = Path(__file__).parent / "network.html"
    _CSS_FILE = Path(__file__).parent / "network.css"

    _HTML_TEMPLATE = _HTML_FILE.read_text()
    _JS_TEMPLATE = _JS_FILE.read_text()
    _CSS_TEMPLATE = _CSS_FILE.read_text()

    def __init__(
        self,
        df: pd.DataFrame,
        lig_mol: Chem.Mol,
        use_coordinates: bool = False,
        flatten_coordinates: bool = True,
        kekulize: bool = False,
        molsize: int = 35,
        rotation: float = 0,
        carbon: float = 0.16,
    ) -> None:
        self.df = df
        self._interacting_atoms: set[int] = {
            atom for atoms in df.index.get_level_values("atoms") for atom in atoms
        }
        mol = deepcopy(lig_mol)
        if kekulize:
            Chem.Kekulize(mol)
        if use_coordinates:
            if flatten_coordinates:
                rdDepictor.GenerateDepictionMatching3DStructure(mol, lig_mol)
        else:
            rdDepictor.Compute2DCoords(mol, clearConfs=True)
        xyz: "NDArray[np.float64]" = mol.GetConformer().GetPositions()
        if rotation:
            theta = np.radians(rotation)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, s], [-s, c]])
            xy, z = xyz[:, :2], xyz[:, 2:3]
            center = xy.mean(axis=0)
            xy = ((xy - center) @ R.T) + center
            xyz = np.concatenate([xy, z], axis=1)
        if carbon:
            self._carbon: dict[str, Any] = {
                "label": " ",
                "shape": "dot",
                "color": self.COLORS["atoms"]["C"],
                "size": molsize * carbon,
            }
        else:
            self._carbon = {"label": " ", "shape": "text"}
        self.xyz = molsize * xyz
        self.mol = mol
        self._multiplier = molsize
        self.options: dict[str, Any] = {}
        self._max_interaction_width = 6
        self._avoidOverlap = 0.8
        self._springConstant = 0.1
        self._bond_color = "black"
        self._default_atom_color = "grey"
        self._default_residue_color = "#dbdbdb"
        self._default_interaction_color = "#dbdbdb"
        self._non_single_bond_spacing = 0.06
        self._dash = [10]
        self._edge_title_formatter = "{interaction}: {distance:.2f}Å"
        self._edge_label_formatter = "{weight_pct:.0f}%"
        # regroup interactions of the same color
        temp = defaultdict(list)
        interactions = set(df.index.get_level_values("interaction").unique())
        for interaction in interactions:
            color = self.COLORS["interactions"].get(
                interaction,
                self._default_interaction_color,
            )
            temp[color].append(interaction)
        self._interaction_types = {
            interaction: "/".join(interaction_group)
            for interaction_group in temp.values()
            for interaction in interaction_group
        }
        # ID for saving to PNG with JS
        self.uuid = uuid4().hex
        self._iframe: str | None = None

    @classmethod
    def from_fingerprint(
        cls,
        fp: "Fingerprint",
        ligand_mol: Chem.Mol,
        kind: Literal["aggregate", "frame"] = "aggregate",
        frame: int = 0,
        display_all: bool = False,
        threshold: float = 0.3,
        **kwargs: Any,
    ) -> "LigNetwork":
        """Helper method to create a ligand interaction diagram from a
        :class:`~prolif.fingerprint.Fingerprint` object.

        Notes
        -----
        Two kinds of diagrams can be rendered: either for a designated frame or
        by aggregating the results on the whole IFP and optionnally discarding
        interactions that occur less frequently than a threshold. In the latter
        case (aggregate), only the group of atoms most frequently involved in
        each interaction is used to draw the edge.

        Parameters
        ----------
        fp : prolif.fingerprint.Fingerprint
            The fingerprint object already executed using one of the ``run`` or
            ``run_from_iterable`` methods.
        lig : rdkit.Chem.rdChem.Mol
            Ligand molecule
        kind : str
            One of ``"aggregate"`` or ``"frame"``
        frame : int
            Frame number (see :attr:`~prolif.fingerprint.Fingerprint.ifp`). Only
            applicable for ``kind="frame"``
        display_all : bool
            Display all occurences for a given pair of residues and interaction, or only
            the shortest one. Only applicable for ``kind="frame"``. Not relevant if
            ``count=False`` in the ``Fingerprint`` object.
        threshold : float
            Frequency threshold, between 0 and 1. Only applicable for
            ``kind="aggregate"``
        kwargs : object
            Other arguments passed to the :class:`LigNetwork` class


        .. versionchanged:: 2.0.0
            Added the ``display_all`` parameter.
        """
        if not hasattr(fp, "ifp"):
            raise RunRequiredError(
                "Please run the fingerprint analysis before attempting to display"
                " results.",
            )
        if kind == "frame":
            df = cls._make_frame_df_from_fp(fp, frame=frame, display_all=display_all)
            return cls(df, ligand_mol, **kwargs)
        if kind == "aggregate":
            df = cls._make_agg_df_from_fp(fp, threshold=threshold)
            return cls(df, ligand_mol, **kwargs)
        raise ValueError(f'{kind!r} must be "aggregate" or "frame"')

    @classmethod
    def _get_records(cls, ifp: "IFP", all_metadata: bool) -> list[dict[str, Any]]:
        records = []
        for (lig_resid, prot_resid), int_data in ifp.items():
            for int_name, metadata_tuple in int_data.items():
                is_bridged_interaction = cls._BRIDGED_INTERACTIONS.get(int_name, None)
                for metadata in metadata_iterator(metadata_tuple, all_metadata):
                    if is_bridged_interaction:
                        distances = [d for d in metadata if d.startswith("distance_")]
                        for distlabel in distances:
                            _, src, dest = distlabel.split("_")
                            if src == "ligand":
                                components = "ligand_water"
                                src = str(lig_resid)
                                atoms = metadata["parent_indices"]["ligand"]
                            elif dest == "protein":
                                components = "water_protein"
                                dest = str(prot_resid)
                                atoms = ()
                            else:
                                components = "water_water"
                                atoms = ()
                            records.append(
                                {
                                    "ligand": src,
                                    "protein": dest,
                                    "interaction": int_name,
                                    "components": components,
                                    "atoms": atoms,
                                    "distance": metadata[distlabel],
                                }
                            )
                    else:
                        records.append(
                            {
                                "ligand": str(lig_resid),
                                "protein": str(prot_resid),
                                "interaction": int_name,
                                "components": "ligand_protein",
                                "atoms": metadata["parent_indices"]["ligand"],
                                "distance": metadata.get("distance", 0),
                            }
                        )
        return records

    @classmethod
    def _make_agg_df_from_fp(
        cls, fp: "Fingerprint", threshold: float = 0.3
    ) -> pd.DataFrame:
        data = []
        for ifp in fp.ifp.values():
            data.extend(cls._get_records(ifp, all_metadata=False))
        df = pd.DataFrame(data)
        # add weight for each atoms, and average distance
        df["weight"] = 1
        df = df.groupby(["ligand", "protein", "interaction", "atoms"]).agg(
            weight=("weight", "sum"),
            distance=("distance", "mean"),
            components=("components", "first"),
        )
        df["weight"] /= len(fp.ifp)
        # merge different ligand atoms of the same residue/interaction group before
        # applying the threshold
        df = df.join(
            df.groupby(level=["ligand", "protein", "interaction"]).agg(
                weight_total=("weight", "sum"),
            ),
        )
        # threshold and keep most occuring ligand atom
        return (
            df[df["weight_total"] >= threshold]
            .drop(columns="weight_total")
            .sort_values("weight", ascending=False)
            .groupby(level=["ligand", "protein", "interaction"])
            .head(1)
            .sort_index()
        )

    @classmethod
    def _make_frame_df_from_fp(
        cls, fp: "Fingerprint", frame: int = 0, display_all: bool = False
    ) -> pd.DataFrame:
        ifp = fp.ifp[frame]
        data = cls._get_records(ifp, all_metadata=display_all)
        df = pd.DataFrame(data)
        df["weight"] = 1
        return df.set_index(["ligand", "protein", "interaction", "atoms"]).reindex(
            columns=["weight", "distance", "components"],
        )

    def _make_carbon(self) -> dict[str, Any]:
        return deepcopy(self._carbon)

    def _make_lig_node(self, atom: Chem.Atom) -> None:
        """Prepare ligand atoms"""
        idx = atom.GetIdx()
        elem = atom.GetSymbol()
        if elem == "H" and idx not in self._interacting_atoms:
            self.exclude.append(idx)
            return
        charge = atom.GetFormalCharge()
        if charge != 0:
            displayed_charge = "{}{}".format(
                "" if abs(charge) == 1 else str(charge),
                "+" if charge > 0 else "-",
            )
            label = f"{elem}{displayed_charge}"
            shape = "ellipse"
        else:
            label = elem
            shape = "circle"
        if elem == "C":
            node = self._make_carbon()
        else:
            node = {
                "label": label,
                "shape": shape,
                "color": "white",
                "font": {
                    "color": self.COLORS["atoms"].get(elem, self._default_atom_color),
                },
            }
        node.update(
            {
                "id": idx,
                "x": float(self.xyz[idx, 0]),
                "y": float(self.xyz[idx, 1]),
                "fixed": True,
                "group": "ligand",
                "borderWidth": 0,
            },
        )
        self._nodes[idx] = node

    def _make_lig_edge(self, bond: Chem.Bond) -> None:
        """Prepare ligand bonds"""
        idx = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        if any(i in self.exclude for i in idx):
            return
        btype = bond.GetBondTypeAsDouble()
        if btype == 1:
            self.edges.append(
                {
                    "from": idx[0],
                    "to": idx[1],
                    "color": self._bond_color,
                    "physics": False,
                    "group": "ligand",
                    "width": 4,
                },
            )
        else:
            self._make_non_single_bond(idx, btype)

    def _make_non_single_bond(self, ids: list[int], btype: float) -> None:
        """Prepare double, triple and aromatic bonds"""
        xyz = self.xyz[ids]
        d = xyz[1, :2] - xyz[0, :2]
        length = np.sqrt((d**2).sum())
        u = d / length
        p = np.array([-u[1], u[0]])
        nodes = []
        dist = self._non_single_bond_spacing * self._multiplier * np.ceil(btype)
        dashes = False if btype in {2, 3} else self._dash
        for perp in (p, -p):
            for point in xyz:
                xy = point[:2] + perp * dist
                id_ = hash(xy.tobytes())
                nodes.append(id_)
                self._nodes[id_] = {
                    "id": id_,
                    "x": xy[0],
                    "y": xy[1],
                    "shape": "text",
                    "label": " ",
                    "fixed": True,
                    "physics": False,
                }
        l1, l2, r1, r2 = nodes
        self.edges.extend(
            [
                {
                    "from": l1,
                    "to": l2,
                    "color": self._bond_color,
                    "physics": False,
                    "dashes": dashes,
                    "group": "ligand",
                    "width": 4,
                },
                {
                    "from": r1,
                    "to": r2,
                    "color": self._bond_color,
                    "physics": False,
                    "dashes": dashes,
                    "group": "ligand",
                    "width": 4,
                },
            ],
        )
        if btype == 3:
            self.edges.append(
                {
                    "from": ids[0],
                    "to": ids[1],
                    "color": self._bond_color,
                    "physics": False,
                    "group": "ligand",
                    "width": 4,
                },
            )

    def _make_interactions(self, mass: int = 2) -> None:
        """Prepare lig-prot interactions"""
        restypes: dict[str, str | None] = {}
        lig_prot_df = self.df[self.df["components"] == "ligand_protein"]
        prot_and_waters: set[str] = (
            set(self.df.index.get_level_values("protein"))
            .union(self.df.index.get_level_values("ligand"))
            .difference(lig_prot_df.index.get_level_values("ligand"))
        )
        for prot_res in prot_and_waters:
            resname = ResidueId.from_string(prot_res).name
            restype = self.RESIDUE_TYPES.get(resname)
            restypes[prot_res] = restype
            color = self.COLORS["residues"].get(restype, self._default_residue_color)
            node = {
                "id": prot_res,
                "label": prot_res,
                "color": color,
                "font": {"color": self._FONTCOLORS.get(restype, "black")},
                "shape": "box",
                "borderWidth": 0,
                "physics": False,
                "mass": mass,
                "group": "protein",
                "residue_type": restype,
            }
            self._nodes[prot_res] = node

        self._calculate_protein_node_coordinates_nx()

        # Continue with interaction processing
        for (lig_res, prot_res, interaction, lig_indices), (
            weight,
            distance,
            components,
        ) in cast(
            Iterable[
                tuple[tuple[str, str, str, tuple[int, ...]], tuple[float, float, str]]
            ],
            self.df.iterrows(),
        ):
            if components.startswith("ligand"):
                if interaction in self._LIG_PI_INTERACTIONS:
                    centroid = self._get_ring_centroid(lig_indices)
                    origin = f"centroid({lig_res}, {prot_res}, {interaction})"
                    self._nodes[origin] = {
                        "id": origin,
                        "x": centroid[0],
                        "y": centroid[1],
                        "shape": "text",
                        "label": " ",
                        "fixed": True,
                        "physics": False,
                        "group": "ligand",
                    }
                else:
                    i = self._DISPLAYED_ATOM.get(interaction, 0)
                    origin = lig_indices[i]
            else:
                # water-water or water-protein
                origin = lig_res
            int_data = {
                "interaction": interaction,
                "distance": distance,
                "weight": weight,
                "weight_pct": weight * 100,
            }
            edge = {
                "from": origin,
                "to": prot_res,
                "title": self._edge_title_formatter.format_map(int_data),
                "interaction_type": self._interaction_types.get(
                    interaction,
                    interaction,
                ),
                "color": self.COLORS["interactions"].get(
                    interaction,
                    self._default_interaction_color,
                ),
                "smooth": {"type": "cubicBezier", "roundness": 0.2},
                "dashes": [10],
                "width": weight * self._max_interaction_width,
                "group": "interaction",
                "components": components,
            }
            if self.show_interaction_data:
                edge["label"] = self._edge_label_formatter.format_map(int_data)
                edge["font"] = self._edge_label_font
            self.edges.append(edge)

    def _calculate_protein_node_coordinates_nx(self) -> None:
        """Calculates 2D coordinates for protein nodes in self._nodes using NetworkX's spring_layout,
        then centers them around the center.
        Updates each protein node with 'x' and 'y' properties.
        """

        # 1. Extract protein nodes
        protein_nodes = [node_id for node_id, node in self._nodes.items() if node.get("group") == "protein"]

        # 2. Build a NetworkX graph with these nodes
        G = nx.Graph()
        G.add_nodes_from(protein_nodes)

        # 3. Get center
        center,width,length = self._get_ligand_center_and_dimensions()

        # 4. Compute spring layout
        pos = nx.spring_layout(G, k=200, fixed=None, center=center)

        # Find the range of current x and y values
        min_x = min(p[0] for p in pos.values())
        max_x = max(p[0] for p in pos.values())
        min_y = min(p[1] for p in pos.values())
        max_y = max(p[1] for p in pos.values())

        current_width = max_x - min_x
        current_height = max_y - min_y

        # Desired total width/height for the layout
        desired_width = 700
        desired_height = 700 # can be adjusted as needed

        # Calculate scaling factors
        scale_x = desired_width / current_width if current_width > 0 else 1
        scale_y = desired_height / current_height if current_height > 0 else 1
        scale_factor = max(scale_x, scale_y)

        # Calculate the mean of the spring layout positions
        layout_mean = np.mean([coords for coords in pos.values()], axis=0)

        for node_id in protein_nodes:
            # Shift to origin, scale, and then shift to center
            original_coords = np.array(pos[node_id])
            scaled_coords = (original_coords - layout_mean) * scale_factor + center

            self._nodes[node_id]["x"] = float(scaled_coords[0])
            self._nodes[node_id]["y"] = float(scaled_coords[1])

    def _calculate_protein_node_coordinates(self) -> None:
        """Calculates 2D coordinates for protein nodes in self._nodes based on their interaction points,
        positioning each residue near the ligand atoms it interacts with.
        Updates each protein node with 'x' and 'y' properties.
        """

        # 1. Extract protein nodes
        protein_nodes = [node_id for node_id, node in self._nodes.items() if node.get("group") == "protein"]
        
        if not protein_nodes:
            return

        # 2. Get ligand (all) center and dimensions for reference
        ligand_center, ligand_width, ligand_height = self._get_ligand_center_and_dimensions()
        base_offset_distance = max(ligand_width, ligand_height) * 0.3 + 10  # minimum distance from interaction point
        
        # 3. Dictionary to store residue positions and track conflicts
        residue_positions = {}
        used_positions = []  # Track positions to avoid overlaps
        min_distance_between_nodes = 40  # Minimum distance between protein nodes
        
        # 4. Process each interaction to position residues near their interaction points
        for (lig_res, prot_res, interaction, lig_indices), (weight, distance, components) in self.df.iterrows():
            if components.startswith("ligand") and prot_res in protein_nodes:
                
                # Get the interaction point on the ligand
                if interaction in self._LIG_PI_INTERACTIONS:
                    # For pi interactions, use ring centroid
                    interaction_point = self._get_ring_centroid(lig_indices)
                else:
                    # For other interactions, use the displayed atom
                    i = self._DISPLAYED_ATOM.get(interaction, 0)
                    interaction_point = self.xyz[list(lig_indices)].mean(axis=0)
                
                # Calculate direction vector from ligand center to interaction point
                direction = interaction_point[:2] - ligand_center
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > 0:
                    # Normalize direction and extend outward
                    direction_unit = direction / direction_norm
                    # Position residue outside the ligand
                    residue_pos = interaction_point[:2] + direction_unit * base_offset_distance

                    print(f"Positioning {prot_res} at {residue_pos}, direction = {direction_unit}")
                else:
                    # Fallback: position at a default offset
                    angle = len(residue_positions) * (2 * np.pi / max(len(protein_nodes), 8))
                    residue_pos = ligand_center + base_offset_distance * np.array([np.cos(angle), np.sin(angle)])
                
                # Check for conflicts with existing positions
                conflict_resolved = False
                max_attempts = 8
                attempt = 0
                
                while not conflict_resolved and attempt < max_attempts:
                    conflict_resolved = True
                    
                    # Check distance to all existing positions
                    for existing_pos in used_positions:
                        if np.linalg.norm(residue_pos - existing_pos) < min_distance_between_nodes:
                            conflict_resolved = False
                            break
                    
                    if not conflict_resolved:
                        # Adjust position by rotating around the interaction point
                        angle_offset = (attempt + 1) * (np.pi / 4)  # 45-degree increments
                        rotation_matrix = np.array([[np.cos(angle_offset), -np.sin(angle_offset)],
                                                  [np.sin(angle_offset), np.cos(angle_offset)]])
                        
                        # Rotate the offset vector
                        offset_vector = direction_unit * base_offset_distance
                        rotated_offset = rotation_matrix @ offset_vector
                        residue_pos = interaction_point[:2] + rotated_offset
                    
                    attempt += 1
                
                # Store the position for this residue (only if not already positioned)
                if prot_res not in residue_positions:
                    residue_positions[prot_res] = residue_pos
                    used_positions.append(residue_pos)
        
        # 5. Handle any remaining unpositioned residues (fallback)
        positioned_residues = set(residue_positions.keys())
        unpositioned_residues = [res for res in protein_nodes if res not in positioned_residues]
        
        if unpositioned_residues:
            # Position remaining residues in a circle around the ligand
            angle_step = 2 * np.pi / len(unpositioned_residues)
            for i, residue in enumerate(unpositioned_residues):
                angle = i * angle_step
                pos = ligand_center + (base_offset_distance + 50) * np.array([np.cos(angle), np.sin(angle)])
                
                # Ensure no conflicts
                while any(np.linalg.norm(pos - existing_pos) < min_distance_between_nodes 
                         for existing_pos in used_positions):
                    angle += np.pi / 8  # Rotate by 22.5 degrees
                    pos = ligand_center + (base_offset_distance + 50) * np.array([np.cos(angle), np.sin(angle)])
                
                residue_positions[residue] = pos
                used_positions.append(pos)
        
        # 6. Update protein node coordinates
        for node_id in protein_nodes:
            if node_id in residue_positions:
                x, y = residue_positions[node_id]
                self._nodes[node_id]["x"] = float(x)
                self._nodes[node_id]["y"] = float(y)
        
        print(f"Positioned {len(residue_positions)} protein residues based on their interaction points")

    def _get_ligand_center_and_dimensions(self):
        """Calculate center and dimensions of the ligand using all atoms"""
        # Get all atom coordinates
        atom_coords = self.xyz[:, :2]  # Only use x,y coordinates
    
        # Find the bounding box
        min_x = np.min(atom_coords[:, 0])
        max_x = np.max(atom_coords[:, 0])
        min_y = np.min(atom_coords[:, 1])
        max_y = np.max(atom_coords[:, 1])
    
        # Calculate center and dimensions
        center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
        width = max_x - min_x
        height = max_y - min_y
    
        return center, width, height

    def _get_ring_centroid(self, indices: tuple[int, ...]) -> "NDArray[np.float64]":
        """Find ring centroid coordinates using the indices of the ring atoms"""
        return self.xyz[list(indices)].mean(axis=0)  # type: ignore[no-any-return]

    def _patch_hydrogens(self) -> None:
        """Patch hydrogens on heteroatoms

        Hydrogen atoms that aren't part of any interaction have been hidden at
        this stage, but they should be added to the label of the heteroatom for
        clarity
        """
        to_patch: defaultdict[int, int] = defaultdict(int)
        for idx in self.exclude:
            h = self.mol.GetAtomWithIdx(idx)
            atom: Chem.Atom = h.GetNeighbors()[0]
            if atom.GetSymbol() != "C":
                to_patch[atom.GetIdx()] += 1
        for idx, nH in to_patch.items():
            node = self._nodes[idx]
            h_str = "H" if nH == 1 else f"H{nH}"
            label = re.sub(r"(\w+)(.*)", rf"\1{h_str}\2", node["label"])
            node["label"] = label
            node["shape"] = "ellipse"

    def _make_graph_data(self) -> None:
        """Prepares the nodes and edges"""
        self.exclude: list[int] = []
        self._nodes: dict[int | str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        # show residues
        self._make_interactions()
        # show ligand
        for atom in self.mol.GetAtoms():
            self._make_lig_node(atom)
        for bond in self.mol.GetBonds():
            self._make_lig_edge(bond)
        self._patch_hydrogens()
        self.nodes = list(self._nodes.values())

    def _get_js(
        self,
        width: str = "100%",
        height: str = "500px",
        div_id: str = "mynetwork",
        fontsize: int = 20,
        show_interaction_data: bool = False,
    ) -> dict[str, Any]:
        """Returns the JavaScript code to draw the network"""
        self.width = width
        self.height = height
        self.show_interaction_data = show_interaction_data
        self._edge_label_font = {"size": fontsize}
        self._make_graph_data()
        options = {
            "width": width,
            "height": height,
            "nodes": {
                "font": {"size": fontsize},
            },
            "physics": {
                "barnesHut": {
                    "avoidOverlap": self._avoidOverlap,
                    "springConstant": self._springConstant,
                },
            },
        }
        options.update(self.options)

        # get the legend buttons
        buttons = self._get_legend_buttons()

        return {
            "div_id": div_id,
            "nodes": json.dumps(self.nodes),
            "edges": json.dumps(self.edges),
            "options": json.dumps(options),
            "js_file_content": self._JS_TEMPLATE,
            "css_file_content": self._CSS_TEMPLATE,
            "buttons": json.dumps(buttons),
        }

    def _get_html(self, **kwargs: Any) -> str:
        """Returns the HTML code to draw the network"""
        js_data = self._get_js(**kwargs)
        return self._HTML_TEMPLATE % js_data

    def _get_legend_buttons(self, height: str = "90px") -> list[dict[str, Any]]:
        """Prepare the legend buttons data"""
        available = {}
        buttons = []
        map_color_restype = {c: t for t, c in self.COLORS["residues"].items()}
        map_color_interactions = {
            self.COLORS["interactions"].get(i, self._default_interaction_color): t
            for i, t in self._interaction_types.items()
        }
        # residues
        for node in self.nodes:
            if node.get("group", "") == "protein":
                color = node["color"]
                available[color] = map_color_restype.get(color, "Unknown")
        available = dict(sorted(available.items(), key=operator.itemgetter(1)))
        for i, (color, restype) in enumerate(available.items()):
            buttons.append(
                {
                    "index": i,
                    "label": restype,
                    "color": color,
                    "fontcolor": self._FONTCOLORS.get(restype, "black"),
                    "group": "residues",
                },
            )
        # interactions
        available.clear()
        for edge in self.edges:
            if edge.get("group", "") == "interaction":
                color = edge["color"]
                available[color] = map_color_interactions[color]
        available = dict(sorted(available.items(), key=operator.itemgetter(1)))
        for i, (color, interaction) in enumerate(available.items()):
            buttons.append(
                {
                    "index": i,
                    "label": interaction,
                    "color": color,
                    "fontcolor": "black",
                    "group": "interactions",
                },
            )

        # update height for legend
        if all("px" in h for h in [self.height, height]):
            h1 = int(re.findall(r"(\d+)\w+", self.height)[0])
            h2 = int(re.findall(r"(\d+)\w+", height)[0])
            self.height = f"{h1 + h2}px"

        return buttons

    @requires("IPython.display")
    def display(self, **kwargs: Any) -> "LigNetwork":
        """Prepare and display the network.

        Parameters
        ----------
        width: str = "100%"
        height: str = "500px"
        fontsize: int = 20
        show_interaction_data: bool = False
        """
        html = self._get_html(**kwargs)
        doc = escape(html)
        self._iframe = (
            f'<iframe id="{self.uuid}" width="{self.width}" height="{self.height}"'
            f' frameborder="0" srcdoc="{doc}"></iframe>'
        )
        return self

    @requires("IPython.display")
    def show(self, filename: str, **kwargs: Any) -> "LigNetwork":
        """Save the network as HTML and display the resulting file"""
        html = self._get_html(**kwargs)
        with open(filename, "w") as f:
            f.write(html)
        self._iframe = (
            f'<iframe id="{self.uuid}" width="{self.width}" height="{self.height}"'
            f' frameborder="0" src="{filename}"></iframe>'
        )
        return self

    def save(self, fp: Union[str, Path, "TextIO"], **kwargs: Any) -> None:
        """Save the network to an HTML file

        Parameters
        ----------
        fp : str or file-like object
            Name of the output file, or file-like object
        """
        html = self._get_html(**kwargs)
        if isinstance(fp, str | Path):
            with open(fp, "w") as f:
                f.write(html)
        elif hasattr(fp, "write") and callable(fp.write):
            fp.write(html)

    @requires("IPython.display")
    def save_png(self) -> Any:
        """Saves the current state of the ligplot to a PNG. Not available outside of a
        notebook.

        Notes
        -----
        Requires calling ``display`` or ``show`` first. The legend won't be exported.

        .. versionadded:: 2.1.0
        """
        return display(
            Javascript(f"""
            var iframe = document.getElementById("{self.uuid}");
            var iframe_doc = iframe.contentWindow.document;
            var canvas = iframe_doc.getElementsByTagName("canvas")[0];
            var link = document.createElement("a");
            link.href = canvas.toDataURL();
            link.download = "prolif-lignetwork.png"
            link.click();
            """),
        )

    def _repr_html_(self) -> str | None:
        if self._iframe:
            return self._iframe
        return None
