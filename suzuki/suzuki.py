import logging
import math
import os
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import List

import numpy as np
from catalystGA import BaseCatalyst, gaussian
from hide_warnings import hide_warnings
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tooltoad.chemutils import (
    _determineConnectivity,
    ac2mol,
    ac2xyz,
    iteratively_determine_bonds,
)
from tooltoad.orca import orca_calculate
from tooltoad.sa import sa_target_score_clipped
from tooltoad.xtb import xtb_calculate

_logger = logging.getLogger(__name__)

rmsd_threshold = 0.25
num_confs = 10
hartree2kcalmol = 627.5094740631

PRE_OPT = {
    "gfn": 2,
    "opt": True,
}

OPT = {
    "B3LYP/G": None,
    "D3BJ": None,
    "3-21G": None,
    "NORI": None,
    "opt": None,
}

SP = {
    "B3LYP/G": None,
    "D3BJ": None,
    "NORI": None,
    "def2-TZVP": None,
}


METALS = "Pd,Fe,Cu,Ni,Ag,Au,Pt"
HEADER1 = ["Conf-ID", "GFN-2 OPT [Hartree]"]
ROW_FORMAT1 = "{:>15}{:>25}"


class SuzukiCatalyst(BaseCatalyst):
    n_ligands = 2
    save_attributes = {
        "energy_difference": "REAL",
        "sq": "TEXT",
        "lin": "TEXT",
        "sq_energy": "REAL",
        "lin_energy": "REAL",
        "sa": "REAL",
    }
    extraLigands = "[METAL:1]>>[*:1](-Br)(-C=C)"
    fragment_energy = -2652.212097391772  # B3LYP-D3/def-TZVP//B3LYP-D3/3-21G
    TARGET = -27.55  # B3LYP-D3/def-TZVP//B3LYP-D3/3-21G

    def __init__(self, metal: Chem.Mol, ligands: List):
        self.energy_difference = math.nan
        self.sq = ""
        self.lin = ""
        self.sq_energy = math.nan
        self.lin_energy = math.nan
        self.sa = math.nan
        super().__init__(metal, ligands)

    @staticmethod
    def _get_multiplicity(mol):
        nUnpairedRadials = 0
        for a in mol.GetAtoms():
            numRadicals = a.GetNumRadicalElectrons()
            if numRadicals % 2 == 1:
                nUnpairedRadials += 1
        multiplicity = nUnpairedRadials + 1
        return multiplicity

    def calculate_energy_difference(
        self,
        n_cores: int = 1,
        envvar_scratch: str = "SCRATCH",
        adjecency_check=False,
        use_hueckel=True,
        disconnect_metal=True,
    ):
        # Setup scrach directory
        scratch = os.environ.get(envvar_scratch, ".")
        calc_dir = Path(scratch)
        start_time = time.time()
        jobid = os.getenv("SLURM_ARRAY_ID", str(uuid.uuid4()))
        calc_dir = calc_dir / jobid
        self.calc_dir = calc_dir
        calc_dir.mkdir(exist_ok=True)

        # Setup logging to stdout
        _logger.setLevel(logging.INFO)
        _logger.addHandler(logging.StreamHandler())
        orca_logger = logging.getLogger("orca")
        orca_logger.setLevel(logging.INFO)
        orca_logger.addHandler(logging.StreamHandler())

        _logger.debug(f"Running on: {socket.gethostname()}")
        _logger.debug(f"Calculation directory: {calc_dir.absolute()}")
        _logger.info(f"Calculating score for {self}\nSMILES: {self.smiles}\n")

        # Embed conformers
        mol4 = self.embed(
            extraLigands=self.extraLigands.replace(
                "METAL", Chem.MolToSmarts(self.metal.atom).strip("[]")
            ),
            chiralTag=Chem.CHI_SQUAREPLANAR,
            permutationOrder=2,
            numConfs=num_confs,
            useRandomCoords=True,
            pruneRmsThresh=rmsd_threshold,
        )
        _logger.info(f"Embedded {mol4.GetNumConformers()} conformers.")
        if mol4.GetNumConformers() == 0:
            raise ValueError("Embedding failed.")

        # Determine connectivity for mol4
        mol4_adj = Chem.GetAdjacencyMatrix(mol4)
        charge = Chem.GetFormalCharge(mol4)
        multiplicity = self._get_multiplicity(mol4)
        mol4_atoms = [a.GetSymbol() for a in mol4.GetAtoms()]
        mol4_results = []
        fail_results = []
        _logger.info(f"{ROW_FORMAT1.format(*HEADER1)}")
        for conf in mol4.GetConformers():
            cid = conf.GetId()
            mol4_coords = conf.GetPositions()

            # Pre-optimization of mol4
            xtb_results = xtb_calculate(
                atoms=mol4_atoms,
                coords=mol4_coords,
                charge=charge,
                multiplicity=multiplicity,
                options=PRE_OPT,
                scr=calc_dir,
                n_cores=min([n_cores, 8]),
            )
            mol4_pre_opt_coords, mol4_pre_energy = (
                xtb_results["opt_coords"],
                xtb_results["electronic_energy"],
            )

            # Check adjacency matrix after pre-optimization of mol4
            if adjecency_check and not self.adjacency_check(
                mol4_adj, mol4_atoms, mol4_pre_opt_coords, use_hueckel, disconnect_metal
            ):
                _logger.warning(
                    f"Change in adjacency matrix after pre-optimization. Skipping conformer {cid}."
                )
                fail_results.append((cid, mol4_pre_energy, mol4_pre_opt_coords))
                continue

            row = [cid, round(mol4_pre_energy, 4)]
            _logger.info(f"{ROW_FORMAT1.format(*row)}")
            mol4_results.append((cid, mol4_pre_energy, mol4_pre_opt_coords))
        if len(mol4_results) == 0:
            _logger.info(
                "Change in adjacency matrix after pre-optimization for all conformers."
            )
            mol4_results = fail_results
            _logger.info("Trying to optimize with higher method.")

        mol4_results.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
        best_mol4_pre_opt_coords = mol4_results[0][2]
        # Optimization of mol4
        mol4_opt_results = orca_calculate(
            atoms=mol4_atoms,
            coords=best_mol4_pre_opt_coords,
            charge=charge,
            multiplicity=multiplicity,
            options=OPT,
            scr=calc_dir,
            n_cores=n_cores,
        )

        mol4_opt_coords, mol4_opt_energy = (
            mol4_opt_results["opt_coords"],
            mol4_opt_results["electronic_energy"],
        )

        self.sq = ac2xyz(mol4_atoms, mol4_opt_coords)

        # Check adjacency matrix after optimization of mol4
        if adjecency_check and not self.adjacency_check(
            mol4_adj, mol4_atoms, mol4_opt_coords, use_hueckel, disconnect_metal
        ):
            raise ValueError("Change in adjacency matrix after optimization of mol4.")

        if math.isnan(mol4_opt_energy):
            raise ValueError("Optimization of mol4 failed.")
        _logger.info(
            f"\nOptimization converged. Energy: {mol4_opt_energy:.04f} [Hartree]."
        )

        # Singlepoint of mol4
        mol4_sp_results = orca_calculate(
            atoms=mol4_atoms,
            coords=mol4_opt_coords,
            charge=charge,
            multiplicity=multiplicity,
            options=SP,
            scr=calc_dir,
            n_cores=n_cores,
        )
        mol4_sp_energy = mol4_sp_results["electronic_energy"]

        self.sq_energy = mol4_sp_energy
        _logger.info(f"Singlepoint Energy: {mol4_sp_energy:.04f} [Hartree].")
        if math.isnan(mol4_sp_energy):
            raise ValueError("Singlepoint of mol4 failed.")

        # Generate mol2
        remove_ids = list(
            mol4.GetSubstructMatch(
                Chem.MolFromSmarts(
                    f"Br-{Chem.MolToSmarts(self.metal.atom)}-C(-[H])=C(-[H])-[H]"
                )
            )
        )
        assert (
            len(remove_ids) == 7
        ), f"Error in generation of mol2: only found remove_ids {remove_ids}"
        remove_ids.remove(mol4.GetSubstructMatch(self.metal.atom)[0])
        substructure_mask = np.ones_like(mol4_atoms, dtype=bool)
        substructure_mask[remove_ids] = False

        mol2_atoms = np.array(mol4_atoms)[substructure_mask]
        mol2_coords = np.array(mol4_opt_coords)[substructure_mask]

        # Determine connectivity for mol2 (convert adjacency matrix of mol4 to adjacency matrix of mol2)
        mol2_adj = np.copy(mol4_adj)
        for idx in sorted(remove_ids, reverse=True):
            mol2_adj = np.delete(mol2_adj, idx, 0)
            mol2_adj = np.delete(mol2_adj, idx, 1)

        # Optimization of mol2
        mol2_opt_results = orca_calculate(
            atoms=mol2_atoms,
            coords=mol2_coords,
            charge=charge,
            multiplicity=multiplicity,
            options=OPT,
            scr=calc_dir,
            n_cores=n_cores,
        )
        mol2_opt_coords, mol2_opt_energy = (
            mol2_opt_results["opt_coords"],
            mol2_opt_results["electronic_energy"],
        )

        self.lin = ac2xyz(mol2_atoms, mol2_opt_coords)
        # Check adjacency matrix after optimization of mol2
        if adjecency_check and not self.adjacency_check(
            mol2_adj, mol2_atoms, mol2_opt_coords, use_hueckel, disconnect_metal
        ):
            raise ValueError("Change in adjacency matrix after optimization of mol2.")

        if math.isnan(mol2_opt_energy):
            raise ValueError("Optimization of mol2 failed.")
        _logger.info(
            f"\nOptimization converged. Energy: {mol2_opt_energy:.04f} [Hartree]."
        )

        # Singlepoint of mol2
        mol2_sp_results = orca_calculate(
            atoms=mol2_atoms,
            coords=mol2_opt_coords,
            charge=charge,
            multiplicity=multiplicity,
            options=SP,
            scr=calc_dir,
            n_cores=n_cores,
        )
        mol2_sp_energy = mol2_sp_results["electronic_energy"]

        self.lin_energy = mol2_sp_energy
        _logger.info(f"Singlepoint Energy: {mol2_sp_energy:.04f} [Hartree].")
        if math.isnan(mol2_sp_energy):
            raise ValueError("Singlepoint of mol2 failed.")

        energy_difference = (
            mol4_sp_energy - mol2_sp_energy - self.fragment_energy
        ) * hartree2kcalmol
        _logger.info(f"Energy difference: {energy_difference:.04f} [kcal/mol].")
        self.energy_difference = energy_difference
        self.timing = time.time() - start_time

        try:
            shutil.rmtree(calc_dir)
        except FileNotFoundError:
            pass

    @staticmethod
    @hide_warnings(out=False)
    def adjacency_check(adj1, atoms, coords, use_hueckel=True, disconnect_metal=False):
        mol = ac2mol(atoms, coords)
        try:
            if use_hueckel:
                iteratively_determine_bonds(mol)
            else:
                mol = _determineConnectivity(mol)
        except ValueError:
            _logger.debug("Molecule contains disconnected fragments")
        if disconnect_metal:
            metal_id = mol.GetSubstructMatch(Chem.MolFromSmarts(f"[{METALS}]"))[0]
            adj1[metal_id, :] = np.zeros(adj1.shape[0])
            adj1[:, metal_id] = np.zeros(adj1.shape[1])
            mol = rdMolStandardize.DisconnectOrganometallics(mol)
            Chem.SanitizeMol(mol)
        adj2 = Chem.GetAdjacencyMatrix(mol)

        _logger.debug(f"Adjacency matrix 1:\n{adj1}")
        _logger.debug(f"Adjacency matrix 2:\n{adj2}")

        return np.array_equal(adj1, adj2)

    def calculate_score(
        self,
        n_cores: int = 1,
        envvar_scratch: str = "SCRATCH",
        adjecency_check: bool = True,
        disconnect_metal: bool = False,
        use_hueckel=True,
        sa: bool = False,
    ):
        self.calculate_energy_difference(
            n_cores=n_cores,
            envvar_scratch=envvar_scratch,
            adjecency_check=adjecency_check,
            use_hueckel=use_hueckel,
            disconnect_metal=disconnect_metal,
        )
        if not sa:
            self.sa = 1
        elif sa == "sa":
            sa1 = sa_target_score_clipped(self.ligands[0].mol)
            sa2 = sa_target_score_clipped(self.ligands[1].mol)
            self.sa = np.mean([sa1, sa2])
        elif sa == "sax":
            sa1 = sa(self.ligands[0].mol)
            sa2 = sa(self.ligands[1].mol)
            self.sa = np.mean([sa1, sa2])
        else:
            raise ValueError(f"Unknown sa option: {sa}")
        self.score = self.sa * gaussian(self.energy_difference, self.TARGET, 6)


class SuzukiCatalystCu(BaseCatalyst):
    """Has a constrained opimization before doing anything else"""

    n_ligands = 2
    save_attributes = {
        "energy_difference": "REAL",
        "sq": "TEXT",
        "lin": "TEXT",
        "sq_energy": "REAL",
        "lin_energy": "REAL",
        "sa": "REAL",
    }
    extraLigands = "[METAL:1]>>[*:1](-Br)(-C=C)"
    fragment_energy = -2652.212097391772  # B3LYP-D3/def-TZVP//B3LYP-D3/3-21G
    TARGET = -27.55  # B3LYP-D3/def-TZVP//B3LYP-D3/3-21G

    def __init__(self, metal: Chem.Mol, ligands: List):
        self.energy_difference = math.nan
        self.sq = ""
        self.lin = ""
        self.sq_energy = math.nan
        self.lin_energy = math.nan
        self.sa = math.nan
        super().__init__(metal, ligands)

    @staticmethod
    def _get_multiplicity(mol):
        nUnpairedRadials = 0
        for a in mol.GetAtoms():
            numRadicals = a.GetNumRadicalElectrons()
            if numRadicals % 2 == 1:
                nUnpairedRadials += 1
        multiplicity = nUnpairedRadials + 1
        return multiplicity

    def calculate_energy_difference(
        self,
        n_cores: int = 1,
        envvar_scratch: str = "SCRATCH",
        adjecency_check=False,
        use_hueckel=True,
        disconnect_metal=True,
    ):
        # Setup scrach directory
        scratch = os.environ.get(envvar_scratch, ".")
        calc_dir = Path(scratch)
        start_time = time.time()
        jobid = os.getenv("SLURM_ARRAY_ID", str(uuid.uuid4()))
        calc_dir = calc_dir / jobid
        self.calc_dir = calc_dir
        calc_dir.mkdir(exist_ok=True)

        # Setup logging to stdout
        _logger.setLevel(logging.INFO)
        _logger.addHandler(logging.StreamHandler())
        orca_logger = logging.getLogger("orca")
        orca_logger.setLevel(logging.INFO)
        orca_logger.addHandler(logging.StreamHandler())

        _logger.debug(f"Running on: {socket.gethostname()}")
        _logger.debug(f"Calculation directory: {calc_dir.absolute()}")
        _logger.info(f"Calculating score for {self}\nSMILES: {self.smiles}\n")

        # Embed conformers
        mol4 = self.embed(
            extraLigands=self.extraLigands.replace(
                "METAL", Chem.MolToSmarts(self.metal.atom).strip("[]")
            ),
            chiralTag=Chem.CHI_SQUAREPLANAR,
            permutationOrder=2,
            numConfs=num_confs,
            useRandomCoords=True,
            pruneRmsThresh=rmsd_threshold,
        )
        _logger.info(f"Embedded {mol4.GetNumConformers()} conformers.")
        if mol4.GetNumConformers() == 0:
            raise ValueError("Embedding failed.")

        # Determine connectivity for mol4
        mol4_adj = Chem.GetAdjacencyMatrix(mol4)
        charge = Chem.GetFormalCharge(mol4)
        multiplicity = self._get_multiplicity(mol4)
        mol4_atoms = [a.GetSymbol() for a in mol4.GetAtoms()]
        mol4_results = []
        fail_results = []
        _logger.info(f"{ROW_FORMAT1.format(*HEADER1)}")
        for conf in mol4.GetConformers():
            cid = conf.GetId()
            mol4_coords = conf.GetPositions()

            # Pre-optimization of mol4
            xtb_results = xtb_calculate(
                atoms=mol4_atoms,
                coords=mol4_coords,
                charge=charge,
                multiplicity=multiplicity,
                options={"gfn": 2},
                scr=calc_dir,
                n_cores=min([n_cores, 8]),
            )
            mol4_pre_opt_coords, mol4_pre_energy = (
                xtb_results["opt_coords"],
                xtb_results["electronic_energy"],
            )

            # Check adjacency matrix after pre-optimization of mol4
            if adjecency_check and not self.adjacency_check(
                mol4_adj, mol4_atoms, mol4_pre_opt_coords, use_hueckel, disconnect_metal
            ):
                _logger.warning(
                    f"Change in adjacency matrix after pre-optimization. Skipping conformer {cid}."
                )
                fail_results.append((cid, mol4_pre_energy, mol4_pre_opt_coords))
                continue

            row = [cid, round(mol4_pre_energy, 4)]
            _logger.info(f"{ROW_FORMAT1.format(*row)}")
            mol4_results.append((cid, mol4_pre_energy, mol4_pre_opt_coords))
        if len(mol4_results) == 0:
            _logger.info(
                "Change in adjacency matrix after pre-optimization for all conformers."
            )
            mol4_results = fail_results
            _logger.info("Trying to optimize with higher method.")

        mol4_results.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
        best_mol4_pre_opt_coords = mol4_results[0][2]
        # Get indices of atoms to constrain
        m_idx = mol4.GetSubstructMatch(Chem.MolFromSmarts("[Cu]"))[0]
        cids = list(np.where(mol4_adj[m_idx])[0])
        cids.append(m_idx)
        cstring = "\n".join([f"{{C {i} C}}" for i in cids])
        constains = f"""%geom
Constraints
{cstring}
end\nend\n\n"""
        # Constrained Optimization of mol4
        mol4_copt_results = orca_calculate(
            atoms=mol4_atoms,
            coords=best_mol4_pre_opt_coords,
            charge=charge,
            multiplicity=multiplicity,
            options=OPT,
            xtra_inp_str=constains,
            scr=calc_dir,
            n_cores=n_cores,
        )

        # Unconstrained Optimization of mol4
        mol4_opt_results = orca_calculate(
            atoms=mol4_atoms,
            coords=mol4_copt_results["opt_coords"],
            charge=charge,
            multiplicity=multiplicity,
            options=OPT,
            xtra_inp_str=constains,
            scr=calc_dir,
            n_cores=n_cores,
        )

        mol4_opt_coords, mol4_opt_energy = (
            mol4_opt_results["opt_coords"],
            mol4_opt_results["electronic_energy"],
        )

        self.sq = ac2xyz(mol4_atoms, mol4_opt_coords)

        # Check adjacency matrix after optimization of mol4
        if adjecency_check and not self.adjacency_check(
            mol4_adj, mol4_atoms, mol4_opt_coords, use_hueckel, disconnect_metal
        ):
            raise ValueError("Change in adjacency matrix after optimization of mol4.")

        if math.isnan(mol4_opt_energy):
            raise ValueError("Optimization of mol4 failed.")
        _logger.info(
            f"\nOptimization converged. Energy: {mol4_opt_energy:.04f} [Hartree]."
        )

        # Singlepoint of mol4
        mol4_sp_results = orca_calculate(
            atoms=mol4_atoms,
            coords=mol4_opt_coords,
            charge=charge,
            multiplicity=multiplicity,
            options=SP,
            scr=calc_dir,
            n_cores=n_cores,
        )
        mol4_sp_energy = mol4_sp_results["electronic_energy"]

        self.sq_energy = mol4_sp_energy
        _logger.info(f"Singlepoint Energy: {mol4_sp_energy:.04f} [Hartree].")
        if math.isnan(mol4_sp_energy):
            raise ValueError("Singlepoint of mol4 failed.")

        # Generate mol2
        remove_ids = list(
            mol4.GetSubstructMatch(
                Chem.MolFromSmarts(
                    f"Br-{Chem.MolToSmarts(self.metal.atom)}-C(-[H])=C(-[H])-[H]"
                )
            )
        )
        assert (
            len(remove_ids) == 7
        ), f"Error in generation of mol2: only found remove_ids {remove_ids}"
        remove_ids.remove(mol4.GetSubstructMatch(self.metal.atom)[0])
        substructure_mask = np.ones_like(mol4_atoms, dtype=bool)
        substructure_mask[remove_ids] = False

        mol2_atoms = np.array(mol4_atoms)[substructure_mask]
        mol2_coords = np.array(mol4_opt_coords)[substructure_mask]

        # Determine connectivity for mol2 (convert adjacency matrix of mol4 to adjacency matrix of mol2)
        mol2_adj = np.copy(mol4_adj)
        for idx in sorted(remove_ids, reverse=True):
            mol2_adj = np.delete(mol2_adj, idx, 0)
            mol2_adj = np.delete(mol2_adj, idx, 1)

        # Get indices of atoms to constrain
        m_idx = np.where(mol2_atoms == "Cu")[0][0]
        cids = list(np.where(mol2_adj[m_idx])[0])
        cids.append(m_idx)
        cstring = "\n".join([f"{{C {i} C}}" for i in cids])
        constains = f"""%geom
Constraints
{cstring}
end\nend\n\n"""
        # Constrained Optimization of mol2
        mol2_copt_results = orca_calculate(
            atoms=mol2_atoms,
            coords=mol2_coords,
            charge=charge,
            multiplicity=multiplicity,
            options=OPT,
            xtra_inp_str=constains,
            scr=calc_dir,
            n_cores=n_cores,
        )
        # Unconstrained Optimization of mol2
        mol2_opt_results = orca_calculate(
            atoms=mol2_atoms,
            coords=mol2_copt_results["opt_coords"],
            charge=charge,
            multiplicity=multiplicity,
            options=OPT,
            scr=calc_dir,
            n_cores=n_cores,
        )
        mol2_opt_coords, mol2_opt_energy = (
            mol2_opt_results["opt_coords"],
            mol2_opt_results["electronic_energy"],
        )

        self.lin = ac2xyz(mol2_atoms, mol2_opt_coords)
        # Check adjacency matrix after optimization of mol2
        if adjecency_check and not self.adjacency_check(
            mol2_adj, mol2_atoms, mol2_opt_coords, use_hueckel, disconnect_metal
        ):
            raise ValueError("Change in adjacency matrix after optimization of mol2.")

        if math.isnan(mol2_opt_energy):
            raise ValueError("Optimization of mol2 failed.")
        _logger.info(
            f"\nOptimization converged. Energy: {mol2_opt_energy:.04f} [Hartree]."
        )

        # Singlepoint of mol2
        mol2_sp_results = orca_calculate(
            atoms=mol2_atoms,
            coords=mol2_opt_coords,
            charge=charge,
            multiplicity=multiplicity,
            options=SP,
            scr=calc_dir,
            n_cores=n_cores,
        )
        mol2_sp_energy = mol2_sp_results["electronic_energy"]

        self.lin_energy = mol2_sp_energy
        _logger.info(f"Singlepoint Energy: {mol2_sp_energy:.04f} [Hartree].")
        if math.isnan(mol2_sp_energy):
            raise ValueError("Singlepoint of mol2 failed.")

        energy_difference = (
            mol4_sp_energy - mol2_sp_energy - self.fragment_energy
        ) * hartree2kcalmol
        _logger.info(f"Energy difference: {energy_difference:.04f} [kcal/mol].")
        self.energy_difference = energy_difference
        self.timing = time.time() - start_time

        try:
            shutil.rmtree(calc_dir)
        except FileNotFoundError:
            pass

    @staticmethod
    @hide_warnings(out=False)
    def adjacency_check(adj1, atoms, coords, use_hueckel=True, disconnect_metal=False):
        mol = ac2mol(atoms, coords)
        try:
            if use_hueckel:
                iteratively_determine_bonds(mol)
            else:
                mol = _determineConnectivity(mol)
        except ValueError:
            _logger.debug("Molecule contains disconnected fragments")
        if disconnect_metal:
            metal_id = mol.GetSubstructMatch(Chem.MolFromSmarts(f"[{METALS}]"))[0]
            adj1[metal_id, :] = np.zeros(adj1.shape[0])
            adj1[:, metal_id] = np.zeros(adj1.shape[1])
            mol = rdMolStandardize.DisconnectOrganometallics(mol)
            Chem.SanitizeMol(mol)
        adj2 = Chem.GetAdjacencyMatrix(mol)

        _logger.debug(f"Adjacency matrix 1:\n{adj1}")
        _logger.debug(f"Adjacency matrix 2:\n{adj2}")

        return np.array_equal(adj1, adj2)

    def calculate_score(
        self,
        n_cores: int = 1,
        envvar_scratch: str = "SCRATCH",
        adjecency_check: bool = True,
        disconnect_metal: bool = False,
        use_hueckel: bool = True,
        sa: str = None,
    ):
        self.calculate_energy_difference(
            n_cores=n_cores,
            envvar_scratch=envvar_scratch,
            adjecency_check=adjecency_check,
            use_hueckel=use_hueckel,
            disconnect_metal=disconnect_metal,
        )
        if not sa:
            self.sa = 1
        elif sa == "sa":
            sa1 = sa_target_score_clipped(self.ligands[0].mol)
            sa2 = sa_target_score_clipped(self.ligands[1].mol)
            self.sa = np.mean([sa1, sa2])
        elif sa == "sax":
            sa1 = sa(self.ligands[0].mol)
            sa2 = sa(self.ligands[1].mol)
            self.sa = np.mean([sa1, sa2])
        else:
            raise ValueError(f"Unknown sa option: {sa}")
        self.score = self.sa * gaussian(self.energy_difference, self.TARGET, 6)


class SuzukiTest(BaseCatalyst):
    n_ligands = 2
    save_attributes = {
        "energy_difference": "REAL",
        "sq": "TEXT",
        "lin": "TEXT",
        "sq_energy": "REAL",
        "lin_energy": "REAL",
        "sa": "REAL",
    }
    extraLigands = f"[{METALS}:1]>>[*:1](-Br)(-C=C)"
    fragment_energy = -2652.212097391772  # B3LYP-D3/def-TZVP//B3LYP-D3/3-21G
    TARGET = -27.55  # B3LYP-D3/def-TZVP//B3LYP-D3/3-21G

    def __init__(self, metal: Chem.Mol, ligands: List):
        self.energy_difference = math.nan
        self.sq = ""
        self.lin = ""
        self.sq_energy = math.nan
        self.lin_energy = math.nan
        self.sa = math.nan
        super().__init__(metal, ligands)

    def calculate_score(
        self,
        n_cores: int = 1,
        envvar_scratch: str = "SCRATCH",
        adjecency_check: bool = True,
        sa=False,
    ):
        import random

        # mock calculate energy_difference
        self.energy_difference = -10.0 * random.random()

        if not sa:
            self.sa = 1
        elif sa == "sa":
            sa1 = sa_target_score_clipped(self.ligands[0].mol)
            sa2 = sa_target_score_clipped(self.ligands[1].mol)
            self.sa = np.mean([sa1, sa2])
        elif sa == "sax":
            sa1 = sa(self.ligands[0].mol)
            sa2 = sa(self.ligands[1].mol)
            self.sa = np.mean([sa1, sa2])
        else:
            raise ValueError(f"Unknown sa option: {sa}")
        self.score = self.sa * gaussian(self.energy_difference, self.TARGET, 6)


OPTAU = {
    "B3LYP/G": None,
    "D3BJ": None,
    "def2-SVP": None,
    "NORI": None,
    "opt": None,
}


class SuzukiCatalystAu(BaseCatalyst):
    n_ligands = 2
    save_attributes = {
        "energy_difference": "REAL",
        "sq": "TEXT",
        "lin": "TEXT",
        "sq_energy": "REAL",
        "lin_energy": "REAL",
        "sa": "REAL",
    }
    extraLigands = "[METAL:1]>>[*:1](-Br)(-C=C)"
    fragment_energy = -2652.212097391772  # B3LYP-D3/def-TZVP//B3LYP-D3/3-21G
    TARGET = -27.55  # B3LYP-D3/def-TZVP//B3LYP-D3/3-21G

    def __init__(self, metal: Chem.Mol, ligands: List):
        self.energy_difference = math.nan
        self.sq = ""
        self.lin = ""
        self.sq_energy = math.nan
        self.lin_energy = math.nan
        self.sa = math.nan
        super().__init__(metal, ligands)

    @staticmethod
    def _get_multiplicity(mol):
        nUnpairedRadials = 0
        for a in mol.GetAtoms():
            numRadicals = a.GetNumRadicalElectrons()
            if numRadicals % 2 == 1:
                nUnpairedRadials += 1
        multiplicity = nUnpairedRadials + 1
        return multiplicity

    def calculate_energy_difference(
        self,
        n_cores: int = 1,
        envvar_scratch: str = "SCRATCH",
        adjecency_check=False,
        use_hueckel=True,
    ):
        # Setup scrach directory
        scratch = os.environ.get(envvar_scratch, ".")
        calc_dir = Path(scratch)
        start_time = time.time()
        jobid = os.getenv("SLURM_ARRAY_ID", str(uuid.uuid4()))
        calc_dir = calc_dir / jobid
        self.calc_dir = calc_dir
        calc_dir.mkdir(exist_ok=True)

        # Setup logging to stdout
        _logger.setLevel(logging.INFO)
        _logger.addHandler(logging.StreamHandler())
        orca_logger = logging.getLogger("orca")
        orca_logger.setLevel(logging.INFO)
        orca_logger.addHandler(logging.StreamHandler())

        _logger.debug(f"Running on: {socket.gethostname()}")
        _logger.debug(f"Calculation directory: {calc_dir.absolute()}")
        _logger.info(f"Calculating score for {self}\nSMILES: {self.smiles}\n")

        # Embed conformers
        mol4 = self.embed(
            extraLigands=self.extraLigands.replace(
                "METAL", Chem.MolToSmarts(self.metal.atom).strip("[]")
            ),
            chiralTag=Chem.CHI_SQUAREPLANAR,
            permutationOrder=2,
            numConfs=num_confs,
            useRandomCoords=True,
            pruneRmsThresh=rmsd_threshold,
        )
        _logger.info(f"Embedded {mol4.GetNumConformers()} conformers.")
        if mol4.GetNumConformers() == 0:
            raise ValueError("Embedding failed.")

        # Determine connectivity for mol4
        mol4_adj = Chem.GetAdjacencyMatrix(mol4)
        charge = Chem.GetFormalCharge(mol4)
        multiplicity = self._get_multiplicity(mol4)
        mol4_atoms = [a.GetSymbol() for a in mol4.GetAtoms()]
        mol4_results = []
        fail_results = []
        _logger.info(f"{ROW_FORMAT1.format(*HEADER1)}")
        for conf in mol4.GetConformers():
            cid = conf.GetId()
            mol4_coords = conf.GetPositions()

            # Pre-optimization of mol4
            xtb_results = xtb_calculate(
                atoms=mol4_atoms,
                coords=mol4_coords,
                charge=charge,
                multiplicity=multiplicity,
                options=PRE_OPT,
                scr=calc_dir,
                n_cores=min([n_cores, 8]),
            )
            mol4_pre_opt_coords, mol4_pre_energy = (
                xtb_results["opt_coords"],
                xtb_results["electronic_energy"],
            )

            # Check adjacency matrix after pre-optimization of mol4
            if adjecency_check and not self.adjacency_check(
                mol4_adj, mol4_atoms, mol4_pre_opt_coords, use_hueckel
            ):
                _logger.warning(
                    f"Change in adjacency matrix after pre-optimization. Skipping conformer {cid}."
                )
                fail_results.append((cid, mol4_pre_energy, mol4_pre_opt_coords))
                continue

            row = [cid, round(mol4_pre_energy, 4)]
            _logger.info(f"{ROW_FORMAT1.format(*row)}")
            mol4_results.append((cid, mol4_pre_energy, mol4_pre_opt_coords))
        if len(mol4_results) == 0:
            _logger.info(
                "Change in adjacency matrix after pre-optimization for all conformers."
            )
            mol4_results = fail_results
            _logger.info("Trying to optimize with higher method.")

        mol4_results.sort(key=lambda x: float("inf") if math.isnan(x[1]) else x[1])
        best_mol4_pre_opt_coords = mol4_results[0][2]
        # Optimization of mol4
        mol4_opt_results = orca_calculate(
            atoms=mol4_atoms,
            coords=best_mol4_pre_opt_coords,
            charge=charge,
            multiplicity=multiplicity,
            options=OPTAU,
            scr=calc_dir,
            n_cores=n_cores,
        )
        print(mol4_opt_results)
        mol4_opt_coords, mol4_opt_energy = (
            mol4_opt_results["opt_coords"],
            mol4_opt_results["electronic_energy"],
        )

        self.sq = ac2xyz(mol4_atoms, mol4_opt_coords)

        # Check adjacency matrix after optimization of mol4
        if adjecency_check and not self.adjacency_check(
            mol4_adj, mol4_atoms, mol4_opt_coords, use_hueckel
        ):
            raise ValueError("Change in adjacency matrix after optimization of mol4.")

        if math.isnan(mol4_opt_energy):
            raise ValueError("Optimization of mol4 failed.")
        _logger.info(
            f"\nOptimization converged. Energy: {mol4_opt_energy:.04f} [Hartree]."
        )

        # Singlepoint of mol4
        mol4_sp_results = orca_calculate(
            atoms=mol4_atoms,
            coords=mol4_opt_coords,
            charge=charge,
            multiplicity=multiplicity,
            options=SP,
            scr=calc_dir,
            n_cores=n_cores,
        )
        mol4_sp_energy = mol4_sp_results["electronic_energy"]

        self.sq_energy = mol4_sp_energy
        _logger.info(f"Singlepoint Energy: {mol4_sp_energy:.04f} [Hartree].")
        if math.isnan(mol4_sp_energy):
            raise ValueError("Singlepoint of mol4 failed.")

        # Generate mol2
        remove_ids = list(
            mol4.GetSubstructMatch(
                Chem.MolFromSmarts(
                    f"Br-{Chem.MolToSmarts(self.metal.atom)}-C(-[H])=C(-[H])-[H]"
                )
            )
        )
        assert (
            len(remove_ids) == 7
        ), f"Error in generation of mol2: only found remove_ids {remove_ids}"
        remove_ids.remove(mol4.GetSubstructMatch(self.metal.atom)[0])
        substructure_mask = np.ones_like(mol4_atoms, dtype=bool)
        substructure_mask[remove_ids] = False

        mol2_atoms = np.array(mol4_atoms)[substructure_mask]
        mol2_coords = np.array(mol4_opt_coords)[substructure_mask]

        # Determine connectivity for mol2 (convert adjacency matrix of mol4 to adjacency matrix of mol2)
        mol2_adj = np.copy(mol4_adj)
        for idx in sorted(remove_ids, reverse=True):
            mol2_adj = np.delete(mol2_adj, idx, 0)
            mol2_adj = np.delete(mol2_adj, idx, 1)

        # Optimization of mol2
        mol2_opt_results = orca_calculate(
            atoms=mol2_atoms,
            coords=mol2_coords,
            charge=charge,
            multiplicity=multiplicity,
            options=OPTAU,
            scr=calc_dir,
            n_cores=n_cores,
        )
        mol2_opt_coords, mol2_opt_energy = (
            mol2_opt_results["opt_coords"],
            mol2_opt_results["electronic_energy"],
        )

        self.lin = ac2xyz(mol2_atoms, mol2_opt_coords)
        # Check adjacency matrix after optimization of mol2
        if adjecency_check and not self.adjacency_check(
            mol2_adj, mol2_atoms, mol2_opt_coords, use_hueckel
        ):
            raise ValueError("Change in adjacency matrix after optimization of mol2.")

        if math.isnan(mol2_opt_energy):
            raise ValueError("Optimization of mol2 failed.")
        _logger.info(
            f"\nOptimization converged. Energy: {mol2_opt_energy:.04f} [Hartree]."
        )

        # Singlepoint of mol2
        mol2_sp_results = orca_calculate(
            atoms=mol2_atoms,
            coords=mol2_opt_coords,
            charge=charge,
            multiplicity=multiplicity,
            options=SP,
            scr=calc_dir,
            n_cores=n_cores,
        )
        mol2_sp_energy = mol2_sp_results["electronic_energy"]

        self.lin_energy = mol2_sp_energy
        _logger.info(f"Singlepoint Energy: {mol2_sp_energy:.04f} [Hartree].")
        if math.isnan(mol2_sp_energy):
            raise ValueError("Singlepoint of mol2 failed.")

        energy_difference = (
            mol4_sp_energy - mol2_sp_energy - self.fragment_energy
        ) * hartree2kcalmol
        _logger.info(f"Energy difference: {energy_difference:.04f} [kcal/mol].")
        self.energy_difference = energy_difference
        self.timing = time.time() - start_time

        try:
            shutil.rmtree(calc_dir)
        except FileNotFoundError:
            pass

    @staticmethod
    def adjacency_check(adj1, atoms, coords, use_hueckel=True):
        mol = ac2mol(atoms, coords)
        try:
            if use_hueckel:
                iteratively_determine_bonds(mol)
            else:
                mol = _determineConnectivity(mol)
        except ValueError:
            return False
        adj2 = Chem.GetAdjacencyMatrix(mol)
        return np.array_equal(adj1, adj2)

    def calculate_score(
        self,
        n_cores: int = 1,
        envvar_scratch: str = "SCRATCH",
        adjecency_check: bool = True,
        use_hueckel=True,
        sa: bool = False,
    ):
        self.calculate_energy_difference(
            n_cores=n_cores,
            envvar_scratch=envvar_scratch,
            adjecency_check=adjecency_check,
            use_hueckel=use_hueckel,
        )
        if not sa:
            self.sa = 1
        elif sa == "sa":
            sa1 = sa_target_score_clipped(self.ligands[0].mol)
            sa2 = sa_target_score_clipped(self.ligands[1].mol)
            self.sa = np.mean([sa1, sa2])
        elif sa == "sax":
            sa1 = sa(self.ligands[0].mol)
            sa2 = sa(self.ligands[1].mol)
            self.sa = np.mean([sa1, sa2])
        else:
            raise ValueError(f"Unknown sa option: {sa}")
        self.score = self.sa * gaussian(self.energy_difference, self.TARGET, 6)
