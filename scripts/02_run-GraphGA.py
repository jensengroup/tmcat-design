import copy
import random

from catalystGA import GA, Ligand
from catalystGA.reproduction_utils import graph_crossover, graph_mutate
from catalystGA.utils import MoleculeOptions
from rdkit import Chem

from suzuki import SuzukiCatalyst


class GraphGA(GA):
    def __init__(
        self,
        mol_options: MoleculeOptions,
        population_size=5,
        n_generations=10,
        maximize_score=True,
        selection_pressure=1.5,
        mutation_rate=0.5,
        **kwargs
    ):
        super().__init__(
            mol_options=mol_options,
            population_size=population_size,
            n_generations=n_generations,
            maximize_score=maximize_score,
            selection_pressure=selection_pressure,
            mutation_rate=mutation_rate,
            **kwargs
        )

    def make_initial_population(self):
        with open("../smiles/pd-starting-population.smi", "r") as f:
            lines = f.readlines()

        population = []
        for line in lines:
            smi = line.rstrip()
            population.append(SuzukiCatalyst.from_smiles(smi))

        assert len(population) == self.population_size
        for i, ind in enumerate(population):
            ind.idx = (0, i)
        return population

    @staticmethod
    def crossover(ind1, ind2):
        ind_type = type(ind1)
        # choose one ligand at random from ind1 and crossover with random ligand from ind2, then replace this ligand in ind1 with new ligand
        ind1_ligands = copy.deepcopy(ind1.ligands)
        new_mol = None
        counter = 0
        while not new_mol:
            idx1 = random.randint(0, len(ind1_ligands) - 1)
            idx2 = random.randint(0, len(ind2.ligands) - 1)
            new_mol = graph_crossover(ind1.ligands[idx1].mol, ind2.ligands[idx2].mol)
            counter += 1
            if counter > 10:
                return None
        try:
            Chem.SanitizeMol(new_mol)
            # this will catch if new_mol has no donor atom
            new_ligand = Ligand(new_mol)
            ind1_ligands[idx1] = new_ligand
            child = ind_type(ind1.metal, ind1_ligands)
            child.assemble()
            return child
        except Exception:
            return None

    @staticmethod
    def mutate(ind):
        # pick one ligand at random, mutate and replace in ligand list
        idx = random.randint(0, len(ind.ligands) - 1)
        new_mol = None
        counter = 0
        while not new_mol:
            new_mol = graph_mutate(ind.ligands[idx].mol)
            counter += 1
            if counter > 10:
                return None
        try:
            Chem.SanitizeMol(new_mol)
            ind.ligands[idx] = Ligand(new_mol)
            ind.assemble()
            return ind
        except Exception:
            return None


if __name__ == "__main__":

    # Set Options for Molecule
    mol_options = MoleculeOptions(
        individual_type=SuzukiCatalyst,
    )

    ga = GraphGA(
        mol_options=mol_options,
        population_size=25,
        n_generations=25,
        mutation_rate=0.5,
        db_location="../data/02_graphGA.sqlite",
    )

    # Run the GA
    results = ga.run()
