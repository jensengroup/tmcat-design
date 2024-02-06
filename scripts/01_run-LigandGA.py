from catalystGA import GA, Ligand, Metal
from catalystGA.reproduction_utils import list_crossover, list_mutate
from catalystGA.utils import MoleculeOptions
from rdkit import Chem

from suzuki import SuzukiCatalyst


class LigandGA(GA):
    def __init__(
        self,
        mol_options: MoleculeOptions,
        population_size=25,
        n_generations=50,
        maximize_score=True,
        selection_pressure=1.5,
        mutation_rate=0.5,
        donor_atoms_smarts_match=True,
        **kwargs,
    ):
        super().__init__(
            mol_options=mol_options,
            population_size=population_size,
            n_generations=n_generations,
            maximize_score=maximize_score,
            selection_pressure=selection_pressure,
            mutation_rate=mutation_rate,
            donor_atoms_smarts_match=donor_atoms_smarts_match,
            **kwargs,
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
    def crossover(ind1: SuzukiCatalyst, ind2: SuzukiCatalyst) -> SuzukiCatalyst:
        """Crossover the ligands of two SuzukiCatalysts."""
        ind_type = type(ind1)
        new_ligands = list_crossover(ind1.ligands, ind2.ligands, n_cutpoints=1)
        child = ind_type(ind1.metal, new_ligands)
        child.assemble()
        return child

    @staticmethod
    def mutate(ind: SuzukiCatalyst) -> SuzukiCatalyst:
        """Mutate one ligand of a SuzukiCatalyst."""
        new_ligands = list_mutate(ind.ligands, ligands_list)
        ind.ligands = new_ligands
        ind.assemble()
        return ind


if __name__ == "__main__":
    # Set Options for Molecule
    mol_options = MoleculeOptions(
        individual_type=SuzukiCatalyst,
    )

    # read ligands
    ligands_list = []
    with open("../smiles/ligands.smi", "r") as f:
        for line in f:
            smi = line.rstrip()
            ligands_list.append(Ligand(Chem.MolFromSmiles(smi)))
    ligands_list = list(set(ligands_list))

    metals_list = [Metal("Pd")]

    ga = LigandGA(
        mol_options=mol_options,
        population_size=25,
        n_generations=10,
        mutation_rate=0.5,
        db_location="../data/01_ligandGA.sqlite",
    )

    # Run the GA
    results = ga.run()
