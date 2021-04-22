import abc


class ProblemDecomposition(abc.ABC):

    def __init__(self):
        self.electronic_structure_solver = None
        self.electron_localization_method = None

    @abc.abstractmethod
    def simulate(self, molecule, fragment_atoms, mean_field=None):
        pass
