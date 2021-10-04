import unittest
import subprocess
import os


def run_notebook_as_test(notebook_path):
    """ Convert python notebook into equivalent script, and run it. Return error if any. """
    try:
        subprocess.run(['jupyter', 'nbconvert', '--to', 'script', notebook_path])
        script_path = os.path.splitext(notebook_path)[0] + '.py'
        subprocess.run(['chmod', '+x', script_path])
        subprocess.run([script_path], check=True)
    except Exception as err:
        raise RuntimeError(err.args)


class TestNotebooks(unittest.TestCase):
    """ Turn target Python notebooks into script, run them as unittests (pass = no errors at runtime) """

    def test_backendbuddy_basics_notebook(self):
        run_notebook_as_test('backendbuddy/1.the_basics.ipynb')

    def test_backendbuddy_noisy_simulation_notebook(self):
        run_notebook_as_test('backendbuddy/3.noisy_simulation.ipynb')

    def test_dmet_notebook(self):
        run_notebook_as_test('./dmet.ipynb')

    def test_vqe_notebook(self):
        run_notebook_as_test('./vqe.ipynb')

    def test_vqe_custom_ansatz_notebook(self):
        run_notebook_as_test('./vqe_custom_ansatz_hamiltonian.ipynb')

    def test_oniom_notebook(self):
        run_notebook_as_test('./problem_decomposition_oniom.ipynb')

    @unittest.skip("Requires qemist cloud access")
    def test_qemist_cloud_hardware_experiments_notebook(self):
        run_notebook_as_test('./qemist_cloud_hardware_experiments_braket.ipynb')


if __name__ == "__main__":
    unittest.main()