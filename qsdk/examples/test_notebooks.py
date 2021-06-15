import unittest
import subprocess


def run_script_as_test(script_path):
    try:
        subprocess.run([script_path], check=True)
    except Exception as e:
        raise


class TestNotebooks(unittest.TestCase):
    """ After Python notebooks have been turned into scripts, run them as unittests """

    def test_dmet_notebook(self):
        run_script_as_test('./dmet.py')

    def test_vqe_notebook(self):
        run_script_as_test('./vqe.py')

    def test_vqe_custom_ansatz_notebook(self):
        run_script_as_test('./vqe_custom_ansatz.py')


if __name__ == "__main__":
    unittest.main()
