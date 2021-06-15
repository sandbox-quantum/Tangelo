import unittest
import subprocess


def run_script_as_test(script_path):
    try:
        subprocess.run([script_path], check=True)
    except Exception as e:
        raise


class TestNotebooks(unittest.TestCase):
    """ After Python notebooks have been turned into scripts, run them as unittests """

    def test_basics_notebook(self):
        run_script_as_test('./1.the_basics.py')

    def test_noisy_simulation_notebook(self):
        run_script_as_test('./3.noisy_simulation.py')


if __name__ == "__main__":
    unittest.main()
