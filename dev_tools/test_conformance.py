import unittest
import pycodestyle


class TestCodeFormat(unittest.TestCase):

    def test_conformance(self):
        style = pycodestyle.StyleGuide(quiet=False, config_file="pycodestyle")
        result = style.check_files(["../qsdk"])
        self.assertEqual(result.total_errors, 0, "Found code style errors and warnings.")
