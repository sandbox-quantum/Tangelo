# Copyright 2021 Good Chemistry Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import numpy as np

from tangelo.toolboxes.molecular_computation.molecule import Molecule, SecondQuantizedMolecule

sup = np.testing.suppress_warnings()
warnings.filterwarnings("ignore", message="Using default_file_mode other than 'r' is deprecated")
warnings.filterwarnings("ignore", message="`np")
warnings.filterwarnings("ignore", category=DeprecationWarning)
sup.filter(np.core)
