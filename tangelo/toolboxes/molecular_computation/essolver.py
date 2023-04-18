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

import abc


class ESSolver(abc.ABC):
    """Instantiate Electronic Structure integration"""
    def __init__(self):
        pass

    @abc.abstractmethod
    def set_basic_data(self, tmol):
        pass

    @abc.abstractmethod
    def compute_mean_field(self, tmol):
        pass

    @abc.abstractmethod
    def get_integrals(self, tmol, mo_coeff=None):
        pass
