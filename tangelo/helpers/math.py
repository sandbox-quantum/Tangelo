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

"""This file provides utility functions to perform mathematical operations."""

import numpy as np


def bool_col_echelon(bool_array):
    """Function to transform a boolean array into its column echelon form
    (transpose of the row-echelon form).

    Args:
        bool_array (array of bool): Self-explanatory.
    """

    pivot = bool_array.shape[1] - 1

    # This is done on matrix where the number of rows > number of columns.
    active_rows = bool_array.shape[0] - bool_array.shape[1] - 1

    # Column by column, starting from the last one, perform gaussian elimination
    # on the columns.
    for row in range(active_rows, -1, -1):

        # Identify if there is a 1 in the selected row and columns.
        if bool_array[row, :pivot+1].max():
            indices = np.where(bool_array[row, :pivot+1])[0]

            # If there are more than one 1 in the selected part, perform
            # gaussian elimination (done with a XOR because the input is a
            # boolean array.
            if len(indices) > 1:
                for i in range(1, len(indices)):
                    bool_array[:, indices[i]] = np.logical_xor(bool_array[:, indices[i]],
                                                               bool_array[:, indices[0]])

            # If there is only one 1 in the selected part, the pivot is
            # decremented.
            if len(indices) > 0:
                bool_array[:, (indices[0], pivot)] = bool_array[:, (pivot, indices[0])]
                pivot -= 1

    return bool_array
