#!/bin/bash

pytest --nbmake "dmet.ipynb"
pytest --nbmake "vqe.ipynb"
pytest --nbmake "vqe_custom_ansatz.ipynb"
pytest --nbmake "user_notebooks/adapt.ipynb"
