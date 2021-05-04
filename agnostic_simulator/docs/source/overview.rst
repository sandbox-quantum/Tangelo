
*******************
Overview
*******************

Welcome to the Agnostic Simulator documentation !

.. contents:: Table of Contents



Purpose
=======

This package aims at providing you with tools to easily generate and simulate quantum circuits on different backends,
from classical simulators (Qulacs, Qiskit, QDK, ...) to actual quantum processors developed by hardware partners
(IonQ, Honeywell...). These backends have different features and performance, and may be appealing for different
reasons, whether they are to be used for stand-alone experiments destined to run on a QPU, or to be used as the compute
backbone of applications relying on the simulation of quantum circuits.

On top of that, this package can host any innovation aiming at improving the resource requirements, reducing the number
of measurements, investigating noise-correction techniques or other forms of post-processing related to quantum circuit simulation.

.. note::
    If you wish to have a very fine control over features available under a specific
    target backend and you are not afraid to write the code for it, then using the corresponding package directly may
    make more sense. I recommend you check what `agnostic_simulator` can do for you first, and don't hesitate
    to request a feature if you think it would make a difference.


Write reusable code and target any supported backend
====================================================

Users can use this package to write highly reusable code, which can be deployed or run with minimal effort on any supported
quantum backend. It also allows them to easily generate backend-specific objects describing the quantum circuit of interest, which can
be useful for collaborations or publications. Quantum circuits can be saved as Python objects using the Python pickle
module, or written directly to a file if they are a human-readable format or other serialized representation.


.. list-table:: Currently supported simulator backends
   :widths: 25 75 25
   :header-rows: 1

   * - Backend
     - Features
     - References
   * - Qulacs
     - statevector simulator, noisy simulator, GPU-enabled
     - https://en.qunasys.com/tools
   * - Qiskit
     - statevector simulator, noisy simulator
     - https://qiskit.org
   * - QDK
     - statevector simulator
     - https://www.microsoft.com/en-us/quantum/development-kit
   * - ProjectQ
     - statevector simulator
     - https://projectq.ch

On top of these simulators, this package facilitates job submission to IonQ and Honeywell QPU services, and is able
to generate code compatible with Microsoft's Azure Quantum services, targeting various QPUs.


This package provides generic data-structures representing "abstract" quantum gates and circuits, which are then parsed and
"translated" to generate backend-specific objects, which can be, for example, a Qiskit QuantumCircuit object, a Q# code file or some
human-readable serialized object. The Simulator class provides a uniform interface to all backends, able to simulate
a quantum circuit, or compute the expectation of an operator with regards to a given state preparation. This allows users
to focus on the nature of their simulation (noise, number of shots...) and their overall application, rather
than low-level code that is specific to a single target backend.

For more details, please don't hesitate to go through the Jupyter notebooks available in the `example` folder or the
docs !
