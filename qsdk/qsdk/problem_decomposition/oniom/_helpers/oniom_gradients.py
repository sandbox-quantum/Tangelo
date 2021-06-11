"""ONIOM energy derivatives with respect nuclear coordinates. useful for the
geometry optimization of an ONIOM model.
"""

import numpy as np
from pyscf import gto, lib


def as_scanner(grad):
    """Define scanner class specific to gradient method passed in. This relies
    on the ONIOM procedure's definition of the energy gradient.

    Args:
        grad (ONIOMGradient): Instance of oniom grad class.

    Returns:
        ONIOM_GradScanner: Gradient scanner for nucleus energy derivatives.
    """

    class ONIOM_GradScanner(grad.__class__, lib.GradScanner):
        """ONIOM gradient scanner class."""

        def __init__(self, g):

            lib.GradScanner.__init__(self, g)

            self.verbose = self.oniom_model.verbose
            self.stdout = self.oniom_model.mol.stdout
            # Need to deal with this for geometric optimizer, relaxing need for Bohr units.
            self.unit = "bohr"

            # Getting all scanners for every fragments.
            for fragment in self.oniom_model.fragments:
                fragment.get_scanners()

        def __call__(self, geometry):

            # Updating the molecule geometry.
            if isinstance(geometry, gto.Mole):
                mol = geometry
            else:
                mol = self.oniom_model.mol.set_geom_(geometry, inplace=False)

            # Update the molecular geometry for all fragments.
            self.oniom_model.update_geometry(mol.atom)

            # Compute energy and gradient.
            e_tot, de = self.kernel()

            # Update molecule attributes.
            self.oniom_model.mol = mol
            self.mol = mol

            return e_tot, de


    return ONIOM_GradScanner(grad)


class ONIOMGradient:
    """ONIOM gradient class. Each layer has its own solver-specific gradient which,
    when implemented as a scanner enables fast updating by saving the density-matrix,
    as well as gradients for SCF fields e.g. at the ONIOM level, we maintain this
    treatment of layer-specific gradients, and bring each together to define a
    high-level ONIOM energy gradient.
    """

    def __init__(self, oniom_model):

        self.oniom_model = oniom_model
        self.mol = self.oniom_model.mol
        self.base = oniom_model

        # Getting all scanners for every fragments if they are not defined yet.
        if not np.product([hasattr(f, "grad_scanners") for f in self.oniom_model.fragments]):
            for fragment in self.oniom_model.fragments:
                fragment.get_scanners()

    def kernel(self):
        """The ONIOM gradient is defined in, for example, https://doi.org/10.1021/cr5004419
        Gradient is defined with respect to atomic coordinates of the system, R = R_SYS.
        dE_ONIOM/dR = dE_LOW[R]/dR + sum_i dE_HIGH_i[R_i]/dR - dE_LOW_i[R_i]/dR
        one can use chain rule to express

        d/dR = dR_i/dR d/dR_i = J(R_i,R) d/dR_i
        dE_ONIOM/dR = dE_LOW[R]/dR + sum_i dE_HIGH_i[R_i]/dR_i . J(R_i,R) - dE_LOW_i[R_i]/dR_i.J(R_i,R)

        Each layer is instantiated with its Jacobian, making this a trivial
        calculation made below here using the einsum method. This conveniently
        handles the different number of atoms in models, system, relying on the
        well-defined functional form of the link-atoms placement to enable proper
        treatment of all atoms in the system.

        Returns:
            float, np.array: Total energy and energy derivatives, Nx3 matrix of
                floats.
        """

        e_tot = 0
        de = np.zeros((len(self.oniom_model.geometry), 3))

        # Computing energy and gradients for each fragments.
        for fragment in self.oniom_model.fragments:

            grad_scanner_low = fragment.grad_scanners[0]
            etmp, dtmp = grad_scanner_low(fragment.mol_low)

            if fragment.solver_high:
                grad_scanner_high = fragment.grad_scanners[1]
                etmp_high, dtmp_high = grad_scanner_high(fragment.mol_high)

                etmp = etmp_high - etmp
                dtmp = dtmp_high - dtmp

            e_tot += etmp
            de += np.einsum('ij,ik->kj', dtmp, fragment.jacobian)

        return e_tot, de

    as_scanner = as_scanner
