"""Docstring"""

import numpy as np
from pyscf import gto, lib


def as_scanner(grad):
    """Define scanner class specific to gradient method passed in.
    This relies on the ONIOM procedure's definition of the
    energy gradient.
    *args*:
        - **grad**: instance of oniom grad class, defined below
    *return*:
        - **ONIOM_GradScanner(grad)**
    """

    class ONIOM_GradScanner(grad.__class__, lib.GradScanner):
        """ONIOM gradient scanner class. """

        def __init__(self, g):
            lib.GradScanner.__init__(self, g)
            self.verbose = self.oniom_model.verbose
            #self.stdout = self.oniom_model.stdout
            #self.max_memory = self.model.max_memory
            self.unit = "AU" #need to deal with this for geometric optimizer, relaxing need for Bohr units

            for fragment in self.oniom_model.fragments:
                fragment.get_scanners()

        def __call__(self, geometry, **kwargs):

            if isinstance(geometry, gto.Mole): #define updated molecular geometry
                mol = geometry
            else:
                mol = self.oniom_model.mol.set_geom_(geometry, inplace=False)

            # Update the molecular geometry for all fragments.
            self.oniom_model.update_geometry(mol.atom)

            # Compute energy and gradient.
            e_tot, de = self.kernel()

            #Update molecule attributes
            self.oniom_model.mol = mol
            self.mol = mol

            return e_tot, de

    return ONIOM_GradScanner(grad)


class ONIOMGradient:
    """ONIOM gradient class. Each layer has its own solver-specific
    gradient which, when implemented as a scanner enables fast updating
    by saving the density-matrix, as well as gradients for SCF fields e.g.
    At the ONIOM level, we maintain this treatment of layer-specific gradients,
    and bring each together to define a high-level ONIOM energy gradient.
    """

    def __init__(self, oniom_model):
        """
        Initialize gradient object. If each layer does not yet have a
        gradient_scanner initialized, this is done at the start.
        *args*:
            - **model**: instance of oniom_model class
        """

        self.oniom_model = oniom_model
        self.mol = self.oniom_model.mol #link to model molecule
        self.base = oniom_model

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
        Each layer is instantiated with its Jacobian, making this a trivial calculation done below here
        using the einsum method. This conveniently handles the different # of atoms in models, system,
        relying on the well-defined functional form of the link-atoms placement to enable proper treatment
        of all atoms in the system.
        *return*:
            - **e_tot**: float, total energy
            - **de**: numpy array of Nx3 float, with N # of atoms in the system
        """

        e_tot = 0
        de = np.zeros((len(self.oniom_model.geometry), 3))

        for fragment in self.oniom_model.fragments:

            grad_scanner_low = fragment.grad_scanners[0]
            etmp, dtmp = grad_scanner_low(fragment.mol_low)

            if fragment.solver_high:
                grad_scanner_high = fragment.grad_scanners[1]
                etmp_high, dtmp_high = grad_scanner_high(fragment.mol_high)

                etmp = etmp_high - etmp
                dtmp = dtmp_high - dtmp

            jacobian = self.oniom_model.get_jacobian(fragment)

            e_tot += etmp
            de += np.einsum('ij,ik->kj', dtmp, jacobian)

        return e_tot, de

    as_scanner = as_scanner
