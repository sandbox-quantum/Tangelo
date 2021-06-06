import numpy as np

from pyscf import lib,gto

def as_scanner(grad):
    """
    Define scanner class specific to gradient method passed in.
    This relies on the ONIOM procedure's definition of the
    energy gradient.
    *args*:
        - **grad**: instance of oniom grad class, defined below
    *return*:
        - **ONIOM_GradScanner(grad)**
    """

    class ONIOM_GradScanner(grad.__class__, lib.GradScanner):
        """
        ONIOM gradient scanner class.
        """
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)
            self.verbose = self.model.verbose
            self.stdout = self.model.stdout
            self.max_memory = self.model.max_memory
            self.unit = 'au' #need to deal with this for geometric optimizer, relaxing need for Bohr units

            self.model.get_scanners()

        def __call__(self, geometry, **kwargs):

            if isinstance(geometry, gto.Mole): #define updated molecular geometry
                mol = geometry
            else:
                mol = self.model.mol.set_geom_(geometry, inplace=False)


            self.model.update_geometry(mol.atom) #update the molecular geometry for all fragments

            e_tot, de = self.kernel() #compute energy and gradient

            self.model.mol = mol #update molecule attributes
            self.mol = mol

            return e_tot, de

    return ONIOM_GradScanner(grad)



class oniom_grad:
    '''
    ONIOM gradient class. Each layer has its own solver-specific
    gradient which, when implemented as a scanner enables fast updating
    by saving the density-matrix, as well as gradients for SCF fields e.g.
    At the ONIOM level, we maintain this treatment of layer-specific gradients,
    and bring each together to define a high-level ONIOM energy gradient.
    '''

    def __init__(self,model):
        '''
        Initialize gradient object. If each layer does not yet have a
        gradient_scanner initialized, this is done at the start.
        *args*:
            - **model**: instance of oniom_model class
        '''

        self.model = model
        self.mol = self.model.mol #link to model molecule
        self.base = model
        if not np.product([hasattr(li, 'grad_scanner') for li in self.model.layers]):
            self.model.get_scanners(True)

    def kernel(self):
        '''
        The ONIOM gradient is defined in, for example, https://doi.org/10.1021/cr5004419
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
        '''

        e_tot = 0
        de = np.zeros((len(self.model.geometry), 3))
        for li in self.model.layers:
            etmp,dtmp = li.grad_scanner(li.mol)
            e_tot += etmp*li.layer_factor
            de += li.layer_factor*np.einsum('ij,ik->kj', dtmp, li.jacobian)

        return e_tot, de

    as_scanner = as_scanner
