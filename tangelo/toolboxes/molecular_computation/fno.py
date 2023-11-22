import time
from functools import reduce

from pyscf import lib, ao2mo, cc, mp, scf
from pyscf.lib import logger
import numpy as np


def do_frozen_core_mp2(mean_field, frozen_occupied):
    """Compute mp2 object and compute correlation energy for full virtual space
    Args:
        mean_field: pyscf UHF mean_field object
        frozen_occupied:
    Returns: mp2 object and mp2 corr. energy for full virtual space
    """

    pt = mp.UMP2(mean_field).set(frozen=frozen_occupied)
    pt.verbose = 0
    fMP2_energy, _ = pt.kernel()

    return pt, fMP2_energy


class FNO(lib.StreamObject):
    def __init__(self, molecule, perturb_theory: mp, frac_fno):
        """This is FNO class which needs HF, MP2 object along with fraction of FNO occupance %

        Args:
            molecule (SecondQuantizedMolecule): Molecule object
            pt: MP2 object
            frac_fno: % of FNO occupation (%alpha, %beta)
        """

        self._scf = molecule.mean_field
        self.scf_mo_coeff = molecule.mean_field.mo_coeff
        # compute the Fock matrix in new mo basis
        self.fock_ao = molecule.mean_field.get_fock()
        self._mp = perturb_theory
        # obtain MP2 aplitudes.  Can be unfolded into t2aa, t2ab, t2bb <= t2
        self.t2 = perturb_theory.t2
        # list of occupied orbitals that are frozen in MP2 calculation
        self.frozen_list = perturb_theory.frozen

        # user data
        self.frac_fno = frac_fno

        # I need to compute this for later use
        self.n_active_occupied = [len(molecule.active_occupied[0]), len(molecule.active_occupied[1])]
        self.n_active_virtual = list()
        self.n_frozen_virtual = list()
        self.n_frozen_occupied = [len(molecule.frozen_occupied[0]), len(molecule.frozen_occupied[1])]
        self.mo_coeff = []
        self.mo_energy = []
        self.fno_occupancy = []

        # I need the total virtual and for FNO truncation
        self.frozen_vrt = list()
        self.tvrt = list()
        self._nocc = list()
        for ispin in range(2):
            self._nocc.append(len(molecule.mo_occ[ispin][molecule.mo_occ[ispin] > 0]))
            tvrt = self._scf.mo_coeff[ispin].shape[1] - self._nocc[ispin]
            self.tvrt.append(tvrt)

    def _compute_mp2_vv_density(self, t2=None):
        '''Create the virtual-virtual block MP2 density for alpha and beta.'''
        if t2 is None:
            t2 = self.t2
        t2aa, t2ab, t2bb = t2

        dvva = lib.einsum('mnae,mnbe->ba', t2aa.conj(), t2aa) * .5
        dvva += lib.einsum('mnae,mnbe->ba', t2ab.conj(), t2ab)
        dvvb = lib.einsum('mnae,mnbe->ba', t2bb.conj(), t2bb) * .5
        dvvb += lib.einsum('mnea,mneb->ba', t2ab.conj(), t2ab)

        return (dvva + dvva.conj().T, dvvb + dvvb.conj().T)

    def _diagonalize_density(self, D, ispin):
        """Diagonalize density and return eval, evec and put no of
        FNO orbitals in appropriate space
        Args:
            D (array): density matrix
        Returns
            array : eigenvectors of density matrix in appropriate space
        """
        # obtain natual orbitals
        occ, unitary = diagonalize_and_reorder(D)

        # find out active number of FNO orbitals
        self.frozen_vrt.append(get_number_of_frozen_vrt(self.frac_fno[ispin], occ))

        self.fno_occupancy.append(occ)
        return unitary

    def compute_mo_coeff(self, unitary, spin):
        """Get transformed mo_coeff using FNO for given spin
        Args:
            unitary (array): Transformation unitary for virtual-virtual block
            spin (int): 0 = alpha, 1 = beta
        Returns:
            array: FNO Transformed mo_coeff
        """
        # get number of occupied orbitals
        n_occupied = self._nocc[spin]
        # get number of active virtual orbitals
        n_active_virtual = self.frozen_vrt[spin]
        # get spin block of scf mo_coeff
        c_scf = self.scf_mo_coeff[spin]
        # slice the virtual block from mo_coeff
        c_aa = c_scf[:, n_occupied:]
        # Transform the vitual block with fno unitary
        c_aa_fno = np.dot(c_aa, unitary)
        # slice the active part of virutal orbitals
        return c_aa_fno[:, :n_active_virtual]

    def compute_aa(self, mo_coeff, spin):
        """Compute FNO mo_coeff and mo_energy for given spin"""
        fock = self.fock_ao[spin]
        # obtain Fock matrix in mo form
        fock_mo = mo_coeff.T.dot(fock).dot(mo_coeff)
        mo_energy_sc, unitary_sc = diagonalize_and_reorder(fock_mo, order=1)
        # print('MO energy in FNO',mo_energy_sc)
        mo_coeff_new = np.dot(mo_coeff, unitary_sc)

        # update the mo_vectors
        scf_mo_coeff = self.scf_mo_coeff[spin]

        nocc_aa = self._nocc[spin]
        fno_vrt_aa = self.frozen_vrt[spin]
        active_space_aa = nocc_aa+fno_vrt_aa

        # update the mo_coeff
        mo_coeff_occ = scf_mo_coeff[:, :nocc_aa]
        mo_coeff_sc = mo_coeff_new[:, :]
        mo_coeff_vrt = scf_mo_coeff[:, active_space_aa:]
        mo_coeff = np.hstack((mo_coeff_occ, mo_coeff_sc, mo_coeff_vrt))

        # update the mo_energy
        E_occ = self._scf.mo_energy[spin][:nocc_aa]
        E_sc = mo_energy_sc[:]
        E_vrt = self._scf.mo_energy[spin][active_space_aa:]

        mo_energy = np.hstack((E_occ, E_sc, E_vrt))

        self.n_active_virtual.append(fno_vrt_aa)
        self.mo_coeff.append(mo_coeff)
        self.mo_energy.append(mo_energy)

        return mo_coeff, mo_energy

    def compute_fno(self):
        """Compute FNO Transformed mo_coeff & mo_energy
        """
        # compute MP2 density matrix of vv block
        Dvv = self._compute_mp2_vv_density()  # Dvv -> [Dvv_aa, Dvv_bb]
        self.mo_coeff = []
        mo_coeff_fno = []
        for spin in [0, 1]:
            # Diagonalize the density
            unitary = self._diagonalize_density(Dvv[spin], spin)
            # do the transformation and slicing etc
            mo_coeff_fno.append(self.compute_mo_coeff(unitary, spin))
            _ = self.compute_aa(mo_coeff_fno[spin], spin)
            n_frozen_virtual = self.tvrt[spin] - self.n_active_virtual[spin]
            self.n_frozen_virtual.append(n_frozen_virtual)

        return self.mo_coeff, self.mo_energy

    def get_active_indices(self):
        pt = self._mp
        self.active_indices = []
        self.frozen_indices = []
        for ispin in [0, 1]:
            # all are False
            moidx = np.zeros(pt.mo_occ[ispin].size, dtype=bool)
            # true starting from first index till active orbital index
            occ_active_vrt_index = self.n_frozen_occupied[ispin] + self.n_active_occupied[ispin] + self.n_active_virtual[ispin]
            moidx[:occ_active_vrt_index] = True
            # find if user has asked for frozen occupied orbitals
            # if no frozen_list is provided skip
            # fno.frozen_list is coming from mp2
            # if user said None
            if self.frozen_list is None:   # this may be run into troubles
                pass
            # below is possible scenario my code is working
            elif isinstance(self.frozen_list, (int, np.integer)):
                moidx[:self.frozen_list] = False
            # when frozen_list is a list, which is not at present
            else:
                moidx[self.frozen_list[ispin]] = False
            # obtain the frozen indices
            active_indices = np.where(moidx)[0]
            frozen_indices = np.where(moidx == False)[0]  # Doesn't work if using np.where(moidx is False)[0]
            # append alpha and beta to the active_indices
            self.active_indices.append(list(active_indices))
            self.frozen_indices.append(list(frozen_indices))

        return self.active_indices

    def compute_full_integrals(self):
        """Compute 1-electron and 2-electron integrals
        The return is formatted as
        [As x As]*2 numpy array h_{pq} for alpha and beta blocks
        [As x As x As x As]*3 numpy array storing h_{pqrs} for alpha-alpha, alpha-beta, beta-beta blocks
        """
        # step 1 : find nao, nmo (atomic orbitals & molecular orbitals)

        # molecular orbitals (alpha and beta will be the same)
        # Lets take alpha blocks to find the shape and things

        # molecular orbitals
        nmo = self.nmo = self.mo_coeff[0].shape[1]
        # atomic orbitals
        nao = self.nao = self.mo_coeff[0].shape[0]

        # step 2 : obtain Hcore Hamiltonian in atomic orbitals basis
        hcore = self._scf.get_hcore()

        # step 3 : obatin two-electron integral in atomic basis
        eri = ao2mo.restore(8, self._scf._eri, nao)

        # step 4 : create the placeholder for the matrices
        # one-electron matrix (alpha, beta)
        self.hpq = []

        # step 5 : do the mo transformation
        # step the mo coeff alpha and beta
        mo_a = self.mo_coeff[0]
        mo_b = self.mo_coeff[1]

        # mo transform the hcore
        self.hpq.append(mo_a.T.dot(hcore).dot(mo_a))
        self.hpq.append(mo_b.T.dot(hcore).dot(mo_b))

        # mo transform the two-electron integrals
        eri_a = ao2mo.incore.full(eri, mo_a)
        eri_b = ao2mo.incore.full(eri, mo_b)
        eri_ba = ao2mo.incore.general(eri, (mo_a, mo_a, mo_b, mo_b), compact=False)

        # Change the format of integrals (full)
        eri_a = ao2mo.restore(1, eri_a, nmo)
        eri_b = ao2mo.restore(1, eri_b, nmo)
        eri_ba = eri_ba.reshape(nmo, nmo, nmo, nmo)

        # # convert this into the order OpenFemion like to receive
        two_body_integrals_a = np.asarray(eri_a.transpose(0, 2, 3, 1), order='C')
        two_body_integrals_b = np.asarray(eri_b.transpose(0, 2, 3, 1), order='C')
        two_body_integrals_ab = np.asarray(eri_ba.transpose(0, 2, 3, 1), order='C')

        # Gpqrs has alpha, alphaBeta, Beta blocks
        self.Gpqrs = (two_body_integrals_a, two_body_integrals_ab, two_body_integrals_b)

        return self.hpq, self.Gpqrs


def diagonalize_and_reorder(matrix_provided, order=-1):
    """Diagonalize the density matrix and sort the eigenvalues and corresponding eigenvectors,"""
    evals, evecs = np.linalg.eigh(matrix_provided)
    idx = evals.argsort()[::order]
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


def get_number_of_frozen_vrt(frac_fno, FNO_occup):
    # Devise a rule to truncate the MO space.
    # Strategy one fraction of total FNO occupancy.
    total_FNO_occ = np.sum(FNO_occup)

    percentage_FNO_to_include = frac_fno / 100

    fno_occ_tmp = 0.0
    FNO_vrt = 0

    for occpancy in FNO_occup:
        thres = fno_occ_tmp / total_FNO_occ
        if thres <= percentage_FNO_to_include:
            FNO_vrt += 1
            fno_occ_tmp += occpancy

    return FNO_vrt


def rhf_get_frozen_mask(fno_cc, spin):
    # Start with all (orbitals to be) False first
    moidx = np.zeros(fno_cc.mo_occ[spin].size, dtype=bool)

    # Starting from first index make index True, Keep n_frozen_virtual as False
    if isinstance(fno_cc.frozen, (int, np.integer)):
        moidx[:fno_cc.frozen+fno_cc.n_active_occupied[spin]+fno_cc.n_active_virtual[spin]] = True
    else:
        moidx[:fno_cc.frozen[spin]+fno_cc.n_active_occupied[spin]+fno_cc.n_active_virtual[spin]] = True

    if fno_cc.frozen_list is None:
        pass
    # check if the frozen_list is an interger
    elif isinstance(fno_cc.frozen_list, (int, np.integer)):
        moidx[:fno_cc.frozen_list] = False  # make n_frozen_occupied as False
    # if frozen_list is given make those list of orbital as False
    else:
        moidx[fno_cc.frozen_list[spin]] = False
    return moidx


def old_get_frozen_mask(fno_cc):
    moidx = np.zeros(fno_cc.mo_occ.size, dtype=bool)  # all are False
    moidx[:fno_cc.frozen+fno_cc.n_active_occupied+fno_cc.n_active_virtual] = True  # true starting from first index till active orbital index
    if fno_cc.frozen_list is None:
        pass
    elif isinstance(fno_cc.frozen_list, (int, np.integer)):
        moidx[:fno_cc.frozen_list] = False
    else:
        moidx[fno_cc.frozen_list] = False  # make the frozen index False
    return moidx


def get_frozen_mask(fno_cc):
    """
    fno_cc:
        open shell CC object
    return :
        index for active occupied & virtual orbitals
    """
    moidxa = rhf_get_frozen_mask(fno_cc, 0)
    moidxb = rhf_get_frozen_mask(fno_cc, 1)

    return moidxa, moidxb


if __name__ == "__main__":
    from tangelo.toolboxes.molecular_computation.molecule import SecondQuantizedMolecule

    xyz_CF2 = """
        C   0.0000 0.0000 0.5932
        F   0.0000 1.0282 -0.1977
        F   0.0000 -1.0282 -0.1977
    """

    mol = SecondQuantizedMolecule(xyz_CF2, 0, 2, basis="cc-pvdz", uhf=True, frozen_orbitals=[[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]])
    print(mol.n_active_ab_electrons)

    pt, frozen_occupied_mp2_etot = do_frozen_core_mp2(mol.mean_field, mol.frozen_occupied)
    fno = FNO(mol, pt, [40, 40])
    fno_mo_coeff, mo_energy = fno.compute_fno()

    active = fno.get_active_indices()
    print(f"The FNO truncated active orbitals are {active}")
    frozen_orbitals = [[i for i in range(mol.n_mos) if i not in active[0]], [i for i in range(mol.n_mos) if i not in active[1]]]
    print(frozen_orbitals)

    mol_frozen = SecondQuantizedMolecule(xyz_CF2, 0, 2, basis="cc-pvdz", uhf=True, frozen_orbitals=frozen_orbitals)
    mol_fno = mol.freeze_mos(frozen_orbitals, inplace=False)
    mol_fno.mo_coeff = fno_mo_coeff
