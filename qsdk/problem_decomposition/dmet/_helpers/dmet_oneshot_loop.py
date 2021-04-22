#   Copyright 2019 1QBit
#   
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
A code for DMET one-shot calculation
"""
import numpy as np
import scipy
from dmet_bath import dmet_fragment_bath
from dmet_onerdm import dmet_low_rdm, dmet_fragment_rdm
from dmet_scf_guess import dmet_fragment_guess
from dmet_scf import dmet_fragment_scf
from dmet_cc_classical import dmet_fragment_cc_classical
from dmet_cc_quantum import dmet_fragment_cc_quantum

def dmet_oneshot_exe(input_dmet, dmet_orbs, orb_list, orb_list2):
    """
    This is the code for one-shot DMET calculation
    :param input_dmet: The dmet input object (dmet_input class)
    :param dmet_orbs: The dmet orbs object (from dmet_orbitals)
    :param orb_list: The number of orbitals for each DMET calculation
    :param orb_list2: Lists of the minimum and maximum of the IAO label for each DMET calculation
    :return: DMET energy and chemical potential
    """

    # Initialize chemical potential
    chemical_potential=0.0

    # Optimize the DMET energy and chemical potential
    dmet_energy, chemical_potential = dmet_chemical_potential(input_dmet, dmet_orbs, orb_list, orb_list2, chemical_potential)

    print(' \t*** DMET Cycle Done *** ')
    print(' \tDMET Energy ( a.u. ) = '+'{:17.10f}'.format(dmet_energy))
    print(' \tChemical Potential   = '+'{:17.10f}'.format(chemical_potential))
    print(' ')

    return dmet_energy, chemical_potential

def dmet_chemical_potential(input_dmet, dmet_orbs, orb_list, orb_list2, chemical_potential):
    """
    Initialize the SCF loop for the chemical potential
    In one-shot DMET calculations, the iteration continues until chemical potential is connverged to 0
    :param input_dmet: The dmet input object (dmet_input class)
    :param dmet_orbs: The dmet orbs object (from dmet_orbitals)
    :param orb_list: The number of orbitals for each DMET calculation
    :param orb_list2: Lists of the minimum and maximum of the IAO label for each DMET calculation
    :param chemical_potential: the chemical potential to be optimized =0 (for consistency over the entire system)
    :return: DMET energy and chemical potential
    """

    # Initialize the energy list and SCF procedure employing newton-raphson algorithm
    energy = []
    chemical_potential = scipy.optimize.newton(dmet_num_electron, chemical_potential, args = (input_dmet, dmet_orbs, orb_list, orb_list2, energy))

    # Get the final energy value
    niter = len(energy)
    dmet_energy = energy[niter-1]

    return dmet_energy, chemical_potential

def dmet_num_electron(chemical_potential, input_dmet, dmet_orbs, orb_list, orb_list2, energy_list):
    """
    Obtain the difference of the number of electrons of the DMET calculation by summing up the trace of RDM for each calculation
    :param input_dmet: The dmet input object (dmet_input class)
    :param chemical_potential: The chemical potential in the previous iteration
    :param dmet_orbs: The dmet orbs object (from dmet_orbitals)
    :param orb_list: The number of orbitals for each DMET calculation
    :param orb_list2: Lists of the minimum and maximum of the IAO label for each DMET calculation
    :param energy_list: List of the DMET energies (For each iteration)
    :return: The difference of the number of electrons
    """

    # Print the iteration number
    niter = len(energy_list)+1
    print(" \tIteration = ", niter)
    print(' \t----------------')
    print(' ')
    # Obtain the number of electrons from DMET calculations
    num_electron_temp = dmet_frag_loop(input_dmet, dmet_orbs, orb_list, orb_list2, energy_list, chemical_potential)
    print(" \tNumber of Active Electrons     = ", dmet_orbs.number_active_electrons)
    print(" \tNumber of the Sum of Electrons = "+'{:12.8f}'.format(num_electron_temp))

    # Obtain the difference of the number of electrons
    number_of_electron_difference = num_electron_temp - dmet_orbs.number_active_electrons
    print(" \tElectron Number Difference     = "+'{:12.8f}'.format(number_of_electron_difference))
    print(' ')

    return number_of_electron_difference

def dmet_frag_loop(input_dmet, dmet_orbs, orb_list, orb_list2, energy_list, chemical_potential):
    """
    The main loop of the one-shot DMET calculation
    :param input_dmet: The dmet input object (dmet_input class)
    :param dmet_orbs: The dmet orbs object (from dmet_orbitals)
    :param orb_list: The number of orbitals for each DMET calculation
    :param orb_list2: Lists of the minimum and maximum of the IAO label for each DMET calculation
    :param energy_list: List of the DMET energies (For each iteration)
    :param chemical_potential: The chemical potential in the previous iteration
    :return: Number of electrons (sum of trace of the RDMs over each fragment)
    """

    # Obtain the one particle RDM from low-level calculation of the entire system
    onerdm_low = dmet_low_rdm(dmet_orbs.active_fock, dmet_orbs.number_active_electrons)

    # Initialize some parameters
    energy_temp = 0.0
    number_of_electron = 0.0

    # Loop over each fragment
    for i, norb in enumerate(orb_list):
        print("\t\tFragment Number : # ", i+1)
        print('\t\t------------------------')
        t_list=[]
        t_list.append(norb)
        temp_list = orb_list2[i]

        # Construct bath orbitals
        bath_orb, e_occupied = dmet_fragment_bath(dmet_orbs.mol_full, t_list, temp_list, onerdm_low)

        # Obtain one particle rdm for a fragment
        norb_high, nelec_high, onerdm_high = dmet_fragment_rdm(t_list, bath_orb, e_occupied, dmet_orbs.number_active_electrons)

        # Calculate matrices for the Hamiltonian of a fragment
        one_ele, fock, two_ele = dmet_orbs.dmet_fragment_hamiltonian(bath_orb, norb_high, onerdm_high)

        # Construct guess orbitals for fragment SCF calculations
        print("\t\tNumber of Orbitals ",norb_high)
        print("\t\tNumber of Electrons ",nelec_high)
        guess_orbitals = dmet_fragment_guess(t_list, bath_orb, chemical_potential, norb_high, nelec_high, dmet_orbs.active_fock)

        # Carry out SCF calculation for a fragment
        mf_fragments, fock_frag_copy, mol_frag = dmet_fragment_scf(t_list, two_ele, fock, nelec_high, \
                                                         norb_high, guess_orbitals, chemical_potential)
        if mf_fragments.converged:
            print("\t\tSCF Converged ")
        else:
            print("\t\tSCF NOT CONVERGED !!!")
            exit()

        print("\t\tSCF Energy                      = "+'{:17.10f}'.format(mf_fragments.e_tot))

        # Perform high-level CC calculation for a fragment and calculate energy for a fragment
        # Perform quantum simulation
        if(input_dmet.quantum == 1):
            print("norb_high = ",norb_high)
            print("nelec_high = ",nelec_high)
            fragment_energy, onerdm_frag, total_energy, total_energy_rdm = dmet_fragment_cc_quantum(input_dmet, mf_fragments, fock_frag_copy, t_list, one_ele, two_ele, fock, norb_high, nelec_high, mol_frag)
        # Perform classical simulation
        else:
            fragment_energy, onerdm_frag, total_energy, total_energy_rdm = dmet_fragment_cc_classical(mf_fragments, fock_frag_copy, t_list, one_ele, two_ele, fock)

        print("\t\tECCSD ( Conventional )          = "+'{:17.10f}'.format(total_energy))
        print("\t\tECCSD ( RDM          )          = "+'{:17.10f}'.format(total_energy_rdm))

        # Sum up the energy
        energy_temp += fragment_energy

        # Sum up the number of electrons
        number_of_electron += np.trace(onerdm_frag[ : t_list[0], : t_list[0]])

        # Print the results
        print("\t\tFragment Energy                 = "+'{:17.10f}'.format(fragment_energy))
        print("\t\tNumber of Electrons in Fragment = "+'{:17.10f}'.format(np.trace(onerdm_frag)))
        print('')

    # add core constant terms to the energy
    energy_temp += dmet_orbs.core_constant_energy
    energy_list.append(energy_temp)
    return number_of_electron

