import unittest
import numpy as np
import itertools

import sys
sys.path.append('../')
# from qsdk.toolboxes.ansatz_generator import _general_unitary_cc
from _general_unitary_cc import *


class UCCGSDTest(unittest.TestCase):


    def test_spin_order(self):
        """Test that spin-ordering is implemented correctly, for both
        openfermion and qiskit orderings"""
        n_orbitals = 6
        p,q,r,s=0,1,2,3
        qiskit = np.array([0,1,2,3]),6+np.array([0,1,2,3])
        fermion = np.array([0,2,4,6]),np.array([1,3,5,7])
        up_q,dn_q = get_spin_ordered(n_orbitals,p,q,r,s,up_down=True)
        up_f,dn_f = get_spin_ordered(n_orbitals,p,q,r,s,up_down=False)

        self.assertEqual(np.linalg.norm(up_q-qiskit[0]),0.0,msg='Spin Up Qiskit-Ordering Fails')
        self.assertEqual(np.linalg.norm(dn_q-qiskit[1]),0.0,msg='Spin Down Qiskit-Ordering Fails')
        self.assertEqual(np.linalg.norm(up_f-fermion[0]),0.0,msg='Spin Up openfermion-Ordering Fails')
        self.assertEqual(np.linalg.norm(dn_f-fermion[1]),0.0,msg='Spin Down openfermion-Ordering Fails')

    def test_factorial(self):
        """Test factorial responds correctly to different input types, and returns valid
        result for valid input."""

        self.assertRaises(ValueError,factorial,-1)
        self.assertRaises(TypeError,factorial,'a')
        self.assertEqual(factorial(4),4*3*2,msg="Invalid Result from factorial")
        self.assertEqual(factorial(0),1,msg="Invalid Result for null factorial")


    def count_doubles_groups(self,n_orbs,up_down=False):
        """General test for number of doubles groups found by generator"""

        selection = np.linspace(0,n_orbs-1,n_orbs,dtype=int)
        groups = np.zeros(5)
        for pp,qq,rr,ss in itertools.product(selection,repeat=4):

            if (pp<qq and pp<rr and pp<ss) and (rr<ss) and (qq!=rr and qq!=ss):
                _ = get_group_1_2(n_orbs,pp,qq,rr,ss,up_down=up_down)    
                groups[0]+=1
            elif qq==rr and pp<ss and pp!=qq and ss!=qq:
                _ = get_group_1_2(n_orbs,pp,qq,rr,ss,up_down=up_down)
                groups[1]+=1
            elif (pp==qq and qq!=rr and rr!=ss and ss!=pp) and rr<ss:
                _ = get_group_3_4(n_orbs,pp,qq,rr,ss,up_down=up_down)
                groups[2]+=1
            elif pp==qq and qq==rr and rr!=ss:
                _ = get_group_3_4(n_orbs,pp,qq,rr,ss,up_down=up_down)
                groups[3]+=1
            elif pp==qq and qq!=rr and rr==ss and pp<ss:     
                _ = get_group_5(n_orbs,pp,qq,rr,ss,up_down=up_down)
                groups[4]+=1
            
            else:
                continue


        self.assertEqual(groups[0],choose(n_orbs,2)*choose(n_orbs-2,2)//2,msg="{:d} orbs: Invalid Group 1 Number".format(n_orbs))
        self.assertEqual(groups[1],n_orbs*choose(n_orbs-1,2),msg="{:d} orbs: Invalid Group 2 Number".format(n_orbs))
        self.assertEqual(groups[2],n_orbs*choose(n_orbs-1,2),msg="{:d} orbs: Invalid Group 3 Number".format(n_orbs))
        self.assertEqual(groups[3],2*choose(n_orbs,2),msg="{:d} orbs: Invalid Group 4 Number".format(n_orbs))
        self.assertEqual(groups[4],choose(n_orbs,2),msg="{:d} orbs: Invalid Group 5 Number".format(n_orbs))
        self.assertEqual(np.sum(groups),get_doubles_number(n_orbs),msg="{:d} orbs: Invalid Total Number".format(n_orbs))

    def test_count_doubles(self):
        """Test for checking number of doubles excitations generated."""

        self.count_doubles_groups(5,up_down=False)
        self.count_doubles_groups(5,up_down=True)
        self.count_doubles_groups(8,up_down=False)

    def test_count_singles(self):
        """Test for checking number of singles excitations generated."""
        self.assertEqual(len(get_singles(5)),get_singles_number(5),msg="5 orbs: Invalid Number of singles")
        self.assertEqual(len(get_singles(10)),get_singles_number(10),msg="5 orbs: Invalid Number of singles")

if __name__ == "__main__":

    unittest.main()