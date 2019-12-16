''' 
This script can be used to compare the parallel and serial esecutions of IMe and LU
and see if the solutions computed by the two algorithms are close.
All the parallelization is done supposing a contiguous allocation.
Launch this as follows:
mpirun -np 4 python3 test/test_IMe_LU_serial_and_parallel.py 
'''


from context import utilities as u
from context import IMe as ime
from context import IMePar as imepar


import os
import numpy as np
import sympy as sy
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve
from mpi4py import MPI
from scalapy import core
import scalapy.routines as rt


#################################################
 # IMe # 
#################################################
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size



if rank==0:
    resources_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..','resources'))
    A = np.load(os.path.join(resources_dir,'A-8.npy'))
    n,_=A.shape
    print('P%d A: \n%s' % (rank,u.stringmat(A)))
    imeinitA = ime.init(A, False)
else:
    imeinitA = None
    n = None

n=comm.bcast(n,root=0)
C=int(n/size)
core.initmpi([1, size] , block_shape=[n, C])
dA = core.DistributedMatrix.from_global_array(imeinitA, rank=0)

dIMe = imepar.factor(dA)


IMe=dIMe.to_global_array(rank=0)
if rank==0:
    print('P%d: IMe parallel factorization:\n%s'%(rank,u.stringmat(IMe)))

    IMeSerial=ime.factor(ime.init(A, False), False) # do not override A because it is used elseware
    print("IMe serial factorization:\n%s"%(u.stringmat(IMeSerial)))
    
    print("ARE IMe serial and parallel factor the same? --> " +str(np.allclose(IMe - IMeSerial, np.zeros((8,8))))+"!!!! \n")


#################################################
 # LU-Scalapack #
#################################################
if rank==0:
    gA4LU = np.asfortranarray(A)
else: 
    gA4LU = None
dA4LU = core.DistributedMatrix.from_global_array(gA4LU, rank=0, block_shape=[8,8])

dLU, ipiv = rt.lu(dA4LU, False)
LU = dLU.to_global_array(rank=0)
if rank==0:
    print('P%d: LU parallel factorization:\n%s'%(rank,u.stringmat(LU)))

    LUSerial,piv=lu_factor(A,False)
    print("LU serial factorization:\n%s"%(u.stringmat(LUSerial)))
    
    print("ARE LU serial and parallel factor the same? --> " +str(np.allclose(LU - LUSerial, np.zeros((8,8))))+"!!!! \n")


#################################################
 # COMPARING THE SOLUTIONS #
#################################################

    np.set_printoptions(linewidth=140)

    b = np.load( os.path.join(resources_dir,'b-8.npy')) 
    print("b: \n"+np.array_repr(b).replace('\n', ''))

    xIMe = ime.solve(IMe,b, False)
    print("xIMe:\n"+str(xIMe))
    xIMeSerial = ime.solve(IMeSerial,b, False)
    print("xIMeSerial:\n"+str(xIMeSerial))
    xLU = lu_solve((LU,piv), b, overwrite_b=False)
    print("xLU:\n"+str(xLU))
    xLUSerial = lu_solve((LUSerial,piv), b, overwrite_b=False)
    print("xLUSerial:\n"+str(xLUSerial))
    
