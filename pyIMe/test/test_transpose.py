
from context import utilities as u
from context import IMe as ime
from context import IMeFT as imeFT


import os
import numpy as np
import sympy as sy
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve
from mpi4py import MPI
from scalapy import core
import scalapy.routines as rt



comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


if rank==0:
    resources_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..','resources'))
    A = np.load(os.path.join(resources_dir,'A-8.npy'))
    (n,_)=A.shape
    print('A : \n%s' % (u.stringmat_compressed(A)))
else :
    A=None
    n=None

N=size

n=comm.bcast(n,root=0)
c=int(n/N)   #number of columns on each processor

core.initmpi([1, size] , block_shape=[n, c])   
dA = core.DistributedMatrix.from_global_array(A, rank=0)

print('P%d portion of dA : \n%s' % (rank, u.stringmat_compressed(dA.local_array)))

dA=dA.T

print('P%d portion of dA.T : \n%s' % (rank, u.stringmat_compressed(dA.local_array)))
