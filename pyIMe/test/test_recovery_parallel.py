''' 
This script launches the serial and parallel IMe factorizations
of a test matrix with simulated faults.
The matrix can be allocated in contiguously or intertwined fashion (alloc='C' or alloc='I').
Faults are specified with the dictionary faults. 
E.g. faults={4:[0,3],6:[2]} means we want to simulate 
two faults at level 4 involving nodes 0 and 3, and a fault at level 6 involving node 2

Launch this as follows:
mpirun -np 7 python3 test/test_recovery_parallel.py 
'''


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


R=3   #number of recovery nodes (the ones hosting the checksums)
#faults={4:[0,3]}
faults={4:[0,3],6:[2]}
alloc='C'   

P=size
N=size-R

if rank==0:
###############################################################
# calling IMe serial initialization of the checksum matrix S  #
###############################################################
    resources_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..','resources'))
    A = np.load(os.path.join(resources_dir,'A-8.npy'))
    (n,_)=A.shape
    print('A : \n%s' % (u.stringmat_compressed(A)))
    #print('cond(A) = %s' % (np.linalg.cond(A)))
    Tn = ime.init(A, False)
    #print('T(n) : \n%s' % (u.stringmat_compressed(Tn)))
    S = ime.cs(Tn,R,N,allocation=alloc)
    (_,q)=S.shape
    m = n+q

###############################################################
# IMe serial fatorization of [Tn|S] with simulated faults #
###############################################################
    TnS=np.concatenate((Tn,S),axis=1)
    #print('[Tn|S] : \n%s' % (u.stringmat_compressed(TnS)))
    #T1S = ime.factor_with_cs(TnS,N,overwrite_a=False,allocation=alloc)
    #print('SERIAL Factorization of [Tn|S]:\n%s' % (u.stringmat_compressed(T1S)))
    T1S = ime.factor_with_cs_and_faults(TnS,R,N,faults, overwrite_a=False,allocation=alloc)
    print('\nSERIAL Factorization of [Tn|S] (despite faults):\n%s' % (u.stringmat_compressed(T1S)))
    print('\n*****************************************************************************\n')
else:
    Tn = None
    S = None
    TnS = None
    n = None
    m = None

n=comm.bcast(n,root=0)
m=comm.bcast(m,root=0)
c=int(n/N)   #number of columns on each processor

#each processor receives n/size complete columns. 
#therefore nprows = 1 --> there is only one processor on the rows.
#and       npcols = size --> the are P=size processors on the columns.
if alloc=='C':
    core.initmpi([1, size] , block_shape=[n, c])   
#since block_shape=[_, c], columns are allocated contiguously (i.e., node0 gets cols [0..c-1], node1 gets [c..2c-1], etc.)
    # distribute the initialized matrix Tn|S
    dTnS = core.DistributedMatrix.from_global_array(TnS, rank=0)
else:
#if I wanted interlaced allocation:  
    core.initmpi([1, size] , block_shape=[n, 1]) 
    if rank < N:
        color = 0
        key = rank
    else:
        color = 1
        key = rank-N
    newcom=comm.Split(color, key)
    context = core.ProcessContext([1, newcom.size],comm=newcom)
    if rank < N:
        dTn = core.DistributedMatrix.from_global_array(Tn, rank=0, context=context)
        dS = None
        if rank==0:
            comm.send(S,N)
    else:
        dTn = None
        if newcom.rank == 0:
            S=comm.recv(S,0)
        dS = core.DistributedMatrix.from_global_array(S, rank=0, block_shape=[n, c],context=context)

# each  processor prints its portion:
#print('P%d: my portion of [Tn|S] is : \n%s' % (rank,u.stringmat_compressed(dTnS.local_array)))

###############################################################
# IMe parallel factorization of [Tn|S] #
###############################################################
if alloc=='C':
    dT1S,_ = imeFT.factor_with_faults(dTnS,None,n,m,faults,overwrite_a=True, allocation=alloc)
    T1S_par = dT1S.to_global_array(rank=0)
else:
    dT1,dS1 = imeFT.factor_with_faults(dTn,dS,n,m,faults,overwrite_a=True, allocation=alloc)
    #print("I'm rank "+str(rank)+" dS1 is "+str(dS1))
    if rank<N:
        T1 = dT1.to_global_array(rank=0)
    else:
        S1 = dS1.to_global_array(rank=0) #corresponds to rank=N in comm_world
    if rank==N:
        comm.send(S1,0)
    if rank==0: 
        S1 = np.empty((n,m-n), dtype=np.float64)       
        S1 = comm.recv(source = N)
        T1S_par=np.concatenate((T1,S1),axis=1)

if rank==0:
    print('PARALLEL Factorization of [Tn|S]:\n%s' % (u.stringmat_compressed(T1S_par)))
    print("ARE IMe serial and parallel factorizations the same? --> " +str(np.allclose(T1S_par, T1S ))+"!!!! \n")












