
import numpy as np
from mpi4py import MPI
from scalapy import core


def factor(A: core.DistributedMatrix, overwrite_a=True):
    """Factorizes the matrix A according to IMe.

    Parameters
    ----------
    A : DistributedMatrix
        The matrix to decompose. ALREADY INITIALIZED with the IMe.init(A) method.
    overwrite_a : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.

    Returns
    -------
    A : DistributedMatrix
        Overwritten by local pieces of the ime-factorized matrix.
    
    """
    A = A if overwrite_a else A.copy()
    #n,_=A.shape
    n,_ = A.global_shape

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    c = int(n/size) # columns per node

    h = np.empty(n-1, dtype=np.float64)

    for l in range(n,1,-1):
        global_idx_lc = l-1
        rank_of_lc = int(global_idx_lc/c)
        if rank == rank_of_lc:
            local_idx_lc = global_idx_lc%c
            lc=A.local_array[0:l-1,local_idx_lc]
        else:
            lc = None
        lc = comm.bcast(lc,root=rank_of_lc)
        lr = A[l-1,0:l-1].to_global_array()

        for i in range(l-1):
            h[i]=1 - lc[i]*lr[0,i]
            for j in range(c):
                if i == A.col_indices()[j]:
                    A.local_array[i,j]=A.local_array[i,j]/h[i]
                    #dA.local_array[i,j]=8.0
                else:
                    if A.col_indices()[j]==l-1:
                        A.local_array[i,j]=(0.0 - A.local_array[l-1,j]*lc[i]) /h[i]
                    else:
                        A.local_array[i,j] = (A.local_array[i,j] - A.local_array[l-1,j]*lc[i]) /h[i]
                        #dA.local_array[i,j]=8.0
        
    return A
    



