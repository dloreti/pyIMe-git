#import IMe as ime
import numpy as np
from mpi4py import MPI
from scalapy import core
import utilities as u
import sympy as sy
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve


def factor_with_faults(A: core.DistributedMatrix, S: core.DistributedMatrix, n:int, m:int, faults:dict, overwrite_a=True, allocation='C'):
    """Factorizes the matrix A according to IMe. 
    If there are more columns than rows, it assumes that the exceeding cols 
    belong to the checksum matrix S and are allocated on recovery nodes. 
    The code is equal to a normal parallel factorization, because IMe
    operates on the columns of A and the cols of S in the same way.

    Parameters
    ----------
    A : DistributedMatrix
        The matrix to decompose. ALREADY INITIALIZED with the IMe.init(A) method.
        contains the checksum matrix S in case of allocation='C'. 
    S : DistributedMatrix
        The matrix of checksums. ALREADY INITIALIZED.
        None if allocation='C' because the checksum matrix is already into A
    faults : dict
            dictionary in the form { Lf1:[Nf1,Nf2,Nf3], Lf2:[Nf4,Nf5], ... }, where
            Lf1 is the level/iteration at which the faults occur, and
            [Nf1, Nf2, Nf3] are the ranks of the nodes fallen at level Lf1
    overwrite_a : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    allocation : char in {'C','I'}
        states if the allocation is Contiguous of Intertwined. 
        I 'I', Then the matrix S should be filled with checksum values.
        Otherwise, if 'C', checksums cols are at the end of the A matrix. S is not used

    Returns
    -------
    A : DistributedMatrix
        Overwritten by local pieces of the iMe factorized matrix.
    S : DistributedMatrix
        Checksum matrix as elaborated by iMe factorization - if allocation='I'
        None - if allocation='C'. Because the facorization of the checksums is already inside A. 
    
    """
    A = A if overwrite_a else A.copy()
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size 

    P = size # --> P: total number of processors (P=N+R)
    c = int(m/P) # columns per node
    R = int((m-n)/c) # recovery nodes
    N = P-R #computing nodes

    h = np.empty(n-1, dtype=np.float64)

    Lambda = np.empty( (N,R), dtype=np.float64)
    if len(faults)>0:
        for i in range(N): 
            for j in range(R):
                Lambda[i,j]=(i+1)**j
    #print('factor_with_cs: Lambda dim=%s \n%s'% (Lambda.shape,u.stringmat(Lambda) ) )
    

    for l in range(n,1,-1): #range(n-1,0,-1): #
        global_idx_lc = l-1
        if allocation=='C':
            rank_of_lc = int(global_idx_lc/c)
        else:# allocation='I'
            rank_of_lc = global_idx_lc%N

        if rank == rank_of_lc:
            if allocation=='C':
                local_idx_lc = global_idx_lc%c
            else:# allocation='I'
                local_idx_lc = int(global_idx_lc/N)
            lc=A.local_array[0:l-1,local_idx_lc]
        else:
            lc = None
        lc = comm.bcast(lc,root=rank_of_lc)
        if allocation=='C' or (allocation=='I' and rank<N):
            lr = A[l-1,0:l-1].to_global_array()
        if allocation=='I':
            if rank==0:
                for q in range(N,P):
                    comm.send(lr,q)
            elif rank>=N:
                lr = np.empty(n, dtype=np.float64)
                lr = comm.recv(source = 0)

        for i in range(l-1):
            h[i]=1 - lc[i]*lr[0,i]
            if allocation=='C' or (allocation=='I' and rank<N):
                for j in range(c):
                    if i == A.col_indices()[j]:
                        A.local_array[i,j]=A.local_array[i,j]/h[i]
                    else:
                        if A.col_indices()[j]==l-1:
                            A.local_array[i,j]=(0.0 - A.local_array[l-1,j]*lc[i]) /h[i]
                        else:
                            A.local_array[i,j] = (A.local_array[i,j] - A.local_array[l-1,j]*lc[i]) /h[i]
            else: #allocation=='I' and rank>=N #on checksum nodes
                for j in range(c):
                    S.local_array[i,j] = (S.local_array[i,j] - S.local_array[l-1,j]*lc[i]) /h[i]
        
        if l in faults: # there is fault at this level
            #gA = A.to_global_array(rank=0)
            #if rank==0:
            #    print("\n- Level: "+str(l))
            #    print('A (l=%d): \n%s' % (l,u.stringmat_compressed(gA)))
        
            Nf = faults.get(l) #list of fallen processor
            f=len(Nf)
            #if allocation='C':
            #    A_original=A.to_global_array(rank=0)
                
            
            if rank in Nf:
                original=np.copy(A.local_array[:,:])
                A.local_array[:,:]=np.zeros((n,c)) # set to zeros to simulate faults
                A_r = (np.take(Lambda[:,:f],Nf,axis=0)).T  
                #print('P%d: A_r: \n%s' % (rank, u.stringmat_compressed(A_r)))
                LU,piv=lu_factor(A_r,False)
                #print('LU: \n%s' % (u.stringmat_compressed(LU)))            
            #if rank < N:
            b=np.zeros((n,c,f),dtype=np.float64)
            for fidx in range(f):
                if rank < N:
                    for j in range(c):
                        b[:,j,fidx] = -A.local_array[:,j]*Lambda[rank,fidx]  
                    (gc,lic,ljc) = A.local_diagonal_indices(allow_non_square=True)
                    #print("P%d %s" % (rank, (gc,lic,ljc) ))
                    for el in range(len(gc)):
                        b[lic[el],ljc[el],fidx] -= Lambda[rank,fidx]             
                elif rank==N+fidx and allocation=='C':
                    b[:,:,fidx] = A.local_array[:,:]
                elif rank==N+fidx and allocation=='I':
                    b[:,:,fidx] = S.local_array[:,:]
                
                b[:,:,fidx]=comm.reduce(b[:,:,fidx],root=Nf[0])#root=Nf[fidx])
                #if rank==Nf[0]:
                #for fidx in range(f):
                    #print('Nf[fidx]=%d b[%d]: \n%s' % (Nf[fidx],fidx, u.stringmat_compressed(b[:,:,fidx])))
            if rank==Nf[0]:
                x=np.zeros((n,c,f),dtype=np.float64)
                for i in range(n):                    
                    for j in range(c):
                        x[i,j,:]=lu_solve((LU,piv),b[i,j,:])
                        A.local_array[i,j]=x[i,j,0]
                print('++++ P%d reconstruction ok? %s'%(rank,np.allclose(original,A.local_array)))
                #print('++++ P%d original \n%s'%(rank,u.stringmat_compressed(original)))
                #print('++++ P%d A.local_array \n%s'%(rank,u.stringmat_compressed(A.local_array)))
                    
                for z in range(1,f):
                    temp=np.zeros((n,c),dtype=np.float64)
                    temp=np.array(x[:,:,z].T)
                    #print('P%d sending x to %d dim = %s x[%d]:\n%s'%(rank,Nf[z],temp.shape,z, u.stringmat_compressed(temp)))
                    comm.Send([temp, MPI.DOUBLE], dest=Nf[z], tag=l)
            if rank!=Nf[0] and rank in Nf:
                temp=np.empty((n,c),dtype=np.float64)
                comm.Recv([temp, MPI.DOUBLE],source=Nf[0], tag=l)
                A.local_array[:,:]=temp
                print('++++ P%d reconstruction ok? %s'%(rank,np.allclose(original,A.local_array)))
                #print('++++ P%d original \n%s'%(rank,u.stringmat_compressed(original)))
                #print('++++ P%d A.local_array \n%s'%(rank,u.stringmat_compressed(A.local_array)))
                #A.local_array[:,:]=temp[:,:]
            
            #if allocation=='C':
                #A_reconstructed = A.to_global_array(rank=0)
                #if rank==0:
                    #if not np.allclose(A_reconstructed, A_original):# - np.eye(n, dtype=np.float64))
                        #print('A_original \n%s'%(u.stringmat_compressed(A_original)))
                        #print('A_reconstructed \n%s'%(u.stringmat_compressed(A_reconstructed)))
                    #else:
                        #print("A correctly reconstructed")
            
    if allocation=='C':
        return A, None
    else:
        return A,S
  


def factor(A: core.DistributedMatrix, S: core.DistributedMatrix, n:int, m:int, overwrite_a=True, allocation='C'):
    """Factorizes the matrix A according to IMe. 
    If there are more columns than rows, it assumes that the exceeding cols 
    belong to the checksum matrix S and are allocated on recovery nodes. 
    The code is equal to a normal parallel factorization, because IMe
    operates on the columns of A and the cols of S in the same way.

    Parameters
    ----------
    A : DistributedMatrix
        The matrix to decompose. ALREADY INITIALIZED with the IMe.init(A) method.
        contains the checksum matrix S in case of allocation='C'. 
    S : DistributedMatrix
        The matrix of checksums. ALREADY INITIALIZED.
        None if allocation='C' because the checksum matrix is already into A
    overwrite_a : boolean, optional
        By default the input matrix is destroyed, if set to False a
        copy is taken and operated on.
    allocation : char in {'C','I'}
        states if the allocation is Contiguous of Intertwined. 
        I 'I', Then the matrix S should be filled with checksum values.
        Otherwise, if 'C', checksums cols are at the end of the A matrix. S is not used

    Returns
    -------
    A : DistributedMatrix
        Overwritten by local pieces of the iMe factorized matrix.
    S : DistributedMatrix
        Checksum matrix as elaborated by iMe factorization - if allocation='I'
        None - if allocation='C'. Because the facorization of the checksums is already inside A. 
    
    """
    A = A if overwrite_a else A.copy()
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size 

    P = size # --> P: total number of processors (P=N+R)
    c = int(m/P) # columns per node
    R = int((m-n)/c) # recovery nodes
    N = P-R #computing nodes

    h = np.empty(n-1, dtype=np.float64)

    for l in range(n,1,-1): #range(n-1,0,-1): #
        global_idx_lc = l-1
        if allocation=='C':
            rank_of_lc = int(global_idx_lc/c)
        else:# allocation='I'
            rank_of_lc = global_idx_lc%N

        if rank == rank_of_lc:
            if allocation=='C':
                local_idx_lc = global_idx_lc%c
            else:# allocation='I'
                local_idx_lc = int(global_idx_lc/N)
            lc=A.local_array[0:l-1,local_idx_lc]
        else:
            lc = None
        lc = comm.bcast(lc,root=rank_of_lc)
        if allocation=='C' or (allocation=='I' and rank<N):
            lr = A[l-1,0:l-1].to_global_array()
        if allocation=='I':
            if rank==0:
                for q in range(N,P):
                    comm.send(lr,q)
            elif rank>=N:
                lr = np.empty(n, dtype=np.float64)
                lr = comm.recv(source = 0)

        for i in range(l-1):
            h[i]=1 - lc[i]*lr[0,i]
            if allocation=='C' or (allocation=='I' and rank<N):
                for j in range(c):
                    if i == A.col_indices()[j]:
                        A.local_array[i,j]=A.local_array[i,j]/h[i]
                    else:
                        if A.col_indices()[j]==l-1:
                            A.local_array[i,j]=(0.0 - A.local_array[l-1,j]*lc[i]) /h[i]
                        else:
                            A.local_array[i,j] = (A.local_array[i,j] - A.local_array[l-1,j]*lc[i]) /h[i]
            else: #allocation=='I' and rank>=N #on checksum nodes
                for j in range(c):
                    S.local_array[i,j] = (S.local_array[i,j] - S.local_array[l-1,j]*lc[i]) /h[i]
        #gA = A.to_global_array(rank=0)
        #if rank==0:
        #    print("h = "+str(h))
        #    print("\n- Level: "+str(l))
        #    print('A (l=%d): \n%s' % (l,u.stringmat_compressed(gA)))
    
    if allocation=='C':
        return A, None
    else:
        return A,S
    




