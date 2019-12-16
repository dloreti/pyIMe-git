import numpy as np
import sympy as sy
import utilities as u
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve
    

class IMeException(Exception):
    pass

def init(A : np.ndarray, overwrite_a=True):
    A = A if overwrite_a else A.copy()
    n,_ = A.shape
    A = A.T
    for i in range(0,n):
        A[i][i]=1/A[i][i]
    for i in range(0,n):
        for j in range(0,n):
            if(i!=j):
                A[i][j]=A[i][j]*A[i][i]
    return A

def cs(A:np.ndarray,R,N,allocation='C'):
    """Initialization function that computes the checksum of A through row-wise weighted sums.

    Parameters
    ----------
    A : np.ndarray
        the n x n matrix for which we want the cs
    R : int
        Number of recovery nodes. Equal to the maximum number of tollerable faults
    C : int
        Number of cols on each node
    allocation : char in ['C','I']
        if the matrix A will be allocated contiguously 'C' or interlaced 'I'

    Returns
    -------
    S : np.ndarray
        the matrix of the checksums. Such matrix has dimension n x CR
    """
    n,_ = A.shape
    c=int(n/N)
    print('c=%d'%(c))
    if isinstance(A[0][0],sy.Rational):
        S = np.empty( (n,c*R), dtype=object)
        Lambda = np.empty( (N,R), dtype=object)
    else:
        S = np.empty( (n,c*R), dtype=np.float64)
        Lambda = np.empty( (N,R), dtype=np.float64)
    for i in range(N): 
        for j in range(R):
            Lambda[i,j]=(i+1)**j
    print('Lambda dim=%s \n%s'% (Lambda.shape,u.stringmat(Lambda) ) )

    if allocation=='I':
        #sum N contiguous columns and allocate in S
        for j in range(c):
            Temp=A[:,j*N:j*N+N]
            Temp=np.dot(Temp,Lambda)
            Temp[j*N:j*N+N,:]+=Lambda
            S[:,j:n:c]=Temp
    else: #allocation=='C'
        #sum N intertwined columns and allocate in S 
        for j in range(c): #building col j of each recovery node
            Temp=A[:,j:n:c]
            Temp=np.dot(Temp,Lambda)
            Temp[j:n:c,:]+=Lambda
            S[:,j:n:c]=Temp
    print('S=A*Lambda dim=%s \n%s'% (S.shape,u.stringmat_compressed(S) ))
    
    return S
 
def factor(A : np.ndarray, overwrite_a=True):
    """Factorizes the matrix A according to IMe.
        Parameters
        ----------
        A : np.ndarray
            The matrix ALREADY INITIALIZED with the init method. It can be a matrix of float64 or sy.Rational. The latter mainly used for testing.

        Returns
        -------
        A : np.arrndarrayay
            The factorized matrix.
        """
    A = A if overwrite_a else A.copy()
    n,_=A.shape
    if isinstance(A[0][0],sy.Rational):
        h = np.empty(n-1, dtype=object)
    else:
        h = np.empty(n-1)

    for l in range(n-1,0,-1):
        for i in range(l):
            h[i]= 1 - A[l][i]*A[i][l]
            A[i][i] = A[i][i] /h[i]
            idx = [x for x in range(n) if x!=i and x!=l]
            for j in idx:
                A[i][j] = (A[i][j] - A[l][j]*A[i][l]) /h[i]
            A[i][l] = (0 - A[l][l]*A[i][l]) /h[i]
        print("h = "+str(h))
        print("- Level: "+str(l))
        print('A (l=%d): \n%s' % (l,u.stringmat(A)))
    return A        

def solve(A : np.ndarray, b : np.array, overwrite_b=True):
    b = b if overwrite_b else b.copy()
    n,_=A.shape
    for l in range(n,0,-1):
        bl=b[l-1]
        for i in range(l-1):
            b[i]=b[i]-A[l-1][i]*bl
        b[l-1]=A[l-1][l-1]*bl
        for i in range(l,n):
            b[i]=b[i]+A[l-1][i]*bl
    return b

def factor_with_cs_and_faults(A : np.ndarray, R:int, N:int, faults:dict, overwrite_a=True,allocation='C'):
    """Factorizes the matrix A|S according to IMe anche checks the validity of the checksums.
    Simulates a series of failures and the recover procedure from them
    This has only testing purpose.
        Parameters
        ----------
        A : np.ndarray
            The matrix ALREADY INITIALIZED with the init method. It can be a matrix of float64 or sy.Rational. The latter mainly used for testing.
        R : int
            number of recovery nodes
        N : int
            number of computing nodes
        faults : dict
            dictionary in the form { Lf1:[Nf1,Nf2,Nf3], Lf2:[Nf4,Nf5], ... }, where
            Lf1 is the level/iteration at which the faults occur, and
            [Nf1, Nf2, Nf3] are the ranks of the nodes fallen at level Lf1
        Returns
        -------
        A : np.ndarray
            The factorized matrix. eventually reconstructed 
        """
    A = A if overwrite_a else A.copy()
    n,m=A.shape
    c=int(n/N)
    if isinstance(A[0][0],sy.Rational):
        h = np.empty(n-1, dtype=object)
    else:
        h = np.empty(n-1)

    # preparing for checksum computation and check...
    if isinstance(A[0][0],sy.Rational):
        S = np.empty( (n,c*R), dtype=object)
        Lambda = np.empty( (N,R), dtype=object)
    else:
        S = np.empty( (n,c*R), dtype=np.float64)
        Lambda = np.empty( (N,R), dtype=np.float64)
        
    for i in range(N): 
        for j in range(R):
            Lambda[i,j]=(i+1)**j
    #print('factor_with_cs: Lambda dim=%s \n%s'% (Lambda.shape,u.stringmat(Lambda) ) )
    
    for l in range(n-1,0,-1):
        
        for i in range(l):
            h[i]= 1 - A[l][i]*A[i][l]
            #print("h["+str(i)+"] = 1 - "+str(A[l][i])+" * "+str(A[i][l])  )
            A[i][i] = A[i][i] /h[i]
            #factorize the cols of cs toghether with those of A
            idx = [x for x in range(n+R*c) if x!=i and x!=l]
            for j in idx:
                A[i][j] = (A[i][j] - A[l][j]*A[i][l]) /h[i]
            A[i][l] = (0 - A[l][l]*A[i][l]) /h[i]
        if l in faults:
            #print("h = "+str(h))
            print("\n- Level: "+str(l))
            print('A (l=%d): \n%s' % (l,u.stringmat_compressed(A)))
        
        ### check the checksums: ###
        if allocation=='I':
            for j in range(c):
                Temp=A[:,j*N:j*N+N]
                Temp=np.dot(Temp,Lambda)
                Temp[j*N:j*N+N,:]+=Lambda
                S[:,j:n:c]=Temp
        else: #allocation=='C'
            for j in range(c):
                Temp=A[:,j:n:c]
                Temp=np.dot(Temp,Lambda)
                Temp[j:n:c,:]+=Lambda
                S[:,j:n:c]=Temp
                
        #print('recomputed S for this level is: (dim=%s) \n%s'% (S.shape,u.stringmat(S) ))
        if isinstance(A[0][0],sy.Rational):
            (c1,c2)=S.shape
            for i in range(c1):
                for j in range(c2):
                    if A[i,n+j]!=S[i,j]:
                        print("INCORRECT CS: A[%d][%d]=%s S=[%d][%d]=%s" % (i,n+j,str(A[i][n+j]),i,j, str(S[i,j])))
        else:
            assert u.allclose(A[:,n:n+R*c], S)# - np.eye(n, dtype=np.float64))
        
        if l in faults: # there is fault at this level
            Nf = faults.get(l) #list of fallen processor
            Na = [v for v in range(N) if v not in Nf] #list of alive processor
            f=len(Nf)
            a=len(Na)
            #which cols I have to reconstruct?
            fcols=[]
            for nf in Nf: #for each fallen node
                if allocation=='I':
                    fcols+=[y for y in range(nf,n,N)]
                else:
                    fcols+=[y for y in range(nf*c,nf*c+c)]
            acols=[v for v in range(n) if v not in fcols]     #alive cols 
            print('************** ')
            print('Level %d: FALLEN nodes=%s alive nodes=%s --> fcols %s , alive cols %s' %(l,Nf,Na,fcols,acols))
            #print('A (l=%d): \n%s' % (l,u.stringmat_compressed(A)))
            # create the A_r matrix of the recovery systems. 
            # It is the same for each n*c recovery system to solve
            A_r = (np.take(Lambda[:,:f],Nf,axis=0)).T  
            print('A_r: \n%s' % (u.stringmat_compressed(A_r)))
            LU,piv=lu_factor(A_r,False)
            #print('LU: \n%s' % (u.stringmat_compressed(LU)))
            
            X_r = np.zeros((n,f*c))
            #print('Lambda: \n%s' % (u.stringmat_compressed(Lambda)))
            TempLambda=np.take(Lambda,Na,axis=0) 
            #print('TempLambda: \n%s' % (u.stringmat_compressed(TempLambda)))
            if allocation=='I':
                for j in range(c):
                    Temp=np.take(A, acols, axis=1)[:,j*a:j*a+a]
                    Temp=np.dot(Temp, TempLambda)
                    Temp[j*N:j*N+N,:]+= Lambda 
                    Temp = (A[:,n+j:m:c]-Temp)[:,:f] #keep only the first f cols -> some wasted flops
                    #print('j=%d Temp: \n%s' % (j,u.stringmat_compressed(Temp)))
                    X_r[:,j::c] = lu_solve((LU,piv), Temp.T, overwrite_b=False).T
                    #print('j=%d x: \n%s' % (j,u.stringmat_compressed(X_r[:,j::c])))
            else:
                #recovery works fine for C allocation
                for j in range(c):
                    Temp=np.take(A, acols, axis=1)[:,j:n:c]
                    Temp=np.dot(Temp, TempLambda)
                    Temp[j:n:c,:]+= Lambda 
                    Temp = (A[:,n+j:m:c]-Temp)[:,:f] #keep only the first f cols -> some wasted flops
                    #print('j=%d Temp: \n%s' % (j,u.stringmat_compressed(Temp)))
                    X_r[:,j::c] = lu_solve((LU,piv), Temp.T, overwrite_b=False).T
                    #print('j=%d x: \n%s' % (j,u.stringmat_compressed(X_r[:,j::c])))
            print("Reconstructed cols: \n%s" % (u.stringmat_compressed(X_r)))
            A_original=A.copy()
            np.put(A,fcols,X_r)
            if u.allclose(A, A_original): # - np.eye(n, dtype=np.float64))
                print("A correctly reconstructed")
            else:
                print("A recontruction error")
            #print('A reconstructed (l=%d): \n%s' % (l,u.stringmat_compressed(A)))
    return A  

def factor_with_cs(A : np.ndarray, N:int, overwrite_a=True,allocation='C'):
    """Factorizes the matrix A|S according to IMe anche checks the validity of the checksums.
     This has only testing purpose.
        Parameters
        ----------
        A : np.ndarray
            The matrix ALREADY INITIALIZED with the init method. It can be a matrix of float64 or sy.Rational. The latter mainly used for testing.

        Returns
        -------
        A : np.arrndarrayay
            The factorized matrix.
        """
    A = A if overwrite_a else A.copy()
    n,m=A.shape
    c=int(n/N)
    R=int((m-n)/c)
    if isinstance(A[0][0],sy.Rational):
        h = np.empty(n-1, dtype=object)
    else:
        h = np.empty(n-1)

    # preparing for checksum computation and check...
    if isinstance(A[0][0],sy.Rational):
        S = np.empty( (n,c*R), dtype=object)
        Lambda = np.empty( (N,R), dtype=object)
    else:
        S = np.empty( (n,c*R), dtype=np.float64)
        Lambda = np.empty( (N,R), dtype=np.float64)
        
    for i in range(N): 
        for j in range(R):
            Lambda[i,j]=(i+1)**j
    #print('factor_with_cs: Lambda dim=%s \n%s'% (Lambda.shape,u.stringmat(Lambda) ) )
    
    
    for l in range(n-1,0,-1):
        for i in range(l):
            h[i]= 1 - A[l][i]*A[i][l]
            A[i][i] = A[i][i] /h[i]
            #factorize the cols of cs toghether with those of A
            idx = [x for x in range(n+R*c) if x!=i and x!=l]
            for j in idx:
                A[i][j] = (A[i][j] - A[l][j]*A[i][l]) /h[i]
            A[i][l] = (0 - A[l][l]*A[i][l]) /h[i]
        #print("h = "+str(h))
        #print("\n- Level: "+str(l))
        #print('A (l=%d): \n%s' % (l,u.stringmat(A)))
        
        ### check the checksums: ###
        if allocation=='I':
            for j in range(c):
                Temp=A[:,j*N:j*N+N]
                Temp=np.dot(Temp,Lambda)
                Temp[j*N:j*N+N,:]+=Lambda
                S[:,j:n:c]=Temp
        else: #allocation=='C'
            for j in range(c):
                Temp=A[:,j:n:c]
                Temp=np.dot(Temp,Lambda)
                Temp[j:n:c,:]+=Lambda
                S[:,j:n:c]=Temp
                
        #print('recomputed S for this level is: (dim=%s) \n%s'% (S.shape,u.stringmat(S) ))
        if isinstance(A[0][0],sy.Rational):
            (c1,c2)=S.shape
            for i in range(c1):
                for j in range(c2):
                    if A[i,n+j]!=S[i,j]:
                        print("INCORRECT CS: A[%d][%d]=%s S=[%d][%d]=%s" % (i,n+j,str(A[i][n+j]),i,j, str(S[i,j])))
        else:
            assert u.allclose(A[:,n:n+R*c], S)# - np.eye(n, dtype=np.float64))
            
    return A  

