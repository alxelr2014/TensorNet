import numpy as np



def construct_w_tensor(n):
    T = np.zeros((2**n))
    j = 1
    for _ in range(n):
        T[j] = 1
        j *= 2
    l = [2 for _ in range(n)]
    return 1/np.sqrt(n)* T.reshape(tuple(l)),l

def mps(T : np.ndarray,n,dims :list):
    Alist = np.empty(n,dtype=np.ndarray)
    prod_dims = np.prod(dims)
    T = T.reshape((1,) + T.shape)
    for i in range(n-1):
        rest_dims = prod_dims//dims[i]
        C = T.reshape((T.shape[0]*dims[i],rest_dims))
        u,s,vh = np.linalg.svd(C,full_matrices=False)
        non_zeros =np.size(s[s > 1e-6])
        Alist[i] =( u @ np.diag(s))[:,:non_zeros]
        T = vh[:non_zeros,:]
        prod_dims = rest_dims
    Alist[n-1]=T
    return Alist


if __name__ == "__main__":
    n = 4
    T,l = construct_w_tensor(n)
    Alist = mps(T,n,l)   
    for A in Alist:
        print(A)
        print() 

            


