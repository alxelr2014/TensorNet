import numpy as np




def construct_w_tensor(n):
    T = np.zeros((2**n))
    j = 1
    for _ in range(n):
        T[j] = 1
        j *= 2
    l = [2 for _ in range(n)]
    return T.reshape(tuple(l)),l

def mps(T : np.ndarray,n,dims :list):
    Alist = np.empty(n,dtype=np.ndarray)
    prod_dims = np.prod(dims)
    T = T.reshape((1,) + T.shape)
    for i in range(n):
        rest_dims = prod_dims//dims[i]
        C = T.reshape((T.shape[0]*dims[i],rest_dims))
        u,s,vh = np.linalg.svd(C,full_matrices=False)
        Alist[i] = u @ np.diag(s)
        T = vh
        prod_dims = rest_dims
    return Alist

if __name__ == "__main__":
    n = 5
    T,l = construct_w_tensor(n)
    Alist = mps(T,n,l)   
    for A in Alist:
        print(A)
        print()     
            
            


# n = 4 # three sites = three legs
# psi = np.zeros(2**n)
# for i in range(n):
#     psi[2**i]=1
# psi = psi / np.linalg.norm(psi)  # random, normalized state vector
# psi = np.reshape(psi, (2, -1))
# U, Lambda, Vh = np.linalg.svd(psi, full_matrices=False)

# print(U)
# print(Lambda)
# print(Vh)
# psi = np.diag(Lambda) @ Vh
# print(psi)
# psi = np.reshape(psi, (4, -1))
# print("--------")
# print(psi)
# U, Lambda, Vh = np.linalg.svd(psi, full_matrices=False)
# print(U)
# print(Lambda)
# print(Vh)
# psi = np.diag(Lambda) @ Vh
# psi = np.reshape(psi[:2][:],(4,2))
# print("--------")
# print(psi)
# U, Lambda, Vh = np.linalg.svd(psi, full_matrices=False)
# print(U)
# print(Lambda)
# print(Vh)
# psi = np.diag(Lambda) @ Vh
# psi = np.reshape(psi[:2][:],(4,1))
# print("--------")
# print(psi)
# U, Lambda, Vh = np.linalg.svd(psi, full_matrices=False)
# print(U)
# print(Lambda)
# print(Vh)
# # for i in range(1,n):
# #     U, Lambda, Vh = np.linalg.svd(psi, full_matrices=False)
# #     psi = np.diag(Lambda) @ Vh
# #     print(i)
# #     print(Lambda)
# #     print(Vh)

# #     psi = np.reshape(psi, (4, -1))
# #     print(psi)
# #     Us = []
# #     if i==1:
# #         U = np.reshape(U, (1, 2, 2)) 
# #     elif i==n-1:
# #         U = np.reshape(U, (2, 2, 1)) 
# #     else:
# #         U = np.reshape(U,(2,2,2))
# #     Us.append(U)