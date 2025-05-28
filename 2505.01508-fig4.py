import matplotlib.pyplot as plt
import numpy as np
import math as mth
from numpy import linalg as LA
from matplotlib.colors import LogNorm
from scipy import optimize
from scipy.integrate import complex_ode
from scipy.linalg import expm, sinm, cosm
import scipy as sp
import copy
from scipy.optimize import curve_fit
import math




def extract_fibonacci_positions(arr):
    # Generate Fibonacci numbers up to the length of the array
    fibs = [0, 1]  # Start with first two Fibonacci numbers
    while fibs[-1] + fibs[-2] < len(arr):
        fibs.append(fibs[-1] + fibs[-2])
    
    # Filter out positions that are out of bounds
    valid_positions = [fib for fib in fibs if fib < len(arr)]
    
    # Return elements at Fibonacci positions
    return arr[valid_positions]

def gammaw(m):
    golden_ratio = (1 + math.sqrt(5)) / 2
    return math.floor((m + 1) * golden_ratio - math.floor(m * golden_ratio)) - 1

def cofactor(A):
    """
    Calculate cofactor matrix of A
    """
    sel_rows = np.ones(A.shape[0],dtype=bool)
    sel_columns = np.ones(A.shape[1],dtype=bool)
    CO = np.zeros_like(A)
    sgn_row = 1
    for row in range(A.shape[0]):
        # Unselect current row
        sel_rows[row] = False
        sgn_col = 1
        for col in range(A.shape[1]):
            # Unselect current column
            sel_columns[col] = False
            # Extract submatrix
            MATij = A[sel_rows][:,sel_columns]
            CO[row,col] = sgn_row*sgn_col*np.linalg.det(MATij)
            # Reselect current column
            sel_columns[col] = True
            sgn_col = -sgn_col
        sel_rows[row] = True
        # Reselect current row
        sgn_row = -sgn_row
    return CO



def matrix_cofactor(matrix):
    return np.linalg.inv(matrix).T * np.linalg.det(matrix)




def adjugate(A):
    """
    Calculate adjugate matrix of A
    """
    return matrix_cofactor(A).T


def fermions1D(L, envelope, hopping=[0, 1], pbc=True):
    ham = np.zeros((L, L), dtype=complex)
    for j, h in enumerate(hopping):
        for i in range(L - j):
            ham[i, i + j] = -envelope((i + i + j - 1) / 2) * h / 2
            ham[i + j, i] = np.conjugate(ham[i, i + j])
        if pbc:
            for i in range(L - j, L):
                ham[i, (i + j) % L] = -envelope((i + i + j - 1) / 2) * h / 2
                ham[(i + j) % L, i] = np.conjugate(ham[i, (i + j) % L])

    return ham


def casimir(s0,s1,s2):
    return np.sqrt(s0**2-s1**2-s2**2+0j)
    


def alpha(s0,s1,s2,t,L):
    return np.cos(np.pi*k*casimir(s0,s1,s2)*t/L)+1j*s0/casimir(s0,s1,s2)*np.sin(np.pi*k*casimir(s0,s1,s2)*t/L)



def alphaconj(s0,s1,s2,t,L):
    return np.cos(np.pi*k*casimir(s0,s1,s2)*t/L)-1j*s0/casimir(s0,s1,s2)*np.sin(np.pi*k*casimir(s0,s1,s2)*t/L)


def beta(s0,s1,s2,t,L):
    return 1j*(s1+1j*s2)/casimir(s0,s1,s2)*np.sin(np.pi*k*t/L*casimir(s0,s1,s2))


def betaconj(s0,s1,s2,t,L):
    return np.conj(1j*(s1+1j*s2)/casimir(s0,s1,s2)*np.sin(np.pi*k*t/L*casimir(s0,s1,s2)))


s0=1
s1=0
s2=0




ss0=0
ss1=1
ss2=0


k=2
h=0





def mob(x, L):
    return ss0 + ss1*np.cos(2*np.pi*k*(x+1)/L)+ss2*np.sin(2*np.pi*k*(x+1)/L)




def Hssd(L):
    return fermions1D(L, lambda i: mob(i, L), hopping=[0, 1], pbc=True)

def H0(L):
    return fermions1D(L, lambda i: 1, hopping=[0, 1], pbc=True)



def energy(state):
    energytot=0
    for x in range(L-1):
        energytot+= -0.5 * (state[x, x + 1] + state[x + 1, x])
    return energytot


def entropy(state, subset, eps=1e-8):
    if len(subset) == 0:
        return 0
    w, v = LA.eigh(state[subset][:, subset])
    e_sum = 0
    for i in range(len(subset)):
        if w[i] > eps and w[i] < (1 - eps):
            e_sum = e_sum - (w[i] * np.log(w[i]) + (1 - w[i]) * np.log(1 - w[i]))
    return e_sum


# parameters    
L=802
T0=L/50
T1=L/10
# gamma=0.1
cycles = 3000
tau=0.2

entropylist=[]
energylist=[]


energyCFT=[]
entropyCFT=[]


nonunitarymat=expm(-tau*T0*H0(L))
#nonunitarymat=expm(-1j*(1-1j*tau)*T0*H0(L))
unitarymat=expm(-1j*T1*Hssd(L))

def unitarylist(m):
    if m==0:
        return unitarymat
    if m==1:
        return nonunitarymat

vals, U = np.linalg.eigh(H0(L))
U = np.matrix(U)
Uf = U[:,: int(L/2)]
Uf2 = U[:,: int(L/2)]

Q,R=LA.qr(Uf)
correlation= (Uf@(Uf.conj().T)).T
Uf=Q
l=int(L/2)
#entropylist.append(entropy(correlation, np.arange(0, l + 1)))
#energylist.append(energy(correlation))


distribution = np.zeros(shape=(int(L)))
distribution[: int(L / 2)] = 1
stateini = np.matrix(np.einsum("ik, kj, k -> ij", U, U.H, distribution))

cycleentries=[0]
for j in range(cycles):
    print(j)
    Q,R=LA.qr(unitarylist(gammaw(j))@Uf)
    Uf=Q
    correlation= (Uf@(Uf.conj().T)).T
    l=int(L/2)
    # print(Uf.H @ Uf)

    
    entropylist.append(entropy(correlation,  np.arange(1, (L // 2) + 1)))
    energylist.append(energy(correlation))


plt.plot(extract_fibonacci_positions(np.array(energylist))-energylist[0],marker='o',markerfacecolor='none', markersize=6, linestyle=' ')
plt.show()
plt.plot(extract_fibonacci_positions(np.array(entropylist)),marker='o',markerfacecolor='none', markersize=6, linestyle=' ')
plt.show()




q=2

matnonuni=np.array([[np.exp(-np.pi*tau*q/L*T0),0],[0,np.exp(np.pi*tau*q/L*T0)]])
mat2=np.array([[alpha(ss0,ss1,ss2,T1,L),beta(ss0,ss1,ss2,T1,L)],[betaconj(ss0,ss1,ss2,T1,L),alphaconj(ss0,ss1,ss2,T1,L)]])

def quasiperiodicsequences(m):
    if m==0:
        return mat2
    if m==1:
        return matnonuni

entropyCFTrandom=[]
energyCFTrandom=[]



hq = 1/q*(h+1/24*(q**2-1))
ncyclemat=np.identity(2, dtype=complex)
for m in range(cycles):
    ncyclemat=quasiperiodicsequences(gammaw(m))@ncyclemat
    zn = (ncyclemat[0,1]/ncyclemat[1,1])
    znb = np.conj(zn)
    energyCFTrandom.append(4*np.pi/L*q*hq*(1+abs(zn)**2)/(1-abs(zn)**2)-np.pi*q**2/(6*L))
    entropyCFTrandom.append(np.log(((2 * np.pi)/L)**(-4) * np.abs( ((-1 + znb)**2 * (-1 + (-1)**q * znb)**2 * ((-((-1 + zn) * znb) / (-1 + znb))**(1/q) + (-1)**q * (1 + (1 - zn * znb) / (-1 + (-1)**q * znb))**(1/q))**2) / ((-1)**q * znb**2 * (-((-1 + zn) * znb) / (-1 + znb))**(-1 + 1/q) * (-1 + zn * znb)**2 * (1 + (1 - zn * znb) / (-1 + (-1)**q * znb))**(-1 + 1/q)) )**2)/12)


entropyCFTrandom=np.array(entropyCFTrandom)
energyCFTrandom=np.array(energyCFTrandom)


plt.plot(extract_fibonacci_positions(np.array(entropyCFTrandom))-(entropyCFTrandom[0]-entropylist[0]), marker='o', markerfacecolor='none', markersize=6,color='blue', linestyle='--')
plt.plot(extract_fibonacci_positions(np.array(entropylist)),marker='x', linestyle=' ',color='red')
plt.savefig('fiboentanglemententropy.pdf')



plt.plot(extract_fibonacci_positions(np.array(energylist))-energylist[0], marker='o', markerfacecolor='none', markersize=6,color='blue', linestyle='--')
plt.plot(extract_fibonacci_positions(np.array(energyCFTrandom))-energyCFTrandom[0],marker='x', linestyle=' ',color='red')
plt.savefig('fiboenergy.pdf')



