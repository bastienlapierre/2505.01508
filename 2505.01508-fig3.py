# This code calculates the equal/unequal time correlation matrix under Floquet measurement.
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
import matplotlib.ticker as ticker

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 22})


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
T1=L/100
# gamma=0.1
cycles = 280


entropylist=[]
energylist=[]


energyCFT=[]
entropyCFT=[]


# taulist=[0.32, 0.8, 1.28, 1.76]
taulist=[0.16, 0.4, 0.64, 0.88]
# taulist=[0.64, 1.6, 2.56, 3.52]
# taulist=[6.4, 16, 25.6, 35.2]



for tau in taulist:
    print(tau)

    nonunitarymat=expm(-tau*H0(L))@expm(-1j*T0*H0(L))
    unitarymat=expm(-1j*T1*Hssd(L))
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
        cycleentries.append(j)
        Q,R=LA.qr(unitarymat@Uf)
        Uf=Q
        Q,R=LA.qr(nonunitarymat@Uf)
        Uf=Q
        correlation= (Uf@(Uf.conj().T)).T
        l=int(L/2)
        # print(Uf.H @ Uf)

        
        entropylist.append(entropy(correlation,  np.arange(1, (L // 2) + 1)))
        energylist.append(energy(correlation))
    

    def totalenergycaputa(T0,T1,L,n):
        matnonuni=np.array([[np.exp(-np.pi*tau*k/L),0],[0,np.exp(np.pi*tau*k/L)]])
        #matnonuni=np.array([[alpha(ss0,ss1,ss2,1j*T1*tau1,L),beta(ss0,ss1,ss2,1j*T1*tau1,L)],[betaconj(ss0,ss1,ss2,1j*T1*tau1,L),alphaconj(ss0,ss1,ss2,1j*T1*tau1,L)]])
        mat1=np.array([[alpha(s0,s1,s2,T0,L),beta(s0,s1,s2,T0,L)],[betaconj(s0,s1,s2,T0,L),alphaconj(s0,s1,s2,T0,L)]])
        mat2=np.array([[alpha(ss0,ss1,ss2,T1,L),beta(ss0,ss1,ss2,T1,L)],[betaconj(ss0,ss1,ss2,T1,L),alphaconj(ss0,ss1,ss2,T1,L)]])
        #ncyclemat=np.linalg.matrix_power(mat2@mat1@matnonuni, n)
        ncyclemat=np.linalg.matrix_power(matnonuni@mat1@mat2, n)
        
        zn = (ncyclemat[0,1]/ncyclemat[1,1])
        hk = 1/k*(h+1/24*(k**2-1))
        return 4*np.pi/L*k*hk*(1+abs(zn)**2)/(1-abs(zn)**2)-np.pi*k**2/(6*L)
    
    
    def newtotalentropycaputa(T0,T1,L,n,tau):
        matnonuni=np.array([[np.exp(-np.pi*tau*k/L),0],[0,np.exp(np.pi*tau*k/L)]])
        mat1=np.array([[alpha(s0,s1,s2,T0,L),beta(s0,s1,s2,T0,L)],[betaconj(s0,s1,s2,T0,L),alphaconj(s0,s1,s2,T0,L)]])
        mat2=np.array([[alpha(ss0,ss1,ss2,T1,L),beta(ss0,ss1,ss2,T1,L)],[betaconj(ss0,ss1,ss2,T1,L),alphaconj(ss0,ss1,ss2,T1,L)]])
        ncyclemat=np.linalg.matrix_power(matnonuni@mat1@mat2, n)
        zn = (ncyclemat[0,1]/ncyclemat[1,1])
        znb = np.conj(zn)
        # hk = 1/k*(h+1/24*(k**2-1))
    
        # Following formula holds only for the GS, and for the EE from x = 0 to x = L/2.
        return np.log(((2 * np.pi)/L)**(-4) * np.abs( ((-1 + znb)**2 * (-1 + (-1)**k * znb)**2 * ((-((-1 + zn) * znb) / (-1 + znb))**(1/k) + (-1)**k * (1 + (1 - zn * znb) / (-1 + (-1)**k * znb))**(1/k))**2) / ((-1)**k * znb**2 * (-((-1 + zn) * znb) / (-1 + znb))**(-1 + 1/k) * (-1 + zn * znb)**2 * (1 + (1 - zn * znb) / (-1 + (-1)**k * znb))**(-1 + 1/k)) )**2)/12
    
    # Simple entropy formula only valid for k even:
    def simplified_entropy(T0,T1,L,n,tau):
            matnonuni=np.array([[np.exp(-np.pi*tau*k/L),0],[0,np.exp(np.pi*tau*k/L)]])
            mat1=np.array([[alpha(s0,s1,s2,T0,L),beta(s0,s1,s2,T0,L)],[betaconj(s0,s1,s2,T0,L),alphaconj(s0,s1,s2,T0,L)]])
            mat2=np.array([[alpha(ss0,ss1,ss2,T1,L),beta(ss0,ss1,ss2,T1,L)],[betaconj(ss0,ss1,ss2,T1,L),alphaconj(ss0,ss1,ss2,T1,L)]])
            ncyclemat=np.linalg.matrix_power(matnonuni@mat1@mat2, n)
            zn = (ncyclemat[0,1]/ncyclemat[1,1])
            znb = np.conj(zn)
            # hk = 1/k*(h+1/24*(k**2-1))
        
            # Following formula holds only for the GS, and for the EE from x = 0 to x = L/2.
            return (1/3)*np.log(np.abs((1-zn) * ((-1)**(k)-zn)) / (1 - np.abs(zn)**2))
    
    def simplified_entropy2(T0,T1,L,n,tau):
            matnonuni=np.array([[np.exp(-np.pi*tau*k/L),0],[0,np.exp(np.pi*tau*k/L)]])
            mat1=np.array([[alpha(s0,s1,s2,T0,L),beta(s0,s1,s2,T0,L)],[betaconj(s0,s1,s2,T0,L),alphaconj(s0,s1,s2,T0,L)]])
            mat2=np.array([[alpha(ss0,ss1,ss2,T1,L),beta(ss0,ss1,ss2,T1,L)],[betaconj(ss0,ss1,ss2,T1,L),alphaconj(ss0,ss1,ss2,T1,L)]])
            ncyclemat=np.linalg.matrix_power(matnonuni@mat1@mat2, n)
            zn = (ncyclemat[0,1]/ncyclemat[1,1])
            znb = np.conj(zn)
            # hk = 1/k*(h+1/24*(k**2-1))
        
            # Following formula holds only for the GS, and for the EE from x = 0 to x = L/2.
            return -np.log(((2 * np.pi)/L)**(4) * np.abs(((np.abs(zn)**2 - 1)**2) / (4 * np.sin(np.pi/k)**2 * (zn - 1)**2 * ((-1)**k * np.conj(zn) - 1)**2))**2)/12

    for j in range(1,cycles+1):
        # entropyCFT.append(newtotalentropycaputa(T0,T1,L,j,tau))
        entropyCFT.append(simplified_entropy(T0,T1,L,j,tau))
        energyCFT.append(totalenergycaputa(T0,T1,L,j))



entropyCFT=np.array(entropyCFT)
energyCFT=np.array(energyCFT)
energylist=np.array(energylist)
entropylist=np.array(entropylist)


entropyCFT=entropyCFT.reshape((len(taulist),len(range(cycles))))
energyCFT=energyCFT.reshape((len(taulist),len(range(cycles))))
energylist=energylist.reshape((len(taulist),len(range(cycles))))
entropylist=entropylist.reshape((len(taulist),len(range(cycles))))


for k in range(len(taulist)):
    plt.plot(np.arange(cycles)[::2],entropyCFT[k,:][::2]-(entropyCFT[k,0]-entropylist[k,0]),color=plt.cm.RdYlBu(k/(len(taulist)-1)))
    plt.plot(np.arange(cycles)[::6],entropylist[k,:][::6],marker='o',color=plt.cm.RdYlBu(k/(len(taulist)-1)),markerfacecolor='none', markersize=6, linestyle=' ')

plt.show()

for k in range(len(taulist)):
    plt.plot(np.arange(cycles)[::2],(energyCFT[k,:][::2]-energyCFT[k,0]),color=plt.cm.RdYlBu(k/(len(taulist)-1)))
    plt.plot(np.arange(cycles)[::6],(energylist[k,:][::6]-energylist[k,0]),marker='o',color=plt.cm.RdYlBu(k/(len(taulist)-1)),markerfacecolor='none', markersize=6, linestyle=' ')
# plt.yticks([0, 0.0003, 0.0006, 0.0009, 0.0012]) # Only for non-heating phase
# Get current axis
ax = plt.gca()

# Force scientific notation
formatter = ticker.ScalarFormatter(useMathText=True)  # Use scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))  # Force scientific notation outside the range [-10, 10]

ax.yaxis.set_major_formatter(formatter)  # Apply formatter

plt.show()


