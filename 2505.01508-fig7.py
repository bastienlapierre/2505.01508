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
from scipy import stats


h=1
k=1

s0=0.5
s1=0.3
s2=0

L=1

cycles=300

t = 0.2
#0.149727
trobi= [0.05,0.08,0.12,0.16,0.17]
#trobi= [0.005, 0.04,0.07,0.17]
listpure=[]
for tau in trobi:
    print(tau)
    
    N = 100  #This is the number of points for the numerical integration
    
    
    ## CFT
    
    
    def casimir(s0,s1,s2):
        return np.sqrt(s0**2-s1**2-s2**2+0j)
        
    
    def alpha(s0,s1,s2,t,L):
        return np.cos(np.pi*k*casimir(s0,s1,s2)*t/L)+1j*s0/casimir(s0,s1,s2)*np.sin(np.pi*k*casimir(s0,s1,s2)*t/L)
    
    
    
    def alphaconj(s0,s1,s2,t,L):
        return np.cos(np.pi*k*casimir(s0,s1,s2)*t/L)-1j*s0/casimir(s0,s1,s2)*np.sin(np.pi*k*casimir(s0,s1,s2)*t/L)
    
    
    def beta(s0,s1,s2,t,L):
        return 1j*(s1+1j*s2)/casimir(s0,s1,s2)*np.sin(np.pi*k*t/L*casimir(s0,s1,s2))
    
    
    def betaconj(s0,s1,s2,t,L):
        return np.conj(1j*(s1+1j*s2))/casimir(s0,s1,s2)*np.sin(np.pi*k*t/L*casimir(s0,s1,s2))
    
    def one_step_time_evolution(s0,s1,t,tau,L,rho,N):
    
        # This is the time evolution. We first evolve with a unitary, then with a dissipative non-unitary, then with a different unitary, then with an amplification. 
    
        #First unitary evolution. The probabilities remain the same
        for i in range(N):
            rho[0,i] = (alpha(s1,s0,s2,t,L)*rho[0,i] + beta(s1,s0,s2,t,L)) / (betaconj(s1,s0,s2,t,L)*rho[0,i] + alphaconj(s1,s0,s2,t,L))
    
        #First non-unitary evolution. We re-normalize the density matrix to one
        for i in range(N):
            rho[1, i] = rho[1, i] * np.exp(2*tau*h) * (1 - np.abs(rho[0,i])**2)**(4*h) / (1 - np.exp(2*tau)*np.abs(rho[0,i])**2)**(4*h)
            rho[0, i] = np.exp(tau) * rho[0,i]
    
        normalization  = np.sum(rho[1])
    
        rho[1] = rho[1] / normalization
    
        #Second unitary evolution
        for i in range(N):
            rho[0,i] = (alpha(s0,s1,s2,t,L)*rho[0,i] + beta(s0,s1,s2,t,L)) / (betaconj(s0,s1,s2,t,L)*rho[0,i] + alphaconj(s0,s1,s2,t,L))
    
        #Second non-unitary evolution
        for i in range(N):
            rho[1, i] = rho[1, i] * np.exp(-2*tau*h) * (1 - np.abs(rho[0,i])**2)**(4*h) / (1 - np.exp(-2*tau)*np.abs(rho[0,i])**2)**(4*h)
            rho[0, i] = np.exp(-tau) * rho[0,i]
        normalization  = np.sum(rho[1])
    
        rho[1] = rho[1] / normalization
    
        return rho
    
    
    
    def Purity(rho, N):
    
        P = 0
    
        for i in range(N):
            for k in range(N):
                P = P + rho[1,i] * rho[1,k] * np.abs( (1 - np.conj(rho[0,i]) * rho[0,k]) / np.sqrt( (1 - np.abs(rho[0,i])**2) * (1 - np.abs(rho[0,k])**2) ) )**(-4*h)
    
                #print((1 - np.abs(rho[0,i])**2) * (1 - np.abs(rho[0,k])**2))
    
        P = np.real(P)
    
        return P
    
    
    
    
    
    #Definition of the initial density matrix
    
    r = 0.7 #Radius of the initial circle
    
    rho = np.zeros((2, N), dtype = complex) #I use first row for the xi's (complex parameter in the unit disk) and the second row for the probabilities
    
    for i in range(N):
    
        rho[0, i] = r*np.exp( ((2*np.pi*i)/N)*1j )
        rho[1, i] = 1/N
    
    
    
    
    P = np.zeros(cycles)
    
    xi = np.zeros(cycles)
    
    #Now we perform the evolution
    
    for i in range(cycles):
    
        P[i] = Purity(rho,N)
    
        xi[i] = np.max(np.abs(rho[0]))
    
        rho = one_step_time_evolution(s0,s1,t,tau,L,rho,N)
        listpure.append(Purity(rho,N))
    


listpure=np.array(listpure)
listpure=listpure.reshape((len(trobi),len(range(cycles))))
color1 = '#8ee5d5'  # Light teal/mint
color2 = '#56b4e9'  # Light blue
color3 = '#2b8cbe'  # Medium blue
color4 = '#213f9a'  # Dark blue
color5 = '#232355'  # Navy blue
plt.semilogy(1-listpure[0, :],color=color1)
plt.semilogy(1-listpure[1, :],color=color2)
plt.semilogy(1-listpure[2, :],color=color3)
plt.semilogy(1-listpure[3, :],color=color4)
plt.semilogy(1-listpure[4, :],color=color5)
plt.savefig('fig7_panela.pdf')










x_c = 0.149727  # The critical point identified
x_data_original = np.array([0.144, 0.1445, 0.145, 0.1455, 0.146, 0.1465, 0.147, 0.1475,
                            0.148, 0.1485, 0.149, 0.1495])
y_data = np.array([313, 327, 344, 364, 387, 416, 452, 500, 567, 673, 874, 1568])

# Calculate distance from critical point
x_data = x_c - x_data_original

# Define power law function
def power_law(x, A, beta):
    return A * x**(-beta)

# Fit the power law function to data
params, covariance = curve_fit(power_law, x_data, y_data)
A, beta = params
beta_error = np.sqrt(covariance[1, 1])

print(f"Power law fit: y = {A:.4f} * (x_c - x)^(-{beta:.4f})")
print(f"Exponent (β): {beta:.4f} ± {beta_error:.4f}")

# Calculate R-squared to assess fit quality
residuals = y_data - power_law(x_data, A, beta)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared: {r_squared:.6f}")

# Create plot for visualization
plt.figure(figsize=(10, 8))

# Log-log plot of data with fit
plt.loglog(x_data, y_data, 'o', label='Data')
x_fit = np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 100)
y_fit = power_law(x_fit, A, beta)
plt.loglog(x_fit, y_fit, 'r-', label=f'Fit: y = {A:.2f} * (x_c - x)^(-{beta:.2f})')

plt.xlabel('x_c - x (log scale)')
plt.ylabel('y (log scale)')
plt.title('Power Law Fit to Diverging Data')
plt.legend()
plt.grid(True, which="both", ls="-")

# Linear plot in log-log scale
plt.figure(figsize=(10, 8))
log_x = np.log10(x_data)
log_y = np.log10(y_data)

# Linear regression on log-transformed data as verification
coeffs = np.polyfit(log_x, log_y, 1)
slope = coeffs[0]
intercept = coeffs[1]

plt.plot(log_x, log_y, 'o', label='Log-transformed data')
plt.plot(log_x, slope*log_x + intercept, 'g--', 
         label=f'Linear fit: log(y) = {slope:.4f}*log(x) + {intercept:.4f}')
plt.xlabel('log(x_c - x)')
plt.ylabel('log(y)')
plt.title('Linear Fit on Log-Transformed Data')
plt.legend()
plt.grid(True)

print(f"Linear fit on log-transformed data: slope = {slope:.4f}")
print(f"This corresponds to a power law exponent of β = {-slope:.4f}")
print(f"Note: This should be close to the curve_fit beta value above")

plt.tight_layout()
plt.show()


# First curve data
x1 = np.array([0.15 - 0.149727, 0.1505 - 0.149727, 0.151 - 0.149727, 
              0.1515 - 0.149727, 0.152 - 0.149727, 0.153 - 0.149727, 
              0.154 - 0.149727, 0.155 - 0.149727, 0.156 - 0.149727, 
              0.157 - 0.149727])
y1 = np.array([1/0.00891557329131687, 1/0.015337680201563848, 1/0.019362341296086542, 
              1/0.022837995164019593, 1/0.02638251, 1/0.03116646, 
              1/0.03541623, 1/0.03927461, 1/0.04283353, 1/0.0461338])

# Second curve data
x2 = np.array([0.149727-0.144, 0.149727-0.1445, 0.149727-0.145, 0.149727-0.1455, 
              0.149727-0.146, 0.149727-0.1465, 0.149727-0.147, 0.149727-0.1475,
              0.149727-0.148, 0.149727-0.1485, 0.149727-0.149, 0.149727-0.1495])
y2 = np.array([313, 327, 344, 364, 387, 416, 452, 500, 567, 673, 874, 1568])

# Function to perform power law fit
def power_law_fit(x, y):
   # Log-transform the data
   log_x = np.log(x)
   log_y = np.log(y)

   # Perform linear regression on log-log data
   slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
   
   return {
       'slope': slope,
       'intercept': np.exp(intercept),
       'r_squared': r_value**2,
       'p_value': p_value,
       'std_err': std_err
   }

# Perform power law fit for both curves
fit1 = power_law_fit(x1, y1)
fit2 = power_law_fit(x2, y2)

# Create the plot
#plt.figure(figsize=(10,10))

# Plot original data points
plt.loglog(x1, y1, marker='o', color='blue')
plt.loglog(x2, y2, marker='o', color='red')

# Generate and plot fitted lines
x1_fit = np.logspace(np.log10(x1.min()), np.log10(x1.max()), 100)
y1_fit = fit1['intercept'] * x1_fit**fit1['slope']
plt.loglog(x1_fit, y1_fit, 'b-', label=f'Fit 1: y = {fit1["intercept"]:.4f} * x^{fit1["slope"]:.4f}')

x2_fit = np.logspace(np.log10(x2.min()), np.log10(x2.max()), 100)
y2_fit = fit2['intercept'] * x2_fit**fit2['slope']
plt.loglog(x2_fit, y2_fit, 'r-', label=f'Fit 2: y = {fit2["intercept"]:.4f} * x^{fit2["slope"]:.4f}')


plt.legend()
plt.savefig('fig7_panelb.pdf')

# Print out the results for both curves
print("Curve 1 Power Law Fit:")
print(f"  Slope: {fit1['slope']}")
print(f"  Coefficient: {fit1['intercept']}")
print(f"  R-squared: {fit1['r_squared']}")
print(f"  P-value: {fit1['p_value']}")

print("\nCurve 2 Power Law Fit:")
print(f"  Slope: {fit2['slope']}")
print(f"  Coefficient: {fit2['intercept']}")
print(f"  R-squared: {fit2['r_squared']}")
print(f"  P-value: {fit2['p_value']}")
