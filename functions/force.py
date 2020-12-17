import numpy as np
from scipy.optimize import fsolve


def Force(E_interp_B1,E_interp_B2,R_B1,R_B2,M,N,delta,n):
    '''
    Computes the normal force between two contacting, elastically anisotropic bodies
    Parameters:
        E_interp_B1: interpolant of values of the plane strain modulus for the material
                     corresponding to body B1
        E_interp_B2: interpolant of values of the plane strain modulus for the material
                     corresponding to body B2
        R_B1: rotation matrix describing the orientation of body B1
        R_B2: rotation matrix describing the orientation of body B2
        M,N: coefficients of the gap function between bodies B1 and B2
        delta: overlap distance between bodies B1 and B2
        n: components of contact normal direction, given as a 3-by-1 2D array
    
    Returns: 
        F: normal contact force between bodies B1 and B2
    '''    
    
    # Get composite plane strain modulus
    Ec = CompositePlainStrainModulus(E_interp_B1,E_interp_B2,R_B1,R_B2,n)
    
    # Compute contact force
    if np.isclose(M,N):
        F = 4/(3*np.sqrt(2))*Ec*M**(-1/2)*delta**(3/2)
    else:
        e = Eccentricity(M,N)
        F = 4*np.pi/3*I1(0,e)**(1/2)/I0(0,e)**(3/2)*Ec*M**(-1/2)*delta**(3/2)
        
    return F
    
    

def CompositePlainStrainModulus(E_interp_B1,E_interp_B2,R_B1,R_B2,n):
    '''
    Retrieves the composite plain strain modulus for two contacting 
    bodies B1 and B2 from precomputed interpolants of the plane strain moduli.
    This is the algorithm described in Algorithm 3 in Mowlavi & Kamrin (2021),
    with the exception that the look-up tables are replaced by interpolants.
    
    Parameters:  
        E_interp_B1: interpolant of values of the plane strain modulus for the material
                     corresponding to body B1
        E_interp_B2: interpolant of values of the plane strain modulus for the material
                     corresponding to body B2
        R_B1: rotation matrix describing the orientation of body B1
        R_B2: rotation matrix describing the orientation of body B2
        n: components of contact normal direction, given as a 3-by-1 2D array
    
    Returns: 
        Ec: composite plain strain modulus
    '''
    
    # Calculate plain strain modulus for B1 and B2 and store in a Python 
    # dictionary E = {'B1': E_B1, 'B2': E_B2}
    E = {}
    for body in ['B1','B2']:
        # Retrieve corresponding look-up table and rotation matrix
        E_interp,R = (E_interp_B1,R_B1) if body=='B1' else (E_interp_B2,R_B2)
        
        # Ensure that n is unit length
        n = n/np.linalg.norm(n)
        
        # Transform the coordinates of n from global to body basis
        n = R.T@n
        
        # Convert these coordinates to Euler angles
        a = np.arctan2(n[1],n[0])%(2*np.pi)
        b = np.arccos(n[2])
        
        # Interpolate corresponding plane strain modulus from look-up table
        E[body] = np.asscalar(E_interp(a,b))
    
    # Compute the composite plain strain modulus from the stored values
    return 1/(1/E['B1'] + 1/E['B2'])


def Eccentricity(M,N):
    fun = lambda e: N/M - I2(0,e)/I1(0,e)
    e0 = 2*np.sqrt(1-M/N)/np.sqrt(3)
    e = np.asscalar(fsolve(fun,e0))
    return e


def I0(m,e):
    tt = np.linspace(0,np.pi,100)
    ff = np.cos(2*m*tt)/np.sqrt(1-e**2*np.cos(tt)**2)
    return np.trapz(ff,tt)

def I1(m,e):
    tt = np.linspace(0,np.pi,100)
    ff = np.sin(tt)**2*np.cos(2*m*tt)/(1-e**2*np.cos(tt)**2)**(3/2)
    return np.trapz(ff,tt)

def I2(m,e):
    tt = np.linspace(0,np.pi,100)
    ff = np.cos(tt)**2*np.cos(2*m*tt)/(1-e**2*np.cos(tt)**2)**(3/2)
    return np.trapz(ff,tt)