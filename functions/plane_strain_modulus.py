import numpy as np
from numba import njit

def PlaneStrainModulusTable(C,N_alpha=40,N_beta=20):
    '''
    Creates a look-up table of values of the plane strain modulus, as 
    described in Algorithm 2 in Mowlavi & Kamrin (2021)
    
    Parameters:  
        C: elasticity tensor
        N_alpha: number of discretization points for alpha
        N_beta: number of discretization points for beta
    
    Returns: 
        EE: look-up table of values of the plane strain modulus for 
            alpha in aa, beta in bb (alpha along columns, beta along rows)
        aa: vector of values of alpha
        bb: vector of values of beta
    '''
        
    aa = np.linspace(0,2*np.pi,N_alpha)
    bb = np.linspace(0,np.pi,N_beta)
    EE = np.zeros((len(bb),len(aa)))
    
    for ia,a in enumerate(aa):
        for ib,b in enumerate(bb):
            
            print('Progress: %d%%\r'%((ia*N_beta+(ib+1))/(N_alpha*N_beta)*100), end="")
                        
            # Construct contact normal vector n
            n = np.array([[np.sqrt(1-np.cos(b)**2)*np.cos(a)],
                          [np.sqrt(1-np.cos(b)**2)*np.sin(a)],
                          [np.cos(b)]])
                         
            # Construct u and v such that (u,v,n) forms an orthonormal basis
            u,v = NormalVectors(n)
            Q = np.concatenate((u,v,n), axis=1)
                        
            # Obtain the Green's function
            hh,tt = Green(C,Q)
                         
            # Calculate the plane strain modulus and store in look-up table
            EE[ib,ia] = 1/(np.pi*np.mean(hh))        
            
    return EE,aa,bb


def NormalVectors(n):
    '''
    Given a vector n in R^3, this function returns two unit-length
    perpendicular vectors u, v in R^3 such that (u,v,n) constitues a 
    right-handed coordinate system
    
    Parameters:
        n: 3-by-1 2D array
    
    Returns:
        u,v: 3-by-1 2D arrays
    '''
    # Flatten n
    n = n.flatten()
    
    # Initialize
    v = np.zeros(3)

    # Find index of non-zero element, and another index
    ind1 = np.flatnonzero(n)[0]
    ind2 = (ind1+1)%3

    # Create first orhtogonal vector
    v[ind2] =  n[ind1]
    v[ind1] = -n[ind2]

    # Create second orhtogonal vector
    u = np.cross(v,n)

    # Normalize
    u = u/np.linalg.norm(u)
    v = v/np.linalg.norm(v)
        
    return np.reshape(u,(-1,1)), np.reshape(v,(-1,1))


def Green(C, Q, N_theta=100):
    '''
    Calculates the angular part of the Green's function, h(theta;0), 
    as described in Algorithm 1 in Mowlavi & Kamrin (2021)
    
    Parameters:  
        C: elasticity tensor
        Q: orientation matrix
        N_theta: number of discretization points for theta
    
    Returns: 
        hh: h(theta;0) evaluated at the values in tt
        tt: vector of values of theta
    '''
    tt = np.linspace(0,2*np.pi,num=N_theta,endpoint=False)
    gg = np.linspace(0,2*np.pi,50)
    
    # Calculate the angular part of the Green's function
    hh = np.zeros(N_theta)
    for it,t in enumerate(tt):
        
        B = np.zeros((len(gg),3,3))
        for ig,g in enumerate(gg):
            
            # Calculate m and n in the (x1,x2,x3) basis
            r = np.array([[np.cos(g)*np.sin(t),-np.cos(g)*np.cos(t),-np.sin(g)]]).T
            s = np.array([[-np.sin(g)*np.sin(t),np.sin(g)*np.cos(t),-np.cos(g)]]).T
            
            # Calculate m and n in the (X1,X2,X3) basis
            r = Q@r
            s = Q@s
            
            # Calculate matrix B
            B[ig,:,:] = 1/(8*np.pi**2)*(ab(r,r,C)-ab(r,s,C)@np.linalg.inv(ab(s,s,C))@ab(s,r,C))
        
        # Integrate matrix B
        B = np.trapz(B,gg,axis=0)
        
        # Calculate h(theta)
        hh[it] = 1/(8*np.pi**2)*Q[:,[2]].T@np.linalg.inv(B)@Q[:,[2]]
        
    return hh,tt


@njit
def ab(a,b,C):
    res = np.zeros((3,3))
    for j in range(3):
        for k in range(3):
            for i in range(3):
                for m in range(3):
                    res[j,k] = res[j,k] + a[i,0]*C[i,j,k,m]*b[m,0]
    return res
