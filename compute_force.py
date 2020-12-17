import numpy as np
from scipy.interpolate import interp2d
import scipy.io as sio
import os

from functions.elasticity_tensor import ElasticityTensor
from functions.plane_strain_modulus import PlaneStrainModulusTable
from functions.force import Force


'''
Parameters
'''

# Materials of bodies B1 and B2
Material_B1,Material_B2 = 'Fe','SiO2'

# Orientations of bodies B1 and B2
R_B1,R_B2 = np.eye(3),np.eye(3)

# Gap function coefficients
M,N = 1e6,2e6

# Overlap distance
delta = 100e-9

# Coordinates of contact normal vector (a,b are Euler angles)
a,b = 0,np.pi/2
n = np.array([[np.cos(a)*np.sqrt(1-np.cos(b)**2)],
              [np.sin(a)*np.sqrt(1-np.cos(b)**2)],
              [np.cos(b)]])


'''
In the first step, typically done offline before the DEM simulation, we 
create a dictionary of interpolants of plain strain modulus values for 
every unique material present in the simulation:
E_interp_dict = {'Material_1': E_interp_1, 'Material_2': E_interp_2, ...}
These interpolants are formed from look-up tables of values of the plain 
strain modulus, which can be precomputed if they don't already exist.
'''

E_interp_dict = {}
SavePath = './stored_lookup_tables/'

for material in [Material_B1,Material_B2]:
    
    # Check if interpolant for material not already present in dictionary
    if material not in E_interp_dict:
        
        # Check if look-up table of plane strain modulus values exists
        if os.path.isfile(SavePath+material+'.mat'):
            
            # Load precomputed look-up table
            print('Loading look-up table from %s' % SavePath+material+'.mat')
            data = sio.loadmat(SavePath+material)
            EE,aa,bb = data['EE'],data['aa'].flatten(),data['bb'].flatten()
        
        else:
            
            # Compute a look-up table of plane strain modulus values
            print('Computing a look-up table for %s' % material)
            C = ElasticityTensor(material)
            EE,aa,bb = PlaneStrainModulusTable(C,N_alpha=60,N_beta=30)
            
            # Save the look-up table for future use
            if not os.path.exists(SavePath): os.makedirs(SavePath)
            sio.savemat(SavePath+material,{'EE':EE,'aa':aa,'bb':bb})
            print('Look-up table saved as %s' % SavePath+material+'.mat')
            
        # Add to dictionary an interpolant of the plane strain modulus
        E_interp_dict[material] = interp2d(aa,bb,EE)
                        

'''
In the second part, done online during the DEM simulation, we calculate 
the normal force between the two contacting bodies by using the 
interpolants corresponding to the materials of these two bodies. 
'''   

# Calculate contact force
F = Force(E_interp_dict[Material_B1],E_interp_dict[Material_B2],R_B1,R_B2,M,N,delta,n)

print('Normal force is %g' % F)