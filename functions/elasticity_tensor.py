import numpy as np

def ElasticityTensor(material):
    '''
    Creates a fourth-order elasticity tensor for a specified material
    
    Parameters: 
        material: 'Fe' (iron) or 
                  'SiO2' (alpha-quartz) or 
                  'ZrO2' (monoclinic zirconia)
    
    Returns:
        C: corresponding elasticity tensor
    '''
                
    def C_Fe():
        # Elastic constants from Simmons and Wang (1971), in Pa
        C = np.zeros((3,3,3,3))
        C[0,0,0,0] = 231.4e9
        C[0,0,1,1] = 134.7e9
        C[0,1,0,1] = 116.4e9
        C[1,1,1,1] = C[0,0,0,0]
        C[2,2,2,2] = C[0,0,0,0]
        C[0,0,2,2] = C[0,0,1,1]
        C[1,1,2,2] = C[0,0,1,1]
        C[1,2,1,2] = C[0,1,0,1]
        C[0,2,0,2] = C[0,1,0,1]
        return AddSymmetries(C)
    
    def C_SiO2():
        # Elastic constants from Heyliger (2003), in Pa
        C = np.zeros((3,3,3,3))
        C[0,0,0,0] = 87.26e9
        C[1,1,1,1] = C[0,0,0,0]
        C[2,2,2,2] = 105.8e9
        C[0,0,1,1] = 6.57e9
        C[0,0,2,2] = 11.95e9
        C[1,1,2,2] = C[0,0,2,2]
        C[1,2,1,2] = 57.15e9
        C[0,2,0,2] = C[1,2,1,2]
        C[0,1,0,1] = 1/2*(C[0,0,0,0]-C[0,0,1,1])
        C[0,0,1,2] = -17.18e9
        C[1,1,1,2] = -C[0,0,1,2]
        C[0,2,0,1] = C[0,0,1,2]
        return AddSymmetries(C)
    
    def C_ZrO2():
        # Elastic constants from Chan (1991), in Pa
        C = np.zeros((3,3,3,3))
        C[0,0,0,0] = 361e9
        C[1,1,1,1] = 408e9
        C[2,2,2,2] = 258e9
        C[1,2,1,2] = 99.9e9
        C[0,2,0,2] = 81.2e9
        C[0,1,0,1] = 126e9
        C[0,0,1,1] = 142e9
        C[0,0,2,2] = 55.0e9
        C[1,1,2,2] = 196e9
        C[0,0,0,2] = -21.3e9
        C[1,1,0,2] = 31.2e9
        C[2,2,0,2] = -18.2e9
        C[1,2,0,1] = -22.7e9
        return AddSymmetries(C)
    
    def AddSymmetries(C):
        '''
        Adds the minor and major symmetries to the 
        fourth-order elasticity tensor C
        '''
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if C[i,j,k,l]!=0:
                            C[k,l,i,j] = C[i,j,k,l];
                            C[j,i,k,l] = C[i,j,k,l];
                            C[i,j,l,k] = C[i,j,k,l];
                        elif C[k,l,i,j]!=0:
                            C[i,j,k,l] = C[k,l,i,j];
                            C[j,i,k,l] = C[k,l,i,j];
                            C[i,j,l,k] = C[k,l,i,j];
                        elif C[j,i,k,l]!=0:
                            C[i,j,k,l] = C[j,i,k,l];
                            C[k,l,i,j] = C[j,i,k,l];
                            C[i,j,l,k] = C[j,i,k,l];
                        elif C[i,j,l,k]!=0:
                            C[i,j,k,l] = C[i,j,l,k];
                            C[k,l,i,j] = C[i,j,l,k];
                            C[j,i,k,l] = C[i,j,l,k];
        return C
    
    options = {'Fe':C_Fe, 'SiO2':C_SiO2, 'ZrO2':C_ZrO2}
    return options[material]()