# AnisotropicGrains

This code accompanies the paper [Contact model for elastically anisotropic bodies and efficient implementation into the discrete-element method](https://link.springer.com/article/10.1007/s10035-021-01104-7) by Mowlavi & Kamrin (2021).

Given two elastically anisotropic bodies of arbitrary shape contacting at a single point (see Figure 1), we provide a contact force law that returns the normal force exerted between the two bodies as a function of their orientations, surface geometries and elastic constants, as well as the direction of contact and overlap distance. This contact force law, which is presented in Section 4.1 of the paper, can be implemented in a straightforward manner into any discrete-element method (DEM) code that already tracks particle orientations.

In order to optimize computational efficiency at runtime, the calculation of the contact force relies on precomputed look-up tables of contact modulus (or plain strain modulus) values, one for each material present in the simulation. Such look-up tables have already been provided in ./stored_look_up_tables/ for iron ('Fe'), quartz ('SiO2') and zirconia ('ZrO2'). Additional materials can be defined in ./functions/elasticity_tensor.py, and the corresponding look-up tables will be automatically calculated the first time they are required, and stored for future use.

![sketch](./sketch.png)

## Main files

* **compute_force** computes the force between two elastic bodies (see Figure 1) for a given orientation of the bodies and contact normal direction. This script is intended as a demonstration of how the contact force would be implemented in a DEM code.

* **compute_pole_plot** computes the force between a rigid plate and an elastic body (see Figure 2) for all possible orientations of the contact normal and displays the resulting orientation-dependence of the force in a pole plot. This script reproduces the middle column of Figures 4 and 5 from the paper.

## Dependencies

[Numba](https://numba.pydata.org): Used to accelerate the computation of the look-up tables. If desired, can be deactivated by removing lines 2 and 128 in ./functions/plane_strain_modulus.py.
