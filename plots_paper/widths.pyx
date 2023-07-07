import numpy as np
import arcs
import angles
import cmath
import lengths 

cimport numpy as np 

from libc.math cimport sin, cos, acos, sqrt, pi


cpdef double Sin(double x):
    return sin(x)

cpdef double Cos(double x):
    return cos(x)

cpdef double Acos(double x):
    return acos(x)

cpdef double Sqrt(double x):
    if (x >= 0) or (np.allclose(x, 0)):
        return cmath.sqrt(x).real
    else:
        return np.nan


cpdef double W_phi(double phi, double r0, double s, double phi0, double theta, double epsilon):
    return arcs.xmais(phi, r0, s, phi0, theta, epsilon) - arcs.xmenos(phi, r0, s, phi0, theta, epsilon)

cpdef double Wc_ex1(double r0, double s, double phi0, double theta, double epsilon):
    return W_phi(angles.phic1_ex(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon).real

cpdef double Wc_ex2(double r0, double s, double phi0, double theta, double epsilon):
    return W_phi(angles.phic2_ex(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon).real 
    
cpdef double W_ex(double s, double theta, double phi0, double epsilon, double r0):
    return Wc_ex1(r0, s, phi0, theta, epsilon)

# @cython.cdivision(True) 
cpdef double Wc_pert(double r0, double s, double phi0, double theta, double epsilon):
    return 2*r0 + r0*Cos(2*phi0)*epsilon + ((3.0/4.0)*r0*(Cos(2*phi0)**2) - ((r0**3)*Sin(2*phi0)/s**2)*(Sin(2*phi0) + 2*Sin(2*phi0)))*(epsilon**2)

