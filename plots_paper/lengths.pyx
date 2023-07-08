# cython: language_level=3

import numpy as np
import cmath
import arcs
import angles

cimport numpy as np


from scipy.integrate import quad
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


# L3:  

cpdef double integrand_L3(phi, double r0, double s, double phi0, double theta, double epsilon):
    cdef double derivadaxmed = (s*((2 - 3*(epsilon**2))*Sin(theta - phi) + epsilon*(epsilon*Sin(theta + 3*phi - 4*phi0) - 2*Cos(theta - phi)*Sin(2*(phi - phi0)))))/(2*((-1 + epsilon*Cos(2*(phi - phi0)))**2))
    cdef double result = Sqrt(arcs.xmed(phi, r0, s, phi0, theta, epsilon)**2 + derivadaxmed**2) 
    return result

cpdef double L3_ex1(double r0, double s, double phi0, double theta, double epsilon):
    return quad(integrand_L3, -angles.phi_4(r0, s, phi0, theta, epsilon), angles.phi_1(r0, s, phi0, theta, epsilon), args=(r0, s, phi0, theta, epsilon))[0] 
    
cpdef double L3_ex2(double r0, double s, double phi0, double theta, double epsilon):
    return quad(integrand_L3, -angles.phi_1(r0, s, phi0, theta, epsilon), angles.phi_4(r0, s, phi0, theta, epsilon), args=(r0, s, phi0, theta, epsilon))[0]

cpdef double L3_in1(double r0, double s, double phi0, double theta, double epsilon):
    return quad(integrand_L3, pi - angles.phi_4(r0, s, phi0, theta, epsilon), pi + angles.phi_1(r0, s, phi0, theta, epsilon), args=(r0, s, phi0, theta, epsilon))[0]

cpdef double L3_in2(double r0, double s, double phi0, double theta, double epsilon):
    return quad(integrand_L3, pi - angles.phi_1(r0, s, phi0, theta, epsilon), pi + angles.phi_4(r0, s, phi0, theta, epsilon), args=(r0, s, phi0, theta, epsilon))[0]

cpdef double L_ex(double s, double theta, double phi0, double epsilon, double r0):
    return L3_ex1(r0, s, phi0, theta, epsilon)

cpdef double L_in(double s, double theta, double phi0, double epsilon, double r0):
    return L3_in1(r0, s, phi0, theta, epsilon)

