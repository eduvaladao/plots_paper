# cython: language_level=3


import numpy as np
import cmath
import scipy.optimize as sciopt 
import angles
import lengths
import widths

cimport numpy as np
cimport cython
cimport scipy.optimize as sciopt

from libc.math cimport sin, pi, cos
from scipy.integrate import quad, fixed_quad, quadrature, romberg 


cpdef double Sin(double x):
    return sin(x)

cpdef double Cos(double x):
    return cos(x)

cpdef double Sqrt(double x):
    if (x >= 0) or (np.allclose(x, 0)):
        return cmath.sqrt(x).real
    else:
        return np.nan


cpdef double sthy(double r0, double phi0, double epsilon, double chute):
    return r0*Sqrt((1 + epsilon*Cos(2*phi0))/(1 - epsilon**2)) 
    
 
cpdef double gex(double s, double phi0, double epsilon, double r0, double rth):
    cdef double theta = 0.
    return (lengths.L_ex(s, theta, phi0, epsilon, r0)/widths.W_ex(s, theta, phi0, epsilon, r0)) - rth 

cpdef double sth_ex(double phi0, double epsilon, double r0, double rth, double chute):
    return sciopt.fsolve(gex, chute, args=(phi0, epsilon, r0, rth), xtol=1e-02, maxfev=100)[0]
    # return sciopt.root(gex, chute, args=(phi0, epsilon, r0, rth), method='lm').x[0] 

cpdef double sigma_ex(double phi0, double epsilon, double r0, double rth, double chute):
    return pi*(sth_ex(phi0, epsilon, r0, rth, chute)**2 - sthy(r0, phi0, epsilon, chute)**2) 
       
cpdef double sigma_ex_medio(double epsilon, double r0, double rth):
    cdef double chute = -r0*(1/Sin(r0 - r0*rth))
    return (1/pi)*quad(sigma_ex, 0.0, pi, args=(epsilon, r0, rth, chute))[0] 

    
cpdef double sigma_pertL2_med_ex(double epsilon, double r0, double rth):
    return (pi/((rth - 1)**2) - (2*pi*(r0**2))/3) + ((3*pi*(rth**2))/(2*((rth - 1)**4)) + (pi*(r0**2)*(4*(rth**2) - 4*rth - 1))/(3*((rth - 1)**2)))*(epsilon**2) 
    
    
# Defining the relative difference between L3crossec and L2crossec
cpdef double delL2_cross_pert_ex(double epsilon, double r0, double rth):
    return (sigma_pertL2_med_ex(epsilon, r0, rth)/sigma_ex_medio(epsilon, r0, rth) - 1.0)

"""cpdef double delL2_cross_pert_in(double epsilon, double r0, double rth, double chute):
    return (sigma_pertL2_med_in(epsilon, r0, rth)/sigma_in_medio(epsilon, r0, rth, chute) - 1.0)"""

"""def delL2_cross_pert_ex(epsilon, r0, rth, chute):
    try:
        return (sigma_pertL2_med_ex(epsilon, r0, rth)/sigma_ex_medio(epsilon, r0, rth, chute) - 1.0)
    except ZeroDivisionError:
        return 0.0"""

