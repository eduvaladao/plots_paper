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
from scipy.integrate import quad 


# Funcs

cpdef double Sin(double x):
    return sin(x)

cpdef double Cos(double x):
    return cos(x)

cpdef double Sqrt(double x):
    if (x >= 0) or (np.allclose(x, 0)):
        return cmath.sqrt(x).real
    else:
        return np.nan


# Changing the order of the parameters to use fsolve for s (L3)

cpdef double L_ex(double s, double phi0, double theta, double epsilon, double r0):
    return lengths.L3_ex(r0, s, phi0, theta, epsilon)

cpdef double L_in(double s, double phi0, double theta, double epsilon, double r0):
    return lengths.L3_in(r0, s, phi0, theta, epsilon)

cpdef double W_ex(double s, double phi0, double theta, double epsilon, double r0):
    return widths.Wc_ex(r0, s, phi0, theta, epsilon)

cpdef double W_in(double s, double phi0, double theta, double epsilon, double r0):
    return widths.Wc_in(r0, s, phi0, theta, epsilon)


# Writing L/W - Rth to find the possible values of s when this expression is zero

cpdef double gex(double s, double phi0, double epsilon, double r0, double rth):
    cdef double theta = 0.
    return (L_ex(s, phi0, theta, epsilon, r0)/W_ex(s, phi0, theta, epsilon, r0)) - rth 
    
cpdef double gin(double s, double phi0, double epsilon, double r0, double rth):
    cdef double theta = 0.
    return (L_in(s, phi0, theta, epsilon, r0)/W_in(s, phi0, theta, epsilon, r0)) - rth 
    
    
# Using fsolve (sciroot) to find the s that will be one of the limits in the integration

cpdef double sth_ex(double phi0, double epsilon, double r0, double rth, double chute):
    # return sciopt.fsolve(gex, chute, args=(phi0, epsilon, r0, rth), xtol=1e-02, maxfev=100)[0]
    return sciopt.root(gex, chute, args=(phi0, epsilon, r0, rth), method='lm').x[0] 
    
cpdef double sth_in(double phi0, double epsilon, double r0, double rth, double chute):
    # return sciopt.fsolve(gin, chute, args=(phi0, epsilon, r0, rth), xtol=1e-02, maxfev=100)[0]
    return sciopt.root(gin, chute, args=(phi0, epsilon, r0, rth), method='lm').x[0] 
    
    
# Integrating first from sthy (sth0 is correct, but we are using sthy for the analytic part) to sth_ex(in)  

cpdef double sthy_index(theta, phi0, epsilon, r0):
    return angles.sthy(r0, phi0, theta, epsilon)  

cpdef double sigma_ex(double phi0, double epsilon, double r0, double rth, double chute):
    cdef double theta = 0.
    return pi*(sth_ex(phi0, epsilon, r0, rth, chute)**2 - sthy_index(theta, phi0, epsilon, r0)**2) 
    
cpdef double sigma_in(double phi0, double epsilon, double r0, double rth, double chute):
    cdef double theta = 0.
    return pi*(sth_in(phi0, epsilon, r0, rth, chute)**2 - sthy_index(theta, phi0, epsilon, r0)**2)   
     
     
# Taking the mean of sigma in phi0 from 0 to pi
       
cpdef double sigma_ex_med(double epsilon, double r0, double rth):
    cdef double chute = pi/(rth - 1)**2 - (2.0*pi*(r0**2))/3.0
    return (1/pi)*quad(sigma_ex, 0.0, pi, args=(epsilon, r0, rth, chute))[0] 
    
cpdef double sigma_in_med(double epsilon, double r0, double rth):
    cdef double chute = pi/(rth + 1)**2 - (2.0*pi*(r0**2))/3.0
    return (1/pi)*quad(sigma_in, 0.0, pi, args=(epsilon, r0, rth, chute))[0]


# sigma_L2 Perturbed Solutions:
    
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

