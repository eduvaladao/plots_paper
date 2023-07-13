# cython: language_level=3

import numpy as np
import cmath


cimport numpy as np


from libc.math cimport sin, cos


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


# Ellipse parametrization

cpdef double a(double epsilon):
    """
    Ellipse parametrization for major semi-axis.
    """
    return 1/Sqrt(1 - epsilon)

cpdef double b(double epsilon):
    """
    Ellipse parametrization for minor semi-axis.
    """
    return 1/Sqrt(1 + epsilon) 


# Parametric equations of an ellipse

cpdef double xell(double phi, double r0, double s, double phi0, double theta, double epsilon):
    return s*Cos(theta) + r0*a(epsilon)*Cos(phi)*Cos(phi0) - r0*b(epsilon)*Sin(phi)*Sin(phi0)

cpdef double yell(double phi, double r0, double s, double phi0, double theta, double epsilon):
    return s*Sin(theta) + r0*a(epsilon)*Cos(phi)*Sin(phi0) + r0*b(epsilon)*Sin(phi)*Cos(phi0)
    
    
# SISell lens equation  

cpdef double part1(double phi, double r0, double s, double phi0, double theta, double epsilon):
    return s*Cos(theta - phi) - epsilon*s*Cos(theta + phi - 2*phi0) 
    
cpdef double part2(double phi, double r0, double s, double phi0, double theta, double epsilon):
    return Sqrt((r0**2)*(1 - epsilon*Cos(2*(phi - phi0))) - (s**2)*(1 - epsilon**2)*(Sin(theta - phi)**2))

cpdef double xmenos(double phi, double r0, double s, double phi0, double theta, double epsilon):
    """
    Inner arc borders for elliptical source.
    """
    return 1 + (1/(1 - epsilon*Cos(2*(phi - phi0))))*(part1(phi, r0, s, phi0, theta, epsilon) - part2(phi, r0, s, phi0, theta, epsilon))

cpdef double xmais(double phi, double r0, double s, double phi0, double theta, double epsilon):
    """
    Outer arc borders for elliptical source.
    """
    return 1 + (1/(1 - epsilon*Cos(2*(phi - phi0))))*(part1(phi, r0, s, phi0, theta, epsilon) + part2(phi, r0, s, phi0, theta, epsilon))  

 
# Arcs ridgeline   

cpdef double xmed(double phi, double r0, double s, double phi0, double theta, double epsilon):
    """
    Arc ridgeline for elliptical source.
    """
    return (xmenos(phi, r0, s, phi0, theta, epsilon) + xmais(phi, r0, s, phi0, theta, epsilon))/2
    
