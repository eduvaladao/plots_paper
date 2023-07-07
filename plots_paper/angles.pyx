import numpy as np
import cmath
import arcs

cimport numpy as np
cimport cython

from libc.math cimport sin, cos, pi


cpdef double Sin(double x):
    return sin(x)

cpdef double Cos(double x):
    return cos(x)

cpdef double Acos(double x):
    return cmath.acos(x).real

cpdef double Sqrt(double x):
    if (x >= 0) or (np.allclose(x, 0)):
        return cmath.sqrt(x).real
    else:
        return np.nan 
        
        
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


# Four possible angles: phi1, phi2, phi3, phi4

cpdef double P1(double r0, double s, double phi0, double theta, double epsilon):
    return (a(epsilon)**2 - b(epsilon)**2)*(r0**2)*((a(epsilon)**2 + b(epsilon)**2)*(r0**2)*Cos(2*phi0) - (s**2)*(2*Cos(2*(theta - phi0)) + Cos(2*phi0))) + ((a(epsilon)**2 - b(epsilon)**2)**2)*(r0**4) + s**4 + (s**2)*(s**2 - (a(epsilon)**2 + b(epsilon)**2)*(r0**2))*Cos(2*theta)     

cpdef double P2(double r0, double s, double phi0, double theta, double epsilon):
    return ((a(epsilon)**2 + b(epsilon)**2)*(s**2) - (a(epsilon)**2 - b(epsilon)**2)*(s**2)*Cos(2*(theta - phi0)) - 2*((a(epsilon)*b(epsilon)*r0)**2))*((r0*(s**2)*Sin(2*theta) - (a(epsilon)**2 - b(epsilon)**2)*(r0**3)*Sin(2*phi0))**2) 

cpdef double P3(double r0, double s, double phi0, double theta, double epsilon):
    return ((a(epsilon)**2 - b(epsilon)**2)**2)*(r0**4) + s**4 - 2*(a(epsilon)**2 - b(epsilon)**2)*(r0**2)*(s**2)*Cos(2*(theta - phi0))
    

cpdef double phi_1(double r0, double s, double phi0, double theta, double epsilon):
    return Acos(Sqrt((P1(r0, s, phi0, theta, epsilon) + Sqrt(2*P2(r0, s, phi0, theta, epsilon)))/(2*P3(r0, s, phi0, theta, epsilon)))) 

cpdef double phi_2(double r0, double s, double phi0, double theta, double epsilon):
    return Acos(-Sqrt((P1(r0, s, phi0, theta, epsilon) - Sqrt(2*P2(r0, s, phi0, theta, epsilon)))/(2*P3(r0, s, phi0, theta, epsilon))))

cpdef double phi_3(double r0, double s, double phi0, double theta, double epsilon):
    return Acos(-Sqrt((P1(r0, s, phi0, theta, epsilon) + Sqrt(2*P2(r0, s, phi0, theta, epsilon)))/(2*P3(r0, s, phi0, theta, epsilon)))) 

cpdef double phi_4(double r0, double s, double phi0, double theta, double epsilon):
    return Acos(Sqrt((P1(r0, s, phi0, theta, epsilon) - Sqrt(2*P2(r0, s, phi0, theta, epsilon)))/(2*P3(r0, s, phi0, theta, epsilon))))     
    
    
cpdef double phic1_ex(double r0, double s, double phi0, double theta, double epsilon):
    return (phi_1(r0, s, phi0, theta, epsilon) - phi_4(r0, s, phi0, theta, epsilon))/2 

cpdef double phic2_ex(double r0, double s, double phi0, double theta, double epsilon):
    return (-phi_1(r0, s, phi0, theta, epsilon) + phi_4(r0, s, phi0, theta, epsilon))/2 
    
