# cython: language_level=3

import numpy as np
import arcs
import angles
import cmath
import lengths 

cimport numpy as np 

from scipy.integrate import quad
from libc.math cimport sin, cos, acos, sqrt, pi


# Funcs

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

# Exact Width

cpdef double W_phi(double phi, double r0, double s, double phi0, double theta, double epsilon):
    return arcs.xmais(phi, r0, s, phi0, theta, epsilon) - arcs.xmenos(phi, r0, s, phi0, theta, epsilon)

cpdef double Wc_ex(double r0, double s, double phi0, double theta, double epsilon):
    return W_phi(angles.phic_ex(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)

cpdef double Wc_in(double r0, double s, double phi0, double theta, double epsilon):
    return W_phi(angles.phic_in(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)


# Perturbed Width
 
cpdef double Wc_pert(double r0, double s, double phi0, double theta, double epsilon):
    return 2*r0 + r0*Cos(2*phi0)*epsilon + ((3.0/4.0)*r0*(Cos(2*phi0)**2) - ((r0**3)*Sin(2*phi0)/s**2)*(Sin(2*phi0) + 2*Sin(2*phi0)))*(epsilon**2) 
    
    
#================
# Renan 
#================ 

cpdef double integrand_area(double phi, double r0, double s, double phi0, double theta, double epsilon):
    """
    Area of each arc
    """
    return W_phi(phi, r0, s, phi0, theta, epsilon) * arcs.xmed(phi, r0, s, phi0, theta, epsilon)


cpdef double Area_ex(double r0, double s, double phi0, double theta, double epsilon):
    return quad(
        integrand_area,
        angles.phii_ex(r0, s, phi0, theta, epsilon),
        angles.phif_ex(r0, s, phi0, theta, epsilon),
        args=(r0, s, phi0, theta, epsilon)
    )[0]


cpdef double Area_in(double r0, double s, double phi0, double theta, double epsilon):
    return quad(
        integrand_area,
        angles.phii_in(r0, s, phi0, theta, epsilon),
        angles.phif_in(r0, s, phi0, theta, epsilon),
        args=(r0, s, phi0, theta, epsilon)
    )[0]


# Widths using the area and the length


cpdef double W1_ex(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_ex(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L1_ex(r0, s, phi0, theta, epsilon))


cpdef double W2_ex(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_ex(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L2_ex(r0, s, phi0, theta, epsilon))


cpdef double W3_ex(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_ex(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L3_ex(r0, s, phi0, theta, epsilon))


cpdef double W4_ex(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_ex(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L4_ex(r0, s, phi0, theta, epsilon))


cpdef double W5_ex(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_ex(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L5_ex(r0, s, phi0, theta, epsilon))


cpdef double W6_ex(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_ex(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L6_ex(r0, s, phi0, theta, epsilon))


cpdef double W7_ex(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_ex(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L7_ex(r0, s, phi0, theta, epsilon))


cpdef double W1_in(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_in(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L1_in(r0, s, phi0, theta, epsilon))


cpdef double W2_in(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_in(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L2_in(r0, s, phi0, theta, epsilon))


cpdef double W3_in(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_in(r0, s, phi0, theta, epsilon) / (np.pi * Lenghts.L3_in(r0, s, phi0, theta, epsilon))


cpdef double W4_in(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_in(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L4_in(r0, s, phi0, theta, epsilon))


cpdef double W5_in(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_in(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L5_in(r0, s, phi0, theta, epsilon))


cpdef double W6_in(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_in(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L6_in(r0, s, phi0, theta, epsilon))


cpdef double W7_in(double r0, double s, double phi0, double theta, double epsilon):
    return 4 * Area_in(r0, s, phi0, theta, epsilon) / (np.pi * lenghts.L7_in(r0, s, phi0, theta, epsilon))


cpdef double W_med_ex(double r0, double s, double phi0, double theta, double epsilon):
    return (1 / (angles.phif_ex(r0, s, phi0, theta, epsilon) - angles.phii_ex(r0, s, phi0, theta, epsilon))) * (
        quad(
            W_phi,
            angles.phii_ex(r0, s, phi0, theta, epsilon),
            angles.phif_ex(r0, s, phi0, theta, epsilon),
            args=(r0, s, phi0, theta, epsilon)
        )[0]
    )


cpdef double W_med_in(double r0, double s, double phi0, double theta, double epsilon):
    return (1 / (angles.phif_in(r0, s, phi0, theta, epsilon) - angles.phii_in(r0, s, phi0, theta, epsilon))) * (
        quad(
            W_phi,
            angles.phii_in(r0, s, phi0, theta, epsilon),
            angles.phif_in(r0, s, phi0, theta, epsilon),
            args=(r0, s, phi0, theta, epsilon)
        )[0]
    )

