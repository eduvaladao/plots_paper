# cython: language_level=3

import numpy as np
import cmath
import arcs
import angles

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


# L3:  

cpdef double integrand_L3(phi, double r0, double s, double phi0, double theta, double epsilon):
    cdef double fator1 = (2 - 3*(epsilon**2))*Sin(theta - phi)
    cdef double numxmed = (s*(fator1 + epsilon*(epsilon*Sin(theta + 3*phi - 4*phi0) - 2*Cos(theta - phi)*Sin(2*(phi - phi0))))) 
    cdef double denxmed = (2*((-1 + epsilon*Cos(2*(phi - phi0)))**2))
    cdef double derivadaxmed = numxmed/denxmed
    cdef double valuex = arcs.xmed(phi, r0, s, phi0, theta, epsilon)**2 + derivadaxmed**2 
    return Sqrt(valuex) 
    
cpdef double L3_ex(double r0, double s, double phi0, double theta, double epsilon):
    return quad(
        integrand_L3, 
        angles.phii_ex(r0, s, phi0, theta, epsilon), 
        angles.phif_ex(r0, s, phi0, theta, epsilon), 
        args=(r0, s, phi0, theta, epsilon)
    )[0] 

cpdef double L3_in(double r0, double s, double phi0, double theta, double epsilon):
    return quad(
        integrand_L3, 
        angles.phii_in(r0, s, phi0, theta, epsilon), 
        angles.phif_in(r0, s, phi0, theta, epsilon), 
        args=(r0, s, phi0, theta, epsilon)
    )[0]  
    
  
# L2 (perturbed solutions for s > sy, theta = 0): 

# L2 of the external arc perturbed up to epsilon^2 for 0 <= phi0 <= pi [!!!]
cpdef double L2_pert_ex(double r0, double s, double phi0, double epsilon):
    return 2*(r0 + Acos(Sqrt(1 - (r0/s)**2))) + ((r0*Cos(2*phi0)*(2*(r0**2)*(3 + Sqrt(s**2 - r0**2)) - 3*(s**2)*(1 + Sqrt(s**2 - r0**2))))/(3*(s**2)*Sqrt(s**2 - r0**2)))*epsilon - (r0/(120*(s**4)*((s**2 - r0**2)**(3/2))))*((15*(s**6)*(1 + Sqrt(s**2 - r0**2)) + 24*(r0**6)*(10 + 13*Sqrt(s**2 - r0**2)) + 5*(r0**2)*(s**4)*(18 + 29*Sqrt(s**2 - r0**2)) - 8*(r0**4)*(s**2)*(45 + 59*Sqrt(s**2 - r0**2)))*Cos(4*phi0) - 15*(16*(r0**6)*Sqrt(s**2 - r0**2) - 24*(r0**4)*(s**2)*Sqrt(s**2 - r0**2) + (r0**2)*(s**4)*(Sqrt(s**2 - r0**2) - 6) + 7*(s**6)*(1 + Sqrt(s**2 - r0**2))) + 240*(r0**2)*Sqrt(s**2 - r0**2)*(2*(r0**4) - 3*(r0**2)*(s**2) + s**4)*Sin(2*phi0)*Sin(2*phi0))*(epsilon**2) 
    
# L2 of the internal arc perturbed up to epsilon^2 for 0 <= phi0 <= pi [!!!]
cpdef double L2_pert_in(double r0, double s, double phi0, double epsilon):
    return 2*(-r0 + Acos(Sqrt(1 - (r0/s)**2))) + ((r0*Cos(2*phi0)*(2*(r0**2)*(3 - Sqrt(s**2 - r0**2)) - 3*(s**2)*(1 - Sqrt(s**2 - r0**2))))/(3*(s**2)*Sqrt(s**2 - r0**2)))*epsilon - (r0/(120*(s**4)*((s**2 - r0**2)**(3/2))))*((15*(s**6)*(1 - Sqrt(s**2 - r0**2)) + 24*(r0**6)*(10 - 13*Sqrt(s**2 - r0**2)) + 5*(r0**2)*(s**4)*(18 - 29*Sqrt(s**2 - r0**2)) - 8*(r0**4)*(s**2)*(45 - 59*Sqrt(s**2 - r0**2)))*Cos(4*phi0) + 15*(16*(r0**6)*Sqrt(s**2 - r0**2) - 24*(r0**4)*(s**2)*Sqrt(s**2 - r0**2) + (r0**2)*(s**4)*(Sqrt(s**2 - r0**2) + 6) + 7*(s**6)*(-1 + Sqrt(s**2 - r0**2))) - 240*(r0**2)*Sqrt(s**2 - r0**2)*(2*(r0**4) - 3*(r0**2)*(s**2) + s**4)*Sin(2*phi0)*Sin(2*phi0))*(epsilon**2) 
    
    
# L5 (perturbed solutions for s > sy, theta = 0):
     
# L5 of the external arc perturbed up to epsilon^2 for 0 <= phi0 <= pi [!!!]
cpdef double L5_pert_ex(double r0, double s, double phi0, double epsilon):
    return 2*(1 + s)*Acos(Sqrt(1 - (r0/s)**2)) + ((r0*(1 + s)*Cos(2*phi0)*(2*(r0**2) - s**2))/((s**2)*Sqrt(s**2 - r0**2)))*epsilon + (r0/(s**3))*((1 + 1/s)*((7*(s**6) - 6*(r0**2)*(s**4) + (24*(r0**4)*(s**2) - 16*(r0**6) - 6*(r0**2)*(s**4) - s**6)*Cos(4*phi0))/(8*((s**2 - r0**2)**(3/2)))) - r0*Acos(Sqrt(1 - (r0/s)**2))*((r0**2)*Sin(2*phi0) + 2*(s**2)*Sin(2*phi0))*Sin(2*phi0))*(epsilon**2)
    
# L5 of the internal arc perturbed up to epsilon^2 for 0 <= phi0 <= pi [!!!]
cpdef double L5_pert_in(double r0, double s, double phi0, double epsilon):
    return 2*(1 - s)*Acos(Sqrt(1 - (r0/s)**2)) + ((r0*(1 - s)*Cos(2*phi0)*(2*(r0**2) - s**2))/((s**2)*Sqrt(s**2 - r0**2)))*epsilon + (r0/(s**3))*(((1 - s)*(7*(s**6) - 6*(r0**2)*(s**4) + (24*(r0**4)*(s**2) - 16*(r0**6) - 6*(r0**2)*(s**4) - s**6)*Cos(4*phi0)))/(8*s*((s**2 - r0**2)**(3/2))) + r0*(r0**2 + 2*(s**2))*Acos(Sqrt(1 - (r0/s)**2))*(Sin(2*phi0)**2))*(epsilon**2) 
    
    
# L6 (perturbed solutions for s > sy, theta = 0):    

# L6 of the external arc perturbed up to epsilon^2 for 0 <= phi0 <= pi/2 [!!!]
cpdef double L6_pert_ex(double r0, double s, double phi0, double epsilon):
    return 2*(1 + Sqrt(s**2 - r0**2))*Acos(Sqrt(1 - (r0/s)**2)) + (r0/((s**2)*((s**2 - r0**2)**(3/2))))*((s**2 - r0**2)*Acos(Sqrt(1 - (r0/s)**2))*(r0*(2*(r0**2) - 3*(s**2))*Cos(2*phi0) - 2*Sqrt(s**2 - r0**2)*(Sin(2*phi0)*(s**2 - 2*(r0**2)) + (r0**2)*Sin(2*phi0))) - (1 + Sqrt(s**2 - r0**2))*Cos(2*phi0)*(2*(r0**4) - 3*(r0**2)*(s**2) + s**4))*epsilon + (r0/(8*(s**4)*((s**2 - r0**2)**(3/2))))*(7*(s**6)*(1 + Sqrt(s**2 - r0**2)) - 8*(r0**4)*(2*(s**2) - r0**2)*Sqrt(s**2 - r0**2) - 6*(r0**2)*(s**4) + (2*(r0**2) - s**2)*(((s**4)*(1 + Sqrt(s**2 - r0**2)) - 4*(r0**4)*(2 + Sqrt(s**2 - r0**2)) + 2*(r0**2)*(s**2)*(4 + Sqrt(s**2 - r0**2)))*Cos(4*phi0) - 4*r0*(s**2 - r0**2)*(2*(r0**2)*Cos(2*phi0)*Sin(2*phi0) - (2*(r0**2) - s**2)*Sin(4*phi0))) + Acos(Sqrt(1 - (r0/s)**2))*(104*(r0**5)*(s**2) - 48*(r0**7) - 65*(r0**3)*(s**4) + 8*r0*(s**6) + r0*(40*(r0**6) - 96*(r0**4)*(s**2) + 71*(r0**2)*(s**4) - 16*(s**6))*Cos(4*phi0) - 4*(s**2 - r0**2)*(2*(r0**2)*Sqrt(s**2 - r0**2)*(7*(s**2) - 6*(r0**2))*Cos(2*phi0)*Sin(2*phi0) + 4*r0*(6*(r0**4) - 7*(r0**2)*(s**2) + s**4)*Sin(2*phi0)*Sin(2*phi0) + Sqrt(s**2 - r0**2)*(4*(r0**4) - 8*(r0**2)*(s**2) + s**4)*Sin(4*phi0))))*(epsilon**2)
    
# L6 of the internal arc perturbed up to epsilon^2 for 0 <= phi0 <= pi/2 [!!!]
cpdef double L6_pert_in(double r0, double s, double phi0, double epsilon):
    return 2*(1 - Sqrt(s**2 - r0**2))*Acos(Sqrt(1 - (r0/s)**2)) + (r0/((s**2)*((s**2 - r0**2)**(3/2))))*(-(2*(r0**4) - 3*(r0**2)*(s**2) + s**4)*(1 - Sqrt(s**2 - r0**2))*Cos(2*phi0) - (s**2 - r0**2)*Acos(Sqrt(1 - (r0/s)**2))*(r0*(2*(r0**2) - 3*(s**2))*Cos(2*phi0) - 2*((s**2 - r0**2)**(3/2))*Sin(2*phi0)))*epsilon + (r0/(8*(s**4)*((s**2 - r0**2)**(3/2))))*(-6*(r0**2)*(s**4) - (r0**2 - 2*(s**2))*8*(r0**4)*Sqrt(s**2 - r0**2) + 7*(s**6)*(1 - Sqrt(s**2 - r0**2)) + (2*(r0**2) - s**2)*(((s**4)*(1 - Sqrt(s**2 - r0**2))) - 4*(r0**4)*(2 - Sqrt(s**2 - r0**2)) + 2*(r0**2)*(s**2)*(4 - Sqrt(s**2 - r0**2)))*Cos(4*phi0) + 4*r0*((r0**2 - s**2)**2)*Sin(4*phi0) + Acos(Sqrt(1 - (r0/s)**2))*((r0**3)*(s**4) + r0*(8*(r0**6) - 8*(r0**4)*(s**2) - 7*(r0**2)*(s**4) + 8*(s**6))*Cos(4*phi0) + 4*Sqrt(s**2 - r0**2)*(2*(r0**6) - (r0**4)*(s**2) - 2*(r0**2)*(s**4) + s**6)*Sin(4*phi0)))*(epsilon**2) 
    
    
# L7 (perturbed solutions for s > sy, theta = 0): 

# L7 of the external arc perturbed up to epsilon^2 for 0 <= phi0 <= pi/2 [!!!]
cpdef double L7_pert_ex(double r0, double s, double phi0, double epsilon):
    return 2*(1 + Sqrt(s**2 - r0**2))*Acos(Sqrt(1 - (r0/s)**2)) + (r0/((s**2)*((s**2 - r0**2)**(3/2))))*((s**2 - r0**2)*Acos(Sqrt(1 - (r0/s)**2))*(r0*(2*(r0**2) - 3*(s**2))*Cos(2*phi0) + 2*Sqrt(s**2 - r0**2)*(Sin(2*phi0)*(s**2 - 2*(r0**2)) + (r0**2)*Sin(2*phi0))) - (1 + Sqrt(s**2 - r0**2))*Cos(2*phi0)*(2*(r0**4) - 3*(r0**2)*(s**2) + s**4))*epsilon + (r0/(8*(s**4)*((s**2 - r0**2)**(3/2))))*(7*(s**6)*(1 + Sqrt(s**2 - r0**2)) - 8*(r0**4)*(2*(s**2) - r0**2)*Sqrt(s**2 - r0**2) - 6*(r0**2)*(s**4) + (2*(r0**2) - s**2)*(((s**4)*(1 + Sqrt(s**2 - r0**2)) - 4*(r0**4)*(2 + Sqrt(s**2 - r0**2)) + 2*(r0**2)*(s**2)*(4 + Sqrt(s**2 - r0**2)))*Cos(4*phi0) + 4*r0*(s**2 - r0**2)*(2*(r0**2)*Cos(2*phi0)*Sin(2*phi0) - (2*(r0**2) - s**2)*Sin(4*phi0))) + Acos(Sqrt(1 - (r0/s)**2))*(104*(r0**5)*(s**2) - 48*(r0**7) - 65*(r0**3)*(s**4) + 8*r0*(s**6) + r0*(40*(r0**6) - 96*(r0**4)*(s**2) + 71*(r0**2)*(s**4) - 16*(s**6))*Cos(4*phi0) - 4*(s**2 - r0**2)*(-2*(r0**2)*Sqrt(s**2 - r0**2)*(7*(s**2) - 6*(r0**2))*Cos(2*phi0)*Sin(2*phi0) + 4*r0*(6*(r0**4) - 7*(r0**2)*(s**2) + s**4)*Sin(2*phi0)*Sin(2*phi0) - Sqrt(s**2 - r0**2)*(4*(r0**4) - 8*(r0**2)*(s**2) + s**4)*Sin(4*phi0))))*(epsilon**2)
    
# L7 of the internal arc perturbed up to epsilon^2 for 0 <= phi0 <= pi/2 [!!!]
cpdef double L7_pert_in(double r0, double s, double phi0, double epsilon):    
    return 2*(1 - Sqrt(s**2 - r0**2))*Acos(Sqrt(1 - (r0/s)**2)) + (r0/((s**2)*((s**2 - r0**2)**(3/2))))*(-(2*(r0**4) - 3*(r0**2)*(s**2) + s**4)*(1 - Sqrt(s**2 - r0**2))*Cos(2*phi0) - (s**2 - r0**2)*Acos(Sqrt(1 - (r0/s)**2))*(r0*(2*(r0**2) - 3*(s**2))*Cos(2*phi0) + 2*((s**2 - r0**2)**(3/2))*Sin(2*phi0)))*epsilon + (r0/(8*(s**4)*((s**2 - r0**2)**(3/2))))*(-6*(r0**2)*(s**4) - (r0**2 - 2*(s**2))*8*(r0**4)*Sqrt(s**2 - r0**2) + 7*(s**6)*(1 - Sqrt(s**2 - r0**2)) - (2*(r0**2) - s**2)*((-(s**4)*(1 - Sqrt(s**2 - r0**2))) + 4*(r0**4)*(2 - Sqrt(s**2 - r0**2)) - 2*(r0**2)*(s**2)*(4 - Sqrt(s**2 - r0**2)))*Cos(4*phi0) + 4*r0*((r0**2 - s**2)**2)*Sin(4*phi0) + Acos(Sqrt(1 - (r0/s)**2))*((r0**3)*(s**4) + (8*(r0**7) - 8*(r0**5)*(s**2) - 7*(r0**3)*(s**4) + 8*r0*(s**6))*Cos(4*phi0) - 4*Sqrt(s**2 - r0**2)*(2*(r0**6) - (r0**4)*(s**2) - 2*(r0**2)*(s**4) + s**6)*Sin(4*phi0)))*(epsilon**2) 
    
    
#==================
# Renan 
#==================    


# L1:

cpdef double L1_ex(double r0, double s, double phi0, double theta, double epsilon):
    cdef double xi_a = arcs.xmed(angles.phii_ex(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)
    cdef double xf_a = arcs.xmed(angles.phif_ex(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)
    cdef double xc = arcs.xmed(angles.phic_ex(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)
    cdef double value1 = xc ** 2 + xf_a ** 2 - 2 * xc * xf_a * Cos(angles.phif_ex(r0, s, phi0, theta, epsilon) - angles.phic_ex(r0, s, phi0, theta, epsilon))
    cdef double value2 = xc ** 2 + xi_a ** 2 - 2 * xc * xi_a * Cos(angles.phii_ex(r0, s, phi0, theta, epsilon) - angles.phic_ex(r0, s, phi0, theta, epsilon))
    return Sqrt(value1) + Sqrt(value2)


cpdef double L1_in(double r0, double s, double phi0, double theta, double epsilon):
    cdef double xi_b = arcs.xmed(angles.phii_in(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)
    cdef double xf_b = arcs.xmed(angles.phif_in(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)
    cdef double xc_b = arcs.xmed(angles.phic_in(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)
    cdef double value1 = xc_b ** 2 + xf_b ** 2 - 2 * xc_b * xf_b * Cos(angles.phif_in(r0, s, phi0, theta, epsilon) - angles.phic_in(r0, s, phi0, theta, epsilon))
    cdef double value2 = xc_b ** 2 + xi_b ** 2 - 2 * xc_b * xi_b * Cos(angles.phii_in(r0, s, phi0, theta, epsilon) - angles.phic_in(r0, s, phi0, theta, epsilon))
    return Sqrt(value1) + Sqrt(value2)


# L2:

cpdef double L2_ex(double r0, double s, double phi0, double theta, double epsilon):
    return quad(
        arcs.xmed,
        angles.phii_ex(r0, s, phi0, theta, epsilon),
        angles.phif_ex(r0, s, phi0, theta, epsilon),
        args=(r0, s, phi0, theta, epsilon)
    )[0]


cpdef double L2_in(double r0, double s, double phi0, double theta, double epsilon):
    return quad(
        arcs.xmed,
        angles.phii_in(r0, s, phi0, theta, epsilon),
        angles.phif_in(r0, s, phi0, theta, epsilon),
        args=(r0, s, phi0, theta, epsilon)
    )[0]


# L4:

# External arc: cpdef doubleining each point (x, y)

cpdef double xi_ex(double r0, double s, double phi0, double theta, double epsilon):
    cdef double phiiex = angles.phii_ex(r0, s, phi0, theta, epsilon)
    return arcs.xmed(phiiex, r0, s, phi0, theta, epsilon) * Cos(phiiex)


cpdef double yi_ex(double r0, double s, double phi0, double theta, double epsilon):
    cdef double phiiex = angles.phii_ex(r0, s, phi0, theta, epsilon)
    return arcs.xmed(phiiex, r0, s, phi0, theta, epsilon) * Sin(phiiex)


cpdef double xc_ex(double r0, double s, double phi0, double theta, double epsilon):
    cdef double phicex = angles.phic_ex(r0, s, phi0, theta, epsilon)
    return arcs.xmed(phicex, r0, s, phi0, theta, epsilon) * Cos(phicex)


cpdef double yc_ex(double r0, double s, double phi0, double theta, double epsilon):
    cdef double phicex = angles.phic_ex(r0, s, phi0, theta, epsilon)
    return arcs.xmed(phicex, r0, s, phi0, theta, epsilon) * Sin(phicex)


cpdef double xf_ex(double r0, double s, double phi0, double theta, double epsilon):
    cdef double phifex = angles.phif_ex(r0, s, phi0, theta, epsilon)
    return arcs.xmed(Angles.phif_ex(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon) * Cos(phifex)


cpdef double yf_ex(double r0, double s, double phi0, double theta, double epsilon):
    cdef double phifex = angles.phif_ex(r0, s, phi0, theta, epsilon)
    return arcs.xmed(phifex, r0, s, phi0, theta, epsilon) * Sin(phifex)


# Determinant


cpdef double M11_ex(double r0, double s, double phi0, double theta, double epsilon):
    return np.linalg.det(
        np.array(
            [
                [xf_ex(r0, s, phi0, theta, epsilon), yf_ex(r0, s, phi0, theta, epsilon), 1],
                [xc_ex(r0, s, phi0, theta, epsilon), yc_ex(r0, s, phi0, theta, epsilon), 1],
                [xi_ex(r0, s, phi0, theta, epsilon), yi_ex(r0, s, phi0, theta, epsilon), 1],
            ]
        )
    )


cpdef double M12_ex(double r0, double s, double phi0, double theta, double epsilon):
    return np.linalg.det(
        np.array(
            [
                [
                    xf_ex(r0, s, phi0, theta, epsilon) ** 2 + yf_ex(r0, s, phi0, theta, epsilon) ** 2,
                    yf_ex(r0, s, phi0, theta, epsilon),
                    1,
                ],
                [
                    xc_ex(r0, s, phi0, theta, epsilon) ** 2 + yc_ex(r0, s, phi0, theta, epsilon) ** 2,
                    yc_ex(r0, s, phi0, theta, epsilon),
                    1,
                ],
                [
                    xi_ex(r0, s, phi0, theta, epsilon) ** 2 + yi_ex(r0, s, phi0, theta, epsilon) ** 2,
                    yi_ex(r0, s, phi0, theta, epsilon),
                    1,
                ],
            ]
        )
    )


cpdef double M13_ex(double r0, double s, double phi0, double theta, double epsilon):
    return np.linalg.det(
        np.array(
            [
                [
                    xf_ex(r0, s, phi0, theta, epsilon) ** 2 + yf_ex(r0, s, phi0, theta, epsilon) ** 2,
                    xf_ex(r0, s, phi0, theta, epsilon),
                    1,
                ],
                [
                    xc_ex(r0, s, phi0, theta, epsilon) ** 2 + yc_ex(r0, s, phi0, theta, epsilon) ** 2,
                    xc_ex(r0, s, phi0, theta, epsilon),
                    1,
                ],
                [
                    xi_ex(r0, s, phi0, theta, epsilon) ** 2 + yi_ex(r0, s, phi0, theta, epsilon) ** 2,
                    xi_ex(r0, s, phi0, theta, epsilon),
                    1,
                ],
            ]
        )
    )


cpdef double M14_ex(double r0, double s, double phi0, double theta, double epsilon):
    return np.linalg.det(
        np.array(
            [
                [
                    xf_ex(r0, s, phi0, theta, epsilon) ** 2 + yf_ex(r0, s, phi0, theta, epsilon) ** 2,
                    xf_ex(r0, s, phi0, theta, epsilon),
                    yf_ex(r0, s, phi0, theta, epsilon),
                ],
                [
                    xc_ex(r0, s, phi0, theta, epsilon) ** 2 + yc_ex(r0, s, phi0, theta, epsilon) ** 2,
                    xc_ex(r0, s, phi0, theta, epsilon),
                    yc_ex(r0, s, phi0, theta, epsilon),
                ],
                [
                    xi_ex(r0, s, phi0, theta, epsilon) ** 2 + yi_ex(r0, s, phi0, theta, epsilon) ** 2,
                    xi_ex(r0, s, phi0, theta, epsilon),
                    yi_ex(r0, s, phi0, theta, epsilon),
                ],
            ]
        )
    )


# (x0, y0) is the position of the center of the circle and rc its radius

cpdef double x0_ex(double r0, double s, double phi0, double theta, double epsilon):
    return M12_ex(r0, s, phi0, theta, epsilon) / (2 * M11_ex(r0, s, phi0, theta, epsilon))


cpdef double y0_ex(double r0, double s, double phi0, double theta, double epsilon):
    return -M13_ex(r0, s, phi0, theta, epsilon) / (2 * M11_ex(r0, s, phi0, theta, epsilon))


cpdef double rc_ex(double r0, double s, double phi0, double theta, double epsilon):
    cdef double value = x0_ex(r0, s, phi0, theta, epsilon) ** 2 + y0_ex(r0, s, phi0, theta, epsilon) ** 2 + M14_ex(r0, s, phi0, theta, epsilon) / M11_ex(r0, s, phi0, theta, epsilon)
    return Sqrt(value)


cpdef double L4_ex(double r0, double s, double phi0, double theta, double epsilon):
    # REMOVE line below if it works...
    # deltaphi_ex = (np.abs(phif_ex(r0, s, phi0, theta, epsilon))+np.abs(phii_ex(r0, s, phi0, theta, epsilon)))
    cdef double value1 = xi_ex(r0, s, phi0, theta, epsilon) ** 2 + yi_ex(r0, s, phi0, theta, epsilon) ** 2
    cdef double value2 = xf_ex(r0, s, phi0, theta, epsilon) ** 2 + yf_ex(r0, s, phi0, theta, epsilon) ** 2
    cdef double ri_ex = Sqrt(value1)  # initial radial coordinate of the arc
    cdef double rf_ex = Sqrt(value2)  # final radial coordinate
    cdef double numerator_ex = ri_ex ** 2 + rf_ex ** 2 - 2 * ri_ex * rf_ex * Cos(angles.deltaphi_ex(r0, s, phi0, theta, epsilon))
    cdef double deltatheta_ex = Acos(1 - numerator_ex / (2 * rc_ex(r0, s, phi0, theta, epsilon) ** 2))
    return (rc_ex(r0, s, phi0, theta, epsilon) * deltatheta_ex)

# Internal arc

# Doubleining each point (x, y).

cpdef double xi_in(double r0, double s, double phi0, double theta, double epsilon):
    cdef double phiiin = angles.phii_in(r0, s, phi0, theta, epsilon)
    return arcs.xmed(phiiin, r0, s, phi0, theta, epsilon) * Cos(phiiin)


cpdef double yi_in(double r0, double s, double phi0, double theta, double epsilon):
    cdef double phiiin = angles.phii_in(r0, s, phi0, theta, epsilon)
    return arcs.xmed(phiiin, r0, s, phi0, theta, epsilon) * Sin(phiiin)


cpdef double xc_in(double r0, double s, double phi0, double theta, double epsilon):
    cdef double phicin = angles.phic_in(r0, s, phi0, theta, epsilon)
    return arcs.xmed(phicin, r0, s, phi0, theta, epsilon) * Cos(phicin)


cpdef double yc_in(double r0, double s, double phi0, double theta, double epsilon):
    cdef double phicin = angles.phic_in(r0, s, phi0, theta, epsilon)
    return arcs.xmed(phicin, r0, s, phi0, theta, epsilon) * Sin(phicin)


cpdef double xf_in(double r0, double s, double phi0, double theta, double epsilon):
    cdef double phifin = angles.phif_in(r0, s, phi0, theta, epsilon)
    return arcs.xmed(Angles.phif_in(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon) * Cos(phifin)


cpdef double yf_in(double r0, double s, double phi0, double theta, double epsilon):
    cdef double phifin = angles.phif_in(r0, s, phi0, theta, epsilon)
    return arcs.xmed(phifin, r0, s, phi0, theta, epsilon) * Sin(phifin)


# Determinant:


cpdef double M11_in(double r0, double s, double phi0, double theta, double epsilon):
    return np.linalg.det(
        np.array(
            [
                [xf_in(r0, s, phi0, theta, epsilon), yf_in(r0, s, phi0, theta, epsilon), 1],
                [xc_in(r0, s, phi0, theta, epsilon), yc_in(r0, s, phi0, theta, epsilon), 1],
                [xi_in(r0, s, phi0, theta, epsilon), yi_in(r0, s, phi0, theta, epsilon), 1],
            ]
        )
    )


cpdef double M12_in(double r0, double s, double phi0, double theta, double epsilon):
    return np.linalg.det(
        np.array(
            [
                [
                    xf_in(r0, s, phi0, theta, epsilon) ** 2 + yf_in(r0, s, phi0, theta, epsilon) ** 2,
                    yf_in(r0, s, phi0, theta, epsilon),
                    1,
                ],
                [
                    xc_in(r0, s, phi0, theta, epsilon) ** 2 + yc_in(r0, s, phi0, theta, epsilon) ** 2,
                    yc_in(r0, s, phi0, theta, epsilon),
                    1,
                ],
                [
                    xi_in(r0, s, phi0, theta, epsilon) ** 2 + yi_in(r0, s, phi0, theta, epsilon) ** 2,
                    yi_in(r0, s, phi0, theta, epsilon),
                    1,
                ],
            ]
        )
    )


cpdef double M13_in(double r0, double s, double phi0, double theta, double epsilon):
    return np.linalg.det(
        np.array(
            [
                [
                    xf_in(r0, s, phi0, theta, epsilon) ** 2 + yf_in(r0, s, phi0, theta, epsilon) ** 2,
                    xf_in(r0, s, phi0, theta, epsilon),
                    1,
                ],
                [
                    xc_in(r0, s, phi0, theta, epsilon) ** 2 + yc_in(r0, s, phi0, theta, epsilon) ** 2,
                    xc_in(r0, s, phi0, theta, epsilon),
                    1,
                ],
                [
                    xi_in(r0, s, phi0, theta, epsilon) ** 2 + yi_in(r0, s, phi0, theta, epsilon) ** 2,
                    xi_in(r0, s, phi0, theta, epsilon),
                    1,
                ],
            ]
        )
    )


cpdef double M14_in(double r0, double s, double phi0, double theta, double epsilon):
    return np.linalg.det(
        np.array(
            [
                [
                    xf_in(r0, s, phi0, theta, epsilon) ** 2 + yf_in(r0, s, phi0, theta, epsilon) ** 2,
                    xf_in(r0, s, phi0, theta, epsilon),
                    yf_in(r0, s, phi0, theta, epsilon),
                ],
                [
                    xc_in(r0, s, phi0, theta, epsilon) ** 2 + yc_in(r0, s, phi0, theta, epsilon) ** 2,
                    xc_in(r0, s, phi0, theta, epsilon),
                    yc_in(r0, s, phi0, theta, epsilon),
                ],
                [
                    xi_in(r0, s, phi0, theta, epsilon) ** 2 + yi_in(r0, s, phi0, theta, epsilon) ** 2,
                    xi_in(r0, s, phi0, theta, epsilon),
                    yi_in(r0, s, phi0, theta, epsilon),
                ],
            ]
        )
    )


# (x0, y0) is the position of the center of the circle and rc its radius

cpdef double x0_in(double r0, double s, double phi0, double theta, double epsilon):
    return M12_in(r0, s, phi0, theta, epsilon) / (2 * M11_in(r0, s, phi0, theta, epsilon))


cpdef double y0_in(double r0, double s, double phi0, double theta, double epsilon):
    return -M13_in(r0, s, phi0, theta, epsilon) / (2 * M11_in(r0, s, phi0, theta, epsilon))


cpdef double rc_in(double r0, double s, double phi0, double theta, double epsilon):
    cdef double value = x0_in(r0, s, phi0, theta, epsilon) ** 2 + y0_in(r0, s, phi0, theta, epsilon) ** 2 + M14_in(r0, s, phi0, theta, epsilon) / M11_in(r0, s, phi0, theta, epsilon)
    return Sqrt(value)


cpdef double L4_in(double r0, double s, double phi0, double theta, double epsilon):
    # deltaphi_in = (np.abs(phif_in(r0, s, phi0, theta, epsilon))+np.abs(phii_in(r0, s, phi0, theta, epsilon)))
    cdef double value1 = xi_in(r0, s, phi0, theta, epsilon) ** 2 + yi_in(r0, s, phi0, theta, epsilon) ** 2
    cdef double value2 = xf_in(r0, s, phi0, theta, epsilon) ** 2 + yf_in(r0, s, phi0, theta, epsilon) ** 2
    cdef double ri_in = Sqrt(value1)  # initial radial coordinate of the arc
    cdef double rf_in = Sqrt(value2)  # final radial coordinate
    cdef double numerator_in = ri_in ** 2 + rf_in ** 2 - 2 * ri_in * rf_in * Cos(angles.deltaphi_in(r0, s, phi0, theta, epsilon))
    cdef double deltatheta_in = Acos(1 - numerator_in / (2 * rc_in(r0, s, phi0, theta, epsilon) ** 2))
    return (rc_in(r0, s, phi0, theta, epsilon) * deltatheta_in)


# L5:

cpdef double L5_ex(double r0, double s, double phi0, double theta, double epsilon):
    return (
        arcs.xmed(angles.phic_ex(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)
        * angles.deltaphi_ex(r0, s, phi0, theta, epsilon)
    )


cpdef double L5_in(double r0, double s, double phi0, double theta, double epsilon):
    return (
        arcs.xmed(angles.phic_in(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)
        * angles.deltaphi_in(r0, s, phi0, theta, epsilon)
    )


# L6:

cpdef double L6_ex(double r0, double s, double phi0, double theta, double epsilon):
    return (
        arcs.xmed(angles.phif_ex(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)
        * angles.deltaphi_ex(r0, s, phi0, theta, epsilon)
    )


cpdef double L6_in(double r0, double s, double phi0, double theta, double epsilon):
    return (
        arcs.xmed(Angles.phif_in(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)
        * angles.deltaphi_in(r0, s, phi0, theta, epsilon)
    )


# L7:

cpdef double L7_ex(double r0, double s, double phi0, double theta, double epsilon):
    return (
        arcs.xmed(angles.phii_ex(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)
        * angles.deltaphi_ex(r0, s, phi0, theta, epsilon)
    )


cpdef double L7_in(double r0, double s, double phi0, double theta, double epsilon):
    return (
        arcs.xmed(angles.phii_in(r0, s, phi0, theta, epsilon), r0, s, phi0, theta, epsilon)
        * angles.deltaphi_in(r0, s, phi0, theta, epsilon)
    )
    
