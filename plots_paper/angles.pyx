# cython: language_level=3

import numpy as np
import cmath


cimport numpy as np
cimport cython


from libc.math cimport sin, cos, pi


# Funcs

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


# Four possible angles: phi1, phi2, phi3, phi4

cpdef double difab(double epsilon):
    return a(epsilon)**2 - b(epsilon)**2 
    
cpdef double sumab(double epsilon):
    return a(epsilon)**2 + b(epsilon)**2 
    

cpdef double P1(double r0, double s, double phi0, double theta, double epsilon):
    cdef double lado1 = difab(epsilon)*(r0**2)*(sumab(epsilon)*(r0**2)*Cos(2*phi0) - (s**2)*(2*Cos(2*(theta - phi0)) + Cos(2*phi0)))
    cdef double lado2 = (difab(epsilon)**2)*(r0**4) + s**4 + (s**2)*(s**2 - sumab(epsilon)*(r0**2))*Cos(2*theta)
    return lado1 + lado2     

cpdef double P2(double r0, double s, double phi0, double theta, double epsilon):
    cdef double lado3 = (sumab(epsilon)*(s**2) - difab(epsilon)*(s**2)*Cos(2*(theta - phi0)) - 2*((a(epsilon)*b(epsilon)*r0)**2))
    cdef double lado4 = ((r0*(s**2)*Sin(2*theta) - difab(epsilon)*(r0**3)*Sin(2*phi0))**2)
    return lado3*lado4 

cpdef double P3(double r0, double s, double phi0, double theta, double epsilon):
    return (difab(epsilon)**2)*(r0**4) + s**4 - 2*difab(epsilon)*(r0**2)*(s**2)*Cos(2*(theta - phi0))
    

cpdef double valuemais(double r0, double s, double phi0, double theta, double epsilon):
    cdef double nmais = P1(r0, s, phi0, theta, epsilon) + Sqrt(2*P2(r0, s, phi0, theta, epsilon))
    return Sqrt(nmais(r0, s, phi0, theta, epsilon)/(2*P3(r0, s, phi0, theta, epsilon)))
    
cpdef double valuemenos(double r0, double s, double phi0, double theta, double epsilon):
    cdef double nmenos = P1(r0, s, phi0, theta, epsilon) - Sqrt(2*P2(r0, s, phi0, theta, epsilon))
    return Sqrt(nmenos(r0, s, phi0, theta, epsilon)/(2*P3(r0, s, phi0, theta, epsilon)))
    

cpdef double phi1(double r0, double s, double phi0, double theta, double epsilon):
    return Acos(valuemais(r0, s, phi0, theta, epsilon)) 

cpdef double phi2(double r0, double s, double phi0, double theta, double epsilon):
    return Acos(-valuemenos(r0, s, phi0, theta, epsilon))

cpdef double phi3(double r0, double s, double phi0, double theta, double epsilon):
    return Acos(-valuemais(r0, s, phi0, theta, epsilon)) 

cpdef double phi4(double r0, double s, double phi0, double theta, double epsilon):
    return Acos(valuemenos(r0, s, phi0, theta, epsilon))     
    
    
# Conditions for possible intersections and quadrant changes    
    
@cython.cdivision(True)
cpdef double theta0(double phi0, double epsilon):
    """
    When the quadrant of the intersection changes
    """
    cdef double nom1 = sumab(epsilon) + difab(epsilon)*Cos(2*phi0)
    cdef double den1 = 2*(a(epsilon)**4 + b(epsilon)**4 + (a(epsilon)**4 - b(epsilon)**4)*Cos(2*phi0))
    return Acos(nom1/den1)

@cython.cdivision(True)
cpdef double sthc(double r0, double phi0, double theta, double epsilon):
    """
    When the quadrant with the center of the arc changes
    """
    return abs(r0*Sqrt(difab(epsilon))*Sqrt(Sin(2*phi0)/Sin(2*theta)))

@cython.cdivision(True)
cpdef double sthx(double r0, double phi0, double theta, double epsilon):
    """
    The ellipse intersects the x-axis
    """
    cdef double nom2 = sumab(epsilon) - difab(epsilon)*Cos(2*phi0)
    cdef double den2 = 1 - Cos(2*theta)
    return r0*Sqrt(nom2/den2)

@cython.cdivision(True)
cpdef double sthy(double r0, double phi0, double theta, double epsilon):
    """
    The ellipse intersects the y-axis
    """
    cdef double nom3 = sumab(epsilon) + difab(epsilon)*Cos(2*phi0)
    cdef double den3 = 1 + Cos(2*theta)
    return r0*Sqrt(nom3/den3)

cpdef double sth0(double r0, double phi0, double theta, double epsilon):
    """
    Limit between two imagens and Einstein ring
    """
    cdef double nom4 = 2*(a(epsilon)**2)*(b(epsilon)**2)
    cdef double den4 = sumab(epsilon) - difab(epsilon)*Cos(2*(theta - phi0))
    return r0*Sqrt(nom4/den4) 
    
    
# Intervals to obtain phi_i (external and internal arc)

cpdef double interval1_phii(double r0, double s, double phi0, double theta, double epsilon):
    """
    0<=phi0<=pi/2 and 0<=theta<=theta0
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    cdef double sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif s0 <= s <= sy:
        return -phi2(r0, s, phi0, theta, epsilon)
    elif sy < s <= sc:
        return -phi4(r0, s, phi0, theta, epsilon)
    elif sc < s <= sx:
        return -phi1(r0, s, phi0, theta, epsilon)
    elif s > sx:
        return phi1(r0, s, phi0, theta, epsilon)
    else:
        print("Bug found!")


cpdef double interval2_phii(double r0, double s, double phi0, double theta, double epsilon):
    """
    0<=phi0<=pi/2 and theta0<theta<=pi/2
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    cdef double sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif sx >= sy:
        if s0 <= s <= sy:
            return -phi1(r0, s, phi0, theta, epsilon)
        elif sy < s <= sx:
            return -phi1(r0, s, phi0, theta, epsilon)
        elif s > sx:
            return phi1(r0, s, phi0, theta, epsilon)
        else:
            print("Bug found!")
    elif sy > sx:
        if sc <= s0:
            if s0 <= s <= sx:
                return -phi1(r0, s, phi0, theta, epsilon)
            elif sx < s <= sy:
                return phi1(r0, s, phi0, theta, epsilon)
            elif s > sy:
                return phi1(r0, s, phi0, theta, epsilon)
            else:
                print("Bug found!")
        elif sc > s0:
            if s0 <= s <= sx:
                return phi4(r0, s, phi0, theta, epsilon)
            elif sx < s <= sc:
                return phi4(r0, s, phi0, theta, epsilon)
            elif sc < s <= sy:
                return phi1(r0, s, phi0, theta, epsilon)
            elif s > sy:
                return phi1(r0, s, phi0, theta, epsilon)
            else:
                print("Bug found!") 
        else: 
            print("Bug found!")
    else: 
        print("Bug found!")


cpdef double interval3_phii(double r0, double s, double phi0, double theta, double epsilon):
    """
    0<=phi0<=pi/2 and pi/2<theta<=pi-theta0
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    # sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif sx >= sy:        
        if s0 <= s <= sy:
            return phi4(r0, s, phi0, theta, epsilon)
        elif sy < s <= sx:
            return phi2(r0, s, phi0, theta, epsilon)
        elif s > sx:
            return phi2(r0, s, phi0, theta, epsilon) 
        else: 
            print("Bug found!")
    elif sy > sx:
        if s0 <= s <= sx:
            return phi4(r0, s, phi0, theta, epsilon)
        elif sx < s <= sy:
            return phi4(r0, s, phi0, theta, epsilon)
        elif s > sy:
            return phi2(r0, s, phi0, theta, epsilon)
        else: 
            print("Bug found!")    
    else: 
        print("Bug found!")


cpdef double interval4_phii(double r0, double s, double phi0, double theta, double epsilon):
    """
    0<=phi0<=pi/2 and pi-theta0<theta<=pi
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    # sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif s0 <= s <= sy:
        return phi4(r0, s, phi0, theta, epsilon)
    elif sy < s <= sx:
        return phi2(r0, s, phi0, theta, epsilon)
    elif s > sx:
        return phi2(r0, s, phi0, theta, epsilon)
    else:
        print("Bug found!") 
   
        
cpdef double interval5_phii(double r0, double s, double phi0, double theta, double epsilon):
    """
    pi/2<phi0<=pi and 0<=theta<=theta0
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    #cdef double sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif s0 <= s <= sy:
        return -phi1(r0, s, phi0, theta, epsilon)
    elif sy < s <= sx:
        return -phi1(r0, s, phi0, theta, epsilon)
    elif s > sx:
        return phi1(r0, s, phi0, theta, epsilon)
    else:
        print("Bug found!") 
 
        
cpdef double interval6_phii(double r0, double s, double phi0, double theta, double epsilon):
    """
    pi/2<phi0<=pi and theta0<theta<=pi/2
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    # sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif sx >= sy:        
        if s0 <= s <= sy:
            return -phi1(r0, s, phi0, theta, epsilon)
        elif sy < s <= sx:
            return -phi1(r0, s, phi0, theta, epsilon)
        elif s > sx:
            return phi1(r0, s, phi0, theta, epsilon) 
        else: 
            print("Bug found!")
    elif sy > sx:
        if s0 <= s <= sx:
            return -phi1(r0, s, phi0, theta, epsilon)
        elif sx < s <= sy:
            return phi1(r0, s, phi0, theta, epsilon)
        elif s > sy:
            return phi1(r0, s, phi0, theta, epsilon)
        else: 
            print("Bug found!")    
    else: 
        print("Bug found!") 
   
        
cpdef double interval7_phii(double r0, double s, double phi0, double theta, double epsilon):
    """
    pi/2<phi0<=pi and pi/2<theta<=pi-theta0
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    cdef double sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif sx >= sy:
        if s0 <= s <= sy:
            return phi4(r0, s, phi0, theta, epsilon)
        elif sy < s <= sx:
            return phi2(r0, s, phi0, theta, epsilon)
        elif s > sx:
            return phi2(r0, s, phi0, theta, epsilon)
        else:
            print("Bug found!")
    elif sy > sx:
        if sc <= s0:
            if s0 <= s <= sx:
                return phi4(r0, s, phi0, theta, epsilon)
            elif sx < s <= sy:
                return phi4(r0, s, phi0, theta, epsilon)
            elif s > sy:
                return phi2(r0, s, phi0, theta, epsilon)
            else:
                print("Bug found!")
        elif sc > s0:
            if s0 <= s <= sx:
                return -phi1(r0, s, phi0, theta, epsilon)
            elif sx < s <= sc:
                return phi1(r0, s, phi0, theta, epsilon)
            elif sc < s <= sy:
                return phi4(r0, s, phi0, theta, epsilon)
            elif s > sy:
                return phi2(r0, s, phi0, theta, epsilon)
            else:
                print("Bug found!") 
        else: 
            print("Bug found!")
    else: 
        print("Bug found!")


cpdef double interval8_phii(double r0, double s, double phi0, double theta, double epsilon):
    """
    pi/2<phi0<=pi and pi-theta0<theta<=pi
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    cdef double sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif s0 <= s <= sy:
        return phi3(r0, s, phi0, theta, epsilon)
    elif sy < s <= sc:
        return phi3(r0, s, phi0, theta, epsilon)
    elif sc < s <= sx:
        return phi2(r0, s, phi0, theta, epsilon)
    elif s > sx:
        return phi2(r0, s, phi0, theta, epsilon)
    else:
        print("Bug found!")



cpdef double phii_ex(double r0, double s, double phi0, double theta, double epsilon):
    if 0 <= phi0 <= pi/2.0:
        if 0 <= theta <= theta0(phi0, epsilon):
            return interval1_phii(r0, s, phi0, theta, epsilon)
        elif theta0(phi0, epsilon) < theta <= pi/2.0:
            return interval2_phii(r0, s, phi0, theta, epsilon)
        elif pi/2.0 < theta <= pi - theta0(phi0, epsilon):
            return interval3_phii(r0, s, phi0, theta, epsilon)
        elif pi - theta0(phi0, epsilon) < theta <= pi:
            return interval4_phii(r0, s, phi0, theta, epsilon)
        else:
            print("Rest of the symmetrical range.")
    elif pi/2.0 < phi0 <= pi:
        if 0 <= theta <= theta0(phi0, epsilon):
            return interval5_phii(r0, s, phi0, theta, epsilon)
        elif theta0(phi0, epsilon) < theta <= pi/2.0:
            return interval6_phii(r0, s, phi0, theta, epsilon)
        elif pi/2.0 < theta <= pi - theta0(phi0, epsilon):
            return interval7_phii(r0, s, phi0, theta, epsilon)
        elif pi - theta0(phi0, epsilon) < theta <= pi:
            return interval8_phii(r0, s, phi0, theta, epsilon)
        else:
            print("Rest of the symmetrical range.")
    else:
        print("Bug found!")   
        
cpdef double phii_in(double r0, double s, double phi0, double theta, double epsilon):
    return phii_ex(r0, s, phi0, theta, epsilon) + pi 
    
    
# Intervals to obtain phi_f (external and internal arc) 

cpdef double interval1_phif(double r0, double s, double phi0, double theta, double epsilon):
    """
    0<=phi0<=pi/2 and 0<=theta<=theta0
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    cdef double sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif s0 <= s <= sy:
        return phi1(r0, s, phi0, theta, epsilon)
    elif sy < s <= sc:
        return phi1(r0, s, phi0, theta, epsilon)
    elif sc < s <= sx:
        return phi4(r0, s, phi0, theta, epsilon)
    elif s > sx:
        return phi4(r0, s, phi0, theta, epsilon)
    else:
        print("Bug found!")


cpdef double interval2_phif(double r0, double s, double phi0, double theta, double epsilon):
    """
    0<=phi0<=pi/2 and theta0<theta<=pi/2
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    cdef double sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif sx >= sy:
        if s0 <= s <= sy:
            return phi2(r0, s, phi0, theta, epsilon)
        elif sy < s <= sx:
            return phi4(r0, s, phi0, theta, epsilon)
        elif s > sx:
            return phi4(r0, s, phi0, theta, epsilon)
        else:
            print("Bug found!")
    elif sy > sx:
        if sc <= s0:
            if s0 <= s <= sx:
                return phi2(r0, s, phi0, theta, epsilon)
            elif sx < s <= sy:
                return phi2(r0, s, phi0, theta, epsilon)
            elif s > sy:
                return phi4(r0, s, phi0, theta, epsilon)
            else:
                print("Bug found!")
        elif sc > s0:
            if s0 <= s <= sx:
                return -phi3(r0, s, phi0, theta, epsilon)
            elif sx < s <= sc:
                return phi3(r0, s, phi0, theta, epsilon)
            elif sc < s <= sy:
                return phi2(r0, s, phi0, theta, epsilon)
            elif s > sy:
                return phi4(r0, s, phi0, theta, epsilon)
            else:
                print("Bug found!") 
        else: 
            print("Bug found!")
    else: 
        print("Bug found!")


cpdef double interval3_phif(double r0, double s, double phi0, double theta, double epsilon):
    """
    0<=phi0<=pi/2 and pi/2<theta<=pi-theta0
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    # sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif sx >= sy:        
        if s0 <= s <= sy:
            return -phi3(r0, s, phi0, theta, epsilon)
        elif sy < s <= sx:
            return -phi3(r0, s, phi0, theta, epsilon)
        elif s > sx:
            return phi3(r0, s, phi0, theta, epsilon) 
        else: 
            print("Bug found!")
    elif sy > sx:
        if s0 <= s <= sx:
            return -phi3(r0, s, phi0, theta, epsilon)
        elif sx < s <= sy:
            return phi3(r0, s, phi0, theta, epsilon)
        elif s > sy:
            return phi3(r0, s, phi0, theta, epsilon)
        else: 
            print("Bug found!")    
    else: 
        print("Bug found!")


cpdef double interval4_phif(double r0, double s, double phi0, double theta, double epsilon):
    """
    0<=phi0<=pi/2 and pi-theta0<theta<=pi
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    # sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif s0 <= s <= sy:
        return -phi3(r0, s, phi0, theta, epsilon)
    elif sy < s <= sx:
        return -phi3(r0, s, phi0, theta, epsilon)
    elif s > sx:
        return phi3(r0, s, phi0, theta, epsilon)
    else:
        print("Bug found!") 
   
        
cpdef double interval5_phif(double r0, double s, double phi0, double theta, double epsilon):
    """
    pi/2<phi0<=pi and 0<=theta<=theta0
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    #cdef double sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif s0 <= s <= sy:
        return phi2(r0, s, phi0, theta, epsilon)
    elif sy < s <= sx:
        return phi4(r0, s, phi0, theta, epsilon)
    elif s > sx:
        return phi4(r0, s, phi0, theta, epsilon)
    else:
        print("Bug found!") 
 
        
cpdef double interval6_phif(double r0, double s, double phi0, double theta, double epsilon):
    """
    pi/2<phi0<=pi and theta0<theta<=pi/2
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    # sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif sx >= sy:        
        if s0 <= s <= sy:
            return phi2(r0, s, phi0, theta, epsilon)
        elif sy < s <= sx:
            return phi4(r0, s, phi0, theta, epsilon)
        elif s > sx:
            return phi4(r0, s, phi0, theta, epsilon) 
        else: 
            print("Bug found!")
    elif sy > sx:
        if s0 <= s <= sx:
            return phi2(r0, s, phi0, theta, epsilon)
        elif sx < s <= sy:
            return phi2(r0, s, phi0, theta, epsilon)
        elif s > sy:
            return phi4(r0, s, phi0, theta, epsilon)
        else: 
            print("Bug found!")    
    else: 
        print("Bug found!") 
   
        
cpdef double interval7_phif(double r0, double s, double phi0, double theta, double epsilon):
    """
    pi/2<phi0<=pi and pi/2<theta<=pi-theta0
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    cdef double sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif sx >= sy:
        if s0 <= s <= sy:
            return -phi3(r0, s, phi0, theta, epsilon)
        elif sy < s <= sx:
            return -phi3(r0, s, phi0, theta, epsilon)
        elif s > sx:
            return phi3(r0, s, phi0, theta, epsilon)
        else:
            print("Bug found!")
    elif sy > sx:
        if sc <= s0:
            if s0 <= s <= sx:
                return -phi3(r0, s, phi0, theta, epsilon)
            elif sx < s <= sy:
                return phi3(r0, s, phi0, theta, epsilon)
            elif s > sy:
                return phi3(r0, s, phi0, theta, epsilon)
            else:
                print("Bug found!")
        elif sc > s0:
            if s0 <= s <= sx:
                return phi2(r0, s, phi0, theta, epsilon)
            elif sx < s <= sc:
                return phi2(r0, s, phi0, theta, epsilon)
            elif sc < s <= sy:
                return phi3(r0, s, phi0, theta, epsilon)
            elif s > sy:
                return phi3(r0, s, phi0, theta, epsilon)
            else:
                print("Bug found!") 
        else: 
            print("Bug found!")
    else: 
        print("Bug found!")


cpdef double interval8_phif(double r0, double s, double phi0, double theta, double epsilon):
    """
    pi/2<phi0<=pi and pi-theta0<theta<=pi
    """
    cdef double s0 = sth0(r0, phi0, theta, epsilon)
    cdef double sx = sthx(r0, phi0, theta, epsilon)
    cdef double sy = sthy(r0, phi0, theta, epsilon)
    cdef double sc = sthc(r0, phi0, theta, epsilon)
    if s < s0:
        return np.nan
    elif s0 <= s <= sy:
        return -phi4(r0, s, phi0, theta, epsilon)
    elif sy < s <= sc:
        return -phi2(r0, s, phi0, theta, epsilon)
    elif sc < s <= sx:
        return -phi3(r0, s, phi0, theta, epsilon)
    elif s > sx:
        return phi3(r0, s, phi0, theta, epsilon)
    else:
        print("Bug found!")



cpdef double phif_ex(double r0, double s, double phi0, double theta, double epsilon):
    if 0 <= phi0 <= pi/2.0:
        if 0 <= theta <= theta0(phi0, epsilon):
            return interval1_phif(r0, s, phi0, theta, epsilon)
        elif theta0(phi0, epsilon) < theta <= pi/2.0:
            return interval2_phif(r0, s, phi0, theta, epsilon)
        elif pi/2.0 < theta <= pi - theta0(phi0, epsilon):
            return interval3_phif(r0, s, phi0, theta, epsilon)
        elif pi - theta0(phi0, epsilon) < theta <= pi:
            return interval4_phif(r0, s, phi0, theta, epsilon)
        else:
            print("Rest of the symmetrical range.")
    elif pi/2.0 < phi0 <= pi:
        if 0 <= theta <= theta0(phi0, epsilon):
            return interval5_phif(r0, s, phi0, theta, epsilon)
        elif theta0(phi0, epsilon) < theta <= pi/2.0:
            return interval6_phif(r0, s, phi0, theta, epsilon)
        elif pi/2.0 < theta <= pi - theta0(phi0, epsilon):
            return interval7_phif(r0, s, phi0, theta, epsilon)
        elif pi - theta0(phi0, epsilon) < theta <= pi:
            return interval8_phif(r0, s, phi0, theta, epsilon)
        else:
            print("Rest of the symmetrical range.")
    else:
        print("Bug found!")   
        
cpdef double phif_in(double r0, double s, double phi0, double theta, double epsilon):
    return phif_ex(r0, s, phi0, theta, epsilon) + pi 
         

# Angle of the center of the arcs
    
cpdef double phic_ex(double r0, double s, double phi0, double theta, double epsilon):
    return (phii_ex(r0, s, phi0, theta, epsilon) + phif_ex(r0, s, phi0, theta, epsilon))/2.0 
    
cpdef double phic_in(double r0, double s, double phi0, double theta, double epsilon):
    return (phii_in(r0, s, phi0, theta, epsilon) + phif_in(r0, s, phi0, theta, epsilon))/2.0 
    
    
# Arc aperture (renan)

cpdef double deltaphi_ex(double r0, double s, double phi0, double theta, double epsilon):

    if phii_ex(r0, s, phi0, theta, epsilon) <= 0:
        return abs(phii_ex(r0, s, phi0, theta, epsilon)) + abs(phif_ex(r0, s, phi0, theta, epsilon))
    else:
        return abs(phif_ex(r0, s, phi0, theta, epsilon)) - abs(phii_ex(r0, s, phi0, theta, epsilon))


cpdef double deltaphi_in(double r0, double s, double phi0, double theta, double epsilon):
    if phii_in(r0, s, phi0, theta, epsilon) <= 0:
        if phif_in(r0, s, phi0, theta, epsilon) >= 0:
            return abs(phif_in(r0, s, phi0, theta, epsilon)) + abs(phii_in(r0, s, phi0, theta, epsilon))
        else:
            return abs(phif_in(r0, s, phi0, theta, epsilon)) - abs(phii_in(r0, s, phi0, theta, epsilon))
    else:
        return abs(phif_in(r0, s, phi0, theta, epsilon)) - abs(phii_in(r0, s, phi0, theta, epsilon))

