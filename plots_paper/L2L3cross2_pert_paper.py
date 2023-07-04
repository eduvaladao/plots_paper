# !/usr/bin/env python
# coding=utf-8

"""
SIS with Elliptical Source
Gráficos de Comprimento, comparando L2, L5, L6, L7 perturbado analítico até epsilon^2 com L3 para phie entre 0 e pi/2, fonte não encosta no eixo y2

Author: Eduardo da Costa Valadão - eduardovaladao98@gmail.com
"""

from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmath
from scipy.integrate import quad
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.gridspec as gridspec
mpl.style.use('classic')

from LWcross import * 


#======================================================================================================================
#====================================================================================================================== 

fig = plt.figure(figsize=(14, 14), layout="constrained")
gs = fig.add_gridspec(ncols=2, nrows=3)


# Defining the relative difference between L3crossec and L2crossec
def delL2_cross_pert_ex(epsilon, r0, rth, chute):
    try:
        return (sigma_pertL2_med_ex(epsilon, r0, rth)/sigma_ex_medio(epsilon, r0, rth, chute) - 1.0)
    except ZeroDivisionError:
        return 0.0

def delL2_cross_pert_in(epsilon, r0, rth, chute):
    return (sigma_pertL2_med_in(epsilon, r0, rth)/sigma_in_medio(epsilon, r0, rth, chute) - 1.0)
    
# Threshold for the validity of the lenghts based on the choice of angles, i.e. source tangent to the y2 axis
def sthy(r0, phi0, epsilon): 
    return r0*np.sqrt((1 + epsilon*np.cos(2*phi0))/(1 - epsilon**2))
def epsilonthy(r0, s, phi0):
    return (np.sqrt(2)*np.sqrt(8*(s**2)*(s**2 - r0**2) + (r0**4)*(1 + np.cos(4*phi0))) - 2*(r0**2)*np.cos(2*phi0))/(4*(s**2))
def r0thy(s, phi0, epsilon):
    return s/np.sqrt((1 + epsilon*np.cos(2*phi0))/(1 - epsilon**2))

props = dict(boxstyle='square', facecolor='white')  

#-------------------------------------------------------------------------------------------------------------------------------
# sig x Rth (R0 variável, epsilon fixo) ex
#-------------------------------------------------------------------------------------------------------------------------------

rthx = np.linspace(1.5, 15, 210)
rth1 = np.linspace(1.5, 7, 70)
rth2 = np.linspace(7, 11, 70)
rth3 = np.linspace(11, 15, 70)
ax1 = fig.add_subplot(gs[0, 0])

plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.08, x, 0.25) for x in rth1], 'darkorchid', lw=2)
plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.05, x, 0.25) for x in rth1], 'darkcyan', lw=2)
plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.01, x, 0.25) for x in rth1], 'crimson', lw=2)

plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.08, x, 0.13) for x in rth2], 'darkorchid', lw=2)
plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.05, x, 0.13) for x in rth2], 'darkcyan', lw=2)
plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.01, x, 0.12) for x in rth2], 'crimson', lw=2)

plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.08, x, 0.10) for x in rth3], 'darkorchid', lw=2)
plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.05, x, 0.09) for x in rth3], 'darkcyan', lw=2)
plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.01, x, 0.08) for x in rth3], 'crimson', lw=2)

plt.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.4)
plt.xlim(1.5, 15)
plt.ylim(-0.15, 0.15)
plt.xlabel(r'$R_{\rm th}$', fontsize=18)
plt.ylabel(r'$\Delta \sigma_{L/W}^{\rm ex}/\sigma_{L_{3}/W}^{\rm ex}}$', fontsize=15)
ax1.grid(False, which='both')
plt.axis('tight')
plt.text(0.596, 0.04, r'$\varepsilon_s = 0.2, R_0 = 0.1, \varphi_e = \pi/6$', fontsize=14, bbox=props, transform=plt.gca().transAxes, ha='left', va='bottom')
ax1.legend(ncol=4, loc="upper center", handlelength=2.0, handletextpad=0.2, columnspacing=0.2)
   

#-------------------------------------------------------------------------------------------------------------------------------
# sig x Rth (R0 fixo, epsilon variável) ex
#-------------------------------------------------------------------------------------------------------------------------------

rthx = np.linspace(1.5, 15, 210)
rth1 = np.linspace(1.5, 7, 70)
rth2 = np.linspace(7, 11, 70)
rth3 = np.linspace(11, 15, 70)
ax1 = fig.add_subplot(gs[0, 1])

plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.08, x, 0.25) for x in rth1], 'darkorchid', lw=2)
plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.05, x, 0.25) for x in rth1], 'darkcyan', lw=2)
plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.01, x, 0.25) for x in rth1], 'crimson', lw=2)

plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.08, x, 0.13) for x in rth2], 'darkorchid', lw=2)
plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.05, x, 0.13) for x in rth2], 'darkcyan', lw=2)
plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.01, x, 0.12) for x in rth2], 'crimson', lw=2)

plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.08, x, 0.10) for x in rth3], 'darkorchid', lw=2)
plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.05, x, 0.09) for x in rth3], 'darkcyan', lw=2)
plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.01, x, 0.08) for x in rth3], 'crimson', lw=2)

plt.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.4)
plt.xlim(1.5, 15)
plt.ylim(-0.15, 0.15)
plt.xlabel(r'$R_{\rm th}$', fontsize=18)
plt.ylabel(r'$\Delta \sigma_{L/W}^{\rm ex}/\sigma_{L_{3}/W}^{\rm ex}}$', fontsize=15)
ax1.grid(False, which='both')
plt.axis('tight')
plt.text(0.596, 0.04, r'$\varepsilon_s = 0.2, R_0 = 0.1, \varphi_e = \pi/6$', fontsize=14, bbox=props, transform=plt.gca().transAxes, ha='left', va='bottom')
ax1.legend(ncol=4, loc="upper center", handlelength=2.0, handletextpad=0.2, columnspacing=0.2)



#-------------------------------------------------------------------------------------------------------------------------------
# sig x R0 (epsilon variável, Rth fixo) ex
#------------------------------------------------------------------------------------------------------------------------------- 

rthx = np.linspace(1.5, 15, 210)
rth1 = np.linspace(1.5, 7, 70)
rth2 = np.linspace(7, 11, 70)
rth3 = np.linspace(11, 15, 70)
ax1 = fig.add_subplot(gs[1, 0])

plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.08, x, 0.25) for x in rth1], 'darkorchid', lw=2)
plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.05, x, 0.25) for x in rth1], 'darkcyan', lw=2)
plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.01, x, 0.25) for x in rth1], 'crimson', lw=2)

plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.08, x, 0.13) for x in rth2], 'darkorchid', lw=2)
plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.05, x, 0.13) for x in rth2], 'darkcyan', lw=2)
plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.01, x, 0.12) for x in rth2], 'crimson', lw=2)

plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.08, x, 0.10) for x in rth3], 'darkorchid', lw=2)
plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.05, x, 0.09) for x in rth3], 'darkcyan', lw=2)
plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.01, x, 0.08) for x in rth3], 'crimson', lw=2)

plt.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.4)
plt.xlim(1.5, 15)
plt.ylim(-0.15, 0.15)
plt.xlabel(r'$R_{\rm th}$', fontsize=18)
plt.ylabel(r'$\Delta \sigma_{L/W}^{\rm ex}/\sigma_{L_{3}/W}^{\rm ex}}$', fontsize=15)
ax1.grid(False, which='both')
plt.axis('tight')
plt.text(0.596, 0.04, r'$\varepsilon_s = 0.2, R_0 = 0.1, \varphi_e = \pi/6$', fontsize=14, bbox=props, transform=plt.gca().transAxes, ha='left', va='bottom')
ax1.legend(ncol=4, loc="upper center", handlelength=2.0, handletextpad=0.2, columnspacing=0.2)
 


#-------------------------------------------------------------------------------------------------------------------------------
# sig x R0 (epsilon fixo, Rth variável) ex
#-------------------------------------------------------------------------------------------------------------------------------

rthx = np.linspace(1.5, 15, 210)
rth1 = np.linspace(1.5, 7, 70)
rth2 = np.linspace(7, 11, 70)
rth3 = np.linspace(11, 15, 70)
ax1 = fig.add_subplot(gs[1, 1])

plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.08, x, 0.25) for x in rth1], 'darkorchid', lw=2)
plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.05, x, 0.25) for x in rth1], 'darkcyan', lw=2)
plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.01, x, 0.25) for x in rth1], 'crimson', lw=2)

plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.08, x, 0.13) for x in rth2], 'darkorchid', lw=2)
plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.05, x, 0.13) for x in rth2], 'darkcyan', lw=2)
plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.01, x, 0.12) for x in rth2], 'crimson', lw=2)

plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.08, x, 0.10) for x in rth3], 'darkorchid', lw=2)
plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.05, x, 0.09) for x in rth3], 'darkcyan', lw=2)
plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.01, x, 0.08) for x in rth3], 'crimson', lw=2)

plt.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.4)
plt.xlim(1.5, 15)
plt.ylim(-0.15, 0.15)
plt.xlabel(r'$R_{\rm th}$', fontsize=18)
plt.ylabel(r'$\Delta \sigma_{L/W}^{\rm ex}/\sigma_{L_{3}/W}^{\rm ex}}$', fontsize=15)
ax1.grid(False, which='both')
plt.axis('tight')
plt.text(0.596, 0.04, r'$\varepsilon_s = 0.2, R_0 = 0.1, \varphi_e = \pi/6$', fontsize=14, bbox=props, transform=plt.gca().transAxes, ha='left', va='bottom')
ax1.legend(ncol=4, loc="upper center", handlelength=2.0, handletextpad=0.2, columnspacing=0.2)
 


#-------------------------------------------------------------------------------------------------------------------------------
# sig x epsilon (R0 fixo, Rth variável) ex
#-------------------------------------------------------------------------------------------------------------------------------

rthx = np.linspace(1.5, 15, 210)
rth1 = np.linspace(1.5, 7, 70)
rth2 = np.linspace(7, 11, 70)
rth3 = np.linspace(11, 15, 70)
ax1 = fig.add_subplot(gs[2, 0])

plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.08, x, 0.25) for x in rth1], 'darkorchid', lw=2)
plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.05, x, 0.25) for x in rth1], 'darkcyan', lw=2)
plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.01, x, 0.25) for x in rth1], 'crimson', lw=2)

plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.08, x, 0.13) for x in rth2], 'darkorchid', lw=2)
plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.05, x, 0.13) for x in rth2], 'darkcyan', lw=2)
plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.01, x, 0.12) for x in rth2], 'crimson', lw=2)

plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.08, x, 0.10) for x in rth3], 'darkorchid', lw=2)
plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.05, x, 0.09) for x in rth3], 'darkcyan', lw=2)
plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.01, x, 0.08) for x in rth3], 'crimson', lw=2)

plt.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.4)
plt.xlim(1.5, 15)
plt.ylim(-0.15, 0.15)
plt.xlabel(r'$R_{\rm th}$', fontsize=18)
plt.ylabel(r'$\Delta \sigma_{L/W}^{\rm ex}/\sigma_{L_{3}/W}^{\rm ex}}$', fontsize=15)
ax1.grid(False, which='both')
plt.axis('tight')
plt.text(0.596, 0.04, r'$\varepsilon_s = 0.2, R_0 = 0.1, \varphi_e = \pi/6$', fontsize=14, bbox=props, transform=plt.gca().transAxes, ha='left', va='bottom')
ax1.legend(ncol=4, loc="upper center", handlelength=2.0, handletextpad=0.2, columnspacing=0.2)



#-------------------------------------------------------------------------------------------------------------------------------
# sig x epsilon (R0 variável, Rth fixo) ex
#------------------------------------------------------------------------------------------------------------------------------- 

rthx = np.linspace(1.5, 15, 210)
rth1 = np.linspace(1.5, 7, 70)
rth2 = np.linspace(7, 11, 70)
rth3 = np.linspace(11, 15, 70)
ax1 = fig.add_subplot(gs[2, 1])

plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.08, x, 0.25) for x in rth1], 'darkorchid', lw=2)
plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.05, x, 0.25) for x in rth1], 'darkcyan', lw=2)
plt.plot(rth1, [delL2_cross_pert_ex(0.10, 0.01, x, 0.25) for x in rth1], 'crimson', lw=2)

plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.08, x, 0.13) for x in rth2], 'darkorchid', lw=2)
plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.05, x, 0.13) for x in rth2], 'darkcyan', lw=2)
plt.plot(rth2, [delL2_cross_pert_ex(0.10, 0.01, x, 0.12) for x in rth2], 'crimson', lw=2)

plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.08, x, 0.10) for x in rth3], 'darkorchid', lw=2)
plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.05, x, 0.09) for x in rth3], 'darkcyan', lw=2)
plt.plot(rth3, [delL2_cross_pert_ex(0.10, 0.01, x, 0.08) for x in rth3], 'crimson', lw=2)

plt.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.4)
plt.xlim(1.5, 15)
plt.ylim(-0.15, 0.15)
plt.xlabel(r'$R_{\rm th}$', fontsize=18)
plt.ylabel(r'$\Delta \sigma_{L/W}^{\rm ex}/\sigma_{L_{3}/W}^{\rm ex}}$', fontsize=15)
ax1.grid(False, which='both')
plt.axis('tight')
plt.text(0.596, 0.04, r'$\varepsilon_s = 0.2, R_0 = 0.1, \varphi_e = \pi/6$', fontsize=14, bbox=props, transform=plt.gca().transAxes, ha='left', va='bottom')
ax1.legend(ncol=4, loc="upper center", handlelength=2.0, handletextpad=0.2, columnspacing=0.2)
 

plt.savefig("Figure3_paper2.pdf")
plt.show()
