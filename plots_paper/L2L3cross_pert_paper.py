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
    
"""# Threshold for the validity of the lenghts based on the choice of angles, i.e. source tangent to the y2 axis
def sthy(r0, phi0, epsilon): 
    return r0*np.sqrt((1 + epsilon*np.cos(2*phi0))/(1 - epsilon**2))
def epsilonthy(r0, s, phi0):
    return (np.sqrt(2)*np.sqrt(8*(s**2)*(s**2 - r0**2) + (r0**4)*(1 + np.cos(4*phi0))) - 2*(r0**2)*np.cos(2*phi0))/(4*(s**2))
def r0thy(s, phi0, epsilon):
    return s/np.sqrt((1 + epsilon*np.cos(2*phi0))/(1 - epsilon**2))"""

props = dict(boxstyle='square', facecolor='white')  

#-------------------------------------------------------------------------------------------------------------------------------
# sig x Rth (R0 variável, epsilon fixo) ex
#-------------------------------------------------------------------------------------------------------------------------------

rthx = np.linspace(1.5, 15, 210)
rth1 = np.linspace(1.5, 7, 70)
rth2 = np.linspace(7, 11, 70)
rth3 = np.linspace(11, 15, 70)
ax1 = fig.add_subplot(gs[0, 0])
plt.plot(rth1, list(map(lambda x:delL2_cross_pert_ex(0.10, 0.08, x, 0.25), rth1)), 'darkorchid', lw=2)
plt.plot(rth1, list(map(lambda x:delL2_cross_pert_ex(0.10, 0.05, x, 0.25), rth1)), 'darkcyan', lw=2)
plt.plot(rth1, list(map(lambda x:delL2_cross_pert_ex(0.10, 0.01, x, 0.25), rth1)), 'crimson', lw=2)
plt.plot(rth2, list(map(lambda x:delL2_cross_pert_ex(0.10, 0.08, x, 0.13), rth2)), 'darkorchid', lw=2)
plt.plot(rth2, list(map(lambda x:delL2_cross_pert_ex(0.10, 0.05, x, 0.13), rth2)), 'darkcyan', lw=2)
plt.plot(rth2, list(map(lambda x:delL2_cross_pert_ex(0.10, 0.01, x, 0.12), rth2)), 'crimson', lw=2)
plt.plot(rth3, list(map(lambda x:delL2_cross_pert_ex(0.10, 0.08, x, 0.10), rth3)), 'darkorchid', lw=2)
plt.plot(rth3, list(map(lambda x:delL2_cross_pert_ex(0.10, 0.05, x, 0.09), rth3)), 'darkcyan', lw=2)
plt.plot(rth3, list(map(lambda x:delL2_cross_pert_ex(0.10, 0.01, x, 0.08), rth3)), 'crimson', lw=2)
plt.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.4)
plt.xlim(1.5, 15)
plt.ylim(-0.15, 0.15)
plt.xlabel(r'$R_{\rm th}$', fontsize=18)
plt.ylabel(r'$\Delta \sigma_{L/W}^{\rm ex}/\sigma_{L_{3}/W}^{\rm ex}}$', fontsize = 15)
ax1.grid(False, which='both')
plt.axis('tight')
plt.text(0.596, 0.04, r'$\varepsilon_s = 0.2, R_0 = 0.1, \varphi_e = \pi/6$', fontsize=14, bbox=props, transform=plt.gca().transAxes, ha='left', va='bottom')
ax1.legend(ncol=4, loc = "upper center", handlelength = 2.0, handletextpad = 0.2, columnspacing = 0.2)   

#-------------------------------------------------------------------------------------------------------------------------------
# sig x Rth (R0 fixo, epsilon variável) ex
#-------------------------------------------------------------------------------------------------------------------------------

ax2 = fig.add_subplot(gs[0, 1])
plt.plot(rth1, list(map(lambda x:delL2_cross_pert_ex(0.20, 0.05, x, 0.25), rth1)), 'darkorchid', lw=2)
plt.plot(rth1, list(map(lambda x:delL2_cross_pert_ex(0.15, 0.05, x, 0.25), rth1)), 'darkcyan', lw=2)
plt.plot(rth1, list(map(lambda x:delL2_cross_pert_ex(0.10, 0.05, x, 0.25), rth1)), 'crimson', lw=2)
plt.plot(rth2, list(map(lambda x:delL2_cross_pert_ex(0.20, 0.05, x, 0.13), rth2)), 'darkorchid', lw=2)
plt.plot(rth2, list(map(lambda x:delL2_cross_pert_ex(0.15, 0.05, x, 0.13), rth2)), 'darkcyan', lw=2)
plt.plot(rth2, list(map(lambda x:delL2_cross_pert_ex(0.10, 0.05, x, 0.12), rth2)), 'crimson', lw=2)
plt.plot(rth3, list(map(lambda x:delL2_cross_pert_ex(0.20, 0.05, x, 0.10), rth3)), 'darkorchid', lw=2)
plt.plot(rth3, list(map(lambda x:delL2_cross_pert_ex(0.15, 0.05, x, 0.09), rth3)), 'darkcyan', lw=2)
plt.plot(rth3, list(map(lambda x:delL2_cross_pert_ex(0.10, 0.05, x, 0.08), rth3)), 'crimson', lw=2)
plt.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.4)
ax2.grid(False, which='both')
plt.axis('tight')
plt.xlim(1.5, 15)
plt.ylim(-0.15, 0.15)
plt.xlabel(r'$R_{\rm th}$', fontsize=18)
plt.ylabel(r'$\Delta \sigma_{L/W}^{\rm ex}/\sigma_{L_{3}/W}^{\rm ex}}$', fontsize = 15)
ax2.text(0.026, 0.04, r'$R_0 = 0.1, S = 0.17, \phi_e = \pi/4$', fontsize=14, bbox=props, transform=plt.gca().transAxes, ha='left', va='bottom')
ax2.legend(ncol=4, loc = "upper center", handlelength = 2.0, handletextpad = 0.2, columnspacing = 0.2)   


#-------------------------------------------------------------------------------------------------------------------------------
# sig x R0 (epsilon variável, Rth fixo) ex
#------------------------------------------------------------------------------------------------------------------------------- 

r0x = np.linspace(0.001, 0.220, 210)
r01 = np.linspace(0.001, 0.060, 70)
r02 = np.linspace(0.060, 0.130, 70)
r03 = np.linspace(0.130, 0.220, 70)
ax3 = fig.add_subplot(gs[1, 0])
plt.plot(r01, list(map(lambda x:delL2_cross_pert_ex(0.20, x, 6, 0.24), r01)), 'darkorchid', lw=2)
plt.plot(r01, list(map(lambda x:delL2_cross_pert_ex(0.15, x, 6, 0.24), r01)), 'darkcyan', lw=2)
plt.plot(r01, list(map(lambda x:delL2_cross_pert_ex(0.10, x, 6, 0.24), r01)), 'crimson', lw=2)
plt.plot(r02, list(map(lambda x:delL2_cross_pert_ex(0.20, x, 6, 0.25), r02)), 'darkorchid', lw=2)
plt.plot(r02, list(map(lambda x:delL2_cross_pert_ex(0.15, x, 6, 0.25), r02)), 'darkcyan', lw=2)
plt.plot(r02, list(map(lambda x:delL2_cross_pert_ex(0.10, x, 6, 0.25), r02)), 'crimson', lw=2)
plt.plot(r03, list(map(lambda x:delL2_cross_pert_ex(0.20, x, 6, 0.27), r03)), 'darkorchid', lw=2)
plt.plot(r03, list(map(lambda x:delL2_cross_pert_ex(0.15, x, 6, 0.27), r03)), 'darkcyan', lw=2)
plt.plot(r03, list(map(lambda x:delL2_cross_pert_ex(0.10, x, 6, 0.27), r03)), 'crimson', lw=2)
plt.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.4)
ax3.grid(False, which='both')
plt.axis('tight')
plt.xlim(0.001, 0.22)
plt.ylim(-0.15, 0.15)  
plt.xlabel(r'$R_0$', fontsize=18)
plt.ylabel(r'$\Delta \sigma_{L/W}^{\rm ex}/\sigma_{L_{3}/W}^{\rm ex}}$', fontsize = 15)
plt.text(0.026, 0.04, r'$\varepsilon_s = 0.15, S = 0.13, \phi_e = \pi/4$', fontsize=14, bbox=props, transform=plt.gca().transAxes, ha='left', va='bottom')
ax3.legend(ncol=4, loc = "upper center", handlelength = 2.0, handletextpad = 0.2, columnspacing = 0.2)   


#-------------------------------------------------------------------------------------------------------------------------------
# sig x R0 (epsilon fixo, Rth variável) ex
#-------------------------------------------------------------------------------------------------------------------------------

r04 = np.linspace(0.130, 0.168, 70)
ax4 = fig.add_subplot(gs[1, 1])
plt.plot(r01, list(map(lambda x:delL2_cross_pert_ex(0.10, x, 7, 0.17), r01)), 'darkorchid', lw=2)
plt.plot(r01, list(map(lambda x:delL2_cross_pert_ex(0.10, x, 5, 0.25), r01)), 'darkcyan', lw=2)
plt.plot(r01, list(map(lambda x:delL2_cross_pert_ex(0.10, x, 3, 0.50), r01)), 'crimson', lw=2)
plt.plot(r02, list(map(lambda x:delL2_cross_pert_ex(0.10, x, 7, 0.18), r02)), 'darkorchid', lw=2)
plt.plot(r02, list(map(lambda x:delL2_cross_pert_ex(0.10, x, 5, 0.26), r02)), 'darkcyan', lw=2)
plt.plot(r02, list(map(lambda x:delL2_cross_pert_ex(0.10, x, 3, 0.50), r02)), 'crimson', lw=2)
plt.plot(r04, list(map(lambda x:delL2_cross_pert_ex(0.10, x, 7, 0.17), r04)), 'darkorchid', lw=2)
plt.plot(r03, list(map(lambda x:delL2_cross_pert_ex(0.10, x, 5, 0.28), r03)), 'darkcyan', lw=2)
plt.plot(r03, list(map(lambda x:delL2_cross_pert_ex(0.10, x, 3, 0.51), r03)), 'crimson', lw=2)
plt.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.4)
ax4.grid(False, which='both')
plt.axis('tight')
plt.xlim(0.001, 0.13)
plt.ylim(-0.15, 0.15)  
plt.xlabel(r'$R_0$', fontsize=18)
plt.ylabel(r'$\Delta \sigma_{L/W}^{\rm ex}/\sigma_{L_{3}/W}^{\rm ex}}$', fontsize = 15)
plt.text(0.026, 0.04, r'$\varepsilon_s = 0.2, R_0 = 0.1, \varphi_e = \pi/6$', fontsize=14, bbox=props, transform=plt.gca().transAxes, ha='left', va='bottom')
ax4.legend(ncol=4, loc = "upper center", handlelength = 2.0, handletextpad = 0.2, columnspacing = 0.2)  


#-------------------------------------------------------------------------------------------------------------------------------
# sig x epsilon (R0 fixo, Rth variável) ex
#-------------------------------------------------------------------------------------------------------------------------------

epsilonx = np.linspace(0.000, 0.800, 210)
epsilon1 = np.linspace(0.000, 0.300, 70)
epsilon2 = np.linspace(0.300, 0.500, 70)
epsilon3 = np.linspace(0.500, 0.800, 70)
ax5 = fig.add_subplot(gs[2, 0])
plt.plot(epsilon1, list(map(lambda x:delL2_cross_pert_ex(x, 0.01, 15, 0.05), epsilon1)), 'darkorchid', lw=2)
plt.plot(epsilon1, list(map(lambda x:delL2_cross_pert_ex(x, 0.01, 8, 0.07), epsilon1)), 'darkcyan', lw=2)
plt.plot(epsilon1, list(map(lambda x:delL2_cross_pert_ex(x, 0.01, 6, 0.11), epsilon1)), 'crimson', lw=2)
plt.plot(epsilon2, list(map(lambda x:delL2_cross_pert_ex(x, 0.01, 15, 0.05), epsilon2)), 'darkorchid', lw=2)
plt.plot(epsilon2, list(map(lambda x:delL2_cross_pert_ex(x, 0.01, 8, 0.07), epsilon2)), 'darkcyan', lw=2)
plt.plot(epsilon2, list(map(lambda x:delL2_cross_pert_ex(x, 0.01, 6, 0.11), epsilon2)), 'crimson', lw=2)
plt.plot(epsilon3, list(map(lambda x:delL2_cross_pert_ex(x, 0.01, 15, 0.05), epsilon3)), 'darkorchid', lw=2)
plt.plot(epsilon3, list(map(lambda x:delL2_cross_pert_ex(x, 0.01, 8, 0.07), epsilon3)), 'darkcyan', lw=2)
plt.plot(epsilon3, list(map(lambda x:delL2_cross_pert_ex(x, 0.01, 6, 0.11), epsilon3)), 'crimson', lw=2)
plt.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.4)
ax5.grid(False, which='both')
plt.axis('tight')
plt.xlim(0, 0.8)
plt.ylim(-0.15, 0.15)  
plt.xlabel(r'$\varepsilon_s$', fontsize=18)
plt.ylabel(r'$\Delta \sigma_{L/W}^{\rm ex}/\sigma_{L_{3}/W}^{\rm ex}}$', fontsize = 15)
ax5.text(0.026, 0.04, r'$R_0 = 0.1, S = 0.17, \phi_e = \pi/4$', fontsize=14, bbox=props, transform=plt.gca().transAxes, ha='left', va='bottom')
ax5.legend(ncol=4, loc = "upper center", handlelength = 2.0, handletextpad = 0.2, columnspacing = 0.2)   


#-------------------------------------------------------------------------------------------------------------------------------
# sig x epsilon (R0 variável, Rth fixo) ex
#------------------------------------------------------------------------------------------------------------------------------- 

ax6 = fig.add_subplot(gs[2, 1])
plt.plot(epsilon1, list(map(lambda x:delL2_cross_pert_ex(x, 0.08, 10, 0.12), epsilon1)), 'darkorchid', lw=2)
plt.plot(epsilon1, list(map(lambda x:delL2_cross_pert_ex(x, 0.05, 10, 0.11), epsilon1)), 'darkcyan', lw=2)
plt.plot(epsilon1, list(map(lambda x:delL2_cross_pert_ex(x, 0.01, 10, 0.11), epsilon1)), 'crimson', lw=2)
plt.plot(epsilon2, list(map(lambda x:delL2_cross_pert_ex(x, 0.08, 10, 0.12), epsilon2)), 'darkorchid', lw=2)
plt.plot(epsilon2, list(map(lambda x:delL2_cross_pert_ex(x, 0.05, 10, 0.11), epsilon2)), 'darkcyan', lw=2)
plt.plot(epsilon2, list(map(lambda x:delL2_cross_pert_ex(x, 0.01, 10, 0.11), epsilon2)), 'crimson', lw=2)
plt.plot(epsilon3, list(map(lambda x:delL2_cross_pert_ex(x, 0.08, 10, 0.12), epsilon3)), 'darkorchid', lw=2)
plt.plot(epsilon3, list(map(lambda x:delL2_cross_pert_ex(x, 0.05, 10, 0.11), epsilon3)), 'darkcyan', lw=2)
plt.plot(epsilon3, list(map(lambda x:delL2_cross_pert_ex(x, 0.01, 10, 0.11), epsilon3)), 'crimson', lw=2)
plt.axhline(y=0, color='grey', ls='--', lw=2, alpha=0.4)
ax6.grid(False, which='both')
plt.axis('tight')
plt.xlim(0, 0.8)
plt.ylim(-0.15, 0.15)  
plt.xlabel(r'$\varepsilon_s$', fontsize=18)
plt.ylabel(r'$\Delta \sigma_{L/W}^{\rm ex}/\sigma_{L_{3}/W}^{\rm ex}}$', fontsize = 15)
plt.text(0.026, 0.04, r'$\varepsilon_s = 0.15, S = 0.13, \phi_e = \pi/4$', fontsize=14, bbox=props, transform=plt.gca().transAxes, ha='left', va='bottom')
ax6.legend(ncol=4, loc = "upper center", handlelength = 2.0, handletextpad = 0.2, columnspacing = 0.2)   

plt.savefig("Figure3_paper2.pdf")
plt.show()
