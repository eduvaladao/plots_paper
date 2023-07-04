import arcs
import angles
import lengths
import widths
import LWcross
import matplotlib


import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed # Porque não sou obrigado ficar esperando...

import warnings
warnings.filterwarnings('ignore')

s = np.linspace(0.1, 1.2, 500)
r0 = np.linspace(1e-5, 0.275, 500)
phi_e = np.linspace(0, np.pi, 500)
e_s = np.linspace(1e-5, 0.9, 500)

rth = np.linspace(4, 100, 30)
cal_1 = np.array(Parallel(n_jobs=30)(delayed(LWcross.sigma_ex_medio)(0.2, 0.08, i) for i in rth))
#cal_3 = np.array(Parallel(n_jobs=30)(delayed(KeetonCross.sig_med)(0.2, i) for i in rth))/2
cal_4 = np.array(Parallel(n_jobs=30)(delayed(LWcross.sigma_ex_medio)(0.2, 0.05, i) for i in rth))
#cal_6 = np.array(Parallel(n_jobs=30)(delayed(KeetonCross.sig_med)(0.2, i) for i in rth))/2
cal_7 = np.array(Parallel(n_jobs=30)(delayed(LWcross.sigma_ex_medio)(0.2, 0.01, i) for i in rth))
#cal_9 = np.array(Parallel(n_jobs=30)(delayed(KeetonCross.sig_med)(0.2, i) for i in rth))/2 

plt.title('Fig. 5.20')
ax1=plt.subplot(111)
plt.loglog(rth, cal_1, ls='--', lw=2, label=r'$R_0 = 0.08, \varepsilon_s=0.2 $' )
plt.loglog(rth, cal_4, ls='--', lw=2, label=r'$R_0 = 0.05 ,\varepsilon_s = 0.2$' )
plt.loglog(rth, cal_7, ls='--', lw=2, label=r'$R_0 = 0.01, \varepsilon_s=0.2$')
#plt.loglog(rth, cal_3, ls='--', lw=2 )
#plt.loglog(rth, cal_6, ls='--', lw=2)
#plt.loglog(rth, cal_9, ls='--', lw=2)
plt.axis([4,100, 5*10**-3, 1])
plt.grid(False, which='both')
plt.axhline(y=0, color='0.6',ls='--',lw=1)
plt.axvline(x=0, color='0.6',ls='--',lw=1)
plt.xlabel(r'$R_{\rm th}$')
plt.ylabel(r'$ \sigma_{L/W} ^{\rm ex}\left(R_0, \varepsilon_s, R_{\rm th}\right)$')
plt.legend(loc="upper right", handlelength=2, ncol=2,  columnspacing=0) 
ax1.set_xticks([5, 10, 20, 30, 50, 70, 100])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.show()
