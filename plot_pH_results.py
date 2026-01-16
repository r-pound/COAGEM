import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def fit_func(x,m,pH,b):
    return m*np.exp(-pH*x) + b

ws_fix = 5.0
T_fix = 297.0
O3_fix = 30.0
I_fix = 100.0

mwHOI = 1.4391E-1
mwI2 = 2.53808E-1
mwICl = 1.6235E-1
mwIBr = 2.06904E-1

fig,ax=plt.subplots(1,3,figsize=(17,5))
fig.subplots_adjust(wspace=0.35)
t_iod = []
HOI = []
I2 = []
ICl = []
IBr = []
r_HOI_I2 = []
e_iodide = []

pH_range=[7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.0,8.1,8.2,8.3,8.4,8.5]

for i in pH_range:
    df = pd.read_csv(f'/users/rp819/scratch/COAGEM/O3{O3_fix}_I{I_fix}_ws{ws_fix}_T{T_fix}_pH{i}.csv')

    t_iod.append(((126.9/143.89)*mwHOI*df['FluxA_HOI'].values[-1]+mwI2*df['FluxA_I2'].values[-1]+\
                  (126.9/162.35)*mwICl*df['Flux_ICl'].values[-1]+\
                  (126.9/206.90)*mwIBr*df['Flux_IBr'].values[-1])*10000)
    HOI.append(((126.9/143.89)*mwHOI*df['FluxA_HOI'].values[-1])*10000)
    I2.append(mwI2*df['FluxA_I2'].values[-1]*10000)
    ICl.append((126.9/162.35)*mwICl*df['Flux_ICl'].values[-1]*10000)
    IBr.append((126.9/206.90)*mwIBr*df['Flux_IBr'].values[-1]*10000)
    r_HOI_I2.append(df['FluxA_I2'].values[-1]/df['FluxA_HOI'].values[-1])
    e_iodide.append(df['I-'].values[-1]/df['I-_bulk'].values[-1])

ax[0].plot(pH_range, t_iod, color='black', label = 'Total')
ax[0].plot(pH_range, HOI, color='orange', label = 'HOI')
ax[0].plot(pH_range, I2, color='blue', label = 'I2')
ax[0].plot(pH_range, ICl, color='green', label = 'ICl')
ax[0].plot(pH_range, IBr, color='purple', label = 'IBr')

ax[0].set_xlabel('pH')
ax[0].set_ylabel('Total iodine emission [kgI m$^2$ s$^{-1}$]')
ax[0].grid()
ax[0].title.set_text('(a)')

#fit equation to the line of just I2 - I think with this something more like a normalised fit 
#this would effectively give a scale factor for us to work with to apply to the emission equations
fit_data = np.asarray(I2)/I2[9]
p0 = (1/8,-0.1,1)
params,cv = curve_fit(fit_func, pH_range, fit_data, p0)
m,pHg,b = params
ax[0].plot(np.arange(7.1,8.6,0.05),fit_func(np.arange(7.1,8.6,0.05),m,pHg,b)*I2[9],'--',color='pink',zorder=4,\
          label='fit')
ax[0].legend(loc = 'upper left')
ax[0].legend()
m = float('%.3g' % m)
pHg = float('%.3g' % pHg)
b = float('%.3g' % b)
ax[0].annotate(f'y={m}*exp(-{pHg}*pH)+{b}',xy=(7.31,7.05e-13))

ax[1].plot(pH_range, r_HOI_I2, color='black')
ax[1].set_xlabel('pH')
ax[1].set_ylabel('I2/HOI emission ratio [unitless]')
ax[1].grid()
ax[1].title.set_text('(b)')

ax[2].plot(pH_range, e_iodide, color='black')
ax[2].set_xlabel('pH')
ax[2].set_ylabel('SML I$^-$ enrichment [unitless]')
ax[2].grid()
ax[2].title.set_text('(c)')

plt.tight_layout()
plt.savefig('/users/rp819/scratch/plots/pH_work/emission_sens.png',dpi=150)
plt.close('all')


