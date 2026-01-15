import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


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

pH_range=np.arange(7.0,8.6,0.1)

for i in pH_range:
    df = pd.read_csv(f'/users/rp819/scratch/COAGEM/O3{O3_fix}_I{I_fix}_ws{ws_fix}_T{T_fix}_pH{i}.csv')

    t_iod.append(((126.9/143.89)*mwHOI*df['FluxA_HOI'].values[-1]+mwI2*df['FluxA_I2'].values[-1]+\
                  (126.9/162.35)*mwICl*df['FluxA_ICl'].values[-1]+\
                  (126.9/206.90)*mwIBr*df['FluxA_IBr'].values[-1])*10000)
    HOI.append(((126.9/143.89)*mwHOI*df['FluxA_HOI'].values[-1])*10000)
    I2.append(mwI2*df['FluxA_I2'].values[-1]*10000)
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
    ax[0].legend(loc = 'upper left')
    ax[0].legend()

    #fit equation to the line of probably just I2


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


