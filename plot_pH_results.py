import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import solve_sml_model as solve
import transfer_calculations as transfer
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
extended = False
if extended:
    pH_range=[5.0,5.25,5.75,6.0,6.25,6.5,6.75,7.0,7.1,7.2,7.3,7.4,\
              7.5,7.6,7.7,7.8,7.9,8.0,8.1,8.2,8.3,8.4,8.5,8.75,9.0,9.25,9.5,9.75,10.0]

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
plt.savefig('/users/rp819/scratch/plots/pH_work/emission_sens_env.png',dpi=150)
plt.close('all')

def bulk_calc(df,u,T,S,spec='HOI',dt_max=0.0001):
    bulk = df[f'{spec}_bulk'].values[-1]
    sml = df[f'{spec}'].values[-1]
    surf_renewal = ((2*(3.42E-3*u+2.7E-3))/(np.pi*(0.235*u+4.935)))*\
                           (bulk - sml)*dt_max

    species_database = solve.define_species_database()
    species_database.loc[species_database['name'] == 'I-', ['conc']] = df['I-'].values[-1]
    h = solve.SML_h(T,species_database,'brown')*10
    spec_comp = solve.get_comp_dict(species_database,spec)
    kw = transfer.kw(spec_comp,T-273.15,u,S)*10
    flux = 0.9*kw*(bulk - sml)
    return flux*dt_max/h + surf_renewal

df = pd.read_csv(f'/users/rp819/scratch/COAGEM/O3{O3_fix}_I{I_fix}_ws{ws_fix}_T{T_fix}_pH7.0.csv')
dfc = pd.read_csv(f'/users/rp819/scratch/COAGEM/ConcAfterChem_O3{O3_fix}_I{I_fix}_ws{ws_fix}_T{T_fix}_pH7.0.csv')
df2 = pd.read_csv(f'/users/rp819/scratch/COAGEM/O3{O3_fix}_I{I_fix}_ws{ws_fix}_T{T_fix}_pH8.0.csv')
df2c = pd.read_csv(f'/users/rp819/scratch/COAGEM/ConcAfterChem_O3{O3_fix}_I{I_fix}_ws{ws_fix}_T{T_fix}_pH8.0.csv')

labels = ['HOI air flux','HOI bulk flux',\
          'I$_2$ air flux',\
          'I$_2$ bulk flux']#,\
          #'I2OH- + H+ => I2',\
          #'I2 + OH- => HOI + I-',\
          #'IO- + H+ => HOI',\
          #'HOI + H+ => H2OI+',\
          #'IO3- + I- + 2 H+ => HOI + HIO2',\
          #'I- + H2OI+ => I2',\
          #'HOI + Br- + H+ => IBr',\
          #'HOI + Cl- + H+ => ICl',\
          #'I- + IBr => I2 + Br-',\
          #'I2Cl- => I2 + Cl-']
ph7 = []
ph8 = []

dt_max=0.0001
species_database = solve.define_species_database()
spec_comp = solve.get_comp_dict(species_database,'O3')
species_database = solve.define_species_database()
species_database.loc[species_database['name'] == 'I-', ['conc']] = df['I-'].values[-1]
h = solve.SML_h(T_fix,species_database,'brown')*10
ref_flux7 = (df['O3_depv'].values[-1]/10)*(dfc['O3g'].values[-1] -\
              transfer.KH(spec_comp,T_fix-273.15,35)*dfc['O3'].values[-1])*dt_max/h

species_database.loc[species_database['name'] == 'I-', ['conc']] = df2['I-'].values[-1]
h = solve.SML_h(T_fix,species_database,'brown')*10
ref_flux8 = (df2['O3_depv'].values[-1]/10)*(df2c['O3g'].values[-1] -\
              transfer.KH(spec_comp,T_fix-273.15,35)*df2c['O3'].values[-1])*dt_max/h

ph7.append((df['FluxA_HOI'].values[-1]*100*dt_max/h)/ref_flux7)
ph8.append((df2['FluxA_HOI'].values[-1]*100*dt_max/h)/ref_flux8)

ph7.append(bulk_calc(dfc,ws_fix,T_fix,35,spec='HOI',dt_max=0.0001)/ref_flux7*-1)
ph8.append(bulk_calc(df2c,ws_fix,T_fix,35,spec='HOI',dt_max=0.0001)/ref_flux8*-1)

ph7.append((df['FluxA_I2'].values[-1]*100*dt_max/h)/ref_flux7)
ph8.append((df2['FluxA_I2'].values[-1]*100*dt_max/h)/ref_flux8)

ph7.append(bulk_calc(dfc,ws_fix,T_fix,35,spec='I2',dt_max=0.0001)/ref_flux7*-1)
ph8.append(bulk_calc(df2c,ws_fix,T_fix,35,spec='I2',dt_max=0.0001)/ref_flux8*-1)

#calculate important species interchange
columns = ['spec 1','spec 2','forward','reverse']
threshold = 1e-5

chem8 = pd.read_csv('./raw_chem_flux_ph8.txt',skiprows=[1],names=columns,delim_whitespace=True)
chem8['net8'] = chem8['forward'] + chem8['reverse']
norm = np.amax(np.abs(chem8['net8']))
chem8['net8'] = chem8['net8']/norm
for index,row in chem8.iterrows():
    if row['net8'] > 0:
        chem8.loc[index,'plot_name'] = f"{row['spec 1']} -> {row['spec 2']}"
    else:
        chem8.loc[index,'plot_name'] = f"{row['spec 2']} -> {row['spec 1']}"
        chem8.loc[index,'net8'] = np.abs(row['net8'])
chem8 = chem8.set_index('plot_name')
chem8 = chem8[chem8.net8 > threshold]
chem8 = chem8.drop(['forward','reverse','spec 1','spec 2'],axis=1)


chem7 = pd.read_csv('./raw_chem_flux_ph7.txt',skiprows=[1],names=columns,delim_whitespace=True)
chem7['net7'] = chem7['forward'] + chem7['reverse']
chem7['net7'] = chem7['net7']/norm
for index,row in chem7.iterrows():
    if row['net7'] > 0:
        chem7.loc[index,'plot_name'] = f"{row['spec 1']} -> {row['spec 2']}"
    else:
        chem7.loc[index,'plot_name'] = f"{row['spec 2']} -> {row['spec 1']}"
        chem7.loc[index,'net7'] = np.abs(row['net7'])
chem7 = chem7.set_index('plot_name')
chem7 = chem7[chem7.net7 > threshold]
chem7 = chem7.drop(['forward','reverse','spec 1','spec 2'],axis=1)

merged_fluxes = pd.concat([chem7,chem8],axis=1)
merged_fluxes = merged_fluxes.sort_values(by=['net7'],ascending=False)


def add_labels(x, y, ax, color):
    for i in range(len(x)):
        ax.text(x[i], y[i]+0.1*y[i], "{:.4f}".format(y[i]), color=color, ha='center')

#bar plot of the percentage change in fluxes through pathways
fig,ax=plt.subplots(1,1,figsize=(10,5))
xvals = np.arange(1,len(ph7)+1)
ax.bar(xvals-0.125,ph7,0.25,color='orange',label='pH7')
add_labels(xvals-0.125, ph7, ax, color = 'orange')
ax.bar(xvals+0.125,ph8,0.25,color='blue',label='pH8')
add_labels(xvals+0.125, ph8, ax, color = 'blue')
ax.set_yscale('log')
ax.set_ylabel('Mixing fluxes normalized to O$_3$ flux in')
ax.legend()
ax.set_xticks(xvals,labels)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('/users/rp819/scratch/plots/pH_work/physical_fluxes_comp.png',dpi=150)
plt.close('all')

fig,ax=plt.subplots(1,1,figsize=(10,5))
xvals = np.arange(1,len(merged_fluxes.index)+1)
ax.bar(xvals-0.125,merged_fluxes['net7'],0.25,color='orange',label='pH7')
ax.bar(xvals+0.125,merged_fluxes['net8'],0.25,color='blue',label='pH8')
ax.set_yscale('log')
ax.set_ylabel('Chemistry fluxes normalised to IO$^-$ -> HOI @ pH 8')
ax.legend()
ax.set_xticks(xvals,merged_fluxes.index)
ax.set_ylim(1e-5,1)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('/users/rp819/scratch/plots/pH_work/chemical_fluxes_comp.png',dpi=150)
plt.close('all')

