import numpy as np
import transfer_calculations as transfer
import pandas as pd
import cantera as ct
import scipy.special as sp
import sys
import os
#####################################################################################################################
def define_species_database():

    """
    define the species that exist in the model and their physcial parameters and starting concentrations

    Double options exist for O3, I2 and HOI to keep track of aqueos and gasseous species for calculating air-sea
    exchange - the gasseous species will not be subject to chemistry but just to keep track of atmospheric
    concentrations \ the flux to atmosphere

    Additionally there is the I- species has a duplicate to account for the "bulk" ocean I- concentration which will
    remain constant and replenish the ocean surface concentration

    HOI gas based on 6.81E-13 average surface concs and I2 on 5.67E-15

    HOI tVar 5862
    """

    names = ["name",     'conc',  "mw",  "KH",  "tVar","C","H","O","N","Br","Cl","F","I","S","rings","db","tb"]
    data = [["O3g"      ,1.25E-9 ,48.0  ,0.01  ,2800  ,0,  0,  3,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["HOIg"     ,2.83E-14,143.91,415   ,0     ,0,  1,  1,  0,  0,   0,   0,  1,  0,  0,0,0],\
            ["I2g"      ,2.35E-16,259.81,2.84  ,4300  ,0,  0,  0,  0,  0,   0,   0,  2,  0,  0,0,0],\
            ["I-_bulk"  ,1.0E-7  ,126.9 ,0.006 ,2300  ,0,  0,  0,  0,  0,   0,   0,  1,  0,  0,0,0],\
            ["HOI_bulk" ,0.0     ,143.91,415   ,0     ,0,  1,  1,  0,  0,   0,   0,  1,  0,  0,0,0],\
            ["I2_bulk"  ,0.0     ,259.81,2.84  ,4300  ,0,  0,  0,  0,  0,   0,   0,  2,  0,  0,0,0],\
            ["I-"       ,1.0E-7  ,126.9 ,0.006 ,2300  ,0,  0,  0,  0,  0,   0,   0,  1,  0,  0,0,0],\
            ["O3"       ,0.0     ,48.0  ,0.01  ,2800  ,0,  0,  3,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["HOI"      ,0.0     ,143.91,415   ,0     ,0,  1,  1,  0,  0,   0,   0,  1,  0,  0,0,0],\
            ["H+"       ,1.0e-8  ,1.01  ,2.6E-4,0     ,0,  1,  0,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["O2"       ,0.0     ,32    ,0     ,0     ,0,  0,  0,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["H2O"      ,55.5    ,18.02 ,0     ,0     ,0,  0,  0,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["OH-"      ,1.0E-6  ,17.01 ,0.29  ,4300  ,0,  1,  1,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["Cl-"      ,0.55    ,35.45 ,2.3E-2,0     ,0,  0,  0,  0,  0,   1,   0,  0,  0,  0,0,0],\
            ["Br-"      ,8.6E-4  ,79.90 ,1.2E-2,0     ,0,  0,  0,  0,  1,   0,   0,  0,  0,  0,0,0],\
            ["IO3-"     ,2.0E-7  ,174.9 ,0     ,0     ,0,  0,  3,  0,  0,   0,   0,  1,  0,  0,0,0],\
            ["I2"       ,0.0     ,259.81,2.84  ,4300  ,0,  0,  0,  0,  0,   0,   0,  2,  0,  0,0,0],\
            ["I2OH-"    ,0.0     ,270.82,0     ,0     ,0,  0,  0,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["I3-"      ,0.0     ,380.71,0     ,0     ,0,  0,  0,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["IO-"      ,0.0     ,142.90,0     ,0     ,0,  0,  1,  0,  0,   0,   0,  1,  0,  0,0,0],\
            ["H2OI+"    ,0.0     ,144.92,0     ,0     ,0,  0,  0,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["HIO2"     ,0.0     ,159.91,0     ,0     ,0,  0,  0,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["IBr"      ,0.0     ,206.81,24.3  ,0     ,0,  0,  0,  0,  1,   0,   0,  1,  0,  0,0,0],\
            ["ICl"      ,0.0     ,162.36,111   ,0     ,0,  0,  0,  0,  0,   1,   0,  1,  0,  0,0,0],\
            ["I2Cl-"    ,0.0     ,289.26,0     ,0     ,0,  0,  0,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["ICl2-"    ,0.0     ,197.81,0     ,0     ,0,  0,  0,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["HOCl"     ,0.0     ,52.46 ,658   ,5900  ,0,  1,  1,  0,  0,   1,   0,  0,  0,  0,0,0],\
            ["HOBr"     ,0.0     ,96.91 ,131   ,0     ,0,  1,  1,  0,  1,   0,   0,  0,  0,  0,0,0],\
            ["BrCl"     ,0.0     ,115.36,0.98  ,5600  ,0,  0,  0,  0,  1,   1,   0,  0,  0,  0,0,0],\
            ["Br2"      ,0.0     ,159.81,0.73  ,4400  ,0,  0,  0,  0,  2,   0,   0,  0,  0,  0,0,0],\
            ["Br2Cl-"   ,0.0     ,195.26,0     ,0     ,0,  0,  0,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["BrCl2-"   ,0.0     ,150.81,0     ,0     ,0,  0,  0,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["Cl2"      ,0.0     ,70.91 ,0.09  ,2000  ,0,  0,  0,  0,  0,   2,   0,  0,  0,  0,0,0],\
            ["OH-_bulk" ,1.0E-6  ,17.01 ,0.29  ,4300  ,0,  1,  1,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["H+_bulk"  ,1.0e-8  ,1.01  ,2.6E-4,0     ,0,  1,  0,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["Br-_bulk" ,8.6E-4  ,79.90 ,1.2E-2,4300  ,0,  0,  0,  0,  1,   0,   0,  0,  0,  0,0,0],\
            ["Cl-_bulk" ,0.55    ,35.45 ,2.3E-2,4300  ,0,  0,  0,  0,  0,   1,   0,  0,  0,  0,0,0],\
            ["IBrg"     ,0       ,206.81,24.3  ,0     ,0,  0,  0,  0,  1,   0,   0,  1,  0,  0,0,0],\
            ["IClg"     ,0       ,162.36,111   ,0     ,0,  0,  0,  0,  0,   1,   0,  1,  0,  0,0,0],\
            ["O3_bulk"  ,0.0     ,48.0  ,0.01  ,2800  ,0,  0,  3,  0,  0,   0,   0,  0,  0,  0,0,0],\
            ["IBr_bulk" ,0.0     ,206.81,24.3  ,0     ,0,  0,  0,  0,  1,   0,   0,  1,  0,  0,0,0],\
            ["ICl_bulk" ,0.0     ,162.36,111   ,0     ,0,  0,  0,  0,  0,   1,   0,  1,  0,  0,0,0],\
            ["HOCl_bulk",0.0     ,52.46 ,658   ,5900  ,0,  1,  1,  0,  0,   1,   0,  0,  0,  0,0,0],\
            ["HOBr_bulk",0.0     ,96.91 ,131   ,0     ,0,  1,  1,  0,  1,   0,   0,  0,  0,  0,0,0],\
            ["Br2_bulk" ,0.0     ,159.81,0.73  ,4400  ,0,  0,  0,  0,  2,   0,   0,  0,  0,  0,0,0],\
            ["Cl2_bulk" ,0.0     ,70.91 ,0.09  ,2000  ,0,  0,  0,  0,  0,   2,   0,  0,  0,  0,0,0],\
            ["HOClg"    ,0.0     ,52.46 ,658   ,5900  ,0,  1,  1,  0,  0,   1,   0,  0,  0,  0,0,0],\
            ["HOBrg"    ,0.0     ,96.91 ,131   ,0     ,0,  1,  1,  0,  1,   0,   0,  0,  0,  0,0,0],\
            ["Br2g"     ,0.0     ,159.81,0.73  ,4400  ,0,  0,  0,  0,  2,   0,   0,  0,  0,  0,0,0],\
            ["Cl2g"     ,0.0     ,70.91 ,0.09  ,2000  ,0,  0,  0,  0,  0,   2,   0,  0,  0,  0,0,0],\
            ["IO3-_bulk",2.0E-7  ,174.9 ,0     ,0     ,0,  0,  3,  0,  0,   0,   0,  1,  0,  0,0,0],\
            ["BrClg"    ,0       ,115.36,0.98  ,5600  ,0,  0,  3,  0,  0,   0,   0,  1,  0,  0,0,0],\
            ["BrCl_bulk",0       ,115.36,0.98  ,5600  ,0,  0,  3,  0,  0,   0,   0,  1,  0,  0,0,0],\
            ["IO-_bulk" ,0.0     ,142.90,0     ,0     ,0,  0,  1,  0,  0,   0,   0,  1,  0,  0,0,0]]

    species_database = pd.DataFrame(data,columns=names)

    return species_database
####################################################################################################################
def O3_Iod_rate(T,scheme):
    """
    central definitions of the O3 + I- rate coefficient
    """
    if scheme == 'brown':
        K = 2.6E11*np.exp((-10600/(8.31*T)))
    elif scheme == 'magi':
        K = np.exp((-8772.2/T)+51.5)
    return K    
#####################################################################################################################
def SML_h(T,species_database,scheme):
    """
    Calculate the depth of the surface microlayer and return result in m from Luhar et.al 2019
    """

    D = 1.1e-6*np.exp(-1896/T)
    K0 = O3_Iod_rate(T,scheme)
    h = np.sqrt(D/(K0 * species_database.loc[species_database['name'] == 'I-', ['conc']].values[0][0]))

    return h
#####################################################################################################################
def Chemistry_solver(species_database,timesteps=100000,T=288,dt_max=0.00001,eqn_file='sml_cantera_base.yaml'):
    """
    Initial implementation for cantera solved SML O3 Iodide chemistry scheme -
    output units in Mol fraction so need to be converted to mol dm-3
    """

    sml = ct.Solution(eqn_file)
    ignore_list = ['HOIg','I2g','I-_bulk','O3g','HOI_bulk','I2_bulk',\
                   'OH-_bulk','H+_bulk','Br-_bulk','Cl-_bulk','IBrg','IClg','O3_bulk',\
                   "IBr_bulk","ICl_bulk","IO3-_bulk","HOClg","HOBrg","Br2g","Cl2g",\
                   "HOCl_bulk","HOBr_bulk","Br2_bulk","Cl2_bulk","BrCl_bulk","BrClg","IO-_bulk"]

    # calculate mass ratios from concentrations in species database
    total_mass = 0
    sml_species = species_database.drop(species_database[species_database.name.isin(ignore_list)].index)
    for index,row in sml_species.iterrows():
            total_mass += row['conc']*row['mw']
    mass_fractions = (sml_species['conc'].values*sml_species['mw'].values)/total_mass

    #create a string in the format of
    y_string = ''
    for i in range(len(mass_fractions)-1):
        y_string = y_string + f'{sml_species["name"].values[i]}:{mass_fractions[i]},'
    y_string = y_string + f'{sml_species["name"].values[-1]}:{mass_fractions[-1]}'
    sml.Y = y_string
    sml.TD = T, 997

    react = ct.IdealGasConstPressureReactor(sml)
    sim = ct.ReactorNet([react])
    sim.verbose = False

    #timestep settings
    t_end = timesteps*dt_max
    states = ct.SolutionArray(sml, extra=['t'])

    #advance chemistry through to max time time, keeping track of states on route
    while sim.time < t_end:
        sim.advance(sim.time + dt_max)
    states.append(react.thermo.state, t=sim.time)

    output = states.to_pandas()
    #get the final row of data for end state of solver
    end_state = output.loc[output.index==output.index.values[-1]]

    #convert back from mass fractions to concentrations and update species_database
    for index,row in species_database.iterrows():
        if row['name'] not in ignore_list:
            species_database.loc[index,'conc'] = (end_state[f'Y_{row["name"]}'].values[0]\
                                        *total_mass)/row['mw']

    return species_database
####################################################################################################################
def get_comp_dict(species_database,name):
    """
    extract data from the species database to create a dictonary in the correct format
    for the transfer calculations code
    """

    #select the right row of data to convert
    row = species_database.loc[species_database['name'] == name]

    #begin to create the right format
    columns = ["mw","KH","tVar","C","H","O","N","Br","Cl","F","I","S","rings","db","tb"]

    spec_dict = {"mw":row['mw'].values[0],\
                 "KH":row['KH'].values[0],\
                 "tVar":row['tVar'].values[0],\
                 "C":row['C'].values[0],\
                 "H":row['H'].values[0],\
                 "O":row['O'].values[0],\
                 "N":row['N'].values[0],\
                 "Br":row['Br'].values[0],\
                 "Cl":row['Cl'].values[0],\
                 "F":row['F'].values[0],\
                 "I":row['I'].values[0],\
                 "S":row['S'].values[0],\
                 "rings":row['rings'].values[0],\
                 "db":row['db'].values[0],\
                 "tb":row['tb'].values[0]}
    return spec_dict
#####################################################################################################################
def new_model(T,u,S,species_database,dt_max,t_total,con_Iod=False,\
              chem_scheme='sml_cantera_base.yaml',ConcAfterChem=False,rate='magi',R=0.9):

    """
    combination of Cen-Lin Tzung-May, cantera chemistry and the ozone deposition equation
     from Pound et al 2019 with Ra and Rb from Chang 2004
    """

    #R = 0.9 #0.9 = open ocean, 0.8 = coastal, 0.7 = esturine waters
    columns = ["Time","FluxA_I2","FluxA_HOI","O3_depv","Flux_ICl","Flux_IBr",\
               "Flux_HOBr","Flux_HOCl","Flux_Br2","Flux_Cl2","Flux_BrCl"] + \
               species_database['name'].values.tolist()
    bulk_mix_specs = ['HOI','I2','Br-','Cl-','O3','I-','IBr','ICl','IO3-','HOBr','HOCl','Br2','Cl2','BrCl']
    air_mix_specs  = ['HOI','I2','IBr','ICl','HOBr','HOCl','Br2','Cl2','BrCl']

    state = pd.DataFrame(columns=columns)
    #parameters for O3 deposition that are a constant for given conditions
    spec_comp = get_comp_dict(species_database,'O3')
    u_star = u*np.sqrt(6.1e-4 + 6.3e-5*u) #as per equation 16 in Johnson 2010  #0.05*u
    u_star_w = 0.0345*u_star #this is an approximation, a more exact way is to use 
    # p = 101325
    # pw = 1025
    #pa = p/(287.058*T_air)
    # u_star_w = ((pa/pw)**0.5)*u_star
    #however for the purposes of this model it is sufficient to approximate at the moment but this can be explored with additional input parameters in the future

    Ra = u/u_star**2
    Rb = (5/u_star)*transfer.Sc_air(spec_comp,T-273.15)**(2.0/3.0)
    alpha = 10**(-0.25-0.013*(T - 273.16))
    Diff = 1.1E-6*np.exp(-1896/T)
    react = O3_Iod_rate(T,rate)

    if ConcAfterChem:
        CAC = pd.DataFrame(columns=["Time"]+species_database['name'].values.tolist())

    for i in range(int(t_total/dt_max)): #calculate total number of timesteps needed
        h = SML_h(T,species_database,rate)*10 #get the depth of the SML and convert to dm
        #advance chemistry
        species_database = Chemistry_solver(species_database,timesteps=1,T=T,\
                            dt_max=dt_max,eqn_file=chem_scheme)
        if ConcAfterChem and (int(np.mod(i,(0.025/dt_max))) == 0):
            names = species_database['name'].values
            row_state = [dt_max*(i+1)]
            for j in range(len(names)):
                row_state.append(species_database.loc[species_database['name'] == \
                                                 names[j], ['conc']].values[0][0])
            row_state = pd.Series(row_state,index=CAC.columns)
            CAC = CAC.append(row_state,ignore_index=True)

        #calculate the change in SML concentration due to surface renewal
        for spec in bulk_mix_specs:
            bulk = species_database.loc[species_database['name'] == \
                                        f'{spec}_bulk', ['conc']].values[0][0]
            sml  = species_database.loc[species_database['name'] == \
                                        f'{spec}', ['conc']].values[0][0]
            #don't need to includes conversion of U [10m wind speed] from m/s to dm/s as
            #this cancels to be a unitless multiplier
            surf_renewal = ((2*(3.42E-3*u+2.7E-3))/(np.pi*(0.235*u+4.935)))*\
                           (bulk - sml)*dt_max
            species_database.loc[species_database['name'] == f'{spec}', ['conc']] = \
                sml + surf_renewal
        
        #calculate moleclar transfer to bulk water
        for spec in bulk_mix_specs:
            spec_comp = get_comp_dict(species_database,spec)
            bulk = species_database.loc[species_database['name'] == \
                                       f'{spec}_bulk', ['conc']].values[0][0]
            sml  = species_database.loc[species_database['name'] == \
                                         f'{spec}', ['conc']].values[0][0]
            #calculate kw in m/s, convert to dm/s from m/s
            kw = transfer.kw(spec_comp,T-273.15,u,S)*10
            flux = R*kw*(bulk - sml)
            species_database.loc[species_database['name'] == f'{spec}', ['conc']]= sml \
                                            + flux*dt_max/h
        #calculate moleclar transfer to atmosphere
        for spec in air_mix_specs:
            spec_comp = get_comp_dict(species_database,spec)
            gas  = species_database.loc[species_database['name'] == f'{spec}g', \
                                        ['conc']].values[0][0]
            sml  = species_database.loc[species_database['name'] == f'{spec}', \
                                       ['conc']].values[0][0]
            # calculate ka and convert to dm/s from m/s
            ka = transfer.ka(spec_comp,u,T-273.15)*10
            flux = ka*(transfer.KH(spec_comp,T-273.15,S)*sml - gas) #molec /dm2 /s
            species_database.loc[species_database['name'] == f'{spec}', \
                                    ['conc']] = sml - flux*dt_max/h

            # return flux of species in mol/cm2/s
            if spec == 'I2':
               FluxA_I2 = flux/100
            elif spec == 'Br2':
               Flux_Br2 = flux/100
            elif spec == 'Cl2':
                Flux_Cl2 = flux/100
            elif spec == 'HOI':
                FluxA_HOI = flux/100
            elif spec == 'HOBr':
                Flux_HOBr = flux
            elif spec == 'HOCl':
                Flux_HOCl = flux
            elif spec == 'ICl':
                Flux_ICl = flux/100
            elif spec == 'IBr':
                Flux_IBr = flux/100
            elif spec == 'BrCl':
                Flux_BrCl = flux/100
            
        spec_comp = get_comp_dict(species_database,'O3')
        gas  = species_database.loc[species_database['name'] == f'O3g', \
                                   ['conc']].values[0][0]
        sml  = species_database.loc[species_database['name'] == f'O3', \
                                   ['conc']].values[0][0]
        a = react*species_database.loc[species_database['name'] == f'I-', \
                                   ['conc']].values[0][0]
        dm = np.sqrt(Diff/a)
        lamb = dm*np.sqrt(a/Diff)
        epsd = np.sqrt(((4*a)/(0.4*u_star_w))*(dm+Diff/(0.4*u_star_w)))
        psi = np.sqrt(1+((0.4*u_star_w*dm)/Diff))
        if epsd >=600:
            # a catch case to stop k0 & k1 giving values of 0 and the model falling over
            epsd = 600
        Rc = 1/(alpha*np.sqrt(a*Diff))*((psi*sp.k1(epsd)*np.sinh(lamb)+\
                                             sp.k0(epsd)*np.cosh(lamb))/\
                                        (psi*sp.k1(epsd)*np.cosh(lamb)+\
                                             sp.k0(epsd)*np.sinh(lamb)))

        Rt = (Ra + Rb + Rc) #s/m
        flux = (1/Rt)*10*(gas - transfer.KH(spec_comp,T-273.15,S)*sml) #molec /dm2 /s

        species_database.loc[species_database['name'] == f'O3', \
                            ['conc']] = sml + flux*dt_max/h
        O3_gain = (1/Rt)*100 # output units in cm/s

        #limit the frequency of timesteps written to file
        if int(np.mod(i,(0.025/dt_max))) == 0:
            if chem_scheme=='sml_cantera_base.yaml':
                row_state = [dt_max*(i+1),FluxA_I2,FluxA_HOI,O3_gain,Flux_ICl,Flux_IBr]
            elif chem_scheme=='sml_cantera_schneider.yaml' or \
                 chem_scheme=='sml_cantera_schneider_Clsen.yaml' or \
                 chem_scheme=='sml_cantera_schneider_Clsen2.yaml':
                row_state = [dt_max*(i+1),FluxA_I2,FluxA_HOI,O3_gain,Flux_ICl,Flux_HOCl]
            else:
                row_state = [dt_max*(i+1),FluxA_I2,FluxA_HOI,O3_gain,Flux_ICl,Flux_IBr,\
                             Flux_HOBr,Flux_HOCl,Flux_Br2,Flux_Cl2,Flux_BrCl]

            names = species_database['name'].values
            for j in range(len(names)):
                row_state.append(species_database.loc[species_database['name'] == \
                                                 names[j], ['conc']].values[0][0])
            row_state = pd.Series(row_state,index=state.columns)
            state = state.append(row_state,ignore_index=True)

        #force reset of H+ to maintain constant pH
        species_database.loc[species_database['name'] == f'H+', ['conc']]  = \
             species_database.loc[species_database['name'] == f'H+_bulk', \
                                        ['conc']].values[0][0]
        species_database.loc[species_database['name'] == f'OH-', ['conc']]  = \
             species_database.loc[species_database['name'] == f'OH-_bulk', \
                                        ['conc']].values[0][0]

        if con_Iod: #if you want to force constant I- in SML
            species_database.loc[species_database['name'] == f'I-', ['conc']]  = \
                 species_database.loc[species_database['name'] == f'I-_bulk', \
                                            ['conc']].values[0][0]

        #zero out O2 as this is a dead end in the system       
        species_database.loc[species_database['name'] == f'O2', ['conc']]  = 0.0

    if ConcAfterChem:
        return state,species_database,CAC
    else:
        return state,species_database
#####################################################################################################################
#############################################################################################
def run_sensitivity(T_range=[296],ws_range=[7],O3_range=[30],I_range=[100],S=35,\
                    chemistry='sml_cantera_base.yaml',outputdir='new_base',con_Iod=False,rate='magi',R=0.9):

    dt_max = 0.0001
    t_total = 1.5

    for O3 in O3_range:
        for I in I_range:
            for ws in ws_range:
                for T in T_range:
                    spec_database = define_species_database()
 
                    spec_database.loc[spec_database['name'] == 'I-', ['conc']] = I*1E-9
                    spec_database.loc[spec_database['name'] == 'I-_bulk', ['conc']] = I*1E-9

                    if outputdir == 'no_chlorine':
                        spec_database.loc[spec_database['name'] == 'Cl-', ['conc']] = 0
                        spec_database.loc[spec_database['name'] == 'Cl-_bulk', ['conc']] = 0
                    if outputdir == 'no_bromine':
                        spec_database.loc[spec_database['name'] == 'Br-', ['conc']] = 0
                        spec_database.loc[spec_database['name'] == 'Br-_bulk', ['conc']] = 0
                    if outputdir == 'no_bromine_chlorine':
                        spec_database.loc[spec_database['name'] == 'Cl-', ['conc']] = 0
                        spec_database.loc[spec_database['name'] == 'Cl-_bulk', ['conc']] = 0
                        spec_database.loc[spec_database['name'] == 'Br-', ['conc']] = 0
                        spec_database.loc[spec_database['name'] == 'Br-_bulk', ['conc']] = 0
                    O3_val = O3*4.15E-11
                    spec_database.loc[spec_database['name']=='O3g',['conc']] = O3_val
                    result,spec_database= new_model(T,ws,S,spec_database,dt_max,t_total,con_Iod=con_Iod,chem_scheme=chemistry,rate=rate,R=R)
                
                    result.to_csv(f'{outputdir}/O3{O3}_I{I}_ws{ws}_T{T}.csv')
#############################################################################################
if __name__ == "__main__":
 
    
    args = sys.argv
    O3 = float(args[1])
    I  = float(args[2])
    ws = float(args[3])
    T  = float(args[4])
    S = 35
    R = 0.9
    dt_max = 0.0001
    t_total = 4
    O3_val = O3*4.15E-11
    chems = ['brown_cl']
    con_Iod = [False]
    rate = ['brown']
    outdir = ['./']
    run_sensitivity(ws_range=[ws],T_range=[T],O3_range=[O3],I_range=[I],\
                    chemistry=f'sml_cantera_{chems[0]}.yaml',S=S,\
                    outputdir=outdir[0],con_Iod=con_Iod[0],rate=rate[0],R=R)