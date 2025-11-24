#=====================================================================================
# Python translation of MixModel code from (ref                                  )
# Origionally Matlab
# 
# Author R. Pound (ryan.pound@york.ac.uk)
#
#=====================================================================================

import numpy as np 
import pandas as pd
import gsw

def create_vert_grid(D,M,Lat):
    Lambda = -1
    Xi = np.linspace(0, M+1)
    dint = (D/Lambda)*np.log(1-Xi*(1-np.exp(Lambda)))
    dgrid = (D/Lambda)*np.log(1-(Xi+0.5/M)*(1-np.exp(Lambda)))
    dgrid[-1] = dint[-1]
    pint = gsw.p_from_z(dint,Lat)
    pgrid = gsw_p_from_z(dgrid,Lat)
    return dint,dgrid,pint,pgrid

def init(hBL=18,Lat=60,Lon=5,delt=1800,RiCrit=0.3,df_init,df_force):
    #initialise the model code parameters
    #Set Model Depth in metres:
    D = np.amax(df_init['depth[m]'].values)

    M = len(df_init['depth[m]'].values)
    dint,dgrid,pint,pgrid = create_vert_grid(D,M,Lat)


    #Define and calculate layer thicknesses and grid-point spacings:
    #Allocate storage space for layer thickness and gridpoint spacing matrices:
    for i in range(0,M):
       deltaint[i] = dint[i+1] - dint[i]
    for i in range(1,M):
	deltagrid[i] = dgrid[i] - dgrid[i-1]
    deltagrid[0] = dgrid[0] - 0

    '''
    Requires initial profile tab-delimited input file 'inprof.txt' in the Matlab search path.
    U,V,T and S are estimates at M+1 gridpoints. Row 1 corresponds to U values, 2 to V, 3 to 
    T and 4 to S. Column 1 to surface value, M+1 to deepest value. External surface forcing
    may be simulated in three ways
    '''
    absolute_salinity = gsw.SA_from_SP(df_init['Salinity'], pgrid, Lon, Lat) 
    potential_temp = gsw.pt_from_t(absolute_salinity,df_init['Temperature'],pgrid,pint[0])
    conservative_temp = gsw.CT_from_t(absolute_salinity,df_init['Temperature'],pgrid)
    buoy = 9.81*((gsw.alpha(absolute_salinity, conservative_temp, pgrid)*\
                 (potential_temp-potential_temp[0]))-\
                 (gsw.beta(absolute_salinity, conservative_temp, pgrid)*\
                 (df_init['Salinity']-df_init['Salinity'].values[0])))
    '''
    Profgrid = [Ugrid Vgrid Tgrid Sgrid Bgrid CH2I2grid]
    '''
    Gradint = np.zeros([M+1,6])
    Gradgrid = np.zeros([M+1,6])
    for i in range(M):
        #could probably put a loop here to calculate gradients based on the length of the columns value of the dataframe and then putting the above into data frames
        Gradint[i,0] = -(df_init['U[m/s]'].values[i+1] - df_init['U[m/s]'].values[i])/deltagrid[i]
        Gradint[i,1] = -(df_init['V[m/s]'].values[i+1] - df_init['V[m/s]'].values[i])/deltagrid[i]
        Gradint[i,2] = -(potential_temp[i+1] - potential_temp[i])/deltagrid[i]
        Gradint[i,3] = -(absolute_salinity[i+1] - absolute_salinity[i])/deltagrid[i]
        Gradint[i,4] = -(buoy[i+1] - buoy[i])/deltagrid[i]
        Gradint[i,5] = -(df_init['CH2I2'].values[i+1] - df_init['CH2I2'].values[i])/deltagrid[i]

    for i in range(1,M):
        Gradgrid[i,0] = -((df_init['U[m/s]'].values[i+1] + df_init['U[m/s]'].values[i])/\
                          2-(df_init['U[m/s]'].values[i] + df_init['U[m/s]'].values[i-1])/2)/deltaint[i]
        Gradgrid[i,1] = -((df_init['V[m/s]'].values[i+1] + df_init['V[m/s]'].values[i])/\
                          2-(df_init['V[m/s]'].values[i] + df_init['V[m/s]'].values[i-1])/2)/deltaint[i]
        Gradgrid[i,2] = -((potential_temp[i+1] + potential_temp[i])/\
                          2-(potential_temp[i] + potential_temp[i-1])/2)/deltaint[i]
        Gradgrid[i,3] = -((absolute_salinity[i+1] + absolute_salinity[i])/\
                          2-(absolute_salinity[i] + absolute_salinity[i-1])/2)/deltaint[i]
        Gradgrid[i,4] = -((buoy[i+1] + buoy[i])/\
                          2-(buoy[i] + buoy[i-1])/2)/deltaint[i]
        Gradgrid[i,5] = -((df_init['CH2I2]'].values[i+1] + df_init['CH2I2'].values[i])/\
                          2-(df_init['CH2I2'].values[i] + df_init['CH2I2'].values[i-1])/2)/deltaint[i]

    Gradgrid[0,0] = -((df_init['U[m/s]'].values[1] + df_init['U[m/s]'].values[0])/\
                      2-(df_init['U[m/s]'].values[0])/deltaint[0]
    Gradgrid[0,1] = -((df_init['V[m/s]'].values[1] + df_init['V[m/s]'].values[0])/\
                      2-(df_init['V[m/s]'].values[0])/deltaint[0]
    Gradgrid[0,2] = -((potential_temp[1] + potential_temp[0])/\
                      2-(potential_temp[0])/deltaint[0]
    Gradgrid[0,3] = -((absolute_salinity[1] + absolute_salinity[0])/\
                      2-(absolute_salinity[0])/deltaint[0]
    Gradgrid[0,4] = -((buoy[1] + buoy[0])/\
                      2-(buoy[0])/deltaint[0]
    Gradgrid[0,5] = -((df_init['CH2I2]'].values[1] + df_init['CH2I2'].values[0])/\
                      2-(df_init['CH2I2'].values[0])/deltaint[0]

    for i in range(6):
        Gradgrid[M,i] = 0.0
        Gradint[M,i] = 0.0

    RunDays = 5.4375
    RunLength = RunDays * 24 * 3600
    T = np.arange(0,RunLength+delt,delt)
    TLength = len(T)
    
    
