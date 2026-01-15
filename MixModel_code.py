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

def mm_locat(hbl,M,dgrid,dint):

    #Locate layer in which boundary layer depth can be found:
    layer = 0
    for k in range(1,M+1):
        if dgrid[k-1] < hBL and dgrid[k] >= hBL:
            if dint[k] > hBL:
		layer = k-0.5
            if dint[k] <= hBL:
                layer = k
        elif dgrid[0] >= hBL
            layer = 1
    if not(layer % 1 == 0):
	k = layer+0.5
    else:
	k=layer
    return [layer k]

def mm_irrad(h,I0):

    #Irradiance I (W/m^2) calculated from I=I0*Sum from i=1 to NI(ri exp(-d/mui))
    #where mui is reciprocal of absorbance of band i,
    #set hB to 0 for no contribution ie. irradiance warming of BL does not
    #contribute to buoyancy forcing, or hB=h for max. contribution

    #Number of Wavelength Bands containing a fraction of the total
    #radiation ri, each with absorption mui after Stramma et al.(1986):
    Irrad = 0
    r = [0.58 0.42]
    mu = [0.35 23]
    for i in range(len(r)):
	Irrad = Irrad + I0 * r[i] * exp( - h / mu[i])
    return Irrad

def mm_evalu(hBL, hBL_init, M, Lat, RiCrit, dint, dgrid, pint, pgrid, deltaint, deltagrid, \
             Profgrid, Gradgrid, Gradint, delt, layer, k, forcing):

    """
    %Define Surface External Forcing:
    %Read table of external forcing values at delt second (start 10 min = 600 s) intervals from
    %input file:
        %Unattenuated surface irradiance I0 in W/m^2 - contributes to Qt and Buoyancy
        %Air Temperature
        %Relative Humidity
        %Precipitation kg/m^2
    """

    Qt = forcing['Qt']
    #surface density of seawater
    Rho0 = gsw_rho #gsw_rho_t_exact #??? potentially what it should now use
    #surface seawater specific heat at constant pressure 
    Cp0 = gsw_cp0
    wt0bar = -Qt/(Rho0*Cp0)

    Ft = df_force[] #fresh water forcing/precipitation 
    Fs = 0 #turbulent saltwater flux due to surface ice
    #Salinities of surface seawater, freshwater, sea ice water:
    S0,Sf,Si = [Profgrid['S'],0,10]
    Rhof = gsw_rho #fresh water density ???
    Rhoice = 990 #gsw_rho_ice

    ws0bar = Ft*S0/Rhof+Fs*(S0-Si)/Rhoice

    #vertical zonal velocity flux
    wu0bar = -df_force['Tauu']/Rho0
    #vertical meridonal velocity flux
    wv0bar = -df_force['Tauv']/Rho0

    #radiative contribution to buoyancy forcing 
    g = 9.81
    alpha0 = gsw_alpha # at surface
    beta0 = gsw_beta # at surface

    Isurf = mm_irrad(0,df_force['I0'])
    IhB = mm_irrad(hBL,df_force['I0'])
    #Exp.coeff. due to T, Density & Specific Heat @ const. press. @hB:
    alphahB = gsw_alpha
    nonpotT = gsw_t_from_pt0
    RhohB = gsw_rho
    CphB = gsw_cp_t_exact

    #Buoyancy Flux Divergence
    wb0bar = g*(alpha0*wt0bar-beta0*ws0bar)

    #Buoyancy contribution due to radiative heating through column from d = 0 to d = hB
    #First calculate Irradiance, potential temperature, density, specific heat & thermal expansion coeff. profiles
    #define arrays for depth?
    for i in range(0,M)
        Id[i] = mm_irrad(dgrid[i],I0)
        nonpotT[i] = gsw_t_from_pt0
        Rhoi[i] = gsw_rho 
        Cpi[i] = gsw_cp_t_exact 
        alphai[i] = gsw_alpha

    BR = 0.55*(g*(alpha0*Isurf/(Rho0*Cp0) - alphahB*IhB/(RhohB*CphB)))
    Bf = -wb0bar+BR

    #Using Monin-Obukhov similarity theory to obtain a length scale:
    kappa = 0.4
    ustar = sqrt(sqrt(wu0bar**2+wv0bar**2))
    Lmo = ustar**3/(kappa*Bf)
    #Calculate contribution of radiative heating contribution to nonlocal heat transport term - possible in any 
    #fraction of the BL from zero to hBL
    wtRbar = -(Isurf/(Rho0*Cp0)-IhB/(RhohB*CphB))
    wtRbar = -(Isurf/(Rho0*Cp0)-IhB/(RhohB*CphB))

    #Following sections modified by Lucy Carpenter in 2008 to incorporate dihalomethanes
    #f)CH2I2 Flux Divergence
    #kCH2I2 = forcing(1,6) / 360000; %convert from cm / hr to m / s
    #flux_CH2I2 = -kCH2I2 * Profgrid(1,6); % in micromol / m^2 / s, where surface CH2I2 concentration, Profgrid(1,6), is in micromol / m^3
    #wCH2I20bar = -flux_CH2I2;

    #pCH2I2(1:10,1)= 0.04/86400;
    #pCH2I2(11:M+1,1) = 0;

    #Calculate empirically-derived non-dimensional flux profiles for scalars and momentum terms
    Zeta = dgrid / Lmo
    #Assign memory for Psi & W Matrices
    PsiM,PsiS,WS,WM = [Zeta,Zeta,Zeta,Zeta]
    #Set empirical constants
    ZetaS,ZetaM,cs,cm,as_,am = [-1,-0.2,98.96,8.38,-28.86,1.26]
    #%And then Calculate vertical turbulent scale, and set fractional extent of surface layer wrt boundary layer
    sigma = dgrid / hBL
    epsilon = 0.1

    #Also calculate the closing non-local convective transport terms, GammaX, 
    #after the Mailhot & Benoit(1982) parameterisation, extended for all
    #forcing conditions. Includes the heat flux term calculated after
    #Deardorff(1972) but ignores nonlocal momentum flux which cannot yet be
    #parameterised in a similar fashion:
    Cstar = 10
    Cs = Cstar*kappa*(kappa*cs*epsilon)**(1/3)
    GammaS = np.zeros((M+1))
    GammaT = np.zeros((M+1))

    for i in range(0,M+1):
        GammaM = 0
        if Zeta[i] >= 0:
	    PsiM[i] = 1+5*Zeta[i];
	    PsiS[i] = PsiM[i];
	    GammaS(i,1) = 0;
			GammaT(i,1) = 0;
		elseif Zeta(i,1) >= ZetaM & Zeta(i,1)  <0
			PsiM(i,1) = (1-16*Zeta(i,1)).^-0.25;
		else
			PsiM(i,1) = (am-cm*Zeta(i,1)).^(-1/3);
		end
		if Zeta(i,1) >= ZetaS & Zeta(i,1) < 0
			PsiS(i,1) = (1-16*Zeta(i,1)).^-0.5;
		elseif Zeta(i,1) < ZetaS
			PsiS(i,1) = (as-cs*Zeta(i,1)).^(-1/3);
      end
      if sigma(i,1) < epsilon & sigma(i+1,1) > epsilon
         PsiSeps = PsiS(i,1);
         PsiMeps = PsiM(i,1);
      elseif i == 1 & sigma(i,1) > epsilon
         PsiSeps = PsiS(i,1);
         PsiMeps = PsiM(i,1);
      end
      %note, the following two expression alternatives are ambiguous in the paper
      if sigma(i,1) > epsilon & sigma(i,1) < 1 & Zeta(i,1) < 0
			WS(i,1) = kappa*ustar/PsiSeps;
			WM(i,1) = kappa*ustar/PsiMeps;
		else
			WS(i,1) = kappa*ustar/PsiS(i,1);
			WM(i,1) = kappa*ustar/PsiM(i,1);

		%if sigma(i,1) > epsilon & sigma(i,1) < 1 & Zeta(i,1) < 0
		%	WS(i,1) = kappa*ustar/(PsiS(i,1)*(epsilon*hBL/Lmo));
		%	WM(i,1) = kappa*ustar/(PsiM(i,1)*(epsilon*hBL/Lmo));
		%else
		%	WS(i,1) = kappa*ustar/(PsiS(i,1)*(sigma(i,1)*hBL/Lmo));
		%	WM(i,1) = kappa*ustar/(PsiM(i,1)*(sigma(i,1)*hBL/Lmo));
      
         if i > 1 & sigma(i-1,1) > epsilon & sigma(i-1,1) <= 1
				WSh=WS(i-1,1);
            WMh = WM(i-1,1);
         elseif sigma(1,1) >= hBL
				WSh=WS(1,1);
            WMh = WM(1,1);
%*****************************************************************************************************************
%NB not sure if the above 2 lines are sufficiently limited to evaluate the velocity scales at h correctly
%21/2/99 - added the condition where first gridpoint was greater than or equal to h
%________________________________________________________________________________
			end
		end
		
      if Zeta(i,1) < 0
         GammaS(i,1) = Cs*ws0bar/(WS(i,1)*hBL);
         %set radiative contribution to convective correction to zero, if assuming its contribution to buoyancy, BR,
         %takes place in entire boundary layer (ie to hBL)
         %wtRbar = 0;
			%wtRbar = -((I0/(Rho0*Cp0))-(Id(i,1)/(Rhoi(i,1)*Cpi(i,1))));
			GammaT(i,1) = Cs*(wt0bar+wtRbar)/(WS(i,1)*hBL);
		end
   end

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
    Gradgrid[0,5] = -((df_init['CH2I2'].values[1] + df_init['CH2I2'].values[0])/\
                      2-(df_init['CH2I2'].values[0])/deltaint[0]

    for i in range(6):
        Gradgrid[M,i] = 0.0
        Gradint[M,i] = 0.0

    RunDays = 5.4375
    RunLength = RunDays * 24 * 3600
    T = np.arange(0,RunLength+delt,delt)
    TLength = len(T)
    #Set convergence tolerance for layer profile loop as the finest grid spacing:
    Tol = deltagrid[0]
    hBL_init = hBL

    layer,k = mm_locat(hBL,M,dgrid,dint)

    for t in range(len(T)):
        print(f'Time is {t}')
    #Ensure that 1st iteration never converges:
    hlast = hBL
    hnow = hBL + 3
    #set inner loop counter:
    i = 2

    while ((abs(hlast-hnow))/deltagrid[k] >= Tol) or (i < 4):
        if t <= 2 and i == 2
            #(pass delt,Force when read from input file):
            PassArg = mm_evalu(hBL, hBL_init, M, Lat, RiCrit, dint, dgrid, pint, pgrid, deltaint, \
                               deltagrid, 0.999*Profgrid, 0.999*Gradgrid, 0.999*Gradint, delt, layer, k, forcing)

         factorH = PassArg(1:M,1:5);
         diffus = PassArg(1:M,7:10);
         RiBulk = PassArg(1:M,6);
         LengthArg = PassArg(1,11);
      elseif t >= 3 & i == 2
         PassArg = mm_evalu(hBL, hBL_init, M, Lat, RiCrit, dint, dgrid, pint, pgrid, deltaint, deltagrid, First_Stab, Gradgrid, Gradint, delt, layer, k, forcing);%(pass delt,Force when read from input file):
         factorH = PassArg(1:M,1:5);
         diffus = PassArg(1:M,7:10);
         RiBulk = PassArg(1:M,6);
         LengthArg = PassArg(1,11);
      else
         PassArg = mm_evalu(hBL, hBL_init, M, Lat, RiCrit, dint, dgrid, pint, pgrid, deltaint, deltagrid, ProfGridHalf, iterGradgrid, iterGradint, delt, layer, k, forcing);%(pass delt,Force when read from input file):
         factorH = PassArg(1:M,1:5);
         diffus = PassArg(1:M,7:10);
         RiBulk = PassArg(1:M,6);
         LengthArg = PassArg(1,11);
###############################################################################################################
def main():
    
    M = np.arange(0,51,1)
    df_init = pd.read_csv('./inprof.txt',names=M,delim_whitespace=True)
    df_init['columns'] = ['U[m/s]','V[m/s]','T','S','CH2I2','duplicate?']
    df_init = df_init.T

    M = np.arange(1,241,1)
    df_force = pd.read_csv('./extforce.txt',header=0,names=M,delim_whitespace=True)
    df_force['columns'] = ['Qt','Ft','Tauu','Tauv','IO','kCH2I2']
    df_force = df_force.T


