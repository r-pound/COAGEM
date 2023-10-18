"""
Python script for calculating liquid / gas transfer velocites for any volatile compound based on physical and chemical data. Python translation of the R code supplied with the paper Johnson, M.T., A numerical scheme to calculate the temperature and salinity dependent air-water transfer velocity for any gas. Ocean Science, 2010.
"""

import numpy as np
import pandas as pd

"""
contents of compound dictionary:

   mw = molecular weight [ g/mol]
   KH = henrys law solubility in M/atm
   tVar = temperature dependence of KH (-delatH/R) in 1/K
   C... etc = number of atoms in molecule
   db = number of double bonds in molecule
   tb = number of triple bonds in molecule
   rings =  number of cyclic features in molecule
"""

def Vb(comp):
    """
    Calculate the molar volume at boiling poing (Vb) using the Schroeder method

    for compounds containing elements other than C,H,O,N,S, Cl, Br, F, and I, the schroeder method 
       won't work so Vb must be specified

    db, tb and rings should be the number of double and triple bonds, and ring features in the moelcule respectively. 
    """
    if comp["rings"]>0:
       ringval=-7
    else:
       ringval = 0

    return (7*(comp["C"]+comp["H"]+comp["O"]+comp["N"]+comp["db"])+\
            14*comp["tb"]+31.5*comp["Br"]+24.5*comp["Cl"]+10.5*comp["F"]+\
            38.5*comp["I"]+21*comp["S"]+ringval)
#################################################################################################################
def C_D(u):
    """
    Calculate the drag coefficient (using smith? from smith?)
    """
    return 1e-4*(6.1+0.63*u)
##################################################################################################################
def n_0(T):
    """
    T = Temperature (C)
    calculate the viscosirty of pure water in cP (mPa s) according to laLiberte 2007 
       (doi:10.1021/je0604075).
    """
    return (T+246)/(137.37+(5.2842*T)+(0.05594*T**2))
###################################################################################################################
def n_sw(T,S):
    """
    T = Temperatuer (C)
    S = Salinity 
    calculate the dynamic viscosity of seawater in cP using the viscosity model / mixing 
       rule approach of LaLiberte 2007
    """
    #define each constituent in seawater per salinity unit per mil and the coefficients determined by
    #   LaLiberte 2007 for each salt 
    data = [['NaCl',0.798,16.22,1.3229,1.4849,0.0074691,30.78,2.0583],\
            ['KCl',0.022,6.4883,1.3175,-0.7785,0.09272,-1.3,2.0811],\
            ['CaCl2',0.033,32.028,0.78792,-1.1495,0.0026995,780860,5.8442],\
            ['MgCl2',0.047,24.032,2.2694,3.7108,0.021853,-1.1236,0.14474],\
            ['MgSO4',0.100,72.269,2.2238,6.6037,0.0079004,3340.1,6.1304]]
    sw_cmf = pd.DataFrame(data,columns=['Name','massfraction','v1','v2','v3','v4','v5','v6'])

    #w_i_ln_n_i_tot is the sum of the product of the mass fraction and the natural log of the 
    #   viscosities contributed by each solute individually (see LaLiberte 2007, equation 8)
    w_i_ln_n_i_tot=[]
    #sum up the mass fractions to get water mass fraction
    w_i_tot=np.sum(sw_cmf["massfraction"]*S/1000)

    #wi_tot is used here as eq 12 in LaLiberte requires (1-w_w), which is equivalent. 
    #    Using this term seems initially counterintuitive - one might expect to use w_i here for each 
    #    solute individually. However,  "at any given solute concentration, the solute viscosity increases 
    #    with the total solute content" so 1-w_w (or w_i_tot) is the correct term - pers. comm. from Marc LaLiberte
    for index,row in sw_cmf.iterrows():
        w_i = row["massfraction"]*S/1000
        n_i = (np.exp(((row['v1']*w_i_tot**row['v2'])+row['v3'])/((row['v4']*T) + 1)))/((row['v5']*(w_i_tot**row['v6']))+1)

        w_i_ln_n_i_tot.append(w_i*np.log(n_i))

    ln_n_m = (1-w_i_tot)*np.log(n_0(T))+np.sum(w_i_ln_n_i_tot)
    return np.exp(ln_n_m)
        
###################################################################################################################
def p_sw(T,S):
    """
    T = Temperature (C)
    S = Salinity 
    Calculate the density of seawater in kg/m3 according to millero and poisson (1981)
    """

    A = 0.824493-(0.0040899*T)+(0.000076438*(T**2))-(0.00000082467*(T**3))+(0.0000000053875*(T**4))
    B = -0.00572466+(0.00010277*T)-(0.0000016546*(T**2))
    C = 0.00048314

    #density of pure water
    po = 999.842594+(0.06793952*T)-(0.00909529*(T**2))+(0.0001001685*(T**3))\
                   -(0.000001120083*(T**4))+(0.000000006536332*(T**5))
    return po+(A*S)+(B*(S**1.5))+(C*S)

###################################################################################################################
def v_sw(T,S):
    """
    T = Temperature (C)
    S = Salinity [psu]
    calculate kinmatic viscosity of seawater in cm/s for Schmidt number calculation
    """

    #multiply convert n_sw to si / 1000 and multiply by 10000 to go from m2/s to cm2/s 
    return 10*n_sw(T,S)/p_sw(T,S)
###################################################################################################################
"""
ALL DIFFUSION COEFFICIENTS CALCULATED IN cm2/s
"""
def diff_HM(comp,T,S):
    """
    T = Temperature (C)
    S = Salinity [psu]
    Hayduk and Minhas (1982) diffusion coefficient calculation
    """
    EpsilonStar = (9.58/Vb(comp)) - 1.12
    return 1.25e-8*((Vb(comp)**-0.19)-0.292)*((T+273.15)**1.52)*(n_sw(T,S)**EpsilonStar)
###################################################################################################################
def diff_WC(comp,T,S):
    """
    T = Temperature (C)
    S = Salinity [psu]
    calculate diffusivity by Wilkie abd Change (1955) method
    associaction factor of solvent (2.6 in the case of water according to Poling 2001; 
           although Wanninkhof suggests 2.26)
    """
    return ((T+273.15)*5.064e-7)/(n_sw(T,S)*Vb(comp)**0.6)
####################################################################################################################
def schmidt(comp,T,S):
    """
    calculate the mean schmidt number in water
    """
    mean_diff = 0.5*(diff_WC(comp,T,S)+diff_HM(comp,T,S))
    sc = v_sw(T,S)/mean_diff

    return  sc
###################################################################################################################
def kw(comp,T,u,S):
    """
    comp = dictionary of the structure/contents of compound
    T = Temperature (C)
    S = Salinity [psu]
    Liquid phase transfer velocity calculations [values in m/s] from nightingale 2000 which
        is an empirical fit
    """
    return ((0.222*u**2+0.333*u)*(schmidt(comp,T,S)/600)**-0.5)/(360000)
####################################################################################################################
def KH0(comp,T=25):
    """
    Calculate Henrys law constant at a given T in pure water from Sander (1999)
    """
    return 12.2/((273.15+T)*comp["KH"]*np.exp(comp["tVar"]*((1/(T+273.15))-0.00335)))
####################################################################################################################
def Ks(comp):
    """
    no description in the code, will update from paper
    """
    theta = 7.3353282561828962e-04 + 3.3961477466551352e-05*np.log(KH0(comp)) - \
            2.4088830102075734E-06*np.log(KH0(comp))**2 + \
            1.5711393120941302E-07*np.log(KH0(comp))**3
    return theta*np.log(Vb(comp))
###################################################################################################################
def K_H_factor(comp,S):
    """
    calculate salinity-dependent salting-out scaling factor for KH0 (see manuscript)
    """
    return 10**(Ks(comp)*S)
####################################################################################################################
def KH(comp,T,S):
    """
    Calculate gas-over-liquid unitless Henry's law constant at a given T and Salinity

    KH input should be M/atm converted from mol/(m3Pa), ~1/101
    """
    return KH0(comp,T)*K_H_factor(comp,S)
####################################################################################################################
def KH_Molar_per_atmosphere(comp,T,S):
    """
    calculate the Henry's law constant in M/atm from Sander data
    applying the salting out factor above
    """
    return (comp["KH"]*np.exp(comp["tVar"]*((1/(T+273.15))-(1/298.15))))/K_H_factor(comp,S)
####################################################################################################################
def D_air(comp,T):
    """
    calculate diffusivity in air [cms/s]
    """
    M_a = 28.97 #molar mass of air
    P = 1 #assuming 1 atmosphere of pressure
    V_a = 20.1 #molar volume of air [cm3/mol]
    M_r = (M_a + comp["mw"]) / (M_a * comp["mw"])
    return (0.001*(T+273.15)**1.75*np.sqrt(M_r))/(P*(V_a**(1/3)+Vb(comp)**(1/3)))**2
###################################################################################################################
def n_air(T):
    """
    dynamic viscosity of saturated air according to Tsiligiris 2008
    """
    SV_0 = 1.715747771e-5
    SV_1 = 4.722402075e-8
    SV_2 = -3.663027156e-10
    SV_3 = 1.873236686e-12
    SV_4 = -8.050218737e-14

    # in N.s/m^2 (Pa.s)
    return SV_0+(SV_1*T)+(SV_2*T**2)+(SV_3*T**3)+(SV_4*T**4)
####################################################################################################################
def p_air(T):
    """
    density of saturated air according to Tsiligiris 2008 in kg/m^3
    """
    SD_0 = 1.293393662
    SD_1 = -5.538444326e-3
    SD_2 = 3.860201577e-5
    SD_3 = -5.2536065e-7
    return SD_0+(SD_1*T)+(SD_2*T**2)+(SD_3*T**3)
######################################################################################################################
def v_air(T):
    """
    calculate kinmatic viscosity of air in cm2/s for Schmidt number calculation
    """
    return 10000*n_air(T)/p_air(T)
#######################################################################################################################
def Sc_air(comp,T):
    """
    calculate the schmidt number of a given gas in air
    """
    return v_air(T)/D_air(comp,T)
########################################################################################################################
def ka(comp,u,T):
    """
    calculate Ka in m/s using the modified jeffrey Ka (Jefferey 2010) which uses smith 1980 Cd
    """
    Cd = 1e-4*(6.1+0.63*u)
    sc = Sc_air(comp,T)
    ra = 13.3*np.sqrt(sc) + np.sqrt(Cd)**(-1) - 5 + np.log(sc)/0.8
    ustar = u*np.sqrt(C_D(u))
    return 1e-3+(ustar/ra)
#####################################################################################################################
def water_side(comp,T,u,S):
    """
    Calculate total transfer velocity from the perspective of water
    """
    rg = 1/(KH(comp,T,S)*ka(comp,u,T))
    rl = 1/kw(comp,T,u,S)
    Kt = 1/(rg+rl)
    return Kt#(kw(comp,T,u,S)**-1+(KH(comp,T,S)*ka(comp,u,T))**-1)**-1
#####################################################################################################################
def air_side(comp,T,u,S):
    """
    Calculate total transfer velocity from the perspective of air 
    """
    rg = 1/ka(comp,u,T)
    rl = KH(comp,T,S)/kw(comp,T,u,S)
    Kt = 1/(rg + rl)
    return Kt#(ka(comp,u,T)**-1 + KH(comp,T,S)/kw(comp,T,u,S))**-1
#####################################################################################################################







































