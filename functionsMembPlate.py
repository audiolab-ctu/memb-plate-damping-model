import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import scipy.special as sps
from alive_progress import alive_bar
from math import pi

# Convention: m, n, l from 1 to R_modes

#Membrane eigennumbers
def Kz_n(n, R):
    z_n = sps.jn_zeros(0, n)
    K_n = z_n[n-1]/R
    return K_n, z_n[n-1]

#Membrane norm
def Norm_Psi(n, R):
    K_n = Kz_n(n, R)[0]
    Norm = np.sqrt(pi)*R*sps.jn(1, K_n*R)
    return Norm

#Membrane eigenfunctions
def Psi_n(r, n, R, Kn_pre, NormPsi_pre):
    K_n = Kn_pre[n]
    Norm = NormPsi_pre[n]
    Psin = sps.jn(0, K_n*r)/Norm
    return Psin

#Derivative of membrane eigenfunctions in r direction
def dPsi_dr_n(r, n, R, Kn_pre, NormPsi_pre):
    K_n = Kn_pre[n]
    Norm = NormPsi_pre[n]
    dPsidrn = -K_n*sps.jn(1, K_n*r)/Norm
    return dPsidrn

#MIntegral of membrane eigenfunction in r direction from 0 to R
def Int_Psi_n(n, R, Kn_pre):
    K_n = Kn_pre[n]
    IntPsi_n = 1/(np.sqrt(pi)*K_n)
    return IntPsi_n

#Integral of membrane eigenfunction in r direction from 0 to Rh
def Int_Psi_Rh_n(Rlim, n, R, Kn_pre, NormPsi_pre):
    k_n = Kn_pre[n]
    Norm = NormPsi_pre[n]
    IntPsi_Rlim_n = Rlim*sps.jv(1, k_n*Rlim)/(Norm*k_n)
    return IntPsi_Rlim_n

#Integral of membrane eigenfunction in r direction from Rh to R
def Int_Psi_Rh_R_m(Rh, n, R, Kn_pre, NormPsi_pre):
    k_n = Kn_pre[n]
    Norm = NormPsi_pre[n]
    IntPsi_n = (R*sps.jv(1, k_n*R) - Rh*sps.jv(1, k_n*Rh))/(Norm*k_n)
    return IntPsi_n

#Plate eigennumbers
def kappaf_m(m, R):
    #Leissa:
    #lam2 = np.array([10.2158, 39.771, 89.104, 158.183, 247.005, 355.568, 483.872, 631.914, 799.702, 987.216])
    #kappa = np.sqrt(lam2[m-1])/R
    #Matlab:
    lam = np.array([3.196220616582541, 6.306437047688423, 9.439499137876405, 12.577130640430655, 15.716438526807485,
                    18.856545522229517, 21.997095157606481, 25.137915406037489, 28.278913109518246, 31.420033447282595,
                    34.561242061850116, 37.702516329133488, 40.843840745108047, 43.985204329896206, 47.126599090148218,
                    50.268019068489792, 53.409459734226090, 56.550917580461665, 59.692389850424462, 62.833874347111291])
    kappa = lam[m-1]/R
    return kappa
    
#Plate norm
def Norm_Phi(m, R):
    kappa_m = kappaf_m(m, R)
    Norm = np.sqrt(pi)*R*np.emath.sqrt((2-(sps.iv(1, kappa_m*R)**2)/(sps.iv(0, kappa_m*R)**2))*(sps.jv(0, kappa_m*R)**2)+(sps.jv(1, kappa_m*R)**2))
    return Norm

#Plate eigenfunctions
def Phi_m(r, m, R, kappa_pre, NormPhi_pre):
    kappa_m = kappa_pre[m]
    Norm = NormPhi_pre[m]
    Phim = (sps.jv(0, kappa_m*r) - sps.iv(0, kappa_m*r)*sps.jv(0, kappa_m*R)/sps.iv(0, kappa_m*R))/Norm
    return Phim

#Integral of plate eigenfunction in r direction from 0 to Rlim 
def Int_Phi_m(Rlim, m, R, kappa_pre, NormPhi_pre):
    kappa_m = kappa_pre[m]
    Norm = NormPhi_pre[m]
    IntPhi_m = Rlim*(sps.jv(1, kappa_m*Rlim) - sps.iv(1, kappa_m*Rlim)*sps.jv(0, kappa_m*R)/sps.iv(0, kappa_m*R))/(Norm*kappa_m)
    return IntPhi_m

#Integral of plate eigenfunction in r direction from Rh to R
def Int_Phi_Rh_R_m(Rh, m, R, kappa_pre, NormPhi_pre):
    kappa_m = kappa_pre[m]
    Norm = NormPhi_pre[m]
    IntPhi_m = (R*sps.jv(1, kappa_m*R) - Rh*sps.jv(1, kappa_m*Rh) - (R*sps.iv(1, kappa_m*R) - Rh*sps.iv(1, kappa_m*Rh))*sps.jv(0, kappa_m*R)/sps.iv(0, kappa_m*R))/(Norm*kappa_m)
    return IntPhi_m

#Integral of both eigenfunctions product from Rh to R
def Int_PsiPhi_Rh_R_nl(n, l, R, Rh, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre):
    K_n = Kn_pre[n]
    kappa_l = kappa_pre[l]
    N_n = NormPsi_pre[n]
    Np_l = NormPhi_pre[l]
    Term1 = (2*K_n*(kappa_l**2)*R*sps.jv(1, K_n*R)*sps.jv(0, kappa_l*R))/(K_n**4 - kappa_l**4)
    Term2 = (Rh*kappa_l*sps.jv(1, kappa_l*Rh)*sps.jv(0, K_n*Rh) - Rh*K_n*sps.jv(0, kappa_l*Rh)*sps.jv(1, K_n*Rh))/(K_n**2 - kappa_l**2)
    Term3 = (-Rh*kappa_l*sps.iv(1, kappa_l*Rh)*sps.jv(0, K_n*Rh) - Rh*K_n*sps.iv(0, kappa_l*Rh)*sps.jv(1, K_n*Rh))/(K_n**2 + kappa_l**2)
    Int_PsiPhi = (Term1 + Term2 - Term3*sps.jv(0, kappa_l*R)/sps.iv(0, kappa_l*R))/(N_n*Np_l)
    return Int_PsiPhi

#Integral of both Membrane eigenfunctions product from Rh to R
def Int_PsiPsi_Rh_R_ml(m, l, R, Rh, Kn_pre, NormPsi_pre):
    k_m = Kn_pre[m]
    k_l = Kn_pre[l]
    NM_m = NormPsi_pre[m]
    NM_l = NormPsi_pre[l]

    if m == l:
        Int_PsiPsi = (R**2*(sps.jv(1, k_l*R))**2 - Rh**2*(sps.jv(0, k_l*Rh))**2 - Rh**2*(sps.jv(1, k_l*Rh))**2)/(2*NM_m*NM_l)
    else:
        Int_PsiPsi = (k_l*Rh*sps.jv(0, k_m*Rh)*sps.jv(1, k_l*Rh) - k_m*Rh*sps.jv(1, k_m*Rh)*sps.jv(0, k_l*Rh) )/(NM_m*NM_l*(k_m**2 - k_l**2))

    return Int_PsiPsi

#Auxiliary variable M_n
def M_n(n, R, zeta, Chi, Kn_pre, NormPsi_pre):
    K_n = Kn_pre[n]
    N_n = NormPsi_pre[n]
    Mn = zeta*4*pi*K_n*R*sps.jv(1, K_n*R)/(N_n*(Chi**2 - K_n**2))
    return Mn

#Auxiliary variable O_mn
def O_mn(m, n, R, Kn_pre, kappa_pre, NormPhi_pre):
    kappa_m = kappa_pre[m]
    Np_m = NormPhi_pre[m]
    K_n = Kn_pre[n]
    Omn = (kappa_m**2)*sps.jv(0, kappa_m*R)/(Np_m*(K_n**4 - kappa_m**4))
    return Omn

#Integration constants alpha_m and beta_m for plate
def alpha_beta_m(m, R, Rh, Zh, C, Zeta, Chi, omega, R_modes, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre):
    L1 = - Chi*sps.jv(1, Chi*R)
    L2 = - Chi*sps.yv(1, Chi*R)
    L3 = - C*Chi*sps.jv(1, Chi*Rh) - sps.jv(0, Chi*Rh)
    L4 = - C*Chi*sps.yv(1, Chi*Rh) - sps.yv(0, Chi*Rh)
    DL = L1*L4 - L2*L3
    Term2Pi = 2*pi*1j*omega*Zh*Int_Phi_m(Rh, m, R, kappa_pre, NormPhi_pre)
    Sum_A = 0
    Sum_B = 0
    for n in range(1,R_modes+1):
        Mn = M_n(n, R, Zeta, Chi, Kn_pre, NormPsi_pre)
        Omn = O_mn(m, n, R, Kn_pre, kappa_pre, NormPhi_pre)
        dPsi_min_Psi = C*dPsi_dr_n(Rh, n, R, Kn_pre, NormPsi_pre) - Psi_n(Rh, n, R, Kn_pre, NormPsi_pre)
        Sum_A = Sum_A + Mn*Omn*(L2*dPsi_min_Psi - L4*dPsi_dr_n(R, n, R, Kn_pre, NormPsi_pre))
        Sum_B = Sum_B + Mn*Omn*(L1*dPsi_min_Psi - L3*dPsi_dr_n(R, n, R, Kn_pre, NormPsi_pre))

    alpha_m = (-L2*Term2Pi + Sum_A)/DL
    beta_m = (L1*Term2Pi - Sum_B)/DL
    return alpha_m, beta_m

#Integration constants alpha_m and beta_m for membrane
def alpha_beta_Memb_m(m, R, Rh, Zh, C, zeta, Chi, omega, R_modes, Kn_pre, NormPsi_pre):
    K_m = Kn_pre[m]
    M_memb = zeta/(Chi**2 - K_m**2)
    L1 = - Chi*sps.jv(1, Chi*R)
    L2 = - Chi*sps.yv(1, Chi*R)
    L3 = - C*Chi*sps.jv(1, Chi*Rh) - sps.jv(0, Chi*Rh)
    L4 = - C*Chi*sps.yv(1, Chi*Rh) - sps.yv(0, Chi*Rh)
    DL = L1*L4 - L2*L3
    
    Term2Pi = 2*pi*1j*omega*Zh*Int_Psi_Rh_n(Rh, m, R, Kn_pre, NormPsi_pre)
    dPsi_min_Psi = C*dPsi_dr_n(Rh, m, R, Kn_pre, NormPsi_pre) - Psi_n(Rh, m, R, Kn_pre, NormPsi_pre)
    alpha_m_x_DL = -L2*Term2Pi + M_memb*(L2*dPsi_min_Psi - L4*dPsi_dr_n(R, m, R, Kn_pre, NormPsi_pre))
    beta_m_x_DL = L1*Term2Pi - M_memb*(L1*dPsi_min_Psi - L3*dPsi_dr_n(R, m, R, Kn_pre, NormPsi_pre))

    return alpha_m_x_DL/DL, beta_m_x_DL/DL

#Auxiliary variable Ph_m
def Ph_m(Rh, m, R, Zh, C, Zeta, Chi, omega, R_modes, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre):
    alpha_m, beta_m = alpha_beta_m(m, R, Rh, Zh, C, Zeta, Chi, omega, R_modes, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)
    Sum_n = 0
    for n in range(1,R_modes+1):
        Mn = M_n(n, R, Zeta, Chi, Kn_pre, NormPsi_pre)
        Omn = O_mn(m, n, R, Kn_pre, kappa_pre, NormPhi_pre)
        Sum_n = Sum_n + Mn*Omn*Psi_n(Rh, n, R, Kn_pre, NormPsi_pre)
    Phm = alpha_m*sps.jv(0, Chi*Rh) + beta_m*sps.yv(0, Chi*Rh) + Sum_n
    return Phm

#Auxiliary variable Ph_m for membrane
def Ph_Memb_m(Rh, m, R, Zh, C, Zeta, Chi, omega, R_modes, Kn_pre, NormPsi_pre):
    alpha_m, beta_m = alpha_beta_Memb_m(m, R, Rh, Zh, C, Zeta, Chi, omega, R_modes, Kn_pre, NormPsi_pre)
    k_m = Kn_pre[m]
    Phm = alpha_m*sps.jv(0, Chi*Rh) + beta_m*sps.yv(0, Chi*Rh) + Psi_n(Rh, m, R, Kn_pre, NormPsi_pre)*Zeta/(Chi**2 - k_m**2)
    return Phm

#<J_0(\chi r)Phi_l(r)> from Rh to R
def Int_JPhi_l(l, R, Rh, Chi, kappa_pre, NormPhi_pre):
    kappa_l = kappa_pre[l]
    Np_l = NormPhi_pre[l]
    Term1 = (2*R*Chi*kappa_l**2*sps.jv(1, Chi*R)*sps.jv(0, kappa_l*R))/(Chi**4 - kappa_l**4)
    Term2 = (Rh*kappa_l*sps.jv(1, kappa_l*Rh)*sps.jv(0, Chi*Rh) - Rh*Chi*sps.jv(0, kappa_l*Rh)*sps.jv(1, Chi*Rh) - R*kappa_l*sps.jv(0, Chi*R)*sps.jv(1, kappa_l*R))/(Chi**2 - kappa_l**2)
    Term3 = (-Rh*kappa_l*sps.iv(1, kappa_l*Rh)*sps.jv(0, Chi*Rh) - Rh*Chi*sps.iv(0, kappa_l*Rh)*sps.jv(1, Chi*Rh) + R*kappa_l*sps.jv(0, Chi*R)*sps.iv(1, kappa_l*R))/(Chi**2 + kappa_l**2)
    IntJPhi = (Term1 + Term2 - Term3*sps.jv(0, kappa_l*R)/sps.iv(0, kappa_l*R))/Np_l
    return IntJPhi

#<J_0(\chi r)Psi_l(r)> from Rh to R
def Int_JPsi_l(l, R, Rh, Chi, Kn_pre, NormPsi_pre):
    k_l = Kn_pre[l]
    NM_l = NormPsi_pre[l]
    Term1 = k_l*Rh*sps.jv(0, Chi*Rh)*sps.jv(1, k_l*Rh)
    Term2 = Chi*Rh*sps.jv(1, Chi*Rh)*sps.jv(0, k_l*Rh)
    Term3 = k_l*R*sps.jv(0, Chi*R)*sps.jv(1, k_l*R)
    IntJPsi = (Term1 - Term2 - Term3)/(NM_l*(Chi**2 - k_l**2))
    return IntJPsi

#<Y_0(\chi r)Phi_l(r)> from Rh to R
def Int_YPhi_l(l, R, Rh, Chi, kappa_pre, NormPhi_pre):
    kappa_l = kappa_pre[l]
    Np_l = NormPhi_pre[l]
    Term1 = (2*R*Chi*kappa_l**2*sps.yv(1, Chi*R)*sps.jv(0, kappa_l*R))/(Chi**4 - kappa_l**4)
    Term2 = (Rh*kappa_l*sps.jv(1, kappa_l*Rh)*sps.yv(0, Chi*Rh) - Rh*Chi*sps.jv(0, kappa_l*Rh)*sps.yv(1, Chi*Rh) - R*kappa_l*sps.yv(0, Chi*R)*sps.jv(1, kappa_l*R))/(Chi**2 - kappa_l**2)
    Term3 = (-Rh*kappa_l*sps.iv(1, kappa_l*Rh)*sps.yv(0, Chi*Rh) - Rh*Chi*sps.iv(0, kappa_l*Rh)*sps.yv(1, Chi*Rh) + R*kappa_l*sps.yv(0, Chi*R)*sps.iv(1, kappa_l*R))/(Chi**2 + kappa_l**2)
    IntYPhi = (Term1 + Term2 - Term3*sps.jv(0, kappa_l*R)/sps.iv(0, kappa_l*R))/Np_l
    return IntYPhi

#<Y_0(\chi r)Psi_l(r)> from Rh to R
def Int_YPsi_l(l, R, Rh, Chi, Kn_pre, NormPsi_pre):
    k_l = Kn_pre[l]
    NM_l = NormPsi_pre[l]
    Term1 = k_l*Rh*sps.yv(0, Chi*Rh)*sps.jv(1, k_l*Rh)
    Term2 = Chi*Rh*sps.yv(1, Chi*Rh)*sps.jv(0, k_l*Rh)
    Term3 = k_l*R*sps.yv(0, Chi*R)*sps.jv(1, k_l*R)
    IntJPsi = (Term1 - Term2 - Term3)/(NM_l*(Chi**2 - k_l**2))
    return IntJPsi

#Plate diagonal matrix UU with D*(kappa_l**4-Kp**4)/(2*pi)
def UU_matrix(R_modes, R, D, Kp, kappa_pre):
    UU = np.zeros((R_modes,R_modes)) #eye matrix - preallocation
    for l in range(1, R_modes+1):
        kappa_l = kappa_pre[l]
        for n in range(1, R_modes+1):
            if l == n:
                UU[l-1, n-1] = D*(Kp**4-kappa_l**4)/(2*pi)
    return UU

#Membrane diagonal matrix UU with T*(K_l**2-Km**2)/(2*pi)
def UU_Memb_matrix(R_modes, R, T, KM, Kn_pre):
    UU = np.zeros((R_modes,R_modes)) #eye matrix - preallocation
    for l in range(1, R_modes+1):
        K_l = Kn_pre[l]
        for n in range(1, R_modes+1):
            if l == n:
                UU[l-1, n-1] = T*(KM**2-K_l**2)/(2*pi)
    return UU

#Vector BB of <p_inc*Phi_l(r)> right side
def BB_vect(R_modes, R, Rh, p_inc, kappa_pre, NormPhi_pre):
    BB = np.zeros((R_modes,1)) #right side - preallocation
    MeanPhiRh = np.zeros((R_modes,1)) #Mean eigenfunction (Rh,R) - preallocation
    MeanPhi = np.zeros((R_modes,1)) #Mean eigenfunction (0,R) - preallocation
    #Right side source vector
    for l in range(1, R_modes+1):
        IntPhi = Int_Phi_m(R, l, R, kappa_pre, NormPhi_pre)
        BB[l-1] = p_inc*IntPhi
        MeanPhiRh[l-1] = 2*Int_Phi_Rh_R_m(Rh, l, R, kappa_pre, NormPhi_pre)/(R**2 - Rh**2)
        MeanPhi[l-1] = 2*IntPhi/(R**2)
 
    return BB, MeanPhiRh.T, MeanPhi.T

#Membrane Vector BB of <p_inc*Psi_l(r)> right side
def BB_Memb_vect(R_modes, R, Rh, p_inc, Kn_pre, NormPsi_pre):
    BB = np.zeros((R_modes,1)) #right side - preallocation
    MeanPsiRh = np.zeros((R_modes,1)) #Mean eigenfunction (Rh,R) - preallocation
    MeanPsi = np.zeros((R_modes,1)) #Mean eigenfunction (0,R) - preallocation
    #Right side source vector
    for l in range(1, R_modes+1):
        IntPsi = Int_Psi_n(l, R, Kn_pre)
        BB[l-1] = p_inc*IntPsi
        MeanPsiRh[l-1] = 2*Int_Psi_Rh_R_m(Rh, l, R, Kn_pre, NormPsi_pre)/(R**2 - Rh**2)
        MeanPsi[l-1] = 2*IntPsi/(R**2)
 
    return BB, MeanPsiRh.T, MeanPsi.T

#Input impedance of the hole and cavity
def func_Impedance_hole_volume(R, Rh, Lh, hc, Rc, c0, rho0, mu, gamma, Cp, lamh, freq):
    """
    Calculate total mechanical, specific, and acoustic impedance of a hole and volume.

    Parameters:
    R - membrane/gap radius [m]
    Rh - hole radius [m]
    Lh - hole length [m]
    Nh - number of holes
    hg - gap thickness [m]
    hc - back cavity thickness [m]
    Rc - back cavity radius [m]
    c0 - adiabatic speed of sound [m/s]
    rho0 - air density [kg/m^3]
    mu - shear dynamic viscosity [Pa.s]
    gamma - ratio of specific heats [-]
    Cp - specific heat at constant pressure [J/(kg.K)]
    lamh - thermal conductivity [W/(m.K)]
    freq - frequency [Hz]

    Returns:
    Ztotm - total mechanical impedance
    Ztots - total specific impedance
    Ztota - total acoustic impedance
    """
    
    omega = 2 * np.pi * freq  # angular frequency [rad/s]
    Sm = np.pi * R**2  # membrane/gap surface [m^2]
    Sh = np.pi * Rh**2  # hole cross-section [m^2]
    Vc = np.pi * Rc**2 * hc  # cavity volume [m^3]
    Sc = 2 * np.pi * Rc * (Rc + hc)  # cavity surface [m^2]
    lh = lamh / (rho0 * c0 * Cp)  # thermal characteristic length [m]
    
    Vcplx = Vc * (1 + (1 - 1j) * (gamma - 1) * Sc * np.sqrt(c0 * lh / omega) / (Vc * np.sqrt(2)))  # equivalent complex volume
        
    # Acoustic impedance of the hole
    lvp=mu/(rho0*c0); #viscous characteristic length [m]
    k0 = omega/c0; #adiabatic sound wave number [1/m]
    kv= np.sqrt(k0/lvp)*(1-1j)/np.sqrt(2) #complex wavenumb.
    Fvh= 1-2*sps.jv(1,kv*Rh)/(kv*Rh*sps.jv(0,kv*Rh))
    Zha = 1j*omega*rho0*Lh/(Fvh*Sh)
    
    # Acoustic compliance of back cavity
    Cva = Vcplx / (rho0 * c0**2)
    ZVca = 1 / (1j * omega * Cva)  # acoustic impedance of back cavity

    #Hole edges correction
    alph = Rh/R
    Z_edgeGap = 1j*omega*rho0*(0.26164-0.353*alph+0.0809*alph**3)/R #added mass
    Z_mech_edge_Cav = (rho0*c0*pi*Rh**2)*(1-sps.jv(1,2*k0*Rh)/(k0*Rh) + 1j*sps.struve(1,2*k0*Rh)/(k0*Rh))
    Z_edge_Cav = Z_mech_edge_Cav/Sh**2
    
    Ztota = Zha + ZVca + Z_edgeGap + Z_edge_Cav  # total acoustic impedance
    Ztots = Ztota * Sm  # total specific impedance
    Ztotm = Ztots * Sm  # total mechanical impedance
    
    return Ztota, Ztots, Ztotm

#Matrix CC - influence of the air-filled system behind the plate
def CC_matrix(R_modes, R, Rh, Lh, hg, hc, Rc, c0, rho0, mu, gamma, Cp, lamh, Chi, Fv, freq, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre):
    omega = 2*pi*freq
    CC = np.zeros((R_modes,R_modes),dtype=complex) #preallocation (or CC matrix in vacuum)
    Ztota = func_Impedance_hole_volume(R, Rh, Lh, hc, Rc, c0, rho0, mu, gamma, Cp, lamh, freq)[0]
    zeta = -(omega**2)*rho0/(Fv*hg)
    C = 2*pi*Rh*hg*Fv*Ztota/(1j*omega*rho0)
    for l in range(1, R_modes+1):
        for m in range(1, R_modes+1):
            alpha_m, beta_m = alpha_beta_m(m, R, Rh, Ztota, C, zeta, Chi, omega, R_modes, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)
            Phm = Ph_m(Rh, m, R, Ztota, C, zeta, Chi, omega, R_modes, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)
            IntPhi_l = Int_Phi_m(Rh, l, R, kappa_pre, NormPhi_pre)
            IJPhi_l = Int_JPhi_l(l, R, Rh, Chi, kappa_pre, NormPhi_pre)
            IYPhi_l = Int_YPhi_l(l, R, Rh, Chi, kappa_pre, NormPhi_pre)
            SumMnOmnIPsiPhinl = 0
            for n in range(1,R_modes+1):
                Mn = M_n(n, R, zeta, Chi, Kn_pre, NormPsi_pre)
                Omn = O_mn(m, n, R, Kn_pre, kappa_pre, NormPhi_pre)
                IPsiPhinl = Int_PsiPhi_Rh_R_nl(n, l, R, Rh, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)
                SumMnOmnIPsiPhinl = SumMnOmnIPsiPhinl + Mn*Omn*IPsiPhinl

            CC[l-1,m-1] = Phm*IntPhi_l + alpha_m*IJPhi_l + beta_m*IYPhi_l + SumMnOmnIPsiPhinl
 
    return CC, C, Ztota, zeta

#Matrix CC for membrane - influence of the air-filled system behind the membrane
def CC_Memb_matrix(R_modes, R, Rh, Lh, hg, hc, Rc, c0, rho0, mu, gamma, Cp, lamh, Chi, Fv, freq, Kn_pre, NormPsi_pre):
    omega = 2*pi*freq
    CC = np.zeros((R_modes,R_modes),dtype=complex) #preallocation (or CC matrix in vacuum)
    Ztota = func_Impedance_hole_volume(R, Rh, Lh, hc, Rc, c0, rho0, mu, gamma, Cp, lamh, freq)[0]
    zeta = -(omega**2)*rho0/(Fv*hg)
    C = 2*pi*Rh*hg*Fv*Ztota/(1j*omega*rho0)
    for l in range(1, R_modes+1):
        for m in range(1, R_modes+1):
            alpha_m, beta_m = alpha_beta_Memb_m(m, R, Rh, Ztota, C, zeta, Chi, omega, R_modes, Kn_pre, NormPsi_pre)
            Phm = Ph_Memb_m(Rh, m, R, Ztota, C, zeta, Chi, omega, R_modes, Kn_pre, NormPsi_pre)
            IntPhi_l = Int_Psi_Rh_n(Rh, l, R, Kn_pre, NormPsi_pre)
            IJPhi_l = Int_JPsi_l(l, R, Rh, Chi, Kn_pre, NormPsi_pre)
            IYPhi_l = Int_YPsi_l(l, R, Rh, Chi, Kn_pre, NormPsi_pre)
            K_m = Kn_pre[m]
            SumMnOmnIPsiPhinl = Int_PsiPsi_Rh_R_ml(m, l, R, Rh, Kn_pre, NormPsi_pre)*zeta/(Chi**2 - K_m**2)
            
            CC[l-1,m-1] = Phm*IntPhi_l + alpha_m*IJPhi_l + beta_m*IYPhi_l + SumMnOmnIPsiPhinl
 
    return CC, C, Ztota, zeta

#Mean displacement and modal coeffs of the moving component, frequency dependent
def xi_mean_m(R_modes, R, ms, D, T, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, freq, membrane, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre):
    #frequency dependent variables
    omega = 2*pi*freq #angular frequency [rad/s]
    Kp = np.sqrt(omega*np.sqrt(ms/D)) #plate wave number [1/m]
    KM = omega*np.sqrt(ms/T) #membrane wave number [1/m]
    k0 = omega/c0; #adiabatic sound wave number [1/m]

    #characteristic lengths
    lh=lamh/(rho0*c0*Cp) #thermal characteristic length [m]
    lvp=mu/(rho0*c0); #viscous characteristic length [m]

    #complex diffusion wavenumbers
    kv=np.sqrt(k0/lvp)*(1-1j)/np.sqrt(2)
    kh=np.sqrt(k0/lh)*(1-1j)/np.sqrt(2)

    #mean particle velocity and temperature change profiles across the air gap thickness
    Fv = 1-(np.tan(kv*hg/2)/(kv*hg/2))
    Fh = 1-(np.tan(kh*hg/2)/(kh*hg/2))

    #complex acoustic pressure wavenumber in the air gap
    Chi=np.emath.sqrt(k0**2*(gamma-(gamma-1)*Fh)/Fv) 
    
    #Matrices and vectors
    xi_m = np.zeros((R_modes,1),dtype=complex) #modal coefficients - preallocation
    
    if membrane:
        BB, MeanPsiRh, MeanPsi = BB_Memb_vect(R_modes, R, Rh, p_inc, Kn_pre, NormPsi_pre)
        UU = UU_Memb_matrix(R_modes, R, T, KM, Kn_pre)
        CC, C, Ztota, zeta = CC_Memb_matrix(R_modes, R, Rh, Lh, hg, hc, Rc, c0, rho0, mu, gamma, Cp, lamh, Chi, Fv, freq, Kn_pre, NormPsi_pre)
        xi_m = np.linalg.solve((UU + CC), BB)
        xi_mean = MeanPsi @ xi_m
        #xi_mean = MeanPsiRh @ xi_m #for condenser microphones
    else:
        BB, MeanPhiRh, MeanPhi = BB_vect(R_modes, R, Rh, p_inc, kappa_pre, NormPhi_pre)
        UU = UU_matrix(R_modes, R, D, Kp, kappa_pre)
        CC, C, Ztota, zeta = CC_matrix(R_modes, R, Rh, Lh, hg, hc, Rc, c0, rho0, mu, gamma, Cp, lamh, Chi, Fv, freq, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)
        xi_m = np.linalg.solve((UU + CC), BB)
        xi_mean = MeanPhi @ xi_m
        #xi_mean = MeanPhiRh @ xi_m #for condenser microphones

    return xi_mean.squeeze(), xi_m.squeeze(), omega, Chi, C, Ztota, zeta

# Mean displacement calculated using Lumped Elements Method (LEM) 
def ximean_LEM(R_modes, R, ms, E, nu, D, T, hp, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, freq, membrane):
    omega = 2*pi*freq #angular frequency [rad/s]
    S = pi*(R**2) #moving component surface [m^2]

    if membrane:
        z1 = sps.jn_zeros(0, 1) #first BesselJ_0 zero
        C1 = 4/((z1**4)*pi*T) # first membrane compliance
        M1 = ((z1**2)*ms*S)/(4) # first membrane mass
        Cinf = 1/(8*T*pi) # quasistatic membrane compliance
        ZC1 = 1/(1j*omega*C1)
        ZM1 = 1j*omega*M1
        Zmemb1 = ZC1 + ZM1 

        C2inf = Cinf - C1 #remaining membrane compliance (see Appendix in [16])
        ZCinf2 = 1/(1j*omega*C2inf)

        Zmove = (Zmemb1*ZCinf2)/(Zmemb1+ZCinf2) # membrane mechanical impedance
    else:
        #Plate lumped elements according to [28]
        alph = 50.265
        bet = 5.78
        c1 = (1-nu**2)*R**2/(alph*E*hp**3) #plate compliance
        m1 = ms*R**2*bet #plate mass
        ZC1 = 1/(1j*omega*c1)
        ZM1 = 1j*omega*m1
        Zmove = ZC1 + ZM1 #plate mechanical impedance

    #Imedances of acoustic elements according to [13,14]
    #Gap
    X0 = R
    beta = np.log(X0/Rh)- 3/4 + (Rh**2)/(X0**2) - (Rh**4)/(4*X0**4) # auxiliary variable
    Rg = (6*mu*pi*(X0**4)*beta)/(hg**3) # gap resistance
    Mg = (rho0*pi*(X0**4)*beta)/(2*hg) #gap mass
    Zg = Rg + 1j*omega*Mg

    #Hole
    Rh_ac = (8*mu*Lh)/(pi*(Rh**4)) # hole resistance
    Mh_ac = (4*rho0*Lh)/(3*pi*(Rh**2)) # hole mass
    Zh = (S**2)*(Rh_ac + 1j*omega*Mh_ac) #transformed hole impedance

    #Back cavity
    Vc = np.pi * Rc**2 * hc  # cavity volume [m^3]
    CV = Vc/(rho0*(c0**2)*S**2) # cavity compliance
    Zc = 1/(1j*omega*CV)

    #total mechanical impedance
    Zmech = Zmove + Zg + Zh + Zc

    #mean displacement
    xi_LEM = -p_inc*S/(1j*omega*Zmech)

    return xi_LEM


#Frequency loop, returns mean displacement and xi_m for all freqs, parallel version
def ximean_frqParallel(R_modes, R, ms, D, T, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, frequencies, membrane):
    #Mean dispacement prealloction
    xi_mean = np.zeros(len(frequencies),dtype=complex)
    xi_m = np.zeros((R_modes, len(frequencies)),dtype=complex)
    #eigennumbers and norms preallocation
    Kn_pre = np.zeros(R_modes+1)
    kappa_pre = np.zeros(R_modes+1)
    NormPsi_pre = np.zeros(R_modes+1)
    NormPhi_pre = np.zeros(R_modes+1)

    for n in range(1,R_modes+1):
        Kn_pre[n] = Kz_n(n, R)[0]
        kappa_pre[n] = kappaf_m(n, R)
        NormPsi_pre[n] = Norm_Psi(n, R)
        NormPhi_pre[n] = Norm_Phi(n, R)

    #frequency loop
    with alive_bar(len(frequencies)) as bar: 
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit each frequency computation as a separate task
            future_to_freq = {executor.submit(xi_mean_m, R_modes, R, ms, D, T, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, freq, membrane, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre): ind_freq for ind_freq, freq in enumerate(frequencies)}
            
            for future in concurrent.futures.as_completed(future_to_freq):
                ind_freq = future_to_freq[future]
                bar()
                try:
                    # Unpack the multiple results from the function
                    xi_mean[ind_freq], xi_m[:,ind_freq], omega, Chi, C, Ztota, zeta = future.result()  # Store all results
                except Exception as exc:
                    print(f"Frequency {frequencies[ind_freq]} generated an exception: {exc}")
  
    return xi_mean, xi_m

#Frequency loop, returns mean displacement and xi_m for all freqs, NON-parallel version
def ximean_frq(R_modes, R, ms, D, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, frequencies, membrane):
    #Mean dispacement
    xi_mean = np.zeros(len(frequencies),dtype=complex)
    xi_m = np.zeros((R_modes, len(frequencies)),dtype=complex)
    #eigennumbers and norms preallocation
    Kn_pre = np.zeros(R_modes+1)
    kappa_pre = np.zeros(R_modes+1)
    NormPsi_pre = np.zeros(R_modes+1)
    NormPhi_pre = np.zeros(R_modes+1)

    for n in range(1,R_modes+1):
        Kn_pre[n] = Kz_n(n, R)[0]
        kappa_pre[n] = kappaf_m(n, R)
        NormPsi_pre[n] = Norm_Psi(n, R)
        NormPhi_pre[n] = Norm_Phi(n, R)

    #frequency loop
    with alive_bar(len(frequencies)) as bar: 
        for ind_freq in range(len(frequencies)):
            bar()
            xi_mean[ind_freq], xi_m[:,ind_freq], omega, Chi, C, Ztota, zeta = xi_mean_m(R_modes, R, ms, D, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, frequencies[ind_freq], membrane, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)

    return xi_mean, xi_m

#Displacement space dependent
def disp_space(R_modes, R, xi_m, membrane, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre):
    r_vect = np.linspace(0, R, 100)
    xi_r = np.zeros((1, len(r_vect)))

    if membrane:
        for n in range(1,R_modes+1):
            xi_r = xi_r + xi_m[n-1]*Psi_n(r_vect, n, R, Kn_pre, NormPsi_pre)
    else:
        for n in range(1,R_modes+1):  
            xi_r = xi_r + xi_m[n-1]*Phi_m(r_vect, n, R, kappa_pre, NormPhi_pre)

    return xi_r.squeeze(), r_vect

#Acoustic pressure space dependent
def press_space(R_modes, R, Rh, xi_m, Ztota, C, zeta, Chi, omega, membrane, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre):
    r_vectRh = np.linspace(Rh, R, 100)
    pg_r = np.zeros((1, len(r_vectRh)))

    if membrane:
        for m in range(1,R_modes+1):
            alpha_m, beta_m = alpha_beta_Memb_m(m, R, Rh, Ztota, C, zeta, Chi, omega, R_modes, Kn_pre, NormPsi_pre)
            k_m = Kn_pre[m]
            Sum_n = zeta*Psi_n(r_vectRh, m, R, Kn_pre, NormPsi_pre)/(Chi**2 - k_m**2)
            pg_r = pg_r + xi_m[m-1]*(alpha_m*sps.jv(0, Chi*r_vectRh) + beta_m*sps.yv(0, Chi*r_vectRh) + Sum_n)
    else:
        for m in range(1,R_modes+1):
            alpha_m, beta_m = alpha_beta_m(m, R, Rh, Ztota, C, zeta, Chi, omega, R_modes, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)
            Sum_n = 0
            for n in range(1,R_modes+1):
                Sum_n = Sum_n + M_n(n, R, zeta, Chi, Kn_pre, NormPsi_pre)*O_mn(m, n, R, Kn_pre, kappa_pre, NormPhi_pre)*Psi_n(r_vectRh, n, R, Kn_pre, NormPsi_pre)
            pg_r = pg_r + xi_m[m-1]*(alpha_m*sps.jv(0, Chi*r_vectRh) + beta_m*sps.yv(0, Chi*r_vectRh) + Sum_n)

 
    ph = pg_r[0,0]
    r_vectp = np.append(0, r_vectRh)
    p_r = np.append(ph, pg_r)
    return p_r.squeeze(), r_vectp


#Plot displacement and pressure
def plot_disp(xi_r, p_r, r_vect, r_vectp, xi_Re_r_Num, xi_Im_r_Num, p_Re_r_Num, p_Im_r_Num, r_vect_Num, save):
    
    label_fontsize = 16 
    other_fontsize = 12
    label_font = {'fontname': 'Times New Roman', 'fontsize': label_fontsize}
    labels_abcd = ['a)', 'b)', 'c)', 'd)']
    plt.rcParams.update({'font.size': other_fontsize})

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

    for i, (ax, data, data_Num, ylabel, title) in enumerate(zip(
        axes.flat,
        [(xi_r.real, r_vect), (xi_r.imag, r_vect), (p_r.real, r_vectp), (p_r.imag, r_vectp)],
        [(xi_Re_r_Num, r_vect_Num), (xi_Im_r_Num, r_vect_Num), (p_Re_r_Num, r_vect_Num), (p_Im_r_Num, r_vect_Num)],
        [r"$\Re[\xi(r)]$ [m]", r"$\Im[\xi(r)]$ [m]", r"$\Re[p(r)]$ [Pa]", r"$\Im[p(r)]$ [Pa]"],
        ["Real part", "Imaginary part", "Real part", "Imaginary part"]
    )):
        values, r_values = data
        values_Num, r_values_Num = data_Num
        
        ax.plot(r_values_Num, values_Num, label="Reference FEM model", color='0.75', linestyle='dashed', linewidth=3)
        ax.plot(-r_values_Num, values_Num, color='0.75', linestyle='dashed', linewidth=3)
        ax.plot(r_values, values, label="Present analytical model", color='black')
        ax.plot(-r_values, values, color='black')
        
        # axis of symmetry
        ax.axvline(0, color='black', linestyle='-.')

        # abcd labels
        ax.text(-0.22, 1.1, labels_abcd[i], transform=ax.transAxes,
        **label_font, va='top', ha='left')

        # Titles
        ax.set_ylabel(ylabel)
        if i < 2:
            ax.set_title(title)

        #Legend
        handles, labels = ax.get_legend_handles_labels()
        if i == 2:
            ax.legend([handles[1],handles[0]], [labels[1],labels[0]], framealpha=1)

        # Real pressure lim
        #if i == 2:
        #   ax.set_ylim(bottom=0)
        
        if i == 0:
            axes[0, 0].set_xlabel("r [m]")
            axes[0, 1].set_xlabel("r [m]")

    for ax in axes[1]:
        ax.set_xlabel("r [m]")

    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.2, wspace=0.3)

    plt.show(block=False)
    plt.pause(0.001) 
    input("hit[enter] to end.")
    plt.close('all') # all open plots are correctly closed after each run

#Plot displacement and pressure at given frequency
def plot_disp_frq(R_modes, R, ms, D, T, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, freq, Comsol_space, save, membrane): 
    #eigennumbers and norms preallocation
    Kn_pre = np.zeros(R_modes+1)
    kappa_pre = np.zeros(R_modes+1)
    NormPsi_pre = np.zeros(R_modes+1)
    NormPhi_pre = np.zeros(R_modes+1)

    for n in range(1,R_modes+1):
        Kn_pre[n] = Kz_n(n, R)[0]
        kappa_pre[n] = kappaf_m(n, R)
        NormPsi_pre[n] = Norm_Psi(n, R)
        NormPhi_pre[n] = Norm_Phi(n, R)

    xi_mean, xi_m, omega, Chi, C, Ztota, zeta = xi_mean_m(R_modes, R, ms, D, T, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, freq, membrane, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)

    xi_r, r_vect = disp_space(R_modes, R, xi_m, membrane, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)
    p_r, r_vectp = press_space(R_modes, R, Rh, xi_m, Ztota, C, zeta, Chi, omega, membrane, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)
    xi_Re_r_Num, r_vect_Num = read_space_data(Comsol_space[0])
    xi_Im_r_Num, r_vect_Num = read_space_data(Comsol_space[1])
    p_Re_r_Num, r_vect_Num = read_space_data(Comsol_space[2])
    p_Im_r_Num, r_vect_Num = read_space_data(Comsol_space[3])
   
    
    plot_disp(xi_r, p_r, r_vect, r_vectp, xi_Re_r_Num, xi_Im_r_Num, p_Re_r_Num, p_Im_r_Num, r_vect_Num, save)

#Plot Mean displacement and xi_m vs frequency
def plot_ximean_m(xi_mean, xi_m, xi_meanLEM, Rh, hg, filenameComs, freq, membrane, save):
    fontsize = 11
    plt.rcParams.update({'font.size': fontsize})
    ticks = np.arange(-pi, 1.1*pi, pi/2)  # from -2π to 2π, step π/2
    labels_phase = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

    freq_num, xi_mean_num = ComsolImport(filenameComs)

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    plt.loglog(freq, abs(xi_meanLEM), color='0.75', linestyle='dashed', linewidth=3, label = 'Lumped-element model')
    plt.loglog(freq_num, abs(xi_mean_num), '.', label = 'Reference FEM model')
    plt.loglog(freq, abs(xi_mean), label = 'Present analytical model')
    plt.grid(True)#, which="both")
    #plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'|$\xi_{mean}$| [m]')
    plt.title('Mean displacement')
    handles, labels = ax1.get_legend_handles_labels()
    plt.legend([handles[2], handles[0], handles[1]], [labels[2], labels[0], labels[1]], framealpha=1)
    if membrane:
        ax1.set_xlim(right=1e6)
        ax1.set_ylim(bottom=1e-13)
    ax = fig.add_subplot(2,1,2, sharex=ax1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels_phase)
    plt.semilogx(freq, np.angle(xi_meanLEM), color='0.75', linestyle='dashed', linewidth=3, label = 'Lumped-element model')
    plt.semilogx(freq_num, np.angle(xi_mean_num), '.', label = 'Reference FEM model')
    plt.semilogx(freq, np.angle(xi_mean), label = 'Present analytical model')
    plt.grid(True)#, which="both")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'phase($\xi_{mean}$) [rad]')
    if membrane:
        ax.set_xlim(right=1e6)

    if save:
        if membrane:
            MP = 'Memb'
        else:
            MP = 'Plate'
        filename = 'xi_mean_' + MP + '_Rh' + str(int(Rh*1e6)) + 'um_hg'+str(int(hg*1e6)) + 'um'
        plt.savefig(filename + ".png", dpi=600, bbox_inches='tight')
        plt.savefig(filename + ".eps", format='eps', bbox_inches='tight')
    
    fig = plt.figure()
    for i in range(xi_m.shape[0]):
        plt.loglog(freq, abs(xi_m[i,:]), label = str(i+1))
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Abs value of Modal coefficients [m]')
    plt.title('Modal coefficients')
    plt.legend(title="modes: ")

    plt.show(block=False)
    plt.pause(0.001) 
    input("hit[enter] to end.")
    plt.close('all') # all open plots are correctly closed after each run

#Plot Mean displacement Analytical, FEM and LEM vs. frequency
def plot_ximean_m_frq(R_modes, R, ms, E, nu, D, T, hp, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, filenameComs, freq, save, membrane):
    xi_mean, xi_m = ximean_frqParallel(R_modes, R, ms, D, T, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, freq, membrane) #parallel version
    #xi_mean, xi_m = ximean_frq(R_modes, R, ms, D, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, freq) #non-parallel version
    xi_meanLEM = ximean_LEM(R_modes, R, ms, E, nu, D, T, hp, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, freq, membrane)

    plot_ximean_m(xi_mean, xi_m, xi_meanLEM, Rh, hg, filenameComs, freq, membrane, save)

# Function to read the data from a file
def read_data(filename):
    try:
        data = np.loadtxt(filename, skiprows=8)  # Skip 9 header lines
        freq = data[:, 0]  # First column is the frequency
        real_parts = data[:, 1]  # 2nd column is the real parts
        imag_parts = data[:, 2]  # 3rd column is the imaginary parts
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        freq = np.nan
        real_parts = np.nan
        imag_parts = np.nan
    return freq, real_parts, imag_parts

# Function to read the space data from a file
def read_space_data(filename):
    try: 
        data = np.loadtxt(filename, skiprows=8)  # Skip 9 header lines
        space = data[:, 0]  # First column is the space coordinate
        space_data = data[:, 1]  # 2nd column is the data column
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        space = np.nan
        space_data = np.nan
    return space_data, space

#Import of Comsol data
def ComsolImport(filenameComs):
 
    freq_num, real, imag = read_data(filenameComs)
    
    xi_num = real + 1j * imag

    return freq_num, xi_num
