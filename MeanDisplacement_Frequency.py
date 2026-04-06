import numpy as np
from pathlib import Path
from functionsMembPlate import plot_ximean_m_frq

#Circular plate/membrane properties
R = 300e-6 #plate radius [m]
hp = 4e-6 #plate thickness [m]
rho_m = 2320 #plate density [kg/m^3]
ms = hp*rho_m #plate mass per unit area [kg/m^2]
E = 160e9 #Young's modulus [Pa]
nu = 0.22 #Poisson's ratio [-]
D = E*hp**3/(12*(1-nu**2)) #flexural rigidity

#Equivalent membrane tension
T = 167 #membrane tension [N/m]

#fluid parameters
rho0=1.18 #air density [kg/m^3] thermoviscous fluid
c0 = 345.9 #adiabatic sound speed [m/s]
mu=1.83e-5 #shear dynamic viscosity [Pa.s]
gamma=1.4 #ratio of specific heats [-]
Cp = 1005 #specific heat at constant pressure [J/(kg.K)]
lamh = 24.4e-3 #thermal conductivity [W/(m.K)]

#Other dimmensions
Rh = 20e-6 #central hole radius [m]
Lh = 10e-6 # central hole length [m]
hg = 5e-6 #gap thickness [m]
hc = 150e-6 #back cavity thickness [m]
Rc = R #back cavity radius [m]

#Membrane or plate
membrane = False #True => membrane, False => plate

#input parameters
p_inc = 1 #incident acoustic pressure [Pa]
freq1 = np.logspace(1, 2.99, 10) #frequency [Hz]
freq2 = np.logspace(3, 4.99, 100) #frequency [Hz]
freq3 = np.logspace(5, 5.99, 2000)#200) #frequency [Hz]
if membrane:
    freq4 = np.logspace(6, np.log10(2e6), 100) #frequency [Hz]
else:
    freq4 = np.logspace(6, np.log10(4e6), 1000) #frequency [Hz]
freq = np.concatenate((freq1, freq2, freq3, freq4))

R_modes = 10 #number of radial modes 

#Display Comsol data ?
display_Comsol = True

#SAVE figure or not
save = False #True -> save, False -> not save

#Path to Comsol data
base_dir = Path(__file__).parent
data_path = base_dir / "Comsol_data"
base_path = Path(data_path)

#Rh=20um, hg=5um
filenameComs_Plate = 'xi_mean_Plate_Rh20um_hg5um.txt' 
filenameComs_Memb = 'xi_mean_Memb_Rh20um_hg5um_To2MHz.txt'

#Rh=100um, hg=10um
#filenameComs_Plate = 'xi_mean_Plate_Rh100um_hg10um.txt'
#filenameComs_Memb = 'xi_mean_Memb_Rh100um_hg10um_To2MHz.txt'

#For display_Comsol = False
filename_NoComsol = ''

if membrane:
    if display_Comsol:
        filenameComs = base_path / filenameComs_Memb
    else:
        filenameComs = filename_NoComsol
else:
    if display_Comsol:
        filenameComs = base_path / filenameComs_Plate
    else:
        filenameComs = filename_NoComsol

if __name__ == "__main__":
    plot_ximean_m_frq(R_modes, R, ms, E, nu, D, T, hp, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, filenameComs, freq, save, membrane)

