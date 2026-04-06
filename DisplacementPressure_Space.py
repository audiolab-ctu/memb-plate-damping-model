import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from functionsMembPlate import xi_mean_m, disp_space, press_space, read_space_data, Kz_n, kappaf_m, Norm_Psi, Norm_Phi

#Circular plate properties
R = 300e-6 #18e-3#300e-6 #plate radius [m]
hp = 4e-6 #25e-6#4e-6 #plate thickness [m]
rho_m = 2329 #plate density [kg/m^3]
ms = hp*rho_m #plate mass per unit area [kg/m^2]
E = 170e9 #Young's modulus [Pa]
nu = 0.28 #Poisson's ratio [-]
D = E*hp**3/(12*(1-nu**2)) #flexural rigidity

T = 167 #membrane tension [N/m]

#fluid parameters
rho0=1.18 #air density [kg/m^3] thermoviscous fluid
c0 = 345.9 #adiabatic sound speed [m/s]
mu=1.83e-5 #shear dynamic viscosity [Pa.s]
gamma=1.4 #ratio of specific heats [-]
Cp = 1005 #specific heat at constant pressure [J/(kg.K)]
lamh = 24.4e-3 #thermal conductivity [W/(m.K)]

#Other dimmensions
Rh = 20e-6 #100e-6 #hole radius [m]
Lh = 10e-6 #hole length [m]
hg = 5e-6 #10e-6 #gap thickness [m]
hc = 150e-6 #back cavity thickness [m]
Rc = R #back cavity radius [m]

#input parameters
p_inc = 1 #incident acousticpressure [Pa]

R_modes = 10 #number of radial modes

#Membrane or plate / frequencies
membrane = False #True => membrane, False => plate

#Display Comsol data ?
display_Comsol = True

#SAVE figure or not
save = False #True -> save, False -> not save

#Path to Comsol data
base_dir = Path(__file__).parent
data_path = base_dir / "Comsol_data"
base_path = Path(data_path)

#Comsol data filenames - choose by commenting:

#Comsol result filename membrane Rh=20um, hg=5um
Comsol_space_memb = [['xi_Re_Memb_freq100kHz_Rh20um_hg5um.txt', 'xi_Re_Memb_freq280kHz_Rh20um_hg5um.txt', 'xi_Re_Memb_freq500kHz_Rh20um_hg5um.txt'],
    ['xi_Im_Memb_freq100kHz_Rh20um_hg5um.txt', 'xi_Im_Memb_freq280kHz_Rh20um_hg5um.txt', 'xi_Im_Memb_freq500kHz_Rh20um_hg5um.txt'],
    ['pg_Re_Memb_freq100kHz_Rh20um_hg5um.txt', 'pg_Re_Memb_freq280kHz_Rh20um_hg5um.txt', 'pg_Re_Memb_freq500kHz_Rh20um_hg5um.txt'],
    ['pg_Im_Memb_freq100kHz_Rh20um_hg5um.txt', 'pg_Im_Memb_freq280kHz_Rh20um_hg5um.txt', 'pg_Im_Memb_freq500kHz_Rh20um_hg5um.txt']]

#Comsol result filename membrane Rh=100um, hg=10um
# Comsol_space_memb = [['xi_Re_Memb_freq100kHz_Rh100um_hg10um.txt', 'xi_Re_Memb_freq280kHz_Rh100um_hg10um.txt', 'xi_Re_Memb_freq500kHz_Rh100um_hg10um.txt'],
#      ['xi_Im_Memb_freq100kHz_Rh100um_hg10um.txt', 'xi_Im_Memb_freq280kHz_Rh100um_hg10um.txt', 'xi_Im_Memb_freq500kHz_Rh100um_hg10um.txt'],
#      ['pg_Re_Memb_freq100kHz_Rh100um_hg10um.txt', 'pg_Re_Memb_freq280kHz_Rh100um_hg10um.txt', 'pg_Re_Memb_freq500kHz_Rh100um_hg10um.txt'],
#      ['pg_Im_Memb_freq100kHz_Rh100um_hg10um.txt', 'pg_Im_Memb_freq280kHz_Rh100um_hg10um.txt', 'pg_Im_Memb_freq500kHz_Rh100um_hg10um.txt']]

#Comsol result filename plate Rh=20um, hg=5um
Comsol_space_plate = [['xi_Re_freq100kHz_Rh20um_hg5um_Poly.txt', 'xi_Re_freq280kHz_Rh20um_hg5um_Poly.txt', 'xi_Re_freq1MHz_Rh20um_hg5um_Poly.txt'],
   ['xi_Im_freq100kHz_Rh20um_hg5um_Poly.txt', 'xi_Im_freq280kHz_Rh20um_hg5um_Poly.txt', 'xi_Im_freq1MHz_Rh20um_hg5um_Poly.txt'],
   ['pg_Re_freq100kHz_Rh20um_hg5um_Poly.txt', 'pg_Re_freq280kHz_Rh20um_hg5um_Poly.txt', 'pg_Re_freq1MHz_Rh20um_hg5um_Poly.txt'],
   ['pg_Im_freq100kHz_Rh20um_hg5um_Poly.txt', 'pg_Im_freq280kHz_Rh20um_hg5um_Poly.txt', 'pg_Im_freq1MHz_Rh20um_hg5um_Poly.txt']]

#Comsol result filename plate Rh=100um, hg=10um
# Comsol_space_plate = [['xi_Re_freq100kHz_Rh100um_hg10um_Poly.txt', 'xi_Re_freq280kHz_Rh100um_hg10um_Poly.txt', 'xi_Re_freq1MHz_Rh100um_hg10um_Poly.txt'],
#      ['xi_Im_freq100kHz_Rh100um_hg10um_Poly.txt', 'xi_Im_freq280kHz_Rh100um_hg10um_Poly.txt', 'xi_Im_freq1MHz_Rh100um_hg10um_Poly.txt'],
#      ['pg_Re_freq100kHz_Rh100um_hg10um_Poly.txt', 'pg_Re_freq280kHz_Rh100um_hg10um_Poly.txt', 'pg_Re_freq1MHz_Rh100um_hg10um_Poly.txt'],
#      ['pg_Im_freq100kHz_Rh100um_hg10um_Poly.txt', 'pg_Im_freq280kHz_Rh100um_hg10um_Poly.txt', 'pg_Im_freq1MHz_Rh100um_hg10um_Poly.txt']]

#For display_Comsol = False
Comsol_space_NoComsol =  [['','',''], ['','',''], ['','',''], ['','','']]

if membrane:
    if display_Comsol:
        Comsol_space = [
            [base_path / f for f in row]
            for row in Comsol_space_memb
        ]
    else:
        Comsol_space = Comsol_space_NoComsol
    freq = [1e5, 2.8e5, 5e5]
    label_frqs = ['100 kHz', '280 kHz', '500 kHz']
    MP = 'Membrane'
else:
    if display_Comsol:
        Comsol_space = [
            [base_path / f for f in row]
            for row in Comsol_space_plate
        ]
    else:
        Comsol_space = Comsol_space_NoComsol
    freq = [1e5, 2.8e5, 1e6]
    label_frqs = ['100 kHz', '280 kHz', '1 MHz']
    MP = 'Plate'

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

#plot init
label_fontsize = 16 
other_fontsize = 12
label_font = {'fontname': 'Times New Roman', 'fontsize': label_fontsize}
labels_abcd = ['a)', 'b)', 'c)', 'd)']
plt.rcParams.update({'font.size': other_fontsize})
fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
linestyles = ['solid', 'dashed', 'dotted']
handles = []


#frequency loop
for ii in range(len(freq)):
    xi_mean, xi_m, omega, Chi, C, Ztota, zeta = xi_mean_m(R_modes, R, ms, D, T, p_inc, c0, Rh, Lh, hg, hc, Rc, rho0, mu, gamma, Cp, lamh, freq[ii], membrane, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)

    xi_r, r_vect = disp_space(R_modes, R, xi_m, membrane, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)
    p_r, r_vectp = press_space(R_modes, R, Rh, xi_m, Ztota, C, zeta, Chi, omega, membrane, Kn_pre, kappa_pre, NormPsi_pre, NormPhi_pre)
    xi_Re_r_Num, r_vect_Num = read_space_data(Comsol_space[0][ii])
    xi_Im_r_Num, r_vect_Num = read_space_data(Comsol_space[1][ii])
    p_Re_r_Num, r_vect_Num = read_space_data(Comsol_space[2][ii])
    p_Im_r_Num, r_vect_Num = read_space_data(Comsol_space[3][ii])

    for i, (ax, data, data_Num, ylabel, title) in enumerate(zip(
        axes.flat,
        [(xi_r.real, r_vect), (xi_r.imag, r_vect), (p_r.real, r_vectp), (p_r.imag, r_vectp)],
        [(xi_Re_r_Num, r_vect_Num), (xi_Im_r_Num, r_vect_Num), (p_Re_r_Num, r_vect_Num), (p_Im_r_Num, r_vect_Num)],
        [r"Re$[\xi(r)]$ [m]", r"Im$[\xi(r)]$ [m]", r"Re$[p(r)]$ [Pa]", r"Im$[p(r)]$ [Pa]"],
        ["Real part", "Imaginary part", "Real part", "Imaginary part"]
    )):
        values, r_values = data
        values_Num, r_values_Num = data_Num

        # Symmetrical plot
        ax.plot(r_values_Num, values_Num, color='0.7', linestyle=linestyles[ii], linewidth=3.5)
        ax.plot(-r_values_Num, values_Num, color='0.7', linestyle=linestyles[ii], linewidth=3.5)
        h, = ax.plot(r_values, values, color='black', linestyle=linestyles[ii], linewidth=1.5)
        if i==2:
            handles.append(h)
        ax.plot(-r_values, values, color='black', linestyle=linestyles[ii], linewidth=1.5)
        

        if ii == len(freq)-1:
            ax.axvline(0, color='black', linestyle='-.', linewidth=1) #axis of symmetry
            ax.text(-0.22, 1.1, labels_abcd[i], transform=ax.transAxes,**label_font, va='top', ha='left') #abcd labels
            ax.set_ylabel(ylabel) #y labels
            #titles
            if i < 2:
                ax.set_title(title)
            #shared x labels
            for ax in axes[1]:
                ax.set_xlabel("r [m]")
            #Legend
            axes[1,0].legend(handles, label_frqs, fontsize=10)#, loc='upper right')
            #adjusting the spacing
            fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.2, wspace=0.3)

if save:
    filename = 'xi_p_' + MP + '_Rh'+str(int(Rh*1e6))+'um_hg'+str(int(hg*1e6))+'um_pinc'+str(p_inc)+'Pa_3frqs'
    plt.savefig(filename+".png", dpi=600, bbox_inches='tight')
    plt.savefig(filename+".eps", format='eps', bbox_inches='tight')

plt.show(block=False)
plt.pause(0.001) 
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run
 