"""
Task: find dispersion of polaritons E(k) in energy range [0, 100] meV  for next material:
    n - Ge with next properties: Ne = 4 * 10 ^ 18 [cm ^ (- 3)], electron mobility 'mu' = 1000 [cm ^ 2 / (V * sec)]
    GaAs with next properties: time life of optical phonon 'tau' = 5 [picoseconds]
    n - GaAs with next properties: Ne = 4 * 10 ^ 17 [cm ^ (- 3)], electron mobility 'mu' = 4000 [cm ^ 2 / (V * sec)]
P.S. when calculating consider wave vector k real
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

'''
A little theoretical material about polariton: 
Polariton are quasi-particles resulting from strong coupling of electromagnetic waves with an electric or magnetic 
dipole-carrying excitation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this unit of code written all material and world constant that we will use for finding 
polaritons dispersion E(k), where E - energy and k - wave vector of polatiton respectively
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ALL VALUES WRITE IN CGS SYSTEM OF UNIT
'''

# GLOBAL CONSTANTS
h = 4.1 * 10 ** -15  # Planck constant eV * s
h_ = h / (2 * np.pi)   # Planck constant / 2pi [eV * s]
m0 = 9.1 * 10 ** -28  # electron mass in state of rest
e = 4.8 * 10 ** -10  # elementary charge
c = 3 * 10 ** 10  # lightspeed [cm / s]

# MATERIALS CONSTANTS
'''
Ge N - TYPE CONSTANTS
~~~~~~~~~~~~~~~~~~~~~
reference: http://www.matprop.ru/Ge
'''
me_Ge = 0.12 * m0  # effective mass of conductivity
epsilon_inf_Ge = 16.2  # high frequency dielectric constant
mu_Ge = 1000  # electron mobility сm ^ 2 / (V * s)
Ne_Ge = 4 * 10 ** 18  # concentration of electrons in n - Ge

'''
GaAs CONSTANTS
~~~~~~~~~~~~~~
reference: http://www.matprop.ru/GaAs
'''
# unalloyed
me_GaAs = 0.063 * m0  # effective electron mass
epsilon_static_GaAs = 12.9  # static dielectric constant
epsilon_inf_GaAs = 10.89  # high frequency dielectric constant
tau_opt_ph_GaAs = 5 * 10 ** (-12)  # lifetime of optical phonon
w_TO = 2 * np.pi * 8.02 * 10 ** 12  # Hz
w_LO = 2 * np.pi * 8.55 * 10 ** 12  # Hz
# alloyed (n - type)
mu_nGaAs = 4000  # electron mobility сm ^ 2 / (V * s)
Ne_nGaAs = 4 * 10 ** 17  # concentration of electrons in n - Ge

'''
A little theoretical information about the description models used and their comparison with our samples 
-------------------------------------------------------------------------------------------------------- 

Theory DRUDE: 
~~~~~~~~~~~~
 eps(w) = eps_inf(1 - (w_plasma ^ 2 / (w * (w - i/tau)))),
 where eps_inf - dielectric constant (high frequency), w - frequency, tau - lifetime of electron,
 w_plasma - plasma frequency.
 w_plasma = [4 * pi * Ne * e ^ 2 / me] ^ (1 / 2)

Theory One - Phonon Resonance:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 eps(w) = eps_inf(1 + (w_LO ^ 2 - w_T0 ^ 2) / (w_TO ^ 2 - w(w - w * i * tau_phonon))), 
 where w_LO - frequency of longitudinal optical phonon, w_TO - frequency of transverse optical phonon,
 tau_phonon - optical phonon lifetime
 
Theory Drude + One - Phonon Resonance: 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 eps(w) = eps_inf(1 + (w_LO ^ 2 - w_T0 ^ 2) / (w_TO ^ 2 - w(w - w * i * gamma)) - (w_plasma ^ 2 / (w * (w - i/tau))))     
 
 In this unit the write necessary law (expressions, formulas, etc) 
 1) n - Ge: non-polar doped semiconductor absorption occurs on free charge carriers (in our case, electrons, because 
 n is the type of doping). Electromagnetic wave interacts with the crystal lattice ===> using Theory Drude (ThD)
 2) GaAs: polar, undoped semiconductor. An electromagnetic wave interacts with the crystal lattice  
 ===> sing One-Phonon Resonance (OPR)
 3) n - GaAs:polar, doped semiconductor. An electromagnetic wave interacts with both the crystal lattice and free charge 
 carriers (electrons) ===> using OPR + TgD
'''

# Create energy range from 0 to 100 meV
E = np.linspace(0.1, 100, 500) * 10 ** -3  # [eV]
w_real_part = E / h_# E = w * hO
w_real_part_GaAs_low = E[:164] / h_
w_real_part_GaAs_hight = E[:325] / h_
w_real_part_nGaAs_low = E[265:314:1] / h_
w_real_part_nGaAs_high = E[:335] / h_



def image_frequency(freq_real: float, freq_image: float):  # create image frequency
    w_i = freq_real + 1j * freq_image
    return w_i


def tau(m_e, mu):  # Calculation lifetime of electron (SI UNIT)
    e_ = 1.6 * 10 ** -19  # coulon
    m_ = m_e * 10 ** -3  # kg
    mu_ = mu * 10 ** -4  # m ^ 2 / (V * sec)
    tau_out = (m_ * mu_) / e_  # 1 / sec
    return tau_out


tau_Ge = tau(me_Ge, mu_Ge)
tau_nGaAs = tau(me_GaAs, mu_nGaAs)


def w_p(ne, eps_inf, me):  # plasma frequency
    mass_const = (4 * np.pi * e ** 2) / me
    w_p_out = np.sqrt(mass_const * ne / eps_inf)
    return w_p_out


def th_drude(eps_inf, w_plasmon, tau_e,
             freq_real, freq_image: float):  # Theory Drude
    freq_ = image_frequency(freq_real, freq_image)
    wp2 = w_plasmon ** 2
    w_ = freq_ * (freq_ + 1j / tau_e)
    eps_return = eps_inf * (1.0 - wp2 / w_)
    return eps_return


def opr(eps_inf, tau_phonon, w_lo, w_to,
        freq_real: float, freq_image: float):  # ONE - PLASMON RESONANCE
    freq_ = image_frequency(freq_real, freq_image)
    numerator = w_lo ** 2 - w_to ** 2
    denominator = w_to ** 2 - freq_ * (freq_ + 1j / tau_phonon)
    opr_return = eps_inf * (1.0 + numerator / denominator)
    return opr_return


def thd_opr(eps_inf, w_plasma, tau_electron, freq_real: float,
            freq_image: float, tau_phonon, w_lo, w_to):  # Theory Drude + One - Phonon Res
    freq_ = image_frequency(freq_real, freq_image)
    numerator = w_lo ** 2 - w_to ** 2
    denominator = w_to ** 2 - freq_ * (freq_ + 1j / tau_phonon)
    w_plasma_sqr = w_plasma ** 2
    w_ = freq_ * (freq_ + 1j / tau_electron)
    thd_opr_return = eps_inf * (1.0 + (numerator / denominator) - (w_plasma_sqr / w_))
    return thd_opr_return


# Find wave vector. This need for calculate wavelength of polariton
'''
For finding wave vector we solve next equation:
k ^ 2 = (w / c) ^ 2 * eps(w),
where k - wave vector, c - ligthspeed [cm / s], eps(w) - dielectric constant, w - frequency
k = (w / c) * (eps(w)) ^ (1 / 2)
NOTE: w = Re(w) + i * Im(w), Re(w) = E / h_
'''


def real_part_wave_equation(x: float):
    if i == 0:
        w_plasma_germany = w_p(Ne_Ge, epsilon_inf_Ge, me_Ge)
        w_ = np.linspace(w_plasma_germany, E[-1] / h_, 100)
        epsilon = th_drude(epsilon_inf_Ge, w_plasma_germany, tau_Ge, w_, x)
        real_part = np.real(((w_ + 1j * x) / c) * np.sqrt(epsilon))
        return real_part
    elif i == 1:
        epsilon = opr(epsilon_inf_GaAs, tau_opt_ph_GaAs, w_LO, w_TO, w_real_part_GaAs_low, x)
        real_part = np.real(((w_real_part_GaAs_low + 1j * x) / c) * np.sqrt(epsilon))
        return real_part
    elif i == 2:
        epsilon = opr(epsilon_inf_GaAs, tau_opt_ph_GaAs, w_LO, w_TO, w_real_part_GaAs_hight, x)
        real_part = np.real(((w_real_part_GaAs_hight + 1j * x) / c) * np.sqrt(epsilon))
        return real_part
    elif i == 3:
        w_plasma_nGaAs = w_p(Ne_nGaAs, epsilon_inf_GaAs, me_GaAs)
        epsilon = thd_opr(epsilon_inf_GaAs, w_plasma_nGaAs, tau_nGaAs, w_real_part_nGaAs_low, x, tau_opt_ph_GaAs, w_LO,
                          w_TO)
        real_part = np.real(((w_real_part_nGaAs_low + 1j * x) / c) * np.sqrt(epsilon))
        return real_part
    elif i == 4:
        w_plasma_nGaAs = w_p(Ne_nGaAs, epsilon_inf_GaAs, me_GaAs)
        epsilon = thd_opr(epsilon_inf_GaAs, w_plasma_nGaAs, tau_nGaAs, w_real_part_nGaAs_high, x, tau_opt_ph_GaAs, w_LO,
                          w_TO)
        real_part = np.real(((w_real_part_nGaAs_high + 1j * x) / c) * np.sqrt(epsilon))
        return real_part


def image_part_wave_equation(x: float):
    if i == 0:
        w_plasma_germany = w_p(Ne_Ge, epsilon_inf_Ge, me_Ge)
        w_ = np.linspace(w_plasma_germany, E[-1] / h_, 100)
        epsilon = th_drude(epsilon_inf_Ge, w_plasma_germany, tau_Ge, w_, x)
        image_part = np.imag(((w_ + 1j * x) / c) * np.sqrt(epsilon))
        return image_part
    elif i == 1:
        epsilon = opr(epsilon_inf_GaAs, tau_opt_ph_GaAs, w_LO, w_TO, w_real_part_GaAs_low, x)
        image_part = np.imag(((w_real_part_GaAs_low + 1j * x) / c) * np.sqrt(epsilon))
        return image_part
    elif i == 2:
        epsilon = opr(epsilon_inf_GaAs, tau_opt_ph_GaAs, w_LO, w_TO, w_real_part_GaAs_hight, x)
        image_part = np.imag(((w_real_part_GaAs_hight + 1j * x) / c) * np.sqrt(epsilon))
        return image_part
    elif i == 3:
        w_plasma_nGaAs = w_p(Ne_nGaAs, epsilon_inf_GaAs, me_GaAs)
        epsilon = thd_opr(epsilon_inf_GaAs, w_plasma_nGaAs, tau_nGaAs, w_real_part_nGaAs_low, x, tau_opt_ph_GaAs, w_LO,
                          w_TO)
        image_part = np.imag(((w_real_part_nGaAs_low + 1j * x) / c) * np.sqrt(epsilon))
        return image_part
    elif i == 4:
        w_plasma_nGaAs = w_p(Ne_nGaAs, epsilon_inf_GaAs, me_GaAs)
        epsilon = thd_opr(epsilon_inf_GaAs, w_plasma_nGaAs, tau_nGaAs, w_real_part_nGaAs_high, x, tau_opt_ph_GaAs, w_LO,
                          w_TO)
        image_part = np.imag(((w_real_part_nGaAs_high + 1j * x) / c) * np.sqrt(epsilon))
        return image_part


def calculation_wave_vector(freq):
    k = real_part_wave_equation(freq)
    return k


def numerical_diff(x: float):
    h = 1
    diff_out = (image_part_wave_equation(x + h) - image_part_wave_equation(x - h)) / (2 * h)
    return diff_out


for i in range(0, 5):
    if i == 0:
        w_plasma_germany = w_p(Ne_Ge, epsilon_inf_Ge, me_Ge)
        w_ = np.linspace(w_plasma_germany, E[-1] / h_, 100)
        np.savetxt('E_Ge', w_ * h_ * 1000)
        initial_guess = w_
    elif i == 1:
        np.savetxt('E_GaA_low', w_real_part_GaAs_low * h_ * 1000)
        initial_guess = w_real_part_GaAs_low
    elif i == 2:
        np.savetxt('E_GaAs_high', w_real_part_GaAs_hight * h_ * 1000)
        initial_guess = w_real_part_GaAs_hight
    elif i == 3:
        np.savetxt('E_nGaAs_low', w_real_part_nGaAs_low * h_ * 1000)
        initial_guess = w_real_part_nGaAs_low
    elif i == 4:
        np.savetxt('E_nGaAs_high', w_real_part_nGaAs_high * h_ * 1000)
        initial_guess = w_real_part_nGaAs_high
    sol = scipy.optimize.newton(image_part_wave_equation, initial_guess,  maxiter=1000,
                                disp=False, full_output=False)
    print('Image part', image_part_wave_equation(sol), '\n')

    wave_vector = calculation_wave_vector(np.array(sol))
    if i == 0:
        np.savetxt('wave_vector_Ge', wave_vector)
    elif i == 1:
        np.savetxt('wave_vector_GaAs_low', wave_vector)
    elif i == 2:
        np.savetxt('wave_vector_GaAs_high', wave_vector)
    elif i == 3:
        np.savetxt('wave_vector_nGaAs_low', wave_vector)
    elif i == 4:
        np.savetxt('wave_vector_nGaAs_high', wave_vector)
