"""
Task: find dispersion of polaritons E(k) in energy range [0, 100] meV  for next material:
    n - Ge with next properties: Ne = 4 * 10 ^ 18 [cm ^ (- 3)], electron mobility 'mu' = 1000 [cm ^ 2 / (V * sec)]
    GaAs with next properties: time life of optical phonon 'tau' = 5 [picoseconds]
    n - GaAs with next properties: Ne = 4 * 10 ^ 17 [cm ^ (- 3)], electron mobility 'mu' = 4000 [cm ^ 2 / (V * sec)]
P.S. when calculating consider wave vector k real
"""

import numpy as np
import matplotlib as plt

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
h = 4.1e-15  # Planck constant eV * s
h_ = h / (2 * np.pi)   # Planck constant / 2pi [eV * s]
m0 = 9.1e-28  # electron mass in state of rest
e = 4.8e-10  # elementary charge

# MATERIALS CONSTANTS
'''
Ge N - TYPE CONSTANTS
~~~~~~~~~~~~~~~~~~~~
reference: http://www.matprop.ru/Ge
'''
me_Ge = 0.12 * m0  # effective mass of conductivity
epsilon_inf_Ge = 16.2  # high frequency dielectric constant
mu_Ge = 1000  # electron mobility сm ^ 2 / (V * s)
Ne_Ge = 4e18  # concentration of electrons in n - Ge
tau_Ge = me_Ge * mu_Ge / (1.6 * 10e-19)  # lifetime of electron

'''
GaAs CONSTANTS
~~~~~~~~~~~~~~
reference: http://www.matprop.ru/GaAs
'''
# unalloyed
me_GaAs = 0.063 * m0  # effective electron mass
epsilon_static_GaAs = 12.9  # static dielectric constant
epsilon_inf_GaAs = 10.89  # high frequency dielectric constant
tau_opt_ph = 5 * 10e-12  # lifetime of optical phonon
w_TO = 8.02e12  # Hz
w_LO = 8.55e12  # Hz
# alloyed (n - type)
mu_nGaAs = 4000  # electron mobility сm ^ 2 / (V * s)
Ne_nGaAs = 4 * 10 ** 17  # concentration of electrons in n - Ge
tau_nGaAs = me_GaAs * mu_nGaAs / (1.6 * 10 ** - 19)  # time of electron life

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


def w_p(ne, eps_inf, me):
    mass_const = (4 * np.pi * e ** 2) / me
    w_p_out = np.sqrt(mass_const * ne / eps_inf)
    return w_p_out


def eps(eps_inf, w_plasmon, tau, freq):
    wp2 = w_plasmon ** 2
    w_ = freq * (freq + 1j / tau)
    eps_return = eps_inf * (1.0 - wp2 / w_)
    return eps_return


def opr(eps_inf, gamma, w_lo, w_to, freq):  # ONE - PLASMON RESONANCE
    numerator = w_lo ** 2 - w_to ** 2
    denominator = w_to ** 2 - freq(freq - freq * 1j * gamma)
    opr_return = eps_inf * (1 + numerator / denominator)
    return opr_return


def thd_opr(eps_inf, w_plasma, tau, freq, gamma, w_lo, w_to,):  # Theory Drude + One - Phonon Resonance
    numerator = w_lo ** 2 - w_to ** 2
    denominator = w_to ** 2 - freq(freq - freq * 1j * gamma)
    w_plasma_sqr = w_plasma ** 2
    w_ = freq * (freq + 1j / tau)
    thd_opr_return = eps_inf * (1 + numerator / denominator - w_plasma_sqr / w_)
    return thd_opr_return


# Find wave vector. This need for calculate wavelength of polariton
