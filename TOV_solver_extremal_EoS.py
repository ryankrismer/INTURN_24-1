#!/usr/bin/env python
# coding: utf-8

# # TOV Solver for an extremal EoS

# In[1]:


# For measuring runtime
import time
start_time = time.time()


# In[2]:


import numpy as np
from scipy.integrate import solve_ivp, odeint, quad
from astropy.constants import G, c
from numpy import pi
import astropy.units as u
import matplotlib.pyplot as plt
import sys
import math
np.set_printoptions(threshold=sys.maxsize)
from scipy import optimize
from sympy import symbols, diff


# In[3]:


# Using natural units
# 197 MeV * fm = 1
# 1 fm = 1 / (197 MeV)
# 3e8 m = 1 s
# 3e8 m / s = 1
# 1 J = 1 kg * (m / s)^2
# 1 J = 1 / 9e16 kg

natural = [(u.MeV, u.fm ** (-1), lambda fm_inverse: 1 / 197 * fm_inverse, lambda MeV: 197 * MeV),
           (u.m, u.s, lambda s: 1 / c.value * s, lambda m: c.value * m),
           (u.kg, u.J, lambda J: c.value ** 2 * J, lambda kg: 1 / c.value ** 2 * kg),
           (u.m / u.s, u.dimensionless_unscaled, lambda dimensionless: 1 / c.value * dimensionless, lambda v: c.value * v),
           (u.fm, u.MeV ** (-1), lambda MeV_inverse: 1 / 197 * MeV_inverse, lambda fm: 197 * fm)
          ]

G = (G.value * (1 * u.m / u.s).to(u.dimensionless_unscaled, equivalencies = natural) ** 2 * (1 * u.m).to(u.fm).to(1 / u.MeV, equivalencies = natural) / (1 * u.kg).to(u.J, equivalencies = natural).to(u.MeV))
print(G)
G = G.value


# ## Functions to find numerical EoS arrays for Fermi gas and strong interaction given some number density (in natural units)

# In[4]:


# Non-rest-mass energy E
def find_E(num_density):
    result = a * (num_density / n_0) ** alpha + b * (num_density / n_0) ** beta
    return result

# Energy density epsilon
def find_epsilon(num_density):
    E = find_E(num_density)
    result = num_density * (E + m_N)
    return result

# Chemical potential
def find_chem_potential(num_density):
    result = a * (alpha + 1) * (num_density / n_0) ** alpha + b * (beta + 1) * (num_density / n_0) ** beta + m_N
    return result

# Pressure
def find_pressure(num_density, epsilon_D):
    epsilon = find_epsilon(num_density)
    chem_potential = find_chem_potential(num_density)

    epsilon_Delta = epsilon_D * u.MeV / (1 * u.fm).to(1 / u.MeV, equivalencies = natural) ** 3
    
    num_density_c = num_density_from_energy_density(epsilon_c.value) * u.MeV ** 3
    chem_potential_c = find_chem_potential(num_density_c)
    pressure_c = - epsilon_c + chem_potential_c * num_density_c
    
    # Piece-wise extremal EoS
    if epsilon.value < epsilon_c.value:
        result = - epsilon + chem_potential * num_density    # Gandolfi nuclear EoS
    else:
        if epsilon.value < epsilon_Delta.value:
            result = pressure_c    # Flat EoS
        else:
            result = pressure_c + (epsilon - epsilon_Delta)    # EoS slope = 1
    
    return result


# In[5]:


# Finding number density as a function of energy density

def energy_density_minus_target_epsilon(n, target_epsilon):
    n = n * u.MeV ** 3
    result = find_epsilon(n).value - target_epsilon
    return result

def num_density_from_energy_density(epsilon):
    b = 10 ** 12
    if energy_density_minus_target_epsilon(0, epsilon) > 0:
        print(f"find_epsilon is {find_epsilon(0 * u.MeV ** 3)} and epsilon is {epsilon}.")
        return
    result = optimize.brentq(energy_density_minus_target_epsilon,
                             0,
                             b,
                             args = (epsilon,))
    return result


# ## Plotting the EoS

# In[6]:


# epsilon = []
# pressure = []

# n = np.logspace(-0.3, 1, num = 200) * n_0

# for num_density in n:
#     epsilon_in_MeV = find_epsilon(num_density).value
#     epsilon_in_MeV_and_fm = epsilon_in_MeV * (1 * u.MeV).to(1 / u.fm, equivalencies = natural) ** 3
#     epsilon.append(epsilon_in_MeV_and_fm.value)
#     pressure_in_MeV = find_pressure(num_density, epsilon_D).value
#     pressure_in_MeV_and_fm = pressure_in_MeV * (1 * u.MeV).to(1 / u.fm, equivalencies = natural) ** 3
#     pressure.append(pressure_in_MeV_and_fm.value)


# In[7]:


# fig, ax = plt.subplots(figsize = (8, 8))

# epsilon_c_MeV_and_fm = epsilon_c * (1 * u.MeV).to(1 / u.fm, equivalencies = natural) ** 3
# epsilon_delta_MeV_and_fm = epsilon_delta * (1 * u.MeV).to(1 / u.fm, equivalencies = natural) ** 3
# ax.set_title("Equation of state", fontsize = 15);
# ax.set_xlabel(r"$\mathrm{Energy\,density}\,\epsilon\,\left(\mathrm{MeV/\,fm^3}\right)$", fontsize = 15);
# ax.set_ylabel(r"$\mathrm{Pressure\,\left(MeV/\,fm^3\right)}$", fontsize = 15);
# ax.set_xlim(0, epsilon_delta_MeV_and_fm.value + 25);
# ax.set_ylim(0, 25);
# ax.plot(epsilon, pressure, label = "EoS");
# ax.axvline(x = epsilon_c_MeV_and_fm.value, color = "red", linewidth = 0.5, label = r"$\epsilon_{c}=150\,\mathrm{MeV/\,fm^3}$");
# ax.axvline(x = epsilon_delta_MeV_and_fm.value, color = "green", linewidth = 0.5, label = r"$\epsilon_{\Delta}=650\,\mathrm{MeV/\,fm^3}$");
# ax.legend(fontsize = 15);

# plt.savefig("Piecewise_EoS.pdf", bbox_inches = "tight");
# plt.savefig("Piecewise_EoS.jpg", bbox_inches = "tight");


# ## Generalized TOV

# In[8]:


# r = distance from center
# m = cumulative mass enclosed within distance r from center
# p = pressure at distance r from center
# epsilon = energy density at distance r from center

# TOV coded to be compatible w/ cgs units, but can't explicitly give units in code b/c solve_ivp() throws errors

def TOV(r, p_and_m, epsilon_D):
    p = p_and_m[0]
    m = p_and_m[1]
    if num_density_from_pressure(p, r, epsilon_D) is None:
        return [0, 0]
    n = num_density_from_pressure(p, r, epsilon_D) * u.MeV ** 3
    epsilon = find_epsilon(n).value
    print(f"number density is {n}")
    rel_factors = (1 + p / epsilon) * (1 + 4 * pi * r ** 3 * p / m) * (1 - 2 * G * m / r) ** (-1)
    p_result = - G * epsilon * m / r ** 2 * rel_factors
    m_result = 4 * pi * r ** 2 * epsilon
    return [p_result, m_result]


# In[9]:


# Finding number density as a function of pressure

def pressure_minus_target_p(n, target_p, epsilon_D):
    n = n * u.MeV ** 3
    result = find_pressure(n, epsilon_D).value - target_p
    return result

def num_density_from_pressure(p, r, epsilon_D):
    b = 10 ** 8
    if pressure_minus_target_p(0, p, epsilon_D) > 0:
        print(f"Distance from center is {(r / u.MeV).to(u.fm, equivalencies = natural).to(u.km)}, w/ find_pressure {find_pressure(0 * u.MeV ** 3, epsilon_D)} and p {p}.")
        return
    result = optimize.brentq(pressure_minus_target_p,
                             0,
                             b,
                             args = (p, epsilon_D))
    return result


# ### Setting central cumulative mass initial condition

# In[10]:


def M_from_r_and_rho(r, rho):
    result = 4 * pi * rho * r ** 3 / 3
    return result


# ### TOV solver function

# In[11]:


# Keep this for all EoS

def solve_TOV(n_central, p_central, epsilon_D):
    epsilon_central = find_epsilon(n_central * u.MeV ** 3).value
    m_central = M_from_r_and_rho(small_r, epsilon_central)
    def reached_surface(t, y, *args):
        return y[0]

    reached_surface.terminal = True
    reached_surface.direction = -1
    
    solution = solve_ivp(TOV,
                         [small_r, R_attempt],
                         [p_central, m_central],
                         events = reached_surface,
                         args = (epsilon_D,)
                        )

    distance_from_center = (solution.t / u.MeV).to(u.fm, equivalencies = natural).to(u.km)
    print(f"Mass is {(solution.y[1][-1] * u. MeV).to(u.J).to(u.kg, equivalencies = natural).to(u.solMass):.2f}.")
    cumulative_mass = (solution.y[1] * u.MeV).to(u.J).to(u.kg, equivalencies = natural).to(u.solMass)
    pressure = solution.y[0]

    result = distance_from_center, cumulative_mass, pressure
    return result


# ### Setting constants for solving TOV

# In[12]:


# Everything in natural units unless otherwise specified
# r_small cannot be 0 b/c it would cause a singularity.

small_r = (0.1 * u.km).to(u.fm).to(1 / u.MeV, equivalencies = natural).value
m_N = 939.565 * u.MeV    # Natural units
n_0 = 0.16 / (1 * u.fm).to(1 / u.MeV, equivalencies = natural) ** 3    # Nuclear saturation density

# Setting nuclear parameters (from Gandolfi et. al.)
a = 13.0 * u.MeV
alpha = 0.49
b = 3.21 * u.MeV
beta = 2.47

# Setting critical energy density corresponding to n_0 for piece-wise EoS
epsilon_c_value = 150    # In MeV / fm^3
epsilon_c = epsilon_c_value * u.MeV / (1 * u.fm).to(1 / u.MeV, equivalencies = natural) ** 3
# Note: need to specify epsilon_delta before solving TOV for piece-wise EoS


# ### Horizontal axis scale

# In[13]:


# Everything in natural units unless otherwise specified

R_attempt = (100 * u.km).to(u.fm).to(1 / u.MeV, equivalencies = natural).value


# ### Solving TOV example

# In[14]:


# # Nuclear saturation number density is 0.16 fm^(-3)
# n_central = 7 * n_0
# print(f"{n_0:.3e}")
# p_central = find_pressure(n_central, epsilon_D).value
# print(f"p_central is {p_central}")
# distance_from_center, cumulative_mass, pressure = solve_TOV(n_central.value, p_central)


# In[15]:


# fig, [ax1, ax2] = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 8))

# ax1.set_title("Cumulative mass vs distance from center", fontsize = 18);
# ax1.set_xlabel("Distance from center (km)", fontsize = 18);
# ax1.set_ylabel(r"Cumulative mass ($\mathrm{M_{\odot}}$)", fontsize = 18);
# # ax1.set_xlim(0, 20);
# # ax1.set_ylim(0, 0.6);
# ax1.plot(distance_from_center, cumulative_mass);
# ax2.set_title("Pressure vs distance from center", fontsize = 18);
# ax2.set_xlabel("Distance from center (km)", fontsize = 18);
# ax2.set_ylabel(r"Pressure $\left(\mathrm{MeV/\,fm^3}\right)$", fontsize = 18);
# # ax2.set_xlim(0, 20);
# ax2.plot(distance_from_center, pressure);


# In[16]:


# print(f"Mass is {np.max(cumulative_mass):.2f}.")


# ## Finding radius for $2.00\,M_{\odot}$

# In[17]:


# Function to find mass and radius for particular number density
def find_mass_radius(num_density, epsilon_D):
    # Solving TOV
    p_central = find_pressure(num_density, epsilon_D).value
    distance_from_center, cumulative_mass, _ = solve_TOV(num_density.value, p_central, epsilon_D)
    print("Solved TOV")

    # Removing NaN and inf values
    cumulative_mass = cumulative_mass[~np.isnan(cumulative_mass)]
    cumulative_mass = cumulative_mass[~np.isinf(cumulative_mass)]

    # Picking out single values
    mass = cumulative_mass[-1].value
    radius = distance_from_center[-1].value
    
    return mass, radius


# In[18]:


# Function to find mass residual and closest density
def find_mass_residual(mass_low, mass_high, n_low, n_high):
    if target_mass - mass_low < mass_high - target_mass:
        mass_closest = mass_low
        n_closest = n_low
    else:
        mass_closest = mass_high
        n_closest = n_high
    
    mass_residual = target_mass - mass_closest

    return mass_residual, n_closest


# In[19]:


# Setting initial densities
n_low = n_0    # Undershoot
n_high = 8 * n_0    # Overshoot


# In[20]:


# Function to find the radius of a 2.00 solar mass neutron star using linear interpolation
def find_radius(epsilon_D):
    # Specifying that we're working w/ external variables
    global n_low
    global n_high
    
    print(f"n_low is {n_low}")
    print(f"n_high is {n_high}")
    
    # Initial mass and radius outputs
    mass_low, _ = find_mass_radius(n_low, epsilon_D)
    mass_high, _ = find_mass_radius(n_high, epsilon_D)
    
    # Ensuring there's not a problem
    if target_mass - mass_low < 0:
        print("Error: mass_low greater than target mass")
        raise KeyboardInterrupt
    
    while mass_high - target_mass < 0:
        print("Error: mass_high less than target mass")
        n_high += n_0
    
    # Finding initial mass residual and closest density
    mass_residual, n_closest = find_mass_residual(mass_low, mass_high, n_low, n_high)
    
    # Repeating until 2.00 solar masses
    while np.abs(mass_residual) > 0.005:
        # Updating density input using linear interpolation
        slope = (n_high - n_low) / (mass_high - mass_low)
        n_attempt = slope * mass_residual + n_closest
    
        # Finding new mass and radius outputs
        mass_result, radius_result = find_mass_radius(n_attempt, epsilon_D)
        print(f"radius result is {radius_result}")
    
        # Updating interpolation values
        if mass_result > mass_low and mass_result < target_mass:
            mass_low = mass_result
            n_low = n_attempt
        else:
            if mass_result < mass_high and mass_result > target_mass:
                mass_high = mass_result
                n_high = n_attempt
            # Just in case it goes over the parabola peak
            else:
                if mass_result > mass_high:
                    mass_high = mass_result
                    n_high = n_attempt
    
        # Updating mass residual and closest density
        mass_residual, n_closest = find_mass_residual(mass_low, mass_high, n_low, n_high)
    
    # Recording radius for 2.00 solar masses
    radius = radius_result * u.km
    print(f"The radius is {radius:.3f}.")
    return radius


# In[21]:


# Setting target mass
target_mass = 2.00    # In solar masses


# ## Minimizing radius of $2.00\,M_{\odot}$ neutron star

# In[22]:


# Setting piece-wise EoS parameters to check
epsilon_low = epsilon_c_value    # This would mean no flat segment
epsilon_delta = np.linspace(epsilon_low, epsilon_low * 10)


# In[23]:


# Creating empty radii array to be populated during iterations
radii = []

# Iterating through EoS parameter array to find radii
for epsilon_delta_value in epsilon_delta:
    radii.append(find_radius(epsilon_delta_value))


# In[ ]:


# Plotting radius vs EoS parameter

fig, ax = plt.subplots(figsize = (8, 8))

ax.set_title(r"Radius vs EoS parameter - piecewise EoS, $2.00\,M_{\odot}$",
             fontsize = 15
            );

ax.set_xlabel(r"$\epsilon_{\Delta}$", fontsize = 15);
ax.set_ylabel("Radius (km)", fontsize = 15);
ax.plot(epsilon_delta, radii, "--o");

# Saving plots if result is notable
plt.savefig("radius_vs_epsilon_delta.jpg", bbox_inches = "tight");
plt.savefig("radius_vs_epsilon_delta.pdf", bbox_inches = "tight");


# In[ ]:


# Finding minimum radius and EoS parameter
min_radius = np.min(radii)
epsilon_delta_constraint = epsilon_delta[np.where(radii == min_radius)] * u.MeV / (1 * u.fm) ** 3
print(f"The minimum radius is {min_radius:.3f}.")
print(f"The value of epsilon_delta that minimizes radius is {epsilon_delta_constraint:.3f}.")


# In[ ]:


# Calculating runtime
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

