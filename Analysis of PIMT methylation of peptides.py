"""
@author: Luis Guerra (luis.guerra@kcl.ac.uk)

Code used to analyse PIMT methylation cascade on H4D24(1-37) and
H4isoD24(1-37) acyl hydrazide peptides.

Data for analysis is hard-coded, and running the script will directly produce
a plot similar to Figure S4.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the system of ODEs
def odes(t, y, k1, k2, k3, k4):
    a, b, c, d = y
    da_dt = np.sqrt(k3**2)*c - np.sqrt(k1**2)*a
    db_dt = np.sqrt(k1**2)*a - np.sqrt(k2**2)*b
    dc_dt = np.sqrt(k2**2)*b - (np.sqrt(k3**2) + np.sqrt(k4**2))*c
    dd_dt = np.sqrt(k4**2)*c
    return [da_dt, db_dt, dc_dt, dd_dt]

# Initial conditions for system of ODEs
a0, b0, c0, d0 = 1, 0, 0, 0
y0 = [a0, b0, c0, d0]

# Load data
time_data = np.array([0, 1, 2, 5, 10, 15, 30, 60])  # Minutes
response_data_raw_ad = np.array([33951651.796, 35115106.792, 34412301.811, 27138928.747, 18475335.046, 12171578.174, 4281524.722, 3796377.306])
response_data_raw_b = np.array([1345335.032, 4010142.584, 6153511.237, 13126563.484, 22081910.190, 29406863.251, 32985012.898, 25027087.631])
response_data_raw_c = np.array([1994330.673, 1850983.691, 1901896.087, 1982003.874, 2712109.514, 3963800.654, 8207471.343, 14859425.390])

# Absolute background subtraction.
total_response = response_data_raw_ad + (response_data_raw_b - response_data_raw_b[0]) + (response_data_raw_c - response_data_raw_c[0])
response_data_ad = response_data_raw_ad / total_response
response_data_b = (response_data_raw_b - response_data_raw_b[0]) / total_response
response_data_c = (response_data_raw_c - response_data_raw_c[0]) / total_response

# Concatenate data for global, non-linear fitting of solutions to Equations S1-S4
time_fit = np.tile(time_data, 3)
response_fit = np.concatenate([response_data_ad, response_data_b, response_data_c])

# Helper function for numerically solving system of ODEs given by Equations S1-S4.
def ode_solver(t, k1, k2, k3, k4):
    sol = solve_ivp(odes, [min(t), max(t)], y0, args=tuple([k1, k2, k3, k4]), t_eval=t)
    a_sol, b_sol, c_sol, d_sol = sol.y
    return a_sol + d_sol, b_sol, c_sol  # return [ad], [b], [c]

# Wrapper for non-linear fitting
def model(t, k1, k2, k3, k4):
    t_fit = t[0:8]
    ad_sol, b_sol, c_sol = ode_solver(t_fit, k1, k2, k3, k4)
    return np.concatenate([ad_sol, b_sol, c_sol])

# Initial guesses for rate constants
k0 = [0.1,0.01,0,0.01]

# Perform global, non-linear fit
popt, pcov = curve_fit(model, time_fit, response_fit, p0=k0, bounds=([0, 0, 0, 0],[np.inf, np.inf, np.inf, np.inf]))

# Fitted rate constants.
k1, k2, k3, k4 = popt

# Half-widths of 95% confidence intervals of fitting parameters, assuming normally-distributed errors.
ci = norm.ppf(0.975) * np.sqrt(np.diag(pcov))
ci1 = ci[0]
ci2 = ci[1]
ci3 = ci[2]
ci4 = ci[3]

# Prepare data for plotting
x_fit = np.linspace(0, 60, 61)
ad_fit, b_fit, c_fit = ode_solver(x_fit, *list(popt))

# Plot data with fits
plt.figure()
plt.box(True)
plt.xlabel('Time (min)')
plt.ylabel('Relative intensity')
plt.plot(x_fit, ad_fit, color='r', linestyle='--', linewidth=1.8)
plt.plot(x_fit, b_fit, color='b', linestyle='--', linewidth=1.8)
plt.plot(x_fit, c_fit, color='g', linestyle='--', linewidth=1.8)
plt.scatter(time_data, response_data_ad, s=80, facecolor='r', edgecolor=[0, 0, 0], alpha=0.75, label='isoAsp + Asp', zorder = 10, clip_on=False)
plt.scatter(time_data, response_data_b, s=80, facecolor='b', edgecolor=[0, 0, 0], alpha=0.75, label='Me-isoAsp', zorder = 10, clip_on=False)
plt.scatter(time_data, response_data_c, s=80, facecolor='g', edgecolor=[0, 0, 0], alpha=0.75, label='Snn', zorder = 10, clip_on=False)
plt.xlim([0, 60])
plt.ylim([-0.02, 1])
plt.legend()

# Display fitted parameters and confidence intervals, with former rounded to
# most significant digits of the latter.
ci1_digits = -int(np.floor(np.log10(abs(ci1))))
k1_rounded = np.round(k1, ci1_digits)
ci1_rounded = np.round(ci1, ci1_digits)
ci2_digits = -int(np.floor(np.log10(abs(ci2))))
k2_rounded = np.round(k2, ci2_digits)
ci2_rounded = np.round(ci2, ci2_digits)
ci3_digits = -int(np.floor(np.log10(abs(ci3))))
k3_rounded = np.round(k3, ci3_digits)
ci3_rounded = np.round(ci3, ci3_digits)
ci4_digits = -int(np.floor(np.log10(abs(ci4))))
k4_rounded = np.round(k4, ci4_digits)
ci4_rounded = np.round(ci4, ci4_digits)
plt.title(('k1 = %.' + str(ci1_digits) + 'f ± %.' + str(ci1_digits) + 'f; k2 = %.'
           + str(ci2_digits) + 'f ± %.' + str(ci2_digits) + 'f; k3 = %.' + str(ci3_digits)
           + 'f ± %.' + str(ci3_digits) + 'f; k4 = %.' + str(ci4_digits) + 'f ± %.' + str(ci4_digits) + 'f') %
          (k1_rounded, ci1_rounded, k2_rounded, ci2_rounded,
           k3_rounded, ci3_rounded, k4_rounded, ci4_rounded))

# Uncomment the following line to output an .eps file of the figure
#plt.savefig('PIMTCascade.eps', format='eps', bbox_inches='tight')

plt.show()
