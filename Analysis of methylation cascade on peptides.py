# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:41:16 2024

@author: Luis Guerra (luis.guerra@kcl.ac.uk)

Code used to analyse Set8/Suv4-20h1 methylation cascade on H4D24(1-37) and
H4isoD24(1-37) acyl hydrazide peptides.

When run, the code first asks for number of replicates being analysed. Then it
ask the user to upload the files of each replicate in separate rounds. This is
to enable background subtraction in each replicate time course by subtracting
signals at their respective 0 min timepoints.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import tkinter as tk
from tkinter import filedialog

# Function to open the file dialog and get file names
def open_files():
    # This will open a dialog box allowing multiple file selection. This
    # script uses .txt representations of deconvoluted MS spectra without
    # centring as described in the SI
    file_paths = filedialog.askopenfilenames(
        title="Select Files",
        filetypes=[("Text Files", "*.txt")] # Option to select text files
    )
    
    # Return the selected file paths.
    return file_paths

# Create the Tkinter main window
root = tk.Tk()
root.withdraw()  # Hide the Tkinter root window

# Initialize empty lists
times_all = []
fractionMe0_all = []
fractionMe1_all = []
fractionMe2_all = []

# Input number of replicates (3 for both H4D24- and H4isoD24-containing 12-mers)
print('\n')
reps = int(input('Type in number of replicates: '))
print('\n')

# Each set of replicate timepoints is treated separately to enable background
# subtraction using the 0 min timepoints.
for _ in range(reps):
    # Select files in a particular replicate using a GUI file dialog. This
    # script uses .txt representations of deconvoluted and centred MS spectra
    # without centring as described in the SI)
    file_paths = filedialog.askopenfilenames(title='Select File to Open',
                                             filetypes=[("All files", "*.*"), ("Text files", "*.txt")])

    H4me0 = []
    H4me1 = []
    H4me2 = []
    timesFromFile = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        timesFromFile.append(file_name)
        
        # Load the data
        data = pd.read_table(file_path, header=None)
        mass = data[0].round(1).values
        counts = data[1].values

        # Sum the intensities of the first nine peaks in the zero-charge state
        # isotopic clusters for peptides with H4K20me0 (Monoisotopic mass = 3991.4 Da), H4K20me1 
        # (Monoisotopic mass = 4005.4 Da), or H4K20me2 (Monoisotopic mass = 4019.4 Da).
        H4me0.append(counts[np.nonzero(np.in1d(mass, np.arange(3991.4, 4000.4, 1)))[0].astype(int)].sum())
        H4me1.append(counts[np.nonzero(np.in1d(mass, np.arange(4005.4, 4014.4, 1)))[0].astype(int)].sum())
        H4me2.append(counts[np.nonzero(np.in1d(mass, np.arange(4019.4, 4028.4, 1)))[0].astype(int)].sum())

    # Check the loading order of the files.
    for time in timesFromFile:
        print(time)
    print('\n')
        
    # Create array of times to match loading order of the files.    
    times = list(map(int, input('Type in times in minutes separated by spaces, matching order of files (i.e. 10 0 20 if timesFromFile lists 10, 0, and 20 min files in that order): ').split()))
    times_all.extend(times)
    zeroInd = times.index(0)

    # Relative background subtraction
    H4me0 = np.array(H4me0)
    H4me1 = np.array(H4me1)
    H4me2 = np.array(H4me2)
    fractionMe0 = H4me0/(H4me0 + H4me1 + H4me2);
    fractionMe1 = H4me1/(H4me0 + H4me1 + H4me2);
    fractionMe2 = H4me2/(H4me0 + H4me1 + H4me2);
    fractionMe0 = fractionMe0 + fractionMe1[zeroInd] + fractionMe2[zeroInd];
    fractionMe1 = fractionMe1 - fractionMe1[zeroInd];
    fractionMe2 = fractionMe2 - fractionMe2[zeroInd];

    # Append results
    fractionMe0_all.extend(fractionMe0)
    fractionMe1_all.extend(fractionMe1)
    fractionMe2_all.extend(fractionMe2)

# Destroy Tkinter root window
root.destroy()

# Plotting the data
plt.figure()
plt.scatter(times_all, fractionMe0_all, s=60, color=[0, 0.4470, 0.7410], edgecolor=[0, 0, 0], alpha=0.75, label='Me0', zorder = 10, clip_on=False)
plt.scatter(times_all, fractionMe1_all, s=60, color=[0.8500, 0.3250, 0.0980], edgecolor=[0, 0, 0], alpha=0.75, label='Me1', zorder = 10, clip_on=False)
plt.scatter(times_all, fractionMe2_all, s=60, color=[0.9290, 0.6940, 0.1250], edgecolor=[0, 0, 0], alpha=0.75, label='Me2', zorder = 10, clip_on=False)
plt.ylim([-0.02, 1])
plt.xlabel('Time (min)')
plt.ylabel('Relative intensity')
plt.title('H4K20 dimethylation by Set8/SUV4-20H1')
plt.legend()

# Model function for global, nonlinear fitting of Equations S5-S7
def model_func(X, k1, k2, p):
    t = X[:, 0]
    Me0 = X[:, 1] * (p * np.exp(-k1 * t) + (1 - p))
    Me1 = X[:, 2] * p * (k1 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    Me2 = X[:, 3] * p * (1 + ((k1 * np.exp(-k2 * t) - k2 * np.exp(-k1 * t))/(k2 - k1)))

    return Me0 + Me1 + Me2

# Set up concatenated vectors to enable global fits.
Me0_Ind = np.concatenate([np.ones(len(times_all)), np.zeros(len(times_all)), np.zeros(len(times_all))])
Me1_Ind = np.concatenate([np.zeros(len(times_all)), np.ones(len(times_all)), np.zeros(len(times_all))])
Me2_Ind = np.concatenate([np.zeros(len(times_all)), np.zeros(len(times_all)), np.ones(len(times_all))])
X = np.column_stack([np.tile(times_all, 3), Me0_Ind, Me1_Ind, Me2_Ind])
Y = np.concatenate([fractionMe0_all, fractionMe1_all, fractionMe2_all])

# Perform the nonlinear fit.
popt, pcov = curve_fit(model_func, X, Y, p0=[0.01, 0.011, 0.8])

# Fitted rate constants and correction factor
k1, k2, p = popt

# Half-widths of 95% confidence intervals of fitting parameters, assuming normally-distributed errors.
ci = norm.ppf(0.975) * np.sqrt(np.diag(pcov))
ci1 = ci[0]
ci2 = ci[1]
ci3 = ci[2]

# Create anonymous functions for plotting.
Me0_func = lambda t: p * np.exp(-k1 * t) + (1 - p)
Me1_func = lambda t: p * (k1 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t))
Me2_func = lambda t: p * (1 + ((k1 * np.exp(-k2 * t) - k2 * np.exp(-k1 * t))/(k2 - k1)))

# Plot the global, nonlinear fits.
t_fit = np.linspace(0, 90, 200)
plt.plot(t_fit, Me0_func(t_fit), color=[0, 0.4470, 0.7410], linestyle='--', linewidth=1.5)
plt.plot(t_fit, Me1_func(t_fit), color=[0.8500, 0.3250, 0.0980], linestyle='--', linewidth=1.5)
plt.plot(t_fit, Me2_func(t_fit), color=[0.9290, 0.6940, 0.1250], linestyle='--', linewidth=1.5)
plt.xlim([0, 90])
plt.box(True)

# Display fitted parameters and confidence intervals, with former rounded to
# most significant digits of the latter.
ci1_digits = -int(np.floor(np.log10(abs(ci1))))
k1_rounded = np.round(k1, ci1_digits)
ci1_rounded = np.round(ci1, ci1_digits)
ci2_digits = -int(np.floor(np.log10(abs(ci2))))
k2_rounded = np.round(k2, ci2_digits)
ci2_rounded = np.round(ci2, ci2_digits)
ci3_digits = -int(np.floor(np.log10(abs(ci3))))
p_rounded = np.round(p, ci3_digits)
ci3_rounded = np.round(ci3, ci3_digits)
plt.title(('k1 = %.' + str(ci1_digits) + 'f ± %.' + str(ci1_digits) + 'f; k2 = %.'
           + str(ci2_digits) + 'f ± %.' + str(ci2_digits) + 'f; p = %.'
                      + str(ci3_digits) + 'f ± %.' + str(ci3_digits) + 'f') %
          (k1_rounded, ci1_rounded, k2_rounded, ci2_rounded, p_rounded, ci3_rounded))


plt.show()

# Uncomment the following line to output an .eps file of the figure
plt.savefig('MethylationCascade.eps', format='eps')