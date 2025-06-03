import numpy as np
import matplotlib.pyplot as plt

inches=2.54
plt.rcParams['path.simplify_threshold'] = 0.111111111111
plt.rcParams['agg.path.chunksize'     ] = 10000
plt.rcParams['figure.figsize'         ] = [8.5/inches, 8.5/inches]
plt.rcParams['xtick.labelsize'        ] = 10
plt.rcParams['ytick.labelsize'        ] = 10
plt.rcParams['legend.fontsize'        ] = 9
plt.rcParams['legend.handletextpad'   ] = 0.1  # Set the default padding
plt.rcParams["legend.borderpad"       ] = 0.2
plt.rcParams['legend.handleheight'    ] = 0.5  # the height of the legend handle
plt.rcParams['legend.handlelength'    ] = 0.5  # the height of the legend handle
plt.rcParams['legend.borderaxespad'   ] = 0.1
plt.rcParams['legend.columnspacing'   ] = 0.5
plt.rcParams['font.family'            ] = 'serif'
plt.rcParams['font.serif'             ] = 'dejavuserif'
plt.rcParams['text.usetex'            ] = True
plt.rcParams['mathtext.fontset'       ] = 'dejavuserif'
plt.rcParams['lines.markersize'       ] = 2




def song2016(Muv, z):
    stellarMass = np.zeros_like(Muv, dtype=float)
    mask_z4 = (z == 4)
    mask_z5 = (z == 5)
    mask_z6 = (z == 6)
    mask_z7 = (z == 7)
    mask_z8 = (z == 8)

    stellarMass[mask_z4] = 10**(-0.54 * Muv[mask_z4] - 1.70)
    stellarMass[mask_z5] = 10**(-0.50 * Muv[mask_z5] - 0.90)
    stellarMass[mask_z6] = 10**(-0.50 * Muv[mask_z6] - 1.04)
    stellarMass[mask_z7] = 10**(-0.50 * Muv[mask_z7] - 1.20)
    stellarMass[mask_z8] = 10**(-0.50 * Muv[mask_z8] - 1.50)
    
    return stellarMass

def stefanon2021(Muv,z):
    stellarMass = np.zeros_like(Muv, dtype=float)
    mask_z4 = (z == 4)
    mask_z5 = (z == 5)

    mask_z6 = (z == 6)
    mask_z7 = (z == 7)
    mask_z8 = (z == 8)
    mask_z9 = (z == 9)
    mask_z10= (z ==10)
    # Song 2016    
    stellarMass[mask_z4 ] = 10**(-0.54 * Muv[mask_z4 ] - 1.70)
    stellarMass[mask_z5 ] = 10**(-0.50 * Muv[mask_z5 ] - 0.90)
    # Stefanon 2021
    stellarMass[mask_z6 ] = 10**(-0.57 * Muv[mask_z6 ] - 1.064)
    stellarMass[mask_z7 ] = 10**(-0.49 * Muv[mask_z7 ] - 1.056)
    stellarMass[mask_z8 ] = 10**(-0.49 * Muv[mask_z8 ] - 1.047)
    stellarMass[mask_z9 ] = 10**(-0.46 * Muv[mask_z9 ] - 1.028)
    stellarMass[mask_z10] = 10**(-0.46 * Muv[mask_z10] - 1.023)

    return stellarMass

