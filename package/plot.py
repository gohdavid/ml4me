"""Module to initialize figure styles in Seaborn.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import numpy as np
from pathlib import Path

def get_size_inches(frac_x, frac_y):
    cm_in_inch = 2.54
    prl_col_size = 8.6 / cm_in_inch
    return [frac_x * prl_col_size, frac_y * 1.0 * prl_col_size]

def rcparams():
    mpl.style.use('default')
    plt.style.use(['science', 'nature', 'no-latex'])
    plt.rcParams.update({
        # 'backend': 'ps',
        # 'savefig.format': 'pdf',

        'font.size': 8,
        'font.family': 'sans-serif',
        # 'font.sans-serif': 'MLMSans10',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'text.usetex': False,
        
        'lines.linewidth':2,
        
        'figure.figsize': get_size_inches(1, 2/3),

        'xtick.direction': 'out',
        'xtick.top': False,
        'xtick.bottom': True,
        'xtick.minor.visible': False,
        'xtick.labelsize': 8,
        'xtick.minor.size': 2,
        'xtick.minor.width': 0.5,
        'xtick.major.pad': 3,
        'xtick.major.size': 3,
        'xtick.major.width': 1,
        
        'ytick.direction': 'out',
        'ytick.right': False,
        'ytick.left': True,
        'ytick.minor.visible': False,
        'ytick.labelsize': 8,
        'ytick.direction': 'out',
        'ytick.minor.size': 2,
        'ytick.minor.width': 0.5,
        'ytick.major.pad': 3,
        'ytick.major.size': 3,
        'ytick.major.width': 1,

        'axes.grid': False,
        'axes.edgecolor': 'black',
        'axes.facecolor': 'white',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.titlesize': 8,
        'axes.titlepad': 5,
        'axes.labelsize': 8,
        'axes.linewidth': 1,
        
        'legend.fontsize': 8,
        
        'figure.facecolor': 'white',
        'figure.dpi': 600,
        
        'savefig.transparent': True
    })

sns.lineplot(x=[0,1],y=[0,1])
rcparams()
plt.close()
    
