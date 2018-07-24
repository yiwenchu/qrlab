# Some plotting defaults

import matplotlib as mpl
import matplotlib.cm
import numpy as np

mpl.rcParams['legend.fontsize'] = 9

# Set colors
Nmax = 10
cmap = mpl.cm.get_cmap(name='spectral')
mpl.rcParams['axes.color_cycle'] = [cmap(i) for i in np.linspace(0, 1.0, Nmax)]

bwr = [(0.0, 0.0, 0.7), (0.1, 0.1, 1.0), (1.0, 1.0, 1.0), (1.0, 0.1, 0.1), (0.7, 0.0, 0.0)]
pcolor_cmap = mpl.colors.LinearSegmentedColormap.from_list('mymap', bwr, gamma=1)
