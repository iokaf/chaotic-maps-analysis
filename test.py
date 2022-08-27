from chaos_maps import ChaoticMap
from chaos_maps.plotting import ChaoticMapPlot

import matplotlib.pyplot as plt

import numpy as np

def step(x, r): 
    x, y = x
    a, b, c = r

    r1 = x + a * x * np.sin(y)
    r2 = y + b * x + c
    
    return r1, r2

map = ChaoticMap(step)
plotter = ChaoticMapPlot(map)
x0 = (1, -3)
r = (2.7, 1, 0.1)

pam_range = np.arange(2.29, 2.73, 0.001)
p_range = (pam_range, 1, 0.1)
le_dict = plotter.lyapunov_exponent_dict(x0, p_range)

# le_x = le_dict.values
le_x, le_y = zip(*le_dict.values())

fig, ax = plt.subplots(2, 1)
ax[0].plot(pam_range, le_x)
ax[0].axhline(y=0, color='g', linestyle='dashed')
ax[0].ylim(-2.5, 2.5)
ax[1].plot(pam_range, le_y)
ax[1].axhline(y=0, color='g', linestyle='dashed')
ax[1].ylim(-2.5, 2.5)

plt.show()
