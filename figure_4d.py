from pred_coding import single_trial
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats.stats import pearsonr 

fig_path = './figures/'
data_path = './data/'

'''
Figure 4 D
'''
Tau_1 = 200 # time constant for u
Tau_2 = 300 # delay parameter Delta_u
Tau_3 = 300 # time constant for v
Tau_5 = 300 # delay paramter Delta_x
a  = 4000
Pi_1 = 1.
Pi_2 = 1.
Pi_3 = 1.
z_threshold = 400
isplot = 1
z_init = 1
evkd_amp = 0
single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_threshold, evkd_amp, z_init, isplot, 1, 1, data_path + 'figure_4d.mat')
plt.savefig(fig_path + 'figure_4d.png')

plt.show()