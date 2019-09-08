from pred_coding import single_trial
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats.stats import pearsonr 

fig_path = './figures/'
data_path = './data/'

'''
Figure 4 C
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
isplot = 0
z_init = 0
smpl_nmbr = 31
trial_nmbr = 10
evkd_amp = np.linspace(2,3.,smpl_nmbr)
sum_u = 0
sum_v = 0
wthdrw_time = np.zeros((smpl_nmbr, trial_nmbr), dtype = float)
# heavy computation :
for i in range(smpl_nmbr):
	for j in range(trial_nmbr):
		print(i,j)
		sum_u, sum_v, wthdrw_time[i,j] = single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_threshold, evkd_amp[i], z_init, isplot, i, 0, '')
sio.savemat(data_path + 'figure_4c.mat',{'wthdrw_time':wthdrw_time, 'evkd_amp':evkd_amp})
plt.errorbar(evkd_amp, np.mean(wthdrw_time, axis = 1), yerr = np.std(wthdrw_time, axis = 1))
plt.savefig(fig_path + 'figure_4c.png')

plt.show()
