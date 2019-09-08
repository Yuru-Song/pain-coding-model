from pred_coding import single_trial
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats.stats import pearsonr 

fig_path = './figures/'
data_path = './data/'

'''
Figure 4 G
'''
Tau_1 = 200 # time constant for u
Tau_2 = 300 # delay parameter Delta_u
Tau_3 = 300 # time constant for v
Tau_5 = 300 # delay paramter Delta_x
a  = 4000
Pi_1 = 1.
Pi_2 = .0
Pi_3 = 1.
z_threshold = 400
isplot = 0
smpl_nmbr = 31
z_init = -np.linspace(1,3,smpl_nmbr)
evkd_amp = 2
sum_u = np.zeros((1,smpl_nmbr),dtype = float)
sum_v = np.zeros((1,smpl_nmbr),dtype = float)
for i in range(smpl_nmbr):
	sum_u[0,i], sum_v[0,i], j = single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_threshold, evkd_amp, z_init[i], isplot, i, 0, '')
cm = plt.cm.get_cmap('viridis')
plt.figure(smpl_nmbr)
plt.scatter(sum_u.ravel(), sum_v.ravel(), c = z_init,cmap=cm)
plt.colorbar()
plt.savefig(fig_path + '/figure_4g.png')
print(pearsonr(sum_u.ravel(),sum_v.ravel()))
sio.savemat(data_path+'figure_4g.mat',{'sum_u':sum_u,'sum_v':sum_v,'z_init':z_init})
