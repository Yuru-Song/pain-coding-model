import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats.stats import pearsonr 
from pred_coding import single_trial

fig_path = './figures/'
data_path = './data/'

'''
Figure 5 C left, placebo, change  Pi_1 = Pi_3 , while Pi_2 = 1
'''
Tau_1 = 200 # time constant for u
Tau_2 = 300 # delay parameter Delta_u
Tau_3 = 300 # time constant for v
Tau_5 = 300 # delay paramter Delta_x
a  = 4000
smpl_nmbr_Pi_13 = 50
Pi_1 = np.linspace(0, 10, smpl_nmbr_Pi_13)
Pi_2 = 1
Pi_3 = np.linspace(0, 10, smpl_nmbr_Pi_13)
z_threshold = 400
isplot = 0
smpl_nmbr_z_init = 51
z_init = -np.linspace(1,3.,smpl_nmbr_z_init)
evkd_amp = 2
rep_nmbr = 10
sum_u = np.zeros((1,rep_nmbr, smpl_nmbr_z_init, smpl_nmbr_Pi_13),dtype = float)
sum_v = np.zeros((1,rep_nmbr, smpl_nmbr_z_init, smpl_nmbr_Pi_13),dtype = float)
for k in range(rep_nmbr):
	for i in range(smpl_nmbr_z_init):
		for j in range(smpl_nmbr_Pi_13):
			sum_u[0,k,i,j], sum_v[0,k,i,j], wthdrw_time = single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1[j], Pi_2, Pi_3[j], z_threshold, evkd_amp, z_init[i], isplot, i, 0, '')
sio.savemat(data_path + 'figure_5c1.mat',{'sum_u':sum_u,'sum_v':sum_v,'z_init':z_init,'Pi_2':Pi_2})	


'''
Figure 5 C right, placebo, change Pi_2, while Pi_1 = Pi_3 = 1
'''
Tau_1 = 200 # time constant for u
Tau_2 = 300 # delay parameter Delta_u
Tau_3 = 300 # time constant for v
Tau_5 = 300 # delay paramter Delta_x
a  = 4000
Pi_1 = 1.
smpl_nmbr_Pi_2 = 50
Pi_2 = np.linspace(0, 10, smpl_nmbr_Pi_2)
Pi_3 = 1.
z_threshold = 400
isplot = 0
smpl_nmbr_z_init = 51
z_init = -np.linspace(1,3,smpl_nmbr_z_init)
evkd_amp = 2
rep_nmbr = 10
sum_u = np.zeros((1,rep_nmbr, smpl_nmbr_z_init, smpl_nmbr_Pi_2),dtype = float)
sum_v = np.zeros((1,rep_nmbr, smpl_nmbr_z_init, smpl_nmbr_Pi_2),dtype = float)

for k in range(rep_nmbr):
	for i in range(smpl_nmbr_z_init):
		for j in range(smpl_nmbr_Pi_2):
			sum_u[0,k,i,j], sum_v[0,k,i,j], wthdrw_time = single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2[j], Pi_3, z_threshold, evkd_amp, z_init[i], isplot, i, 0, '')

sio.savemat(data_path + 'figure_5c2.mat',{'sum_u':sum_u,'sum_v':sum_v,'z_init':z_init,'Pi_2':Pi_2})	

