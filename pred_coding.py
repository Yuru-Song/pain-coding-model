import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from scipy.stats.stats import pearsonr 


def single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_thrshld, evkd_amp, z_init, isplot, fig_num, is_save, save_name):
	timestep = 10000
	noise_std_1 = 0.003 #standard deviation of noise 
	noise_std_2 = 0.003
	noise_std_3 = 0.003
	#initialize the array with size: timestep*1 
	e_1 = np.zeros((timestep,1),dtype = np.float32) 
	x = np.zeros((timestep,1),dtype = np.float32)
	z = np.zeros((timestep,1),dtype = np.float32)
	u = np.zeros((timestep,1),dtype = np.float32)
	e_2 = np.zeros((timestep,1),dtype = np.float32)
	v = np.zeros((timestep,1),dtype = np.float32)
	z[0] = z_init
	x[np.int64(.4*timestep):np.int64(.45*timestep)] = x[np.int64(.4*timestep):np.int64(.45*timestep)] + evkd_amp
	#use Euler method to calculate Ordinal Differential Equations
	is_reset_z = 0
	for i in range(1,timestep):	

		# define tau_4, I chose 3*x[i] because 3 makes results look good. You can try other values.
		Tau_4 = a/(1+np.exp(x[i])) 

		# before the integration of z reaches z_threshold
		if (np.sum(z[max(i-3*Tau_5, 0):i,:]))< z_threshold and is_reset_z < 1:
			#if timestep is less than Tau_5, then x(t-Tau_5) = 0 in the original equations.
			if i < Tau_5:
				z[i,0] = z[i-1,0]*(1 - 1./Tau_4) + np.random.normal(0,noise_std_3) #Euler method
			else:
				z[i,0] = z[i-1,0]*(1 - 1./Tau_4) + 1./Tau_4*x[i-Tau_5,0] + np.random.normal(0,noise_std_3) #Euler method
		#after the integration of z reaches z_threshold, set z=0
		else:
			is_reset_z = 1
			z[i,0] = 0

		e_1[i,0] = np.abs(x[i,0] - z[i,0])
		u[i,0] = (1 - 1./Tau_1)*u[i-1] + Pi_1*e_1[i-1,0]*1./Tau_1 + np.random.normal(0,noise_std_1) #Euler method

		#if timestep is less than Tau_2, then x(t- Tau_2) = 0 in the original equations.
		if i > Tau_2:
			e_2[i,0] = u[i-Tau_2,0] 
			v[i,0] = (1 - 1./Tau_3)*v[i-1] + Pi_2*e_2[i-1,0]*1./Tau_3 + Pi_3*z[i-1,0]*1./Tau_3+np.random.normal(0,noise_std_2) #Euler method
		else:
			v[i,0] = (1 - 1./Tau_3)*v[i-1] + Pi_3*z[i-1,0]*1./Tau_3 + np.random.normal(0,noise_std_2) #Euler method

	#show the withdraw timepoint
	j  = np.max(np.where(z>0))
	print('Paw withdraw at timestep: '+str(j))

	if isplot > 0:
		t = (np.arange(timestep)-j)/1000.
		plt.figure(fig_num)
		plt.plot(t,u,c = 'b',label = 'u')
		plt.plot(t,v,c = 'r',label = 'v')
		plt.plot(t,z,c = 'g',label = 'z')
		plt.plot(t,x, c= 'm',label = 'x')
		plt.legend()
	if is_save > 0:
		sio.savemat(save_name, {'u':u, 'v':v, 'z':z, 'x':x})

	return sum(u[1:j])/j, sum(v[j:-1])/(timestep-j), j

#######################################################################################
# Main part of the program
#######################################################################################


'''
Figure 4 A
'''
Tau_1 = 200 # time constant for u
Tau_2 = 300 # delay parameter Delta_u
Tau_3 = 300 # time constant for v
Tau_5 = 300 # delay paramter Delta_x
a  = 4000
Pi_1 = 1.
Pi_2 = 1.
Pi_3 = 1.
z_threshold = 200
isplot = 1
z_init = 0
evkd_amp = 1
single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_threshold, evkd_amp, z_init, isplot, 1, 1, 'figure_4a.mat')
plt.savefig('figure_4a')



'''
Figure 4 B
# '''
# Tau_1 = 200 # time constant for u
# Tau_2 = 300 # delay parameter Delta_u
# Tau_3 = 300 # time constant for v
# Tau_5 = 300 # delay paramter Delta_x
# a  = 4000
# Pi_1 = 1.
# Pi_2 = 1.
# Pi_3 = 1.
# z_threshold = 500
# isplot = 0
# z_init = 0
# smpl_nmbr = 51
# evkd_amp = np.linspace(1.5,3.,smpl_nmbr)
# sum_u = np.zeros((1,smpl_nmbr),dtype = float)
# sum_v = np.zeros((1,smpl_nmbr),dtype = float)
# # wthdrw_time = np.zeros((1,smpl_nmbr),dtype = float)
# for i in range(smpl_nmbr):
# 	sum_u[0,i], sum_v[0,i],wthdrw_time = single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_threshold, evkd_amp[i], z_init, isplot, i, 0, '')
# sio.savemat('figure_4b.mat',{'sum_u':sum_u,'sum_v':sum_v,'evkd_amp':evkd_amp})
# cm = plt.cm.get_cmap('viridis')
# plt.figure(smpl_nmbr)
# plt.scatter(sum_u.ravel(), sum_v.ravel(), c = evkd_amp,cmap=cm)
# plt.colorbar()



'''
Figure 4 C
'''
# Tau_1 = 200 # time constant for u
# Tau_2 = 300 # delay parameter Delta_u
# Tau_3 = 300 # time constant for v
# Tau_5 = 300 # delay paramter Delta_x
# a  = 4000
# Pi_1 = 1.
# Pi_2 = 1.
# Pi_3 = 1.
# z_threshold = 400
# isplot = 0
# z_init = 0
# smpl_nmbr = 31
# trial_nmbr = 10
# evkd_amp = np.linspace(2,3.,smpl_nmbr)
# sum_u = 0
# sum_v = 0
# wthdrw_time = np.zeros((smpl_nmbr, trial_nmbr), dtype = float)
# for i in range(smpl_nmbr):
# 	for j in range(trial_nmbr):
# 		print(i,j)
# 		sum_u, sum_v, wthdrw_time[i,j] = single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_threshold, evkd_amp[i], z_init, isplot, i, 0, '')
# sio.savemat('figure_4c.mat',{'wthdrw_time':wthdrw_time, 'evkd_amp':evkd_amp})
# plt.errorbar(evkd_amp, np.mean(wthdrw_time, axis = 1), yerr = np.std(wthdrw_time, axis = 1))




'''
Figure 4 D
'''
# Tau_1 = 200 # time constant for u
# Tau_2 = 300 # delay parameter Delta_u
# Tau_3 = 300 # time constant for v
# Tau_5 = 300 # delay paramter Delta_x
# a  = 4000
# Pi_1 = 1.
# Pi_2 = 1.
# Pi_3 = 1.
# z_threshold = 400
# isplot = 1
# z_init = 1
# evkd_amp = 0
# single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_threshold, evkd_amp, z_init, isplot, 1, 1, 'figure_4d.mat')




'''
Figure 4 E
'''
# Tau_1 = 200 # time constant for u
# Tau_2 = 300 # delay parameter Delta_u
# Tau_3 = 300 # time constant for v
# Tau_5 = 300 # delay paramter Delta_x
# a  = 4000
# Pi_1 = 1.
# Pi_2 = 1.
# Pi_3 = 1.
# z_threshold = 400
# isplot = 0
# smpl_nmbr = 51
# z_init = np.linspace(.5,2.,smpl_nmbr)
# evkd_amp = 0
# sum_u = np.zeros((1,smpl_nmbr),dtype = float)
# sum_v = np.zeros((1,smpl_nmbr),dtype = float)
# for i in range(smpl_nmbr):
# 	sum_u[0,i], sum_v[0,i], j = single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_threshold, evkd_amp, z_init[i], isplot, i, 0, '')
# sio.savemat('figure_4e.mat',{'sum_u':sum_u,'sum_v':sum_v,'evkd_amp':evkd_amp})	
# cm = plt.cm.get_cmap('viridis')
# plt.figure(smpl_nmbr)
# plt.scatter(sum_u.ravel(), sum_v.ravel(), c = z_init,cmap=cm)
# plt.colorbar()



'''
Figure 4 F
'''
# Tau_1 = 200 # time constant for u
# Tau_2 = 300 # delay parameter Delta_u
# Tau_3 = 300 # time constant for v
# Tau_5 = 300 # delay paramter Delta_x
# a  = 4000
# Pi_1 = 1.
# Pi_2 = 1.
# Pi_3 = 1.
# z_threshold = 400
# isplot = 1
# z_init = -1
# evkd_amp = 2
# single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_threshold, evkd_amp, z_init, isplot, 1, 1, 'figure_4f.mat')




'''
Figure 4 G
'''
# Tau_1 = 200 # time constant for u
# Tau_2 = 300 # delay parameter Delta_u
# Tau_3 = 300 # time constant for v
# Tau_5 = 300 # delay paramter Delta_x
# a  = 4000
# Pi_1 = 1.
# Pi_2 = .0
# Pi_3 = 1.
# z_threshold = 400
# isplot = 0
# smpl_nmbr = 31
# z_init = -np.linspace(1,3,smpl_nmbr)
# evkd_amp = 2
# sum_u = np.zeros((1,smpl_nmbr),dtype = float)
# sum_v = np.zeros((1,smpl_nmbr),dtype = float)
# for i in range(smpl_nmbr):
# 	sum_u[0,i], sum_v[0,i], j = single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_threshold, evkd_amp, z_init[i], isplot, i, 0, '')
# cm = plt.cm.get_cmap('viridis')
# plt.figure(smpl_nmbr)
# plt.scatter(sum_u.ravel(), sum_v.ravel(), c = z_init,cmap=cm)
# plt.colorbar()
# print(pearsonr(sum_u.ravel(),sum_v.ravel()))
# sio.savemat('figure_4g.mat',{'sum_u':sum_u,'sum_v':sum_v,'z_init':z_init})

# plt.show()
