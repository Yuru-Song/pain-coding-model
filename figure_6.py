#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:48:11 2019

@author: helenkemprecos
"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio

fig_path = './figures/'
data_path = './data/'

'''
noise for spontaneous : 0.0009
noise for evoked and placebo = 0.003
'''

def single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_thrshld, evkd_amp, z_init, isplot, fig_num, is_save, save_name, z_constant, u_delay):
    timestep = 10000
    noise_std_1 = 0.0003    # standard deviation of noise 
    noise_std_2 = 0.0003
    noise_std_3 = 0.0003
	#initialize the array with size: timestep*1 
    e_1 = np.zeros((timestep,1),dtype = np.float32) 
    x = np.zeros((timestep,1),dtype = np.float32)
    z = np.zeros((timestep,1),dtype = np.float32)
    u = np.zeros((timestep,1),dtype = np.float32)
    e_2 = np.zeros((timestep,1),dtype = np.float32)
    v = np.zeros((timestep,1),dtype = np.float32)
    x[np.int64(.4*timestep):np.int64(.45*timestep)] = x[np.int64(.4*timestep):np.int64(.45*timestep)] + evkd_amp
	#use Euler method to calculate Ordinal Differential Equations
    is_reset_z = 0
    
    if z_constant != 'none':
        z[0] = z_constant
    else:
        z[0] = z_init
        
    for i in range(1,timestep):	

		# define tau_4, I chose 3*x[i] because 3 makes results look good. You can try other values.
        Tau_4 = a/(1+np.exp(x[i])) 

		# before the integration of z reaches z_threshold
        if z_constant != 'none' and is_reset_z < 1:
            if i < .4*timestep:
                z[i,0] = z_constant
            elif (np.sum(z[max(i-3*Tau_5, 0):i,:])) < z_threshold and is_reset_z < 1:
    			#if timestep is less than Tau_5, then x(t-Tau_5) = 0 in the original equations.
                if i < Tau_5:
                    z[i,0] = z[i-1,0]*(1 - 1./Tau_4) + np.random.normal(0,noise_std_3) #Euler method
                else:
                    z[i,0] = z[i-1,0]*(1 - 1./Tau_4) + 1./Tau_4*x[i-Tau_5,0] + np.random.normal(0,noise_std_3) #Euler method
    		#after the integration of z reaches z_threshold, set z=0
            else:
                is_reset_z = 1
                z[i,0] = 0

        else:
            if (np.sum(z[max(i-3*Tau_5, 0):i,:])) < z_threshold and is_reset_z < 1:
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

    #return u, v, x, z, time
    return sum(u[1:j])/j, sum(v[j:-1])/(timestep-j), j

#######################################################################################
# Main part of the program
#######################################################################################


'''
Figure 6, evoked
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
smpl_nmbr = 25
evkd_amp = np.linspace(2, 4.,smpl_nmbr)
smpl_u = 8               #i
u_delay = np.linspace(50, 400, smpl_u)
rep_nmbr = 10             #j
sum_u = np.zeros((1,rep_nmbr, smpl_nmbr, smpl_u),dtype = float)
sum_v = np.zeros((1,rep_nmbr, smpl_nmbr, smpl_u),dtype = float)
z_constant = 'none'

for k in range(rep_nmbr):
    for i in range(smpl_nmbr):
        for j in range(smpl_u):
            sum_u[0,k,i,j], sum_v[0,k,i,j], j = single_trial(
                    Tau_1, Tau_2, Tau_3, Tau_5, a, 
                    Pi_1, Pi_2, Pi_3, 
                    z_threshold, evkd_amp[i], 
                    z_init, isplot, i, 0, '', 
                    z_constant, u_delay[j])

sio.savemat(data_path + 'figure_6evoked.mat',
            {'sum_u':sum_u,'sum_v':sum_v,
            'z_init':z_init, 'u_delay':u_delay})


'''
Figure 6, spontaneous
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
smpl_nmbr_z_init = 25
z_init = np.linspace(0.7, 1.5, smpl_nmbr_z_init)
evkd_amp = 0
smpl_u = 8               #i
u_delay = np.linspace(50, 400, smpl_u)
rep_nmbr = 10            #j
sum_u = np.zeros((1,rep_nmbr, smpl_nmbr_z_init, smpl_u),dtype = float)
sum_v = np.zeros((1,rep_nmbr, smpl_nmbr_z_init, smpl_u),dtype = float)
z_constant = 'none'

for k in range(rep_nmbr):
    for i in range(smpl_nmbr_z_init):
        for j in range(smpl_u):
            sum_u[0,k,i,j], sum_v[0,k,i,j], j = single_trial(
                    Tau_1, Tau_2, Tau_3, Tau_5, a, 
                    Pi_1, Pi_2, Pi_3, 
                    z_threshold, evkd_amp, 
                    z_init[i], isplot, i, 0, '', 
                    z_constant, u_delay[j])

sio.savemat(data_path + 'figure_6spon.mat',
            {'sum_u':sum_u,'sum_v':sum_v,
            'z_init':z_init, 'u_delay':u_delay})

'''
Figure 6,  placebo
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
smpl_nmbr_z_init = 25
z_init = -np.linspace(3, 1, smpl_nmbr_z_init)
evkd_amp = 2
smpl_u = 8              #i
u_delay = np.linspace(50, 400, smpl_u)
rep_nmbr = 10            #j
sum_u = np.zeros((1,rep_nmbr, smpl_nmbr_z_init, smpl_u),dtype = float)
sum_v = np.zeros((1,rep_nmbr, smpl_nmbr_z_init, smpl_u),dtype = float)
z_constant = 'none'

for k in range(rep_nmbr):
    for i in range(smpl_nmbr_z_init):
        for j in range(smpl_u):
            sum_u[0,k,i,j], sum_v[0,k,i,j], j = single_trial(
                    Tau_1, Tau_2, Tau_3, Tau_5, a, 
                    Pi_1, Pi_2, Pi_3, 
                    z_threshold, evkd_amp, 
                    z_init[i], isplot, i, 0, '', 
                    z_constant, u_delay[j])

sio.savemat(data_path + 'figure_6placebo.mat',
            {'sum_u':sum_u,'sum_v':sum_v,
             'z_init':z_init, 'u_delay':u_delay})


plt.show()
