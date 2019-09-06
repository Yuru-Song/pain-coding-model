import numpy as np 
import matplotlib.pyplot as plt 

def single_trial(Tau_1, Tau_2, Tau_3, Tau_5, a, Pi_1, Pi_2, Pi_3, z_thrshld, evkd_amp, z_init, isplot):
	timestep = 10000
	noise_std_1 = 0.001 #standard deviation of noise 
	noise_std_2 = 0.001
	noise_std_3 = 0.001
	#initialize the array with size: timestep*1 
	e_1 = np.zeros((timestep,1),dtype = np.float32) 
	x = np.zeros((timestep,1),dtype = np.float32)
	z = np.zeros((timestep,1),dtype = np.float32)
	u = np.zeros((timestep,1),dtype = np.float32)
	e_2 = np.zeros((timestep,1),dtype = np.float32)
	v = np.zeros((timestep,1),dtype = np.float32)
	z[0] = z_init
	x[np.int64(.4*timestep):np.int64(.5*timestep)] = x[np.int64(.4*timestep):np.int64(.5*timestep)] + evkd_amp
	#use Euler method to calculate Ordinal Differential Equations
	for i in range(1,timestep):	

		# define tau_4, I chose 3*x[i] because 3 makes results look good. You can try other values.
		Tau_4 = a/(1+np.exp(3.*x[i])) 

		# before the integration of z reaches z_threshold
		if np.sum(z[:i,:]) < z_threshold:
			#if timestep is less than Tau_5, then x(t-Tau_5) = 0 in the original equations.
			if i < Tau_5:
				z[i,0] = z[i-1,0]*(1 - 1./Tau_4) + np.random.normal(0,noise_std_3) #Euler method
			else:
				z[i,0] = z[i-1,0]*(1 - 1./Tau_4) + 1./Tau_4*x[i-Tau_5,0] + np.random.normal(0,noise_std_3) #Euler method
		#after the integration of z reaches z_threshold, set z=0
		else:
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
		plt.plot(t,u,c = 'b',label = 'u')
		plt.plot(t,v,c = 'r',label = 'v')
		plt.plot(t,z,c = 'g',label = 'z')
		plt.plot(t,x, c= 'm',label = 'x')
		plt.legend()
		plt.show()

	return sum(u[1:j]), sum(v[j:-1])

