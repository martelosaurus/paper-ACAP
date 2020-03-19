import numpy as np
import matplotlib.pyplot as plt

t1 = 1.
n = 20
h = t1/n
t_plot = np.linspace(0.,t1,n)

def tvprk4(f,t1,y1):
	def g(t,z):
		return -f(t1-t,z)
	z0 = y1
	y = y1*np.ones(n)
	z = z0*np.ones(n)
	for j in range(0,n-1):
		k1 = h*g(t_plot[j],z[j])
		k2 = h*g(t_plot[j]+.5*h,z[j]+.5*k1)
		k3 = h*g(t_plot[j]+.5*h,z[j]+.5*k2)
		k4 = h*g(t_plot[j]+h,z[j]+k3)
		z[j+1] = z[j]+(k1/6.)+(k2/3.)+(k3/3.)+(k4/6.)
		y[n-j-2] = z[j+1]
	return y 

def f(t,y):
	return -t*y

y_plot = tvprk4(f,t1,np.exp(-.5))
plt.plot(t_plot,y_plot,'.b')
plt.show()
