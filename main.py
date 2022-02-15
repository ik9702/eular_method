import numpy as np
import scipy
import matplotlib.pyplot as plt
from module import eular_man as em, sci_method as sm


a = 10  #m
b = 10  #b
c = 6   #k
d = 0.1 #e
e = 50  #a
f = 2   #o

X_0 = 0
V_0 = 1
t_0 = 0
t_end = 99
d_t = 10**-3
Vol = int((t_end-t_0+1)/d_t)
para = np.array([[a, b, c, d, e, f], [X_0, V_0, t_0, t_end, d_t, 0]])


s_m = sm.sci_method(para)
e_m = em.eular_method(para)
error = np.zeros((2, Vol))
error[1,:] = np.sqrt((s_m[1]-e_m[1])**2)

error[0,:] = s_m[0]


plt.plot(e_m[0], e_m[1])
plt.plot(s_m[0], s_m[1])
plt.plot(error[0], error[1])



plt.show()










