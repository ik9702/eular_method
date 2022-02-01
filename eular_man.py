import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


sp.init_printing(use_latex=True)

print ("hmm")


A = 10
B = 10
C = 6
D = 0.1
E = 50
F = 2
d_t = 10**-3
rng = 100
vol = int(rng/d_t)


V = np.zeros(vol)
V[0]=1
X = np.zeros(vol)
T = np.zeros(vol)
for t in range(1,vol):
    V[t] = (1-(d_t*B)/A)*V[t-1]+(np.sin(F*(t-1)*d_t)*E/A-C*X[t-1]/A-D*(X[t-1]**3)/A)*d_t

for t in range(1,vol):
    X[t] = X[t-1] + V[t-1]*d_t
    T[t] = d_t*t

t, x = sp.symbols('t x')
x = sp.symbols('f', cls=sp.Function) 
x(t)

deq = sp.Eq(A*x(t).diff(t,2) + B*x(t).diff() + C*x(t) + D*(x(t)**3)-E*sp.sin(F*t), x(t))
print(deq)
eq = sp.dsolve(deq, x(t), ics={x(0):0, sp.diff(x(t),t).subs(t,0):1})

#sp.plot(deq.rhs, (t,0,10))

plt.plot(T, X,'r,')
plt.axis([0,rng,-5,5])
plt.show()


'''
for t in range(vol):
    print(t*d_t, X[t])
    
    잘 되는가??
    
    groung control to major Tom
    '''

