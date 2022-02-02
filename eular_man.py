import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


sp.init_printing(use_latex=True)

print ("hmm")


A = 10  #m
B = 10  #b
C = 6   #k
D = 0.1 #e
E = 50  #a
F = 2   #o

X0 = 1
V0 = 0
d_t = 10**-3
rng = 100
vol = int(rng/d_t)


V = np.zeros(vol)
V[0]=V0
X = np.zeros(vol)
X[0]=X0
T = np.zeros(vol)
for t in range(1,vol):
    V[t] = (1-(d_t*B)/A)*V[t-1]+(np.sin(F*(t-1)*d_t)*E/A-C*X[t-1]/A-D*(X[t-1]**3)/A)*d_t
    print("V 생성중...", int(100*t/vol),"%")
print("V 생성중...", 100,"%")    
print("완료")

for t in range(1,vol):
    X[t] = X[t-1] + V[t-1]*d_t
    T[t] = d_t*t
    print("X 생성중...", int(100*t/vol),"%")
print("X 생성중...", 100,"%")    
print("완료")    

# t, x = sp.symbols('t x')
# x = sp.symbols('f', cls=sp.Function) 
# x(t)

# deq = sp.Eq(A*x(t).diff(t,2) + B*x(t).diff() + C*x(t) + D*(x(t)**3)-E*sp.sin(F*t), x(t))
# print(deq)
# eq = sp.dsolve(deq, x(t), ics={x(0):0, sp.diff(x(t),t).subs(t,0):1})

# sp.plot(deq.rhs, (t,0,10))

plt.plot(T, X,'r,')
plt.axis([0,rng,-5,5])
plt.show()


