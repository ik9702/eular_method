from re import A
import numpy as np
import matplotlib.pyplot as plt




def eular_method(P):
    A = P[0,0]
    B = P[0,1]
    C = P[0,2]
    D = P[0,3]
    E = P[0,4]
    F = P[0,5]

    X0 = P[1,0]
    V0 = P[1,1]
    D_t = P[1,4]
    rng = P[1,3]-P[1,2]
    vol = int(rng/D_t)
    
    V = np.zeros(vol)
    V[0]=V0
    X = np.zeros(vol)
    X[0]=X0
    T = np.zeros(vol)
    # T배열 생성
    for t in range(0, vol):
        T[t] = D_t*t
    # V배열 생성
    for t in range(0, vol-1):
        V[t+1] = V[t] -(B*V[t]+C*X[t]+D*X[t]**3-E*np.sin(F*T[t]))*D_t/A
        X[t+1] = X[t] + V[t]*D_t        
    result = np.array([T, X])
    return result

    # plt.plot(T, X,'r,')
    # plt.axis([0,rng,-5,5])
    # plt.show()
    
    




