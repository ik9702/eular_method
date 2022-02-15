import scipy as sp
import numpy as np
from scipy.integrate import solve_ivp



def sci_method(P):
    A = P[0,0]
    B = P[0,1]
    C = P[0,2]
    D = P[0,3]
    E = P[0,4]
    F = P[0,5]

    X_0 = P[1,0]
    V_0 = P[1,1]
    t_0 = int(P[1,2])
    t_end = int(P[1,3])
    D_t = P[1,4]
    rng = int(t_end - t_0)
    vol = P[1,3]-P[1,2]+1
    vol /= P[1,4]
    vol = int(vol)
    
    def dXdt(t, X):
        return X[1], -B*X[1]/A-C*X[0]/A-D*X[0]**3/A+E*np.sin(F*t)/A
    
    
    X0 = (X_0, V_0)
    sol = solve_ivp( dXdt, (t_0, t_end), X0, t_eval = np.linspace(t_0, t_end, vol))
    result = np.array([sol.t, sol.y[0]])
    return result




    






# plt.plot(sol.t, sol.y[0])
# plt.xlabel('t')
# plt.ylabel('X')
# plt.show()

