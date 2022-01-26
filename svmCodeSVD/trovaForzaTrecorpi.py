#!/usr/bin/env python
import numpy as np
from scipy.optimize import minimize

from subprocess import Popen, PIPE

def f(x):
    energyGoal = -0.00130345
    d = 4.5
    R0 = 0.05
    process = Popen(["./twoBound", "100", "1e5", str(R0), str(x[0]),str(d)], stdout=PIPE)
    (energy, err) = process.communicate()
    exit_code = process.wait()
    print(energy)

    return((float(energy)-energyGoal)**2)

def funzione(x):
    energyGoal = -0.00130345
    aGoal = 100.234
    aGoal = 7.32629
    weightEnergy = 1
    weightA = 1

    process = Popen(["./twoBound", "100", "1e5", str(x[0]), str(x[1])], stdout=PIPE)
    (energy, err) = process.communicate()
    exit_code = process.wait()

    print(energy)

    process = Popen(["./twoScattering", "100", "1e5", str(x[0]), str(x[1])], stdout=PIPE)
    (a, err) = process.communicate()
    exit_code = process.wait()

    print(a)

    return(weightEnergy * (float(energy)-energyGoal)**2+weightA * (float(a)-aGoal)**2)


v0 = -1.2013e4
r0 = 1 
x0 = np.array([v0])
#x0 = np.array([r0,v0])
res = minimize(f, x0, method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': True})
#res = minimize(funzione, x0, method='nelder-mead',
#                       options={'xtol': 1e-8, 'disp': True})
print(res.x)
#print(funzione(res.x))
print(f(res.x))
