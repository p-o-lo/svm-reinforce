import os
import sys
import tempfile


outFile = 'tmp.input'
inFile  = "per3BodyForce.input"
twoBodyFile = 'realTwoParameter.dat'
resFile = '3BodyForcesRealPotential_r0.8.dat'

def newInputFile(inputFile,twoBodyPar,a): 
	with open(outFile,'w') as newFile :
		with open(inputFile) as baseFile :
			for line in baseFile :
 				if '#' in line :
					continue
				if '%' in line :
					continue
				if 'threeBodyParameters' in line :
					continue
				if 'twoBodyParameters' in line :
					continue
				newFile.write(line)
			newFile.write(twoBodyPar)
                        r0=0.8
			newFile.write("threeBodyParameters   %.3f %.3f\n" % (a[0],r0))
			# newFile.write("threeBodyParameters   %.3f %.3f\n" % (a[0],a[1]))
			newFile.write("% END OF FILE\n")

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brentq
from subprocess import Popen, PIPE

def funzione(x,twoBodyPar):
	energyGoal = -8.48
	newInputFile(inFile,twoBodyPar,x)
	process = Popen(["./svmThree", outFile], stdout=PIPE)
	(energy, err) = process.communicate()
	exit_code = process.wait()

	energy = float(energy)
	print(type(energy))
	print(energy)

	distanza = abs(energy-energyGoal)
	distanza=round(distanza,5)
	print(distanza)
	return(distanza)

v0 = 1500
r0 = 0.8
x0 = np.array([v0])
# x0 = np.array([v0, r0])
with open(resFile,'w') as dataFile :
	with open(twoBodyFile) as twoPrm :
		for line in twoPrm :
			res = minimize(funzione, x0, args=(line), method='nelder-mead',
						   options={'xtol': 1e-4, 'disp': True, 'fatol': 1e-4, 'xatol' : 1e-4})
			print(res.x)
			
			dataFile.write(line)
			dataFile.write("threeBodyParameters   %.3f %.3f\n" % (res.x[0],r0))
			# dataFile.write("threeBodyParameters   %.3f %.3f\n" % (res.x[0],res.x[1]))
			process = Popen(["./svmThree", outFile], stdout=PIPE)
			(energy, err) = process.communicate()
			exit_code = process.wait()
			energy = float(energy)
			dataFile.write("# The energy of the Trizium is %.3f\n" % (energy))
                        dataFile.flush()
