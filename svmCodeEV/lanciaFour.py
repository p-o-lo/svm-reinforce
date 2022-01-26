import os
import sys


with open(sys.argv[1]) as prmFile:
    directory = 'nuovoFourBody'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for line1 in prmFile: 
        line2 = prmFile.next()
        line3 = prmFile.next()
        campo = line1.split()
        nomeFile = 'fourBody_r03.5_%s' % campo[5] 
        with open('%s/%s.input' % (directory,nomeFile),'w') as outFile :
            with open(sys.argv[2]) as inFile :
                for line in inFile :
                    if '#' in line :
                            continue
                    if '%' in line :
                            continue
                    if 'threeBodyParameters' in line :
                            continue
                    if 'twoBodyParameters' in line :
                            continue
                    if 'dataFile' in line :
                            continue
                    outFile.write(line)
            outFile.write(line1)
            outFile.write(line2)
            outFile.write("dataFile             %s\n" % nomeFile)
