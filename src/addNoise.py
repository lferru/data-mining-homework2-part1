#    name: addNoise.py
#  author: molloykp (Oct 2019)
# purpose: Accept a numpy npy file and add Gaussian noise
#          Parameters:

import numpy as np
from numpy import save
from numpy import load
import argparse

def main():
    np.random.seed(1671)

    # make the script read the parameters (fill this out)
    # call the input matrix inMatrix
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputFile', type=str)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--outputFile', type=str)
    parms = parser.parse_args()
    inMatrix = load(parms.inputFile)
    # matrix must be floating point to add values
    # from the Gaussian
    inMatrix = inMatrix.astype('float32')
    inMatrix += np.random.normal(0,parms.sigma,(inMatrix.shape))
    inMatrix = inMatrix.astype('int')

    # noise may have caused values to go outside their allowable
    # range
    inMatrix[inMatrix < 0] = 0
    inMatrix[inMatrix > 255] = 255
    
    #save the perturbed matrix in a file

    save(parms.outputFile, inMatrix)

if __name__ == '__main__':
    main()
