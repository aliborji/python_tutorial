# from scipy import signal
import numpy as np
# import timeit
# from timeit import default_timer as timer

import time

umax = np.iinfo(np.uint8).max

def conv1D(M):
    # kernel = np.array([-1,0,1])
    # kernelH = np.tile(kernel, (rows, 1))
    # kernelV = np.tile(kernel, (1, cols))
    # Dx = scipy.signal.convolve2d(M, kernelH, 'same')
    # Dy = scipy.signal.convolve2d(M, kernelV, 'same')

    # testing
    # M = np.zeros((10,10)) #, dtype='uint16')
    # M[:5,:] = -1
    # M[5:,:] = 1
    # M = M.T
    # print(M)


    # assuming ZERO PADDING 
    # horizontal derivative
    aH = np.zeros((M.shape[1], M.shape[1]))
    i,j = np.indices(aH.shape)
    aH[i==j-1] = -1  
    aH[i==j+1] = 1
    # print(aH)    
    Dx = M.dot(aH)

    # vertical derivative
    aV = np.zeros((M.shape[0], M.shape[0]))
    i,j = np.indices(aV.shape)
    aV[i==j-1] = -1
    aV[i==j+1] = 1
    # print(aV)    
    Dy = M.T.dot(aV)

    return Dx, Dy.T    # endd =


if __name__ == '__main__':
    rows, cols = list(map(int, input('Enter rows and cols separated by space: ').split(',')))
    M = np.random.randint(0, umax, size=(rows, cols), dtype='uint8')

    start = time.time() #timeit.timeit()
    Dx, Dy = conv1D(M)
    print(F"Total taken time: {time.time() - start}")


    Dx_Max = Dx.max()
    Dy_Max = Dy.max()

    # print(Dx)
    # print(Dy)

    print(F"Max of Dx: {Dx_Max}, Max of Dy: {Dy_Max}")


