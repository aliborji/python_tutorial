import numpy as np
import time

umax = np.iinfo(np.uint8).max

def conv1D(M):

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

    return Dx, Dy.T    


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




    # Refs:
    # https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
    # https://stackoverflow.com/questions/18026541/make-special-diagonal-matrix-in-numpy
    # https://docs.python.org/2/library/timeit.html
    # https://stackoverflow.com/questions/15616742/vectorized-way-of-calculating-row-wise-dot-product-two-matrices-with-scipy
    # https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html
