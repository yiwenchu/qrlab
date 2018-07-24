import sys
import numpy as np

def perform_convolution(data, kernel):
    '''
    Perform a convolution between <data> array and <kernel>.
    Keep output array the same length as data
    '''
    klen = len(kernel)
    dlen = len(data)

    if klen > 2 * dlen:
        pad_len = int((klen-2*dlen) / 2.0)
        t_kernel = kernel
        t_data = np.pad(data, pad_len, mode='constant',
                        constant_values=(data[0], data[-1]))
    else:
        pad_len = int((2*dlen-klen) / 2.0)
        t_kernel = np.pad(kernel, pad_len, mode='constant')
        t_data = data

    convolved_data = np.convolve(t_data, t_kernel, 'valid')
    c_mid = int(len(convolved_data)/2.0)
    convolved_data = convolved_data[int(np.floor(c_mid-dlen/2.0)):
                                    int(np.floor(c_mid+dlen/2.0))]

    return convolved_data

if __name__ == '__main__':
    kernel = np.loadtxt('test_kernel.csv')
    data = np.concatenate((np.zeros(5000),
                           np.ones(5000)))
    blah = perform_convolution(data, kernel)

    print 'Data length %d, result length %d' % (len(data), len(blah))
    plt.figure()
    plt.plot(data)
    plt.plot(blah, 'ks')