import numpy as np
import time
import matplotlib.pyplot as plt

class Demodulator:
    def __init__(self, nsamples, period):
        self.nsamples = nsamples
        self.period = period
        if (nsamples % period) != 0:
            raise ValueError('Number of samples should be a multiple of the IF period')

    def reserve_IQ(self, dtype):
        self.I = np.zeros([self.nsamples/self.period,], dtype=dtype)
        self.Q = np.zeros([self.nsamples/self.period,], dtype=dtype)

    def demodulate(self, ar):
        pass

    def __call__(self, ar):
        return self.demodulate(ar)

class DemodulatorInt(Demodulator):

    def __init__(self, nsamples, period):
        Demodulator.__init__(self, nsamples, period)
        phis = np.linspace(0, 2*np.pi, period, endpoint=False)
        self._cosphi = np.round(8000 * np.cos(phis)).astype(np.int32)
        self._sinphi = np.round(8000 * np.sin(phis)).astype(np.int32)
        self._norm = 8000. / self.period
        self.reserve_IQ(dtype=np.int32)

    def demodulate(self, ar):
        ar2 = ar.reshape((len(ar) / self.period, self.period))
        np.dot(ar2, self._cosphi, self.I[:len(ar2)])
        self.I /= self._norm
        np.dot(ar2, self._sinphi, self.Q[:len(ar2)])
        self.Q /= self._norm

class DemodulatorFloat(Demodulator):

    def __init__(self, nsamples, period):
        Demodulator.__init__(self, nsamples, period)
        phis = np.linspace(0, 2*np.pi, period, endpoint=False)
        self._cosphi = np.cos(phis).astype(np.float32)
        self._sinphi = np.sin(phis).astype(np.float32)
        self.reserve_IQ(dtype=np.float32)

    def demodulate(self, ar):
        ar2 = ar.reshape((len(ar) / self.period, self.period))
        np.dot(ar2, self._cosphi, self.I[:len(ar2)])
        np.dot(ar2, self._sinphi, self.Q[:len(ar2)])

class DemodulatorComplex(Demodulator):
    '''
    Class to perform complex demodulation by calculating sig * exp(-i phi).
    This class can directly average the IQ values over several perios of the
    IF frequency by specifying <avg_periods>.
    '''

    def __init__(self, nsamples, period, avg_periods=1, weight_func=1):
        Demodulator.__init__(self, nsamples, period)
        self.avg_periods = avg_periods

        # Number of samples for one data point
        self.samples_per_point = period * avg_periods
        if (nsamples % self.samples_per_point) != 0:
            raise ValueError('Number of samples needs to be multiple of period and avg_cycles')

        phis = np.linspace(0, 2*np.pi * avg_periods, self.samples_per_point, endpoint=False)
        self._exp_iphi = np.exp(1j * phis) / avg_periods * weight_func
        self._exp_iphi = self._exp_iphi.astype(np.complex64)
        self.IQ = np.zeros([self.nsamples/self.samples_per_point,], dtype=np.complex64)

    def demodulate(self, ar):
        ar2 = ar.reshape((len(ar) / self.samples_per_point, self.samples_per_point))
        np.dot(ar2, self._exp_iphi, self.IQ[:len(ar2)])

class ArrayWindow:
    '''
    Object to hold a description of a measurement window in a data array.
    Contains start / length and optional weighting array.
    '''

    def __init__(self, start, length, weight=None):
        self.start = start
        self.length = length
        self.weight = weight

    def get(self, ar):
        '''
        Return window from an array, optionally weighted.
        '''
        d = ar[self.start:self.start+self.length]
        if self.weight:
            np.multiply(d, self.weight, d)
        return d

class ReferencedMeasurement:
    '''
    Class to implement processing of a referenced measurement.

    The process function receives an array <ar> which should contain two
    measured signals. One of them from window1.start and the other at
    window2.start. Demol
    '''

    def __init__(self, window1, window2, period, cyclelen, robins, demod=DemodulatorComplex):
        self.window1 = window1
        self.window2 = window2
        self.demod1 = demod(len(window1), period)
        self.demod2 = demod(len(window2), period)
        self.cyclelen = cyclelen
        self.robins = robins
        self.nmeasurements = cyclelen * robins
        self.reset()

    def reset(self):
        self.summed = np.zeros([cyclelen,], dtype=np.complex128)
        self.cur_index = 0

    def process(self, ar):
        w1 = self.window1.get(ar)
        d1 = self.demod1(w1)
        w2 = self.window2.get(ar)
        d2 = self.demod1(w2)
        rel = np.average(d1) / np.average(d2)
        self.summed[self.cur_index % self.cyclelen] += rel
        self.cur_index += 1
        if self.cur_index == self.nmeasurements:
            self.done()

    def done(self):
        self.summed /= self.robins
        print 'Averaged data: %s' % (self.summed, )

IF_PERIOD = 20
DEMOD_MAP = {
    'int': DemodulatorInt,
    'float': DemodulatorFloat,
    'complex': DemodulatorComplex,
}
