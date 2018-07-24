import matplotlib.pyplot as plt
import mclient
import numpy as np

def moving_average(a, n=20) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Setup alazar according to settings in GUI
alz = mclient.instruments['alazar']
alz.setup_channels()
alz.setup_clock()
alz.setup_trigger()


if 0:
    alz.setup_shots(1)
    buf = alz.take_raw_shots()
    plt.figure()
    nsamp = alz.get_nsamples()
    plt.plot(buf[:nsamp], label='A')
    plt.plot(buf[nsamp:2*nsamp], label='B')
    plt.suptitle('Raw single shot')
    plt.legend()
    plt.xlabel('Time [ns]')

if 0:
    alz.setup_shots(1)
    buf = alz.take_demod_shots()
    buf2 = moving_average(buf)
    plt.figure()
    plt.suptitle('Demodulated single shot')

    plt.subplot(211)
    plt.plot(np.real(buf), label='Iraw')
    plt.plot(np.imag(buf), label='Qraw ')
    plt.plot(np.real(buf2), label='IMA')
    plt.plot(np.imag(buf2), label='QMA ')
    plt.xlabel('IF period #')
    plt.legend()

    plt.subplot(212)
    plt.plot(np.real(buf), np.imag(buf), label='IQraw')
    plt.plot(np.real(buf2), np.imag(buf2), label='IQMA')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.legend()

if 1:
    alz.setup_avg_shot(100000)
    buf = alz.take_avg_shot(timeout=50000)

    plt.figure()
    plt.suptitle('Average demodulated shot')

    plt.subplot(211)
    plt.plot(np.abs(buf))
    plt.xlabel('IF period #')

    plt.subplot(212)
    plt.plot(np.real(buf), np.imag(buf))
    plt.xlabel('I')
    plt.ylabel('Q')

if not plt.isinteractive():
    plt.show()
