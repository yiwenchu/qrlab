# Check raw FPGA data
# Channel 0 should be the reference
# Channel 1 should be the signal
# Make sure the signals have reasonable amplitudes, between 400 - 1000

import numpy as np
import time
import plotting
import matplotlib.pyplot as plt

execfile('fpgarunro.py')
m.iqrel = False
m.raw = True
m.start_exp()
m.stop_exp()

raw0 = m.load_data(m.get_fpgadata_fn(1)).reshape((m.readout.naverages, m.readout.ref_len))
raw1 = m.load_data(m.get_fpgadata_fn(2)).reshape((m.readout.naverages, m.readout.acq_len))

ax = plt.figure().add_subplot(111)
plt.suptitle('Average shots')
ax.plot(np.average(raw0,0), label='avg ref')
ax.plot(np.average(raw1,0), label='avg sig')
ax.legend()

fig = plt.figure()
plt.suptitle('All shots')
ax = fig.add_subplot(211)
plotting.pcolormesh(raw0, cb=True, ax=ax)
axsig = fig.add_subplot(212)
plotting.pcolormesh(raw1, cb=True, ax=axsig)
