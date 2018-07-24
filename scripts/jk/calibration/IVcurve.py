# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 16:24:05 2014

@author: rsl
"""

import mclient
import numpy as np
import matplotlib.pyplot as plt

k = mclient.instruments['keithley']
yoko = mclient.instruments['qubit_yoko']

# set configuration for keithley

k.set_trigger_continuous(False)
k.set_averaging(True)
k.set_averaging_count(10)
k.set_mode_curr_dc()
k.set_range(0.01) # 10 mA
#k.set_range_auto()

yoko.do_set_source_range('1E+0')
yoko.set_output_state(1)

# loop for yoko
voltages = np.linspace(-0.1, 0.1, 26)

num_av = 10
currents = []
for v in voltages:
    yoko.set_voltage(v, range='FIX')
    time.sleep(0.5)

    tv = 0
    for i in range(num_av):
        tv += k.get_readval()
    tv /= num_av

    currents.append(tv)
yoko.set_voltage(0)

currents = np.array(currents)*1e6
voltages = np.array(voltages)*1e3

fig = plt.figure()

ax = fig.add_subplot(311)
ax.plot(voltages, currents,'ro')

p = np.polyfit(voltages, currents,1)
ax.plot(voltages, np.polyval(p, voltages))

ax.set_ylabel('FBL Current (uA)')

ax = fig.add_subplot(312)
residue = np.polyval(p, voltages) - currents
ax.plot(voltages, residue)

ax.set_ylabel('residuals')

ax = fig.add_subplot(313)
ax.plot(voltages, np.abs(residue/currents))
ax.set_ylabel('frac. residuals')



ax.set_xlabel('Yoko Voltage (mV)')

resistance = 1000.0/p[0]

plt.suptitle('IV curve, resistance: %0.1f Ohms' % (resistance))


#data = [[0,0],
#[02,5.32],
#[04,11.06],
#[10,27.6],
#[20,55.28],
#[22,60.81],
#[24,66.34],
#[26,71.86],
#[28,77.35],
#[30,82.84],
#[32,88.31],
#[34,93.78],
#[36,99.27],
#[50,137.68]]

#import numpy as np
#import matplotlib.pyplot as pl
#
#data=np.array(data)

#v = data[:,0] * 1e-3
#i = data[:,1] * 1e-6
#
