import time
import matplotlib.pyplot as plt
import calibrator
import numpy as np

# This script assumes some instruments have already been created:
# - awg: Arbitrary Waveform Generator
# - sa: Spectrum Analyzer

rfsrc = bnc2
F0 = 7.526e9 + 50e6
awg = awg1
chans = (1,2)
channel_amp = 1.0

#rfsrc = bnc3
#F0 = 8.3e9
#awg = awg1
#chans = (3,4)
#channel_amp = 1.2

#rfsrc = ag2
#F_ge = 7260.48e6
#F0 = F_ge+50e6
#awg = awg2
#chans = (1,2)
#channel_amp = 1.2

rfsrc.set_frequency(F0)
vspec.set_rf_on(True)
vspec.set_Navg(10)
vspec.set_delay(0.03)

#print 'Calibrating spectrum analyzer...'
#sa.find_offset(F0, freqrange=0.5e6, N=1001, plot=True)
#print '  offset = %.03f MHz' % (sa.df0/1e6, )

#print 'Calibrating LO leakage...'bricks = labbrick.find_labbricks()
cawg = calibrator.AWGCalibrator(awg, vspec, channel_amp=channel_amp, delay=0.1)
cawg.calibrate_offsets(F0, chans, iterations=6)

fs = []
amps =  []
phases = []
periods = (20,)
for period in periods:
    f = 1e9 / period
    print 'Balancing IQ for single sideband at -%.03f MHz...' % (f / 1e6, )
    fs.append(f)
    amp, phase = cawg.calibrate_sideband_phase(chans, F0 - f, F0 + f, period, plot=True)
    amps.append(amp)
    phases.append(phase)

vspec.set_rf_on(False)
