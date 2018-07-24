import numpy as np
from pulseseq import sequencer, pulselib
import mclient
import fpgapulses

import fpgameasurement
REPRATE_DELAY = 1000000      # Delay to get a reasonable repetition rate
SELECTIVE = True             # Will calibrate long cavity pulse
UPDATE = True

disps = np.linspace(0.0, 2.0, 26)

m = fpgameasurement.FPGAMeasurement('dispcal', xs=disps, probabilities=True, fit_func='displacement_cal', fit_func_kwargs=dict(n=0))

s = sequencer.Sequence()
s.append(sequencer.Delay(256, label='start'))
for disp in disps:
#    s.append(sequencer.Constant(1000, 1, chan='m0'))
    s.append(sequencer.Delay(REPRATE_DELAY))
    if not SELECTIVE:
        s.append(m.cavity.displace(disp))
    else:
        s.append(m.cavity.displace_selective(disp))
    s.append(sequencer.Delay(200))
    s.append(m.qubit.rotate_selective(np.pi, 0))
    s.append(fpgapulses.LongMeasurementPulse())
    s.append(sequencer.Delay(REPRATE_DELAY))

s.append(sequencer.Delay(256, jump='start'))

m.set_seq(s, len(disps))

m.start_exp()
m.plot_se()
m.save_fig()

if UPDATE:
    ampfit = 1.0 / m.fit_params["dispscale"].value
    u_ampfit = 0.5 / m.fit_params["dispscale"].value**2*m.fit_params["dispscale"].stderr 
    if u_ampfit < 0.1 * ampfit:
        if SELECTIVE:
            oldpiamp =  mclient.instruments[m.cavity.insname].get_pi_amp_selective()
            newpiamp = oldpiamp * ampfit
            mclient.instruments[m.cavity.insname].set_pi_amp_selective(newpiamp)
        else:
            oldpiamp =  mclient.instruments[m.cavity.insname].get_pi_amp()
            newpiamp = oldpiamp * ampfit
            mclient.instruments[m.cavity.insname].set_pi_amp(newpiamp)            
    else:
        print("Returned a bad fit. I did not update displacement")
