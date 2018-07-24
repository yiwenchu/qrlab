from pulseseq.sequencer import *
from pulseseq.pulselib import *

from opt_ro_cal import Optimal_RO_Cal as ORC


class Optimal_Readout_Power(ORC):
    '''
    This is the simplest possible use of ORC.  We compare a pi pulse
    to no pi pulse and sweep readout power.

    We're not actually sweeping a sequence parameter, so the generated
    sequence is only two elements.  Swept params is a dummy list of one
    element.
    '''
    def __init__(self, qubit_info, powers, plen=None, amp=1.0, **kwargs):
        self.qubit_info = qubit_info
        self.plen = plen
        self.amp = amp
        super(Optimal_Readout_Power, self).__init__(infos=qubit_info,
            powers=powers, swept_params=[0],  **kwargs)

    def generate(self):
        r = self.qubit_info.rotate
        s = Sequence()

        for i in [1,0]:
            s.append(Trigger(250))
            if self.plen is None:
                s.append(r(np.pi*i,0))
            else:
                s.append(Constant(self.plen, i*self.amp, chan=r.chans[0]))
            s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        return s.render()