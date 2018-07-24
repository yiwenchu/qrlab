from measurement import Measurement2D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fit
import pulseseq.sequencer_subroutines as ss
import time
from PyQt4 import QtGui
import objectsharer as objsh

import fitting_programs as fp

SPEC   = 0
POWER  = 1

class SwapFastFlux(Measurement2D):
    '''

    '''
    def __init__(self, qubit_info, flux_amps, flux_lengths,
                 flux_chan=None, qubit_flux=0, ro_flux=0,
                 pulse_delay=100, pre_delay=40, post_delay=40,
                 seq=None, postseq=None, freq_delay=0.1, do_analyze=True,
                 kernel_path=None,
                 **kwargs):

        self.qubit_info = qubit_info

        self.flux_lengths = np.array(flux_lengths)
        self.flux_amps = np.array(flux_amps)

        self.flux_chan = flux_chan

        self.qubit_flux = qubit_flux
        self.ro_flux = ro_flux

        self.pulse_delay = pulse_delay
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        self.post_ro_flux = 2000

        self.freq_delay = freq_delay
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        self.xs = self.flux_amps
        self.ys = self.flux_lengths

        self.do_analyze = do_analyze
        self.kernel_path = kernel_path
        self.residuals = False
        self.res_vert = True

        npoints = len(self.flux_lengths) * len(self.flux_amps)
        super(SwapFastFlux, self).__init__(npoints, infos=qubit_info,
                    residuals=True, res_vert=False, res_horz=True, **kwargs)
        self.data.create_dataset('flux lengths', data=self.flux_lengths)
        self.data.create_dataset('flux amps', data=self.flux_amps)

    def generate(self):
        s = Sequence()
        r = self.qubit_info.rotate

        flux_pulse = lambda duration, flux_level: Constant(duration, flux_level, chan=self.flux_chan)

        q_pi = r(np.pi,0)
        q_length = q_pi.get_length()
        qubit_pulse = Combined([
                q_pi,
                flux_pulse(max(self.pulse_delay, q_length), self.qubit_flux),
            ], align=ALIGN_RIGHT)

        for fl in self.flux_lengths:
            for fa in self.flux_amps:

#                print 'flux amp: %0.3f, flux length: %0.3f' % (fa, fl)

                s.append(self.seq)
                s.append(qubit_pulse)


                if fl == 0:
                    s.append(Join([
                            flux_pulse(self.pre_delay, self.qubit_flux),
                            flux_pulse(self.post_delay, self.ro_flux)
                            ])
                            )
                else:
                    s.append(Join([
                            flux_pulse(self.pre_delay, self.qubit_flux),
                            flux_pulse(fl, fa),
                            flux_pulse(self.post_delay, self.ro_flux)
                            ])
                            )


                if self.postseq is not None:
                    s.append(self.postseq)

                s.append(self.get_readout_pulse())

#                s.append(Combined([
#                        Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
#                        Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
#                        flux_pulse(self.readout_info.pulse_len, self.ro_flux)
#                        ]))
#
#                s.append(flux_pulse(self.post_ro_flux, self.ro_flux))
#
#                s.append(flux_pulse(250, self.qubit_flux))

        s = self.get_sequencer(s)
        seqs = s.render()

        seqs = ss.flatten_waveforms(seqs)

        if self.kernel_path:
            ss.fast_fluxy_goodness(seqs,self.flux_chan,self.kernel_path,join_all=False)

        return seqs

    def update(self, avg_data):
        import matplotlib.colorbar as clb
        zs = self.get_ys(avg_data)
        fig = self.get_figure()
        fig.axes[0].clear()
        fig.axes[1].clear()
        if hasattr(self, 'xs') and hasattr(self, 'ys'):
            zs = zs.reshape((len(self.ys), len(self.xs)))
#            xs, ys = self.get_plotxsys()
            xs, ys = self.xs, self.ys
            extent = (min(self.xs), max(self.xs), min(self.ys), max(self.ys))
            im = fig.axes[0].imshow(zs,
                                interpolation='nearest',
                                extent=extent, aspect='auto',
                                origin='lower'
                                )

#                fig.axes[0].pcolormesh(xs, ys, np.transpose(zs)) # seems like i need this
            fig.axes[0].set_xlim(xs.min(), xs.max())
            fig.axes[0].set_ylim(ys.min(), ys.max())

#            clb.on_mappable_changed(im)
            fig.colorbar(im, cax=fig.axes[1])
            fig.canvas.draw()
        else:
            logging.warning('Unable to plot 2D array without xs and ys')


    def analyze(self, data=None, fig=None):
        pass
#        self.fig = self.get_figure()
#        ax_data, ax_res = self.fig.axes

