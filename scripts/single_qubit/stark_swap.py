import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import lmfit
from lib.math import fitter
import time


def analysis_stark_swap(meas, data=None, fig=None, detune=None, txt=''):
    xs = np.array(meas.delays, dtype=np.float) / 1e3 # us
#    if meas.calib ==0:
#        pass
#    else:
#        ds = xs[1]-xs[0]
#        xs = np.concatenate(([xs[0]-ds*2, xs[0]-ds], xs))
#            
#    
    ys, meas_fig = meas.get_ys_fig(data, fig)
    
    if data is not None:
        ys = data
    if fig is None:
        fig = meas_fig

    fig.axes[0].plot(xs, ys, 'ks', ms=3)


    f = fitter.Fitter('exp_decay_sine2')


    p = f.get_lmfit_parameters(xs, ys)
#    if meas.fix_freq:
#        p['f'].value = meas.detune/1e6
#        p['f'].vary = not meas.fix_freq
    result = f.perform_lmfit(xs, ys, print_report=True, plot=False)
    ys_fit = f.eval_func()
    
    p = result.params

    pi_amp = 1 / (2.0 * p['f'].value)
#    tau = result.params['tau'].value   
    
    txt += 'Amp = %0.3f +/- %0.4f\n' % (p['A'].value, p['A'].stderr)
#    txt += 'f = %0.4f +/- %0.5f\n' % (p['f'].value, p['f'].stderr)
    txt += 'phase = %0.3f +/- %0.4f\n' % (p['dphi'].value, p['dphi'].stderr)
#    txt += 'period = %0.4f\n' % (1.0 / p['f'].value)
    txt += 'tau = %0.3f +/- %0.4f\n' % (p['tau'].value, p['tau'].stderr)
    txt += 'pi amp = %0.4f; pi/2 amp = %0.4f' % (pi_amp, pi_amp/2.0)

    fig.axes[0].plot(xs, ys_fit, label=txt)
    fig.axes[1].plot(xs, (ys_fit-ys), marker='s')

    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Residuals')
    fig.axes[1].set_xlabel('Time [us]')

    fig.canvas.draw()

    return result.params
    
def analysis_phonon_T1(meas, data=None, fig=None, detune=None, txt=''):
    xs = np.array(meas.delays, dtype=np.float) / 1e3 # us
    ys, meas_fig = meas.get_ys_fig(data, fig)
    if data is not None:
        ys = data
    if fig is None:
        fig = meas_fig

    fig.axes[0].plot(xs, ys, 'ks', ms=3)


    f = fitter.Fitter('exp_decay')

    p = f.get_lmfit_parameters(xs, ys)
#    if meas.fix_freq:
#        p['f'].value = meas.detune/1e6
#        p['f'].vary = not meas.fix_freq
    result = f.perform_lmfit(xs, ys, print_report=True, plot=False)
    ys_fit = f.eval_func()

#    pi_amp = 1 / (2.0 * p['f'].value)
    tau = result.params['tau']
    txt += 'tau = %0.3f us +\- %0.4f us' % (tau.value, tau.stderr)

    fig.axes[0].plot(xs, ys_fit, label=txt)
    fig.axes[1].plot(xs, (ys_fit-ys), marker='s')

    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Residuals')
    fig.axes[1].set_xlabel('Time [us]')

    fig.canvas.draw()

    return result.params

def analysis_phonon_T2(meas, data=None, fig=None, detune=None, txt=''):
    xs = np.array(meas.delays, dtype=np.float) / 1e3 # us
    ys, meas_fig = meas.get_ys_fig(data, fig)
    
    if data is not None:
        ys = data
    if fig is None:
        fig = meas_fig
    if detune is None:
        detune = meas.detune/1e3

    fig.axes[0].plot(xs, ys, 'ks', ms=3)


    f = fitter.Fitter('exp_decay_sine')

    p = f.get_lmfit_parameters(xs, ys)
    if meas.fix_freq:
        p['f'].value = meas.detune/1e6
        p['f'].vary = not meas.fix_freq
    result = f.perform_lmfit(xs, ys, p=p, print_report=True, plot=False)
    ys_fit = f.eval_func()

    f = result.params['f'].value * 1e3
    tau = result.params['tau'].value

    txt += 'tau = %0.3f us\n' % (tau)
    txt += 'f = %0.4f kHz\n' % (f)
    txt += 'software df = %0.3f kHz\n' % (detune)
    txt += 'frequency detuning = %0.4f kHz' % (detune-f)  

    fig.axes[0].plot(xs, ys_fit, label=txt)
    fig.axes[1].plot(xs, (ys_fit-ys), marker='s')

    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Residuals')
    fig.axes[1].set_xlabel('Time [us]')

    fig.canvas.draw()

    return result.params

def analysis_swap_rabi(meas, data=None, fig=None, txt=''):
    xs = np.array(meas.rabi_amps, dtype=np.float)
    ys, meas_fig = meas.get_ys_fig(data, fig)
    if data is not None:
        ys = data
    if fig is None:
        fig = meas_fig

    fig.axes[0].plot(xs, ys, 'ks', ms=3)


    f = fitter.Fitter('sine')

#    p = f.get_lmfit_parameters(xs, ys)
#    if meas.fix_freq:
#        p['f'].value = meas.detune/1e6
#        p['f'].vary = not meas.fix_freq
    result = f.perform_lmfit(xs, ys, print_report=True, plot=False)
    p = result.params
    ys_fit = f.eval_func()

    pi_amp = 1 / (2.0 * p['f'].value)

    txt += 'Amp = %0.3f +/- %0.4f\n' % (p['A'].value, p['A'].stderr)
    txt += 'f = %0.4f +/- %0.5f\n' % (p['f'].value, p['f'].stderr)
    txt += 'phase = %0.3f +/- %0.4f\n' % (p['dphi'].value, p['dphi'].stderr)
    txt += 'period = %0.4f\n' % (1.0 / p['f'].value)
    txt += 'pi amp = %0.4f; pi/2 amp = %0.4f' % (pi_amp, pi_amp/2.0)

    fig.axes[0].plot(xs, ys_fit, label=txt)
    fig.axes[1].plot(xs, (ys_fit-ys), marker='s')

    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Residuals')
    fig.axes[1].set_xlabel('Rabi Amplitude')

    fig.canvas.draw()

    return result.params

class stark_swap(Measurement1D):

    def __init__(self, qubit_info, delays, amp=0, sigma=100, qubit_pi = True,
                 stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5,
                 calib = 0, calib_amp = 0, calib_pi = 0,               
                 fit_type='exp_decay_sine', fix_freq=None, seq2 = None,
                 seq=None, postseq=None, **kwargs):
        self.qubit_info = qubit_info
        if calib == 0:
            self.delays = np.array(np.round(delays), dtype=int)
            self.delays_r = np.array(np.round(delays), dtype=int)
        else:
            ds = delays[1]-delays[0]
            self.delays_r = np.array(np.round(delays), dtype=int)
            self.delays = np.array(np.round(np.concatenate(([delays[0]-ds*2, delays[0]-ds], delays))), dtype=int)
        self.xs = self.delays / 1e3        # For plotting purposes
        self.amp = amp
        self.sigma = sigma
        self.qubit_pi = qubit_pi
        self.stark_chan = stark_chan
        self.stark_mkr = stark_mkr
        self.stark_ofs = stark_ofs
        self.stark_bufwidth = stark_bufwidth
        self.calib = calib
        self.calib_pi = calib_pi
        self.calib_amp = calib_amp
        self.fit_type=fit_type
        self.fix_freq = fix_freq
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.seq2 = seq2
        self.postseq = postseq

        super(stark_swap, self).__init__(len(self.delays), infos=qubit_info, **kwargs)
        self.data.create_dataset('delays', data=self.delays)
        self.data.set_attrs(
            amp = amp,
            sigma = sigma,
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        self.s = Sequence()

        r = self.qubit_info.rotate
        
        if self.calib == 1:
            self.s.append(self.seq)
            self.s.append(self.get_readout_pulse())            

            self.s.append(self.seq)
            self.s.append(r(np.pi, X_AXIS))
            self.s.append(self.get_readout_pulse())
            self.s.append(Delay(30e3))
            
        if self.calib == 2:
            self.s.append(self.seq)
            self.s.append(GaussSquare(self.calib_pi, self.calib_amp, self.sigma, chan = self.stark_chan))
            self.s.append(r(np.pi, X_AXIS))
            self.s.append(self.get_readout_pulse())
            
            self.s.append(self.seq)
            self.s.append(GaussSquare(self.calib_pi, self.calib_amp, self.sigma, chan = self.stark_chan))
            self.s.append(self.get_readout_pulse())
            
        for i, dt in enumerate(self.delays_r):
            self.s.append(self.seq)
            self.s.append(self.seq2)
            
            if self.qubit_pi is True:
                self.s.append(r(np.pi, X_AXIS))

            self.s.append(GaussSquare(dt, self.amp, self.sigma, chan = self.stark_chan))

            if self.postseq is not None:
                self.s.append(self.postseq)
                
#            s.append(Delay(1000))
            self.s.append(self.get_readout_pulse())

        s = self.get_sequencer(self.s)
        s.add_marker(self.stark_mkr, self.stark_chan, ofs=self.stark_ofs, bufwidth=self.stark_bufwidth)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        txt = 'amp = %0.4f \n' % (self.amp)
        txt += 'sigma = %0.1f ns\n' % (self.sigma)
        self.fit_params = analysis_stark_swap(self, data, fig, txt=txt)
#        analysis_stark_swap(self, data, fig)
        return self.fit_params

# class vacuum_rabi(Measurement1D):
#     def __init__(self, qubit_info, delays, amps, sigma=50, 
#                  stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5,             
#                  fit_type='exp_decay_sine', fix_freq=None,
#                  seq=None, postseq=None, **kwargs):

#         self.qubit_info = qubit_info
#         self.delays = np.array(np.round(delays), dtype=int)
#         self.xs = self.delays / 1e3        # For plotting purposes
#         self.num_delays = len(self.delays)
#         self.amps = amps
#         self.num_amps = len(self.amps)
#         self.sigma = sigma
#         self.stark_chan = stark_chan
#         self.stark_mkr = stark_mkr
#         self.stark_ofs = stark_ofs
#         self.stark_bufwidth = stark_bufwidth
#         self.fit_type=fit_type
#         self.fix_freq = fix_freq
#         if seq is None:
#             seq = Trigger(250)
#         self.seq = seq
#         self.postseq = postseq

#         super(vacuum_rabi, self).__init__(1, infos=qubit_info, **kwargs)
#         self.data.create_dataset('amps', data=self.amps)
#         self.data.create_dataset('delays', data=self.delays)
#         self.avg_pps = self.data.create_dataset('avg_pp', shape=[self.num_steps, self.num_freq_pts])
#         self.data.set_attrs(
#             sigma = sigma,
#             rep_rate=self.instruments['funcgen'].get_frequency()
#         )


        
class phonon_T1(Measurement1D):

    def __init__(self, qubit_info, delays, n_swaps = 1, phonon_pi=1e3, amp=0, sigma=100, qubit_pi = True, second_swap = True,
                 stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5, calib = 0,
                 fit_type='exp_decay_sine', fix_freq=None,
                 seq=None, postseq=None, **kwargs):
        self.qubit_info = qubit_info
        self.delays = np.array(np.round(delays), dtype=int)
        self.xs = delays / 1e3        # For plotting purposes
        self.n_swaps = n_swaps
        self.phonon_pi = phonon_pi
        self.amp = amp
        self.sigma = sigma
        self.stark_chan = stark_chan
        self.stark_mkr = stark_mkr
        self.stark_ofs = stark_ofs
        self.stark_bufwidth = stark_bufwidth
        self.fit_type=fit_type
        self.fix_freq = fix_freq
        self.calib = calib
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.qubit_pi = qubit_pi
        self.second_swap = second_swap

        npoints = len(delays)
        if self.calib == 1:
            npoints += 2
        super(phonon_T1, self).__init__(npoints, infos=qubit_info, **kwargs)
        self.data.create_dataset('delays', data=delays)
        self.data.set_attrs(
            amp = amp,
            sigma = sigma,
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        self.s = Sequence()

        r = self.qubit_info.rotate

        if self.calib == 1:
            self.s.append(self.seq)
            self.s.append(self.get_readout_pulse())            

            self.s.append(self.seq)
            self.s.append(r(np.pi, X_AXIS))
            self.s.append(self.get_readout_pulse())

        for i, dt in enumerate(self.delays):
            self.s.append(self.seq)
            
            for j in range(self.n_swaps):
#                self.s.append(r(np.pi, X_AXIS))
                if self.qubit_pi is True:
                    self.s.append(r(np.pi, X_AXIS))
                self.s.append(GaussSquare(self.phonon_pi/np.sqrt(j+1), self.amp, self.sigma, chan = self.stark_chan))
                
            self.s.append(Delay(dt))
            
            if self.second_swap is True:
                for j in range(self.n_swaps-1)[::-1]:
                    
                    self.s.append(GaussSquare(self.phonon_pi/np.sqrt(j+2), self.amp, self.sigma, chan = self.stark_chan))
                    self.s.append(r(np.pi, X_AXIS))
                        
                self.s.append(GaussSquare(self.phonon_pi, self.amp, self.sigma, chan = self.stark_chan))

            if self.postseq is not None:
                self.s.append(self.postseq)
                
#            s.append(Delay(1000))
            self.s.append(self.get_readout_pulse())

        s = self.get_sequencer(self.s)
        s.add_marker(self.stark_mkr, self.stark_chan, ofs=self.stark_ofs, bufwidth=self.stark_bufwidth)
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        ys = super(phonon_T1, self).get_ys(data)
        if self.calib == 1:
            return (ys[2:] - ys[0])/(ys[1]-ys[0])
        return ys

    def analyze(self, data=None, fig=None):
        txt = 'phonon pi = %0.5f ns\n' % (self.phonon_pi)
        txt += 'amp = %0.5f\n' % (self.amp)
        txt += 'sigma = %0.1f ns\n' % (self.sigma)
        txt += 'nswaps = %d\n' % (self.n_swaps)
        self.fit_params = analysis_phonon_T1(self, data, fig, txt = txt)
        return self.fit_params
        
class phonon_T2(Measurement1D):

    def __init__(self, qubit_info, delays, detune=0, phonon_pi=1e3, amp=0, sigma=100, 
                 stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5,
                 necho = 0,
                 fit_type='exp_decay_sine', fix_freq=None,
                 seq=None, postseq=None, **kwargs):
        self.qubit_info = qubit_info
        self.delays = np.array(np.round(delays), dtype=int)
        self.xs = delays / 1e3        # For plotting purposes
        self.detune = detune
        self.phonon_pi = phonon_pi
        self.amp = amp
        self.sigma = sigma
        self.stark_chan = stark_chan
        self.stark_mkr = stark_mkr
        self.stark_ofs = stark_ofs
        self.stark_bufwidth = stark_bufwidth
        self.necho = necho
        self.fit_type=fit_type
        self.fix_freq = fix_freq
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        super(phonon_T2, self).__init__(len(delays), infos=qubit_info, **kwargs)
        self.data.create_dataset('delays', data=delays)
        self.data.set_attrs(
            amp = amp,
            sigma = sigma,
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        self.s = Sequence()

        r = self.qubit_info.rotate
        swap = GaussSquare(self.phonon_pi, self.amp, self.sigma, chan = self.stark_chan)

        for i, dt in enumerate(self.delays):
            self.s.append(self.seq)

            self.s.append(r(np.pi/2, X_AXIS))

            self.s.append(swap)
            if self.necho == 0:
                self.s.append(Delay(dt))
            else:
                tau = int(np.round(dt / (2 * self.necho)))
                e = r(np.pi, X_AXIS)
                
                self.s.append(Delay(tau))

                for i in range(self.necho - 1):
                    self.s.append(swap)
                    self.s.append(e)
                    self.s.append(swap)
                    self.s.append(Delay(2*tau))
                    
                self.s.append(swap)
                self.s.append(e)
                self.s.append(swap)
                self.s.append(Delay(tau))
            self.s.append(swap)

            angle = dt * 1e-9 * self.detune * 2 * np.pi
            self.s.append(r(np.pi/2, angle))

            if self.postseq is not None:
                self.s.append(self.postseq)
                
#            s.append(Delay(1000))
            self.s.append(self.get_readout_pulse())

        s = self.get_sequencer(self.s)
        s.add_marker(self.stark_mkr, self.stark_chan, ofs=self.stark_ofs, bufwidth=self.stark_bufwidth)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
#        self.fit_params = analysis(self, data, fig)
        txt = 'phonon pi = %0.5f ns\n' % (self.phonon_pi)
#        txt += 'software df = %0.3f kHz\n' % (self.detune/1e3)
        txt += 'amp = %0.3f\n' % (self.amp)
        txt += 'sigma = %0.1f ns\n' % (self.sigma)
        analysis_phonon_T2(self, data, fig, txt = txt)
#        return self.fit_params

        

        
class swap_Rabi(Measurement1D):

    def __init__(self, qubit_info, rabi_amps, phonon_pi=1e3, amp=0, sigma=100, delay = 0,
                 stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5,
                 r_axis=0,
                 fit_type='exp_decay_sine', fix_freq=None,
                 seq=None, postseq=None, **kwargs):
        self.qubit_info = qubit_info
        self.xs = rabi_amps        # For plotting purposes
        self.rabi_amps = rabi_amps
        self.phonon_pi = phonon_pi
        self.amp = amp
        self.sigma = sigma
        self.delay = delay
        self.stark_chan = stark_chan
        self.stark_mkr = stark_mkr
        self.stark_ofs = stark_ofs
        self.stark_bufwidth = stark_bufwidth
        self.r_axis = r_axis
        self.fit_type=fit_type
        self.fix_freq = fix_freq
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        super(swap_Rabi, self).__init__(len(rabi_amps), infos=qubit_info, **kwargs)
        self.data.create_dataset('rabi_amps', data=rabi_amps)
        self.data.set_attrs(
            amp = amp,
            sigma = sigma,
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        self.s = Sequence()

        r = self.qubit_info.rotate

        for i, r_amp in enumerate(self.rabi_amps):
            self.s.append(self.seq)

            self.s.append(GaussSquare(self.phonon_pi, self.amp, self.sigma, chan = self.stark_chan))
            self.s.append(Delay(self.delay))
#            s.append(Repeat(self.qubit_info.rotate(0, self.r_axis, amp=amp), self.repeat_pulse))
            self.s.append(r(0, self.r_axis, amp=r_amp))
            
            if self.postseq is not None:
                self.s.append(self.postseq)
            self.s.append(self.get_readout_pulse())


        s = self.get_sequencer(self.s)
        s.add_marker(self.stark_mkr, self.stark_chan, ofs=self.stark_ofs, bufwidth=self.stark_bufwidth)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
#        self.fit_params = analysis(self, data, fig)
        txt = 'phonon pi = %0.5f ns\n' % (self.phonon_pi)
#        txt += 'software df = %0.3f kHz\n' % (self.detune/1e3)
        txt += 'amp = %0.3f\n' % (self.amp)
        txt += 'sigma = %0.1f ns\n' % (self.sigma)
        analysis_swap_rabi(self, data, fig = fig, txt = txt)
#        return self.fit_params
        
def analysis_PStemperature(meas, data=None, fig=None, txt=''):
    ys, fig = meas.get_ys_fig(data, fig)
    ys = data
    xs = meas.rabi_amps

    fig.axes[0].plot(xs, ys, 'ks', ms=3)



    r = fitter.Fitter('sine')
    result = r.perform_lmfit(xs, ys, print_report=False, plot=False)
    p = result.params
    # ys_fit = r.test_values(xs, p=p)
    ys_fit = r.eval_func(xs, params=p)

    pi_amp = 1 / (2.0 * p['f'].value)

    txt += 'Amp = %0.3f +/- %0.4f\n' % (p['A'].value, p['A'].stderr)
    txt += 'f = %0.4f +/- %0.5f\n' % (p['f'].value, p['A'].stderr)
    txt += 'phase = %0.3f +/- %0.4f\n' % (p['dphi'].value, p['dphi'].stderr)
    fig.axes[0].plot(xs, ys_fit, label=txt)
    fig.axes[1].plot(xs, ys-ys_fit, marker='s')

    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[0].set_xlabel('Pulse amplitude')
    fig.axes[0].legend(loc=0)

    fig.canvas.draw()
    return result.params

class phonon_swap_temperature(Measurement1D):

    def __init__(self, ef_info, ge_info, rabi_amps, update=False,
                 phonon_pi=1e3, amp=0, sigma=100, 
                 stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5,
                 seq=None, r_axis=0, fix_phase=False,
                 fix_period=None, postseq=None,
                 fit_type='sine',
                 **kwargs):

        self.ge_info = ge_info
        self.ef_info = ef_info
        self.rabi_amps = rabi_amps
        self.xs = rabi_amps
        self.phonon_pi = phonon_pi
        self.amp = amp
        self.sigma = sigma
        self.stark_chan = stark_chan
        self.stark_mkr = stark_mkr
        self.stark_ofs = stark_ofs
        self.stark_bufwidth = stark_bufwidth
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.fix_phase = fix_phase
        self.fix_period = fix_period
        self.r_axis = r_axis
        self.fit_type = fit_type
        self.infos = [ge_info, ef_info]

        super(phonon_swap_temperature, self).__init__(2*len(rabi_amps), infos=self.infos, **kwargs)
        self.data.create_dataset('amps', data=rabi_amps)

    def generate(self):
        r_ge = self.ge_info.rotate
        r_ef = self.ef_info.rotate

        s = Sequence()

        for i, r_amp in enumerate(self.rabi_amps):
            s.append(self.seq)
            
#            s.append(GaussSquare(self.phonon_pi, self.amp, self.sigma, chan = self.stark_chan))

            s.append(r_ge(np.pi,0))
            s.append(r_ef(0, 0, amp=r_amp))
            s.append(r_ge(np.pi,0))

            if self.postseq is not None:
                s.append(self.postseq)
            s.append(self.get_readout_pulse())


        for i, r_amp in enumerate(self.rabi_amps):
            s.append(self.seq)
            
#            s.append(GaussSquare(self.phonon_pi, self.amp, self.sigma, chan = self.stark_chan))

            s.append(Constant(r_ge(np.pi,0).get_length(),0,chan='foo'))
            s.append(r_ef(0, 0, amp=r_amp))
            s.append(r_ge(np.pi,0))

            if self.postseq is not None:
                s.append(self.postseq)
            s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        s.add_marker(self.stark_mkr, self.stark_chan, ofs=self.stark_ofs, bufwidth=self.stark_bufwidth)

        seqs = s.render()

        return seqs

    def analyze(self, data=None, fig=None):
        data1 = data[:len(self.rabi_amps)]
        self.fit_params = analysis_PStemperature(self, data=data1, fig=fig)
        amp1 = self.fit_params['A'].value

        data2 = data[len(self.rabi_amps):]
        txt = 'phonon pi = %0.5f ns\n' % (self.phonon_pi)
#        txt += 'software df = %0.3f kHz\n' % (self.detune/1e3)
        txt += 'amp = %0.3f\n' % (self.amp)
        txt += 'sigma = %0.1f ns\n' % (self.sigma)
        self.fit_params = analysis_PStemperature(self, data=data2, fig=fig, txt=txt)
        amp2 = self.fit_params['A'].value
        
        population = np.abs(amp2/amp1)
        print 'Polarization ~ %0.03f' % (population )
        self.data.create_dataset('Polarization', data=population)

    def update(self, avg_data):
        ys = self.get_ys(avg_data)
        ys1 = ys[:len(self.rabi_amps)]
        ys2 = ys[len(self.rabi_amps):]

        fig = self.get_figure()
        fig.axes[0].clear()
        if hasattr(self, 'xs'):
            fig.axes[0].plot(self.xs, ys1)
            fig.axes[0].plot(self.xs, ys2)
        else:
            fig.axes[0].plot(ys1)
            fig.axes[0].plot(ys2)
        fig.canvas.draw()

        return 1

def analysis_phonon_fock(meas, data=None, fig=None, detune=None, txt=''):
    xs = np.array(meas.delays, dtype=np.float) / 1e3 # us
    ys, meas_fig = meas.get_ys_fig(data, fig)
    if data is not None:
        ys = data
    if fig is None:
        fig = meas_fig

    fig.axes[0].plot(xs, ys, 'ks', ms=3)


    f = fitter.Fitter('exp_decay_sine')


    p = f.get_lmfit_parameters(xs, ys)
#    if meas.fix_freq:
#        p['f'].value = meas.detune/1e6
#        p['f'].vary = not meas.fix_freq
    result = f.perform_lmfit(xs, ys, print_report=True, plot=False)
    ys_fit = f.eval_func()
    
    p = result.params

    pi_amp = 1 / (2.0 * p['f'].value)
#    tau = result.params['tau'].value   
    
    txt += 'Amp = %0.3f +/- %0.4f\n' % (p['A'].value, p['A'].stderr)
#    txt += 'f = %0.4f +/- %0.5f\n' % (p['f'].value, p['f'].stderr)
    txt += 'phase = %0.3f +/- %0.4f\n' % (p['dphi'].value, p['dphi'].stderr)
#    txt += 'period = %0.4f\n' % (1.0 / p['f'].value)
    txt += 'tau = %0.3f +/- %0.4f\n' % (p['tau'].value, p['tau'].stderr)
    txt += 'pi amp = %0.4f; pi/2 amp = %0.4f' % (pi_amp, pi_amp/2.0)

    fig.axes[0].plot(xs, ys_fit, label=txt)
    fig.axes[1].plot(xs, (ys_fit-ys), marker='s')

    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Residuals')
    fig.axes[1].set_xlabel('Time [us]')

    fig.canvas.draw()

    return result.params

class phonon_fock(Measurement1D):

    def __init__(self, qubit_info, delays, n_swaps=1, phonon_pi=1e3, amp=0, sigma=100, 
                 stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5,
                 fit_type='exp_decay_sine', fix_freq=None, calib = 0,
                 seq=None, postseq=None, **kwargs):
        self.qubit_info = qubit_info
        if calib == 0:
            self.delays = np.array(np.round(delays), dtype=int)
            self.delays_r = np.array(np.round(delays), dtype=int)
        else:
            ds = delays[1]-delays[0]
            self.delays_r = np.array(np.round(delays), dtype=int)
            self.delays = np.array(np.round(np.concatenate(([delays[0]-ds*2, delays[0]-ds], delays))), dtype=int)
        self.xs = self.delays / 1e3        # For plotting purposes
        self.n_swaps = n_swaps
        self.phonon_pi = phonon_pi
        self.amp = amp
        self.sigma = sigma
        self.stark_chan = stark_chan
        self.stark_mkr = stark_mkr
        self.stark_ofs = stark_ofs
        self.stark_bufwidth = stark_bufwidth
        self.fit_type=fit_type
        self.fix_freq = fix_freq
        self.calib = calib
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        super(phonon_fock, self).__init__(len(self.delays), infos=qubit_info, **kwargs)
        self.data.create_dataset('delays', data=self.delays)
        self.data.set_attrs(
            amp = amp,
            sigma = sigma,
            phonon_pi=phonon_pi,
            n_swaps = n_swaps,
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        self.s = Sequence()

        r = self.qubit_info.rotate

        if self.calib == 1:
            self.s.append(self.seq)
            self.s.append(self.get_readout_pulse())            

            self.s.append(self.seq)
            self.s.append(r(np.pi, X_AXIS))
            self.s.append(self.get_readout_pulse())
            self.s.append(Delay(30e3))

        for i, dt in enumerate(self.delays_r):
            self.s.append(self.seq)
            for j in range(self.n_swaps):
                self.s.append(r(np.pi, X_AXIS))
                self.s.append(GaussSquare(self.phonon_pi/np.sqrt(j+1), self.amp, self.sigma, chan = self.stark_chan))

            self.s.append(GaussSquare(dt, self.amp, self.sigma, chan = self.stark_chan))

            if self.postseq is not None:
                self.s.append(self.postseq)
                
#            s.append(Delay(1000))
            self.s.append(self.get_readout_pulse())

        s = self.get_sequencer(self.s)
        s.add_marker(self.stark_mkr, self.stark_chan, ofs=self.stark_ofs, bufwidth=self.stark_bufwidth)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        txt = 'phonon pi = %0.5f ns\n' % (self.phonon_pi)
        txt += 'amp = %0.3f \n' % (self.amp)
        txt += 'sigma = %0.1f ns\n' % (self.sigma)
        txt += 'n_swaps = %d \n' % (self.n_swaps)
        self.fit_params = analysis_stark_swap(self, data, fig, txt=txt)
        return self.fit_params
        

     
#######################        
def analysis_phonon_drive(meas, data=None, fig=None, txt=''):
    xs = np.array(meas.drive_amps, dtype=np.float)
    ys = data
    
    if fig is None:
        fig = plt.figure()

    fig.axes[0].plot(xs, ys, 'ks', ms=3, label = txt)


#    fig.axes[0].plot(xs, ys_fit, label=txt)

    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Intensity [AU]')

    fig.canvas.draw()

    return result.params        
        
class phonon_drive(Measurement1D): #BROKEN

    def __init__(self, qubit_info, spec_params, drive_amps, plen = 50e3, phonon_pi=1e3, amp=0, sigma=100, 
                 stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5,
                 r_axis=0,
                 fit_type='exp_decay_sine', fix_freq=None,
                 seq=None, postseq=None, **kwargs):
        self.qubit_info = qubit_info
        [self.spec_brick, self.spec_info] = spec_params
        self.xs = drive_amps        # For plotting purposes
        self.drive_amps = drive_amps
        self.plen = plen
        self.phonon_pi = phonon_pi
        self.amp = amp
        self.sigma = sigma
        self.stark_chan = stark_chan
        self.stark_mkr = stark_mkr
        self.stark_ofs = stark_ofs
        self.stark_bufwidth = stark_bufwidth
        self.r_axis = r_axis
        self.fit_type=fit_type
        self.fix_freq = fix_freq
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        super(phonon_drive, self).__init__(len(drive_amps), infos=qubit_info, **kwargs)
        self.data.create_dataset('drive_amps', data=drive_amps)
        self.data.set_attrs(
            amp = amp,
            sigma = sigma,
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        self.s = Sequence()

#        r = self.qubit_info.rotate

        for amp in enumerate(self.drive_amps):
            self.spec_brick.set_power(amp)
            time.sleep(0.25)
            
        self.s.append(self.seq)
        
        self.s.append(Constant(self.plen, 1, chan=self.spec_info.marker_channel))

        self.s.append(GaussSquare(self.phonon_pi, self.amp, self.sigma, chan = self.stark_chan))

        if self.postseq is not None:
            self.s.append(self.postseq)
        self.s.append(self.get_readout_pulse())

        s = self.get_sequencer(self.s)
        s.add_marker(self.stark_mkr, self.stark_chan, ofs=self.stark_ofs, bufwidth=self.stark_bufwidth)
        seqs = s.render()
        return seqs

    def measure(self):

        alz = self.instruments['alazar']
        weight_func = alz.get_weight_func()

        IQg = self.readout_info.IQg
        IQe = self.readout_info.IQe

        xs = []
        IQs = []
        reals = []


        self.setup_measurement() #Stops AWGs, Fg loads seq
        self.avg_data = None
        self.pp_data = None
        self.shot_data = None
        started = not self.cyclelen==1 # False

        # spectroscopy loop
        for amp in enumerate(self.drive_amps):
            self.spec_brick.set_power(amp)
            time.sleep(0.25)
            if not started:
                self.start_awgs()
                self.start_funcgen()
                started = True

            ret = self.acquisition_loop(alz, fast=(self.cyclelen==1))

            if self.use_IQge:
                real = self.complex_to_real(ret)
            else:
                real = np.abs(ret)

            if self.subtraction:
                real = - real[1] + real[0]
                ret = - ret[1] + ret[0]
            else:
                real = real[0]
                ret = ret[0]

            xs.append(amp)

            IQs.append(ret)
            reals.append(real)

            self.xs = np.array(xs)
            self.update(np.array(reals))

        self.IQs[:] = np.array(IQs)
        self.reals[:] = np.array(reals)

        if self.analyze_data:
            self.analyze()

        if self.savefig:
            self.save_fig()
        return self.drive_amps, self.IQs[:], self.reals[:]

        self.analyze()

    def analyze(self, data=None, fig=None):
#        self.fit_params = analysis(self, data, fig)
        txt = 'phonon pi = %0.5f ns\n' % (self.phonon_pi)
#        txt += 'software df = %0.3f kHz\n' % (self.detune/1e3)
        txt += 'amp = %0.3f\n' % (self.amp)
        txt += 'sigma = %0.1f ns\n' % (self.sigma)
        txt += 'pulse length = 0.3e ns\n'  % (self.plen)
        analysis_swap_rabi(self, self.reals, fig = fig, txt = txt)
#        return self.fit_params
        
 #####################
def analysis_phonon_SSB_spec(meas, data=None, fig=None, txt = ''):
    ys, fig = meas.get_ys_fig(data, fig)
    xs = meas.detunings
    #####
    fit = fitter.Fitter('lorentzian')
    p = fit.get_lmfit_parameters(xs, ys)
    result = fit.perform_lmfit(xs, ys, print_report=False, plot=False, p=p)
    datafit = fit.eval_func()

    x0 = result.params['x0']

    fit_label = txt + 'center = %0.4f MHz,' % (x0/1e6) + ' width = %0.4f MHz' % (result.params['w']/1e6)
    ######

    fig.axes[0].plot(xs/1e6, ys, marker='s', ms=3, label='')
    fig.axes[0].plot(xs/1e6, datafit, ls='-', ms=3, label=fit_label) #####

    # fig.axes[0].plot(xs/1e6, ys)
    fig.axes[0].set_xlabel('Detuning (MHz)')
    fig.axes[0].set_ylabel('Intensity (AU)')
    # fig.canvas.draw()
    fig.axes[0].legend(loc='best')

    return result.params

class phonon_SSB_spec(Measurement1D):

    def __init__(self, phonon_info, detunings, drive_amp = None,
                 phonon_pi=1e3, amp=0, sigma=100, 
                 stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5,
                 qubit_info = None, simul_drive_amp = None,
                 seq=None, seq2 = None, simulseq = None, postseq = None, calib = 0, **kwargs):
        self.phonon_info = phonon_info
        self.drive_amp = drive_amp
        self.phonon_pi = phonon_pi
        self.amp = amp
        self.sigma = sigma
        self.stark_chan = stark_chan
        self.stark_mkr = stark_mkr
        self.stark_ofs = stark_ofs
        self.stark_bufwidth = stark_bufwidth
        self.qubit_info = qubit_info
        self.simul_drive_amp = simul_drive_amp
        self.calib = calib
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.seq2 = seq2
        self.simulseq = simulseq
        self.postseq = postseq        
        self.detunings = detunings
        self.xs = detunings / 1e6       # For plot
#        self.bgcor = bgcor

        npoints = len(detunings)
        if self.calib == 1:
            npoints += 2
        super(phonon_SSB_spec, self).__init__(npoints, residuals=False, infos=(phonon_info, qubit_info), **kwargs)
        self.data.create_dataset('detunings', data=detunings)
        self.data.set_attrs(
            drive_amp = self.drive_amp,
            phonon_pi = self.phonon_pi,
            amp = self.amp,
            sigma = self.sigma,
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        s = Sequence()

        r = self.qubit_info.rotate

        ro = Combined([
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
        ])
        
        if self.calib == 1:
            s.append(self.seq)
            s.append(ro)            

            s.append(self.seq)
            s.append(r(np.pi, X_AXIS))
            s.append(ro)
#            self.s.append(Delay(30e3))

#        if self.bgcor:
#            plen = self.qubit_info.rotate.base(np.pi, 0).get_length()
#            s.append(self.seq)
#            s.append(Delay(plen))
#            s.append(ro)

        for i, df in enumerate(self.detunings):
            g = DetunedSum(self.phonon_info.rotate.base, self.phonon_info.w_selective, chans=self.phonon_info.sideband_channels)
            if df != 0:
                period = 1e9 / df
            else:
                period = 1e50
            if self.drive_amp is None:
                g.add(self.phonon_info.pi_amp_selective, period)
            else:
                g.add(self.drive_amp, period)

            s.append(self.seq)
            s.append(self.seq2)

            if self.simulseq is not None:
                s.append(Combined((g(), self.simulseq), align = 2))
            elif self.simul_drive_amp is not None:
                g.add(self.simul_drive_amp, 1e9/(self.phonon_info.deltaf-self.qubit_info.deltaf))
                s.append(g())
            else: 
                s.append(g())
            
            s.append(GaussSquare(self.phonon_pi, self.amp, self.sigma, chan = self.stark_chan))
            
            if self.postseq is not None:
                s.append(self.postseq)
            
            s.append(ro)

        s = self.get_sequencer(s)
        s.add_marker(self.stark_mkr, self.stark_chan, ofs=self.stark_ofs, bufwidth=self.stark_bufwidth)
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        ys = super(phonon_SSB_spec, self).get_ys(data)
        if self.calib == 1:
            return (ys[2:] - ys[0])/(ys[1]-ys[0])
        return ys

    def analyze(self, data=None, fig=None):
        txt = 'phonon pi = %0.5f ns\n' % (self.phonon_pi)
        txt += 'amp = %0.3f \n' % (self.amp)
        txt += 'sigma = %0.1f ns\n' % (self.sigma)
        txt += 'drive amp = %0.3e \n' % (self.drive_amp)
#        txt += 'qubit drive amp = %0.3e \n' % (self.simul_drive_amp)
        self.fit_params = analysis_phonon_SSB_spec(self, data, fig, txt)
        return self.fit_params ####
        
####################        
def analysis_phonon_displacement(meas, data=None, fig=None, detune=None, txt=''):
    xs = np.array(meas.delays, dtype=np.float) / 1e3 # us
    ys, meas_fig = meas.get_ys_fig(data, fig)
    if data is not None:
        ys = data
    if fig is None:
        fig = meas_fig

    fig.axes[0].plot(xs, ys, 'ks', ms=3)


    f = fitter.Fitter('exp_decay_sine')


    p = f.get_lmfit_parameters(xs, ys)
#    if meas.fix_freq:
#        p['f'].value = meas.detune/1e6
#        p['f'].vary = not meas.fix_freq
    result = f.perform_lmfit(xs, ys, print_report=True, plot=False)
    ys_fit = f.eval_func()
    
    p = result.params

    pi_amp = 1 / (2.0 * p['f'].value)
#    tau = result.params['tau'].value   
    
    txt += 'Amp = %0.3f +/- %0.4f\n' % (p['A'].value, p['A'].stderr)
#    txt += 'f = %0.4f +/- %0.5f\n' % (p['f'].value, p['f'].stderr)
    txt += 'phase = %0.3f +/- %0.4f\n' % (p['dphi'].value, p['dphi'].stderr)
#    txt += 'period = %0.4f\n' % (1.0 / p['f'].value)
    txt += 'tau = %0.3f +/- %0.4f\n' % (p['tau'].value, p['tau'].stderr)
    txt += 'pi amp = %0.4f; pi/2 amp = %0.4f' % (pi_amp, pi_amp/2.0)

    fig.axes[0].plot(xs, ys_fit, label=txt)
    fig.axes[1].plot(xs, (ys_fit-ys), marker='s')

    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Residuals')
    fig.axes[1].set_xlabel('Time [us]')

    fig.canvas.draw()

    return result.params

class phonon_displacement(Measurement1D):

    def __init__(self, qubit_info, delays, drive_amp = 0, phonon_pi=1e3, amp=0, sigma=100, 
                 stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5,
                 fit_type='exp_decay_sine', fix_freq=None,
                 seq=None, postseq=None, **kwargs):
        self.qubit_info = qubit_info
        self.delays = np.array(np.round(delays), dtype=int)
        self.xs = delays / 1e3        # For plotting purposes
        self.drive_amp = drive_amp
        self.phonon_pi = phonon_pi
        self.amp = amp
        self.sigma = sigma
        self.stark_chan = stark_chan
        self.stark_mkr = stark_mkr
        self.stark_ofs = stark_ofs
        self.stark_bufwidth = stark_bufwidth
        self.fit_type=fit_type
        self.fix_freq = fix_freq
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        super(phonon_displacement, self).__init__(len(delays), infos=qubit_info, **kwargs)
        self.data.create_dataset('delays', data=delays)
        self.data.set_attrs(
            amp = amp,
            sigma = sigma,
            phonon_pi=phonon_pi,
            drive_amp = drive_amp,
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        self.s = Sequence()

        r = self.qubit_info.rotate

        for i, dt in enumerate(self.delays):
            self.s.append(self.seq)
            r = self.qubit_info.rotate_selective
            self.s.append(r(self.drive_amp, 0))

            self.s.append(GaussSquare(dt, self.amp, self.sigma, chan = self.stark_chan))

            if self.postseq is not None:
                self.s.append(self.postseq)
                
#            s.append(Delay(1000))
            self.s.append(self.get_readout_pulse())

        s = self.get_sequencer(self.s)
        s.add_marker(self.stark_mkr, self.stark_chan, ofs=self.stark_ofs, bufwidth=self.stark_bufwidth)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        txt = 'phonon pi = %0.5f ns\n' % (self.phonon_pi)
        txt += 'amp = %0.3f \n' % (self.amp)
        txt += 'sigma = %0.1f ns\n' % (self.sigma)
        txt += 'drive amp = %0.3f \n' % (self.drive_amp)
        self.fit_params = analysis_phonon_displacement(self, data, fig, txt=txt)
#        analysis_stark_swap(self, data, fig)
        return self.fit_params
        
class phonon_displacement_SSB(Measurement1D):

    def __init__(self, phonon_info, delays, drive = 0, phonon_pi=1e3, amp=0, sigma=100, 
                 stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5,
                 fit_type='exp_decay_sine', fix_freq=None, calib = 0, qubit_info = None, 
                 seq=None, seq2 = None, postseq=None, **kwargs):
        self.phonon_info = phonon_info
        if calib == 0:
            self.delays = np.array(np.round(delays), dtype=int)
            self.delays_r = np.array(np.round(delays), dtype=int)
        else:
            ds = delays[1]-delays[0]
            self.delays_r = np.array(np.round(delays), dtype=int)
            self.delays = np.array(np.round(np.concatenate(([delays[0]-ds*2, delays[0]-ds], delays))), dtype=int)
        self.xs = self.delays / 1e3        # For plotting purposes
        self.drive = drive
        self.drive_amp = np.abs(drive)
        self.drive_phi = np.angle(drive)
        self.phonon_pi = phonon_pi
        self.amp = amp
        self.sigma = sigma
        self.stark_chan = stark_chan
        self.stark_mkr = stark_mkr
        self.stark_ofs = stark_ofs
        self.stark_bufwidth = stark_bufwidth
        self.fit_type=fit_type
        self.fix_freq = fix_freq
        self.calib = calib
        self.qubit_info = qubit_info
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.seq2 = seq2
        self.postseq = postseq

        super(phonon_displacement_SSB, self).__init__(len(self.delays), infos=[phonon_info, qubit_info], **kwargs)
        self.data.create_dataset('delays', data=self.delays)
        self.data.set_attrs(
            amp = amp,
            sigma = sigma,
            phonon_pi=phonon_pi,
            drive = drive,
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        self.s = Sequence()
        
        r = self.qubit_info.rotate
        p = self.phonon_info.rotate_selective

        if self.calib == 1:
            self.s.append(self.seq)
            self.s.append(self.seq2)
            self.s.append(self.get_readout_pulse())            

            self.s.append(self.seq)
            self.s.append(self.seq2)
            self.s.append(r(np.pi, X_AXIS))
            self.s.append(self.get_readout_pulse())
            self.s.append(Delay(30e3))

        for i, dt in enumerate(self.delays_r):
            
            self.s.append(self.seq)
            self.s.append(p(0, self.drive_phi, amp = self.drive_amp))
            self.s.append(self.seq2) #(we want to cool the qubit after the phonon drive)
            self.s.append(GaussSquare(dt, self.amp, self.sigma, chan = self.stark_chan))

            if self.postseq is not None:
                self.s.append(self.postseq)
                
#            s.append(Delay(1000))
            self.s.append(self.get_readout_pulse())

        s = self.get_sequencer(self.s)
        s.add_marker(self.stark_mkr, self.stark_chan, ofs=self.stark_ofs, bufwidth=self.stark_bufwidth)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        txt = 'phonon pi = %0.5f ns\n' % (self.phonon_pi)
        txt += 'amp = %0.3f \n' % (self.amp)
        txt += 'sigma = %0.1f ns\n' % (self.sigma)
        txt += 'drive amp = %0.5f+%0.5fj \n' % (self.drive.real, self.drive.imag)
#        self.fit_params = analysis_phonon_displacement(self, data, fig, txt=txt)
        self.fit_params = analysis_stark_swap(self, data, fig, txt=txt)
        return self.fit_params
        
class phonon_wigner(Measurement1D):

    def __init__(self, phonon_info, delays, drive = 0, drive_phi = 0, phonon_pi=1e3, amp=0, sigma=100, 
                 stark_chan=3, stark_mkr = '3m1', stark_ofs = -85, stark_bufwidth = 5,
                 fit_type='exp_decay_sine', fix_freq=None, calib = 0, qubit_info = None, 
                 seq=None, statePrep = None, postseq=None, post_cooling = False,  **kwargs):
        self.phonon_info = phonon_info
        if calib == 0:
            self.delays = np.array(np.round(delays), dtype=int)
            self.delays_r = np.array(np.round(delays), dtype=int)
        else:
            ds = delays[1]-delays[0]
            self.delays_r = np.array(np.round(delays), dtype=int)
            self.delays = np.array(np.round(np.concatenate(([delays[0]-ds*2, delays[0]-ds], delays))), dtype=int)
        self.xs = self.delays / 1e3        # For plotting purposes
        self.drive = drive
        self.drive_amp = np.abs(drive)
        self.drive_phi = np.angle(drive)
        self.phonon_pi = phonon_pi
        self.amp = amp
        self.sigma = sigma
        self.stark_chan = stark_chan
        self.stark_mkr = stark_mkr
        self.stark_ofs = stark_ofs
        self.stark_bufwidth = stark_bufwidth
        self.fit_type=fit_type
        self.fix_freq = fix_freq
        self.calib = calib
        self.qubit_info = qubit_info
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.statePrep = statePrep
        self.postseq = postseq
        self.post_cooling = post_cooling

        super(phonon_wigner, self).__init__(len(self.delays), infos=[phonon_info, qubit_info], **kwargs)
        self.data.create_dataset('delays', data=self.delays)
        self.data.set_attrs(
            amp = amp,
            sigma = sigma,
            phonon_pi=phonon_pi,
            drive = drive,
            rep_rate=self.instruments['funcgen'].get_frequency()
        )

    def generate(self):
        self.s = Sequence()
        
        r = self.qubit_info.rotate
        p = self.phonon_info.rotate_selective

        if self.calib == 1:
            self.s.append(self.seq)
#            self.s.append(self.seq2)
            self.s.append(self.get_readout_pulse())            

            self.s.append(self.seq)
#            self.s.append(self.seq2)
            self.s.append(r(np.pi, X_AXIS))
            self.s.append(self.get_readout_pulse())
#            self.s.append(Delay(30e3))

        for i, dt in enumerate(self.delays_r):
            
            self.s.append(self.seq)
            self.s.append(self.statePrep) #this is the sequence for preparing the state we want
            self.s.append(p(0, self.drive_phi, amp = self.drive_amp))
            self.s.append(GaussSquare(dt, self.amp, self.sigma, chan = self.stark_chan))

            if self.postseq is not None:
                self.s.append(self.postseq)
                
#            s.append(Delay(1000))
            self.s.append(self.get_readout_pulse())
            
            if self.post_cooling:
                self.s.append(GaussSquare(100e3, self.amp, self.sigma, chan = self.stark_chan))

        s = self.get_sequencer(self.s)
        s.add_marker(self.stark_mkr, self.stark_chan, ofs=self.stark_ofs, bufwidth=self.stark_bufwidth)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        txt = 'phonon pi = %0.5f ns\n' % (self.phonon_pi)
        txt += 'amp = %0.3f \n' % (self.amp)
        txt += 'sigma = %0.1f ns\n' % (self.sigma)
        txt += 'drive amp = %0.5f+%0.5fj \n' % (self.drive.real, self.drive.imag)
#        self.fit_params = analysis_phonon_displacement(self, data, fig, txt=txt)
        self.fit_params = analysis_stark_swap(self, data, fig, txt=txt)
        return self.fit_params