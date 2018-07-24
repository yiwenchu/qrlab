# -*- coding: utf-8 -*-
"""
AutoDeconv
"""
import numpy as np
import matplotlib.pyplot as plt

from measurement import Measurement
from pulseseq.sequencer_subroutines import *
from pulseseq.sequencer import *
from pulseseq.pulselib import *

import fitting_programs as fp
import time

'''
do_auto_deconv = True
if do_auto_deconv:
    adexp = Auto_Deconv(fs,
                        qubit_info=ancilla,
                        flux_chan = 1)
    adexp.setup(load=False)

    #adexp.take_trace()
    #adexp.deconv_analysis()
    '''
class Auto_Deconv(Measurement):

    def __init__(self, fs, awg,
                 flux_chan=2,
                 marker_chan='2m1',
                 total_time_ns = 25e3, # in ns
                 v_awg_hittite_chan = None,
                 v_awg_initial = 0.0,
                 v_awg_final = 0.500,
                 v_yoko=0,
                 **kwargs):
        '''
        We actually need a qubit/flux info to convert awg voltages to
        DAC values, for now those numbers mean DAC values, not voltages
        '''


        #fs = FastScope('TCPIP0::172.28.141.133::inst0::INSTR')

        #awg1 = AWG.AWG('GPIB0::1::INSTR')
        #awg1 = AWG.AWG('TCPIP0::172.28.140.179::4101::SOCKET')
        self.awg = awg

        if total_time_ns > 2**16:
            raise ValueError('cannot take data for > 2^16 ns with 1 ns sampling')
        self.fs_total_time_range = total_time_ns/1e9

        self.fs_numpts = self.fs_total_time_range * 1e9 # 1 sample/ns
        if self.fs_numpts > 2**16:
            raise ValueError('cannot have more than 65536 points')
        self.fs_averages = 1000

        self.fs = fs
#        self.qubit_info = qubit_info
        self.fc = flux_chan
        self.marker_chan = marker_chan
        self.hc = v_awg_hittite_chan
        self.vf = v_awg_final
        self.vi = v_awg_initial
        self.vy = v_yoko
        self.kwargs = kwargs

        max_voltage = max([self.vy+self.vf,self.vy+self.vi])
        min_voltage = min([self.vy+self.vf,self.vy+self.vi])
        self.voltage_range = max_voltage - min_voltage
        self.voltage_offset = (max_voltage+min_voltage)/2
        #these numbers give us 1ns steps, which we need for deconv

        self.xs = []
        self.ys = []

        super(Auto_Deconv, self).__init__(0)


    def generate(self, pattern = 'heaviside', kernel_path = None):

        s = Sequence()
        s.append(Trigger(250))
        s.append(Combined([Constant(250, 1, chan=self.marker_chan),
                           Constant(250, self.vi, chan=int(self.marker_chan[0]))
                           ]))

        delay = self.fs_total_time_range * 10**9 / 2.0

        if pattern == 'heaviside':
            s.append(smartConstant(delay-250,self.vi,self.fc))
            s.append(smartConstant(2*delay,self.vf,self.fc))
            s.append(smartConstant(500,self.vi,self.fc))
            #the extra 500 is to make sure that our sequence doesn't get
            #screwed up when the FS doesn't sync perfectly
        elif pattern == 'testcomb':
            for _ in range(5):
                s.append(smartConstant(delay/5,self.vi,self.fc))
                s.append(smartConstant(delay/5,self.vf,self.fc))

        #channelInitDelay(s, self.qubit_info.channels)

        s = Sequencer(s)
        seqs = s.render()

        seqs = flatten_waveforms(seqs) #Used to be after FFG

        if kernel_path:
            fast_fluxy_goodness(seqs,self.fc,kernel_path,join_all =False)#Need to deal with that join at.
        #s.plot_seqs(seqs)
        #plt.show()
#        seqs = flatten_waveforms(seqs)

        self.seqs = seqs

        #print self.seqs[1][0].get_length()
        self.load(self.seqs, run=True)
#        self.play_sequence(load=True)

        #postProcess

    def setup(self):
#        '''setup configures the instrument and loads the pattern'''
#        self.qubit_info.fg.set_periods_us(self.fs_total_time_range*10**6 * 10)

        '''setup yoko'''

        '''setup awg'''
#        self.qubit_info.awg.set_channel_amplitude(1,self.fc)
        self.awg.do_set_amplitude(1,self.fc)

        '''align fastscope'''
        self.fs.reset()
        self.fs.set_trig_frontpanel(level=0.5)

        self.fs.set_time_range(self.fs_total_time_range)
        self.fs.set_voltage_range(1.3*.5*self.voltage_range)#Change these
        self.fs.set_voltage_offset(.9*self.voltage_offset*0.5)

        self.fs.set_averaging(1000)
        self.fs.set_num_pts_manual()
        self.fs.set_num_pts(self.fs_numpts)

    def take_trace(self):
        self.fs.take_trace()

        self.xs = self.fs.x
        self.ys = self.fs.y

        plt.plot(self.xs,self.ys)
        return self.xs, self.ys

    def deconv_analysis(self, plot_all=False, plot_results = True,save_kernel=False,
                        params=None):
        '''deconvolution parameters'''
        if params == None:
            self.trunc_freq_MHz = 25.0#25.0
            self.filter_cutoff = 75#75#75
            self.filter_sigma = 25#25
            self.poly_order = 20
            self.peak_shift = 0  #This should always be zero unless you really
                                 #Know what you're doing.
        else:
            self.trunc_freq_MHz = params['trunc_freq_MHz']
            self.filter_cutoff = params['filter_cutoff']
            self.filter_sigma = params['filter_sigma']
            self.poly_order = params['poly_order']
            self.peak_shift = params['peak_shift']
        #The fit trunc point needs to be far enough out the H(f) starts sloping
        #up again.
        self.time_step_in_seconds = 10**-9 #Hopefully you took ns data.

        '''Here do the high-level deconvolution algorithm'''

        '''Scale the data to be between 0 and 1'''
        scaled_data = self.scale_data(self.ys, plot=plot_all)

        '''Calculate the time domain transfer function (derivative)'''
        h_t = self.take_coarse_deriv(scaled_data,plot=plot_all)

        '''FFT into the frequency domain transfer function'''
        h_w, self.freq_axis = self.find_h_w(h_t,plot=False)

        '''Truncate the low-frequency data, which we'll keep directly, and
        use a polynomial fit for the medium/high frequency data'''
        trunc_index_for_fit = np.abs(self.trunc_freq_MHz - self.freq_axis).argmin()
        hf_poly = self.fit_to_high_freq(h_w,plot=plot_all)
        h_smooth = np.concatenate((h_w[:trunc_index_for_fit], hf_poly))

        '''Filter this response function such that all high frequency response
        is identity'''
        h_smooth_filtered = self.filter_h_smooth(h_smooth, plot=plot_all)

        '''Generate a time-domain convolution kernel from the new frequency
        response function'''
        kernel = self.generate_kernel(h_smooth_filtered,plot=plot_all)
        self.kernel = kernel

        if plot_results:
            convolved_time_response = np.convolve(scaled_data, kernel, mode='same')

            if plot_all:
                plt.figure()
                plt.plot(scaled_data, label='scaled data')
                plt.plot(convolved_time_response, label='convolved data')
                plt.title('SCALED DATA BEFORE AND AFTER DECONV')
                plt.legend(loc='best')
                plt.show()

            ranges = [range(self.peak_start-4000,self.peak_start-10),
                      range(self.peak_start+10,self.peak_start+200),
                      range(self.peak_start+10,int(self.peak_start+2000))]
            titles = ['Before the step',
                      'Immediately after the step',
                      'Long after the step']
            for idx,rang in enumerate(ranges):
                plt.subplot(len(ranges),1,idx)
                plt.plot(scaled_data[rang], label='scaled data')
                plt.plot(convolved_time_response[rang], label='convolved data')
                plt.title(titles[idx])
                plt.legend(loc='best')
            plt.tight_layout()
            #plt.title('CLOSEUPS:  Imm. After, Long after, Long Before')
            plt.show()

        if save_kernel:
            '''save_kernel should be a string to a csv file'''
            np.savetxt(save_kernel, np.asarray(kernel), delimiter=",")

    def generate_setup_take_trace(self, avg_time, kernel_path=None):
        self.generate(kernel_path=kernel_path)
        self.setup()
        print 'averaging on fast scope...'
        for i in np.arange(avg_time):
            time.sleep(1)
        print 'finished averaging on fast scope...'
        return self.take_trace()

    def generate_kernel_and_compare(self, kernel_path, avg_time = 200,
                                    plot_verbose = True, params=None):

        x_old, y_old = self.generate_setup_take_trace(avg_time, kernel_path=None)
        self.data.create_dataset('x raw', data=x_old)
        self.data.create_dataset('y raw', data=y_old)

        self.deconv_analysis(plot_all=plot_verbose,plot_results=plot_verbose,
                             save_kernel=kernel_path, params=params)

        x_new, y_new = self.generate_setup_take_trace(avg_time, kernel_path=kernel_path)
        self.data.create_dataset('x deconv', data=x_new)
        self.data.create_dataset('y deconv', data=y_new)

        ranges = [[0, len(y_new)-1],
                  [len(y_new)/2 - 3000, len(y_new)/2 + 10],
                  [len(y_new)/2 - 10, len(y_new)/2 + 3000]
               ]
        titles = ['full range','before the step','after the step']
        for idx, val in enumerate(ranges):
            r_min,r_max = val
            plt.figure()
            plt.plot(x_old[r_min:r_max]*1e9,y_old[r_min:r_max], label='no deconv')
            plt.plot(x_new[r_min:r_max]*1e9,y_new[r_min:r_max], label='with deconv')

            avg_old = np.average(y_old[r_min:r_max])
            plt.plot([x_old[r_min]*1e9, x_old[r_max]*1e9],[avg_old, avg_old], label='no deconv avg')

            avg_new = np.average(y_new[r_min:r_max])
            plt.plot([x_new[r_min]*1e9,x_new[r_max]*1e9],[avg_new,avg_new], label='deconv avg')

            plt.title(titles[idx])
            plt.legend(loc='best')
            plt.show()

        for i in [1,2]:
            r_min,r_max = ranges[i]
            data = y_new[r_min:r_max]
            avg = np.average(data)
            plt.figure()
            plt.plot(x_new[r_min:r_max]*1e9,(data-avg))
            plt.title('(deviation) ' + titles[i])
            plt.show()

#==============================================================================
#       Deconvolution subroutines
#==============================================================================

    def scale_data(self, raw_data, plot=False):
        '''scale the effective heaviside from 0 to 1'''
        offset = min(raw_data) # np.average(raw_data)
        scaled_data = raw_data - offset
        amplitude = max(scaled_data) #np.average(scaled_data)
        scaled_data/=amplitude

        len_raw_data = len(raw_data)
        min_avg_amp = np.average(raw_data[:int(len_raw_data / 5.0)])
        max_avg_amp = np.average(raw_data[-int(len_raw_data / 5.0):])

        offset = min_avg_amp
        amplitude = max_avg_amp - min_avg_amp

        scaled_data = (raw_data - offset) / np.float(amplitude)

        if plot:
            plt.figure()
            plt.plot(scaled_data)
            plt.title('SCALED TIME DOMAIN DATA')
            plt.xticks([])
            plt.show()
        return scaled_data

    def take_coarse_deriv(self,scaled_data,plot=True):
        '''Take the coarse derivative without dividing by the time step and
        pad the extra point with zeros.'''
        numpts = len(scaled_data)
        coarse_derivs = [scaled_data[i+1]-scaled_data[i] for i in range(numpts-1)]
        coarse_derivs.append(0)
#        coarse_derivs = np.abs(coarse_derivs) #BAD LINE

        peak_idx = np.argmax(np.abs(coarse_derivs))
#        peak_idx = np.argmax(coarse_derivs)

        self.peak_start = peak_idx + self.peak_shift #normally the transition is under 1 ns

        #Rotate the time derivatives such that the peak begins at the start of the vector
        h_t = np.roll(coarse_derivs,-self.peak_start)
        if plot:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(coarse_derivs)
            plt.title('COARSE DERIVATIVE')
            plt.xticks([])
            plt.subplot(2,1,2)
            plt.plot(h_t)
            plt.title('COARSE DERIVATIVE ROTATED (H_T)')
            plt.xticks([])
            plt.show()
        return h_t


    def find_h_w(self,h_t,plot=False):
        '''Take the fft.  We also generate the corresponding frequencies.'''
        h_w = np.fft.rfft(h_t)  #IGOR generates the one-sided fft for real inputs, so does numpy rfft
        freq_axis = np.fft.fftfreq(len(h_t), self.time_step_in_seconds)
        #freq_axis = freq_axis[:len(h_w)-1] # take only non-neg frequency
        freq_axis = freq_axis[:len(h_w)] # take only non-neg frequency
        freq_axis *= 1L*1e-6
        #h_w = h_w[:-1] #I Don't think this should be present, not tested.  1/28/14 jacob
        if plot:
            plt.figure()
            plt.plot(freq_axis[0:200],h_w[0:200])
            plt.title('H_W')
            plt.xticks([])
            plt.show()
        return h_w, freq_axis

    def fit_to_high_freq(self,h_w,plot=False):
        '''create a polynomial to describe the high frequency components of the
        response function'''
        trunc_index_for_fit = np.abs(self.trunc_freq_MHz - self.freq_axis).argmin()
        #print 'trunc idx',trunc_index_for_fit
        hf_h_w = h_w[trunc_index_for_fit:-5]
        hf_freq_axis = self.freq_axis[trunc_index_for_fit:-5]
        hf_freq_axisb = self.freq_axis[trunc_index_for_fit:]

        real_fit = np.polyfit(hf_freq_axis, np.real(hf_h_w), self.poly_order)
        imag_fit = np.polyfit(hf_freq_axis, np.imag(hf_h_w), self.poly_order)
        poly = np.poly1d(real_fit)+1j*np.poly1d(imag_fit)
        hf_poly = poly(hf_freq_axis)

        if plot:
            plt.figure()
            plt.subplot(3,1,1)
            plt.plot(hf_freq_axis, np.real(hf_h_w), 'r-')
            plt.plot(hf_freq_axis, np.real(hf_poly),'k-',linewidth=2)
            plt.ylim(-0.5,1.1)
            plt.xticks([])
            plt.title('HF H(f) fit: real')
            plt.subplot(3,1,2)
            plt.plot(hf_freq_axis, np.imag(hf_h_w), 'r-')
            plt.plot(hf_freq_axis, np.imag(hf_poly),'k-',linewidth=2)
            plt.ylim(-0.5,0.5)
            plt.xticks([])
            plt.title('HF H(f) fit: imag')
            plt.subplot(3,1,3)
            plt.plot(hf_freq_axis, np.abs(hf_h_w), 'r-')
            plt.plot(hf_freq_axis, np.abs(hf_poly),'k-',linewidth=2)
            plt.ylim(0,1.1)
            #plt.xticks([])
            plt.title('HF H(f) fit: abs')
            plt.show()
        return poly(hf_freq_axisb)#hf_poly

    def filterfunc(self,h_f, fi):
        f0 = self.filter_cutoff
        sigma = self.filter_sigma
        return 1+(0.5)*(1/h_f - 1)*(np.tanh((fi-f0)/sigma)+1)

    def filter_h_smooth(self,h_smooth, plot=False):
        '''Apply the filter function that pushes all high frequency response
        to unity'''
        idxs = range(len(h_smooth))

        h_smooth_filtered = [h_smooth[idx] * \
                            self.filterfunc(h_smooth[idx],
                                            self.freq_axis[idx]) \
                             for idx in idxs]

        #Blakes IGOR code actually takes the abs of h_f here.  Not sure why.
        #This changes the transfer function dramatically.  See also the line
        #in filter_h_smooth which takes the abs.
#        h_smooth_filtered = np.abs([h_smooth[idx] * \
#                            self.filterfunc(np.abs(h_smooth[idx]),
#                                            self.freq_axis[idx]) \
#                             for idx in idxs])

        if plot:
            plt.figure()
            plt.ylim(-0.5,2)

            plt.plot(self.freq_axis[:20000],h_smooth[:20000])
            plt.plot(self.freq_axis[:20000],h_smooth_filtered[:20000])
            plt.title('H_W SMOOTH, FILTERED & UNFILTERED')
            plt.ylim(0,1.1)
            plt.xlim(0,500)
            plt.show()

        return h_smooth_filtered

    def generate_kernel(self,h_smooth_filtered, plot=False):
        '''Finally, generate the kernel'''
        g_f = 1.0/np.array(h_smooth_filtered)

        kernel = np.fft.irfft(g_f) #Igor ifft takes N+1 complex points into 2N real points
        kernel /= sum(kernel)

        kernel = np.roll(kernel,len(kernel)/2)
#
#        time_axis = self.time_step_in_seconds*np.arange(self.fs_numpts-2)
#        time_axis -= self.peak_start * self.time_step_in_seconds
#        time_axis *= 1e9
        if plot:
            plt.figure()
            plt.plot(kernel[len(kernel)/2-50:len(kernel)/2+50],'ro-')
            plt.ylim(-0.01,0.01)
            plt.title('TIME DOMAIN CONVOLUTION KERNEL')
            plt.xticks([])
            plt.show()
        return kernel