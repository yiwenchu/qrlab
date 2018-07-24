import mclient
from mclient import instruments
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from pulseseq import sequencer, pulselib

mpl.rcParams['figure.figsize']=[6,4]

qubit_info = mclient.get_qubit_info('qubit_info')
qubit_ef_info = mclient.get_qubit_info('qubit_ef_info')
vspec = instruments['vspec']
awg1 = instruments['AWG1']
qubit_brick = instruments['qubit_brick']
qubit_ef_brick = instruments['qubit_ef_brick']
va_lo = instruments['va_lo']
funcgen = instruments['funcgen']
alazar = instruments['alazar']
spec_brick = instruments['spec_brick']
spec_info = mclient.get_qubit_info('spec_info')
cavity_info = mclient.get_qubit_info('cavity_info')

field = 0.0
temp = 'cd'
#voltage = laser_info.get_DCOffset()


################################################################################################################################################
from scripts.single_qubit import T1measurement, T2measurement
# from scripts.single_qubit import T1measurement_QP, T2measurement_QP
# from scripts.single_qubit import FT1measurement, EFT2measurement, GFT2measurement
# from scripts.single_qubit import efrabi
# from scripts.single_qubit import efrabi_QP
# from scripts.single_qubit import QPdecay
from scripts.single_qubit import rabi

def try_twice(func, N=2, **kwargs):
    for i in range(N):
        try:
            return func(**kwargs)
        except Exception, e:
            print 'Error %s' % (e,)
            pass
    print 'Failed to do %s %s times...' % (func, N)


# work in progress. For looping over multiple qubits
# def T1T2Loop(qubit_params):
# 	# from scripts.single_qubit.t1t2_plotting import do_T1_plot, do_T2_plot, do_T2echo_plot
# 	T1s={}
# 	T2s={}
# 	T2Es={}
# 	rep_rates = [500]

# 	for qubit in enumerate(qubit_params)	
# 	    T1s[qubit] = {'t1s':[], 't1s_err':[], 'ofs':[], 'ofs_err':[], 'amps':[], 'amps_err':[],}
# 	    T2s[qubit] = {'t2s':[], 't2s_err':[], 't2freqs':[], 't2freqs_err':[], 'amps':[], 'amps_err':[], 't22s':[], 't22s_err':[], 't22freqs':[], 't22freqs_err':[], 'amp2s':[], 'amp2s_err':[],}
# 	    T2Es[qubit] = {'t2es':[], 't2es_err':[]}
	
# 	for i in range(1000): #set number of repetitions.
# 		for qubit, params in enumerate(qubit_params)
# 			qubit_info = params[1] 
# 			qubit_freq = params[2]


# 	        if 1:
# 	            for rep_rate in rep_rates:
# 	                funcgen.set_frequency(rep_rate)
# 	                do_T1_plot(qubit_info, 500, np.concatenate((np.linspace(0, 10e3, 21), np.linspace(11e3, 60e3, 50))), T1s[qubit_info], 300*(qubit_ind+1))
# 	                do_T2_plot(qubit_info, 500, np.linspace(0, 10e3, 101), 1000e3, T2s[qubit_info], 301*(qubit_ind+1), double_freq=False)
# 	                do_T2echo_plot(qubit_info, 500, np.linspace(1e3, 20e3, 101), 500e3, T2Es[qubit_info], 302*(qubit_ind+1))

def do_ROspec_plot(qubit_info, n_avg, freqs, ro_powers, ro_fits, fig_num, var=None):
    from scripts.single_cavity import rocavspectroscopy
    alazar.set_naverages(n_avg)
    rospec = rocavspectroscopy.ROCavSpectroscopy(qubit_info, ro_powers, freqs) #qubit_pulse=np.pi/2
    rospec.measure()
    plt.close()

    ro_fits['x0s'].append(rospec.fit_params[0][2])
    ro_fits['x0s_err'].append(rospec.fit_params[1][2])
    ro_fits['As'].append(rospec.fit_params[0][1])
    ro_fits['As_err'].append(rospec.fit_params[1][1])
    ro_fits['ws'].append(rospec.fit_params[0][3])
    ro_fits['ws_err'].append(rospec.fit_params[1][3])
    if var!=None:
        ro_fits['vars'].append(var)
    plt.figure(fig_num)
    plt.clf()
    if ro_fits['vars']==[]:
        plt.subplot(311).axis(xmin=-len(ro_fits['x0s'])*0.10, xmax=len(ro_fits['x0s'])*1.10)
        plt.errorbar(range(len(ro_fits['x0s'])),ro_fits['x0s'],ro_fits['x0s_err'],fmt='go')
    else:
        xmin=min(ro_fits['vars'])
        xmax=max(ro_fits['vars'])
        plt.subplot(311).axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax))
        plt.errorbar(ro_fits['vars'],ro_fits['x0s'],ro_fits['x0s_err'],fmt='go')
    plt.xlabel("Measurement iterations")
    plt.ylabel("Center frequency(MHz)")

    if ro_fits['vars']==[]:
        plt.subplot(312).axis(xmin=-len(ro_fits['As'])*0.10, xmax=len(ro_fits['As'])*1.10)
        plt.errorbar(range(len(ro_fits['As'])),ro_fits['As'],ro_fits['As_err'],fmt='go')
    else:
        xmin=min(ro_fits['vars'])
        xmax=max(ro_fits['vars'])
        plt.subplot(312).axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax))
        plt.errorbar(ro_fits['vars'],ro_fits['As'],ro_fits['As_err'],fmt='go')
    plt.xlabel("Measurement iterations")
    plt.ylabel("Amplitude")

    if ro_fits['vars']==[]:
        plt.subplot(313).axis(xmin=-len(ro_fits['ws'])*0.10, xmax=len(ro_fits['ws'])*1.10)
        plt.errorbar(range(len(ro_fits['ws'])),ro_fits['ws'],ro_fits['ws_err'],fmt='go')
    else:
        xmin=min(ro_fits['vars'])
        xmax=max(ro_fits['vars'])
        plt.subplot(313).axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax))
        plt.errorbar(ro_fits['vars'],ro_fits['ws'],ro_fits['ws_err'],fmt='go')
    plt.xlabel("Measurement iterations")
    plt.ylabel("Width")
    return rospec


def do_spec_plot(qubit_info, n_avg, freqs, spec_params, spec_fits, fig_num, plen=50000, amp=0.01,var=None):
    from scripts.single_qubit import spectroscopy as spectroscopy
    alazar.set_naverages(n_avg)
    s = spectroscopy.Spectroscopy(qubit_info, freqs, spec_params,
                                     plen, amp, plot_seqs=False,subtraction = False) #1=1ns5
    s.measure()
    plt.close()
    spec_fits['x0s'].append(s.fit_params['x0'].value)
    spec_fits['x0s_err'].append(s.fit_params['x0'].stderr)
    spec_fits['ofs'].append(s.fit_params['ofs'].value)
    spec_fits['ofs_err'].append(s.fit_params['ofs'].stderr)
    spec_fits['ws'].append(s.fit_params['w'].value)
    spec_fits['ws_err'].append(s.fit_params['w'].stderr)
    if var!=None:
        spec_fits['vars'].append(var)
    plt.figure(fig_num)
    plt.clf()
    if spec_fits['vars']==[]:
        plt.subplot(311).axis(xmin=-len(spec_fits['x0s'])*0.10, xmax=len(spec_fits['x0s'])*1.10)
        plt.errorbar(range(len(spec_fits['x0s'])),spec_fits['x0s'],spec_fits['x0s_err'],fmt='go')
    else:
        xmin=min(spec_fits['vars'])
        xmax=max(spec_fits['vars'])
        plt.subplot(311).axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax))
        plt.errorbar(spec_fits['vars'],spec_fits['x0s'],spec_fits['x0s_err'],fmt='go')
    plt.xlabel("Measurement iterations")
    plt.ylabel("Center frequency(MHz)")

    if spec_fits['vars']==[]:
        plt.subplot(312).axis(xmin=-len(spec_fits['ofs'])*0.10, xmax=len(spec_fits['ofs'])*1.10)
        plt.errorbar(range(len(spec_fits['ofs'])),spec_fits['ofs'],spec_fits['ofs_err'],fmt='go')
    else:
        xmin=min(spec_fits['vars'])
        xmax=max(spec_fits['vars'])
        plt.subplot(312).axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax))
        plt.errorbar(spec_fits['vars'],spec_fits['ofs'],spec_fits['ofs_err'],fmt='go')
    plt.xlabel("Measurement iterations")
    plt.ylabel("Offset")

    if spec_fits['vars']==[]:
        plt.subplot(313).axis(xmin=-len(spec_fits['ws'])*0.10, xmax=len(spec_fits['ws'])*1.10)
        plt.errorbar(range(len(spec_fits['ws'])),spec_fits['ws'],spec_fits['ws_err'],fmt='go')
    else:
        xmin=min(spec_fits['vars'])
        xmax=max(spec_fits['vars'])
        plt.subplot(313).axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax))
        plt.errorbar(spec_fits['vars'],spec_fits['ws'],spec_fits['ws_err'],fmt='go')
    plt.xlabel("Measurement iterations")
    plt.ylabel("Width")
    return s

def do_T1(qubit_info, delays, double_exp = False):
    from scripts.single_qubit import T1measurement
    t1 = T1measurement.T1Measurement(qubit_info, delays)
    t1.data.set_attrs(field_current=field)
    t1.data.set_attrs(temperature=temp)
#    t1.data.set_attrs(laser_power=voltage)
    t1.measure()
    plt.close()
    return t1
    

def do_T1_plot(qubit_info, n_avg, delays, t1_fits, fig_num, double_exp = False, var=None):
    alazar.set_naverages(n_avg)
    t1 = do_T1(qubit_info, delays)
    t1_fits['t1s'].append(t1.fit_params['tau'].value)
    t1_fits['t1s_err'].append(t1.fit_params['tau'].stderr)
    t1_fits['ofs'].append(t1.fit_params['ofs'].value)
    t1_fits['ofs_err'].append(t1.fit_params['ofs'].stderr)
    t1_fits['amps'].append(t1.fit_params['A'].value)
    t1_fits['amps_err'].append(t1.fit_params['A'].stderr)
    if var!=None:
        t1_fits['vars'].append(var)
    plt.figure(fig_num)
    plt.clf()
    if t1_fits['vars']==[]:
        plt.subplot(211).axis(xmin=-len(t1_fits['t1s'])*0.10, xmax=len(t1_fits['t1s'])*1.10)
        plt.errorbar(range(len(t1_fits['t1s'])),t1_fits['t1s'],t1_fits['t1s_err'],fmt='go')
    else:
        xmin=min(t1_fits['vars'])
        xmax=max(t1_fits['vars'])
        plt.subplot(211).axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax))
        plt.errorbar(t1_fits['vars'],t1_fits['t1s'],t1_fits['t1s_err'],fmt='go')
    plt.xlabel("Measurement iterations")
    plt.ylabel("T1(us)")
    if t1_fits['vars']==[]:
        plt.subplot(212).axis(xmin=-len(t1_fits['t1s'])*0.10, xmax=len(t1_fits['t1s'])*1.10)
        plt.errorbar(range(len(t1_fits['amps'])),t1_fits['amps'],t1_fits['amps_err'],fmt='go')
    else:
        xmin=min(t1_fits['vars'])
        xmax=max(t1_fits['vars'])
        plt.subplot(212).axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax))
        plt.errorbar(t1_fits['vars'],t1_fits['amps'],t1_fits['amps_err'],fmt='go')
    plt.xlabel("Measurement iterations")
    plt.ylabel("Amplitude")
    
def do_T1_phonon(qubit_info, delays, amp, piLength, sigma = 10):
    from scripts.single_qubit import stark_swap 
    t1 = stark_swap.phonon_T1(qubit_info, 
            delays, phonon_pi = piLength, amp = amp,
            sigma = sigma,
           )
    t1.measure()
    plt.close()
    return t1

def do_T1_phonon_plot(qubit_info, n_avg, delays, amp, piLength, t1_fits, fig_num, sigma = 10, var=None):
    alazar.set_naverages(n_avg)
    t1 = do_T1_phonon(qubit_info, delays, amp, piLength, sigma)
    t1_fits['t1s'].append(t1.fit_params['tau'].value)
    t1_fits['t1s_err'].append(t1.fit_params['tau'].stderr)
    t1_fits['ofs'].append(t1.fit_params['ofs'].value)
    t1_fits['ofs_err'].append(t1.fit_params['ofs'].stderr)
    t1_fits['amps'].append(t1.fit_params['A'].value)
    t1_fits['amps_err'].append(t1.fit_params['A'].stderr)
    if var!=None:
        t1_fits['vars'].append(var)
    plt.figure(fig_num)
    plt.clf()
    if t1_fits['vars']==[]:
        plt.subplot(211).axis(xmin=-len(t1_fits['t1s'])*0.10, xmax=len(t1_fits['t1s'])*1.10)
        plt.errorbar(range(len(t1_fits['t1s'])),t1_fits['t1s'],t1_fits['t1s_err'],fmt='go')
    else:
        xmin=min(t1_fits['vars'])
        xmax=max(t1_fits['vars'])
        plt.subplot(211).axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax))
        plt.errorbar(t1_fits['vars'],t1_fits['t1s'],t1_fits['t1s_err'],fmt='go')
    plt.xlabel("Measurement iterations")
    plt.ylabel("T1(us)")
    if t1_fits['vars']==[]:
        plt.subplot(212).axis(xmin=-len(t1_fits['t1s'])*0.10, xmax=len(t1_fits['t1s'])*1.10)
        plt.errorbar(range(len(t1_fits['amps'])),t1_fits['amps'],t1_fits['amps_err'],fmt='go')
    else:
        xmin=min(t1_fits['vars'])
        xmax=max(t1_fits['vars'])
        plt.subplot(212).axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax))
        plt.errorbar(t1_fits['vars'],t1_fits['amps'],t1_fits['amps_err'],fmt='go')
    plt.xlabel("Measurement iterations")
    plt.ylabel("Amplitude")

def do_T2(qubit_info, delays, detune, fix_freq=None, fit_type='exp_decay_sine',):
    from scripts.single_qubit import T2measurement
    t2 = T2measurement.T2Measurement(qubit_info, delays, detune=detune, fix_freq = fix_freq, fit_type = fit_type)
    t2.data.set_attrs(field_current=field)
    t2.data.set_attrs(temperature=temp)
#    t2.data.set_attrs(laser_power=voltage)
    t2.measure()
    plt.close()
    return t2

def do_T2_plot(qubit_info, n_avg, delays, detune, t2_fits, fig_num, fix_freq=None, fit_type='exp_decay_sine', var=None):
    alazar.set_naverages(n_avg)
    t2 = do_T2(qubit_info, delays, detune, fix_freq, fit_type)

    if (t2!=None):
        t2_fits['t2s'].append(t2.fit_params['tau'].value)
        t2_fits['t2s_err'].append(t2.fit_params['tau'].stderr)
        t2_fits['t2freqs'].append(t2.fit_params['f'].value*1000 - detune/1e6)
        t2_fits['t2freqs_err'].append(t2.fit_params['f'].stderr*1000.0)
        t2_fits['amps'].append(t2.fit_params['A'].value)
        t2_fits['amps_err'].append(t2.fit_params['A'].stderr)
        # if double_freq == True:
        #     t2_fits['t22s'].append(t2.fit_params['tau2'].value)
        #     t2_fits['t22s_err'].append(t2.fit_params['tau2'].stderr)
        #     t2_fits['t22freqs'].append(t2.fit_params['freq2'].value*1000 -detune/1e6)
        #     t2_fits['t22freqs_err'].append(t2.fit_params['freq2'].stderr*1000.0)
        #     t2_fits['amp2s'].append(t2.fit_params['amp2'].value)
        #     t2_fits['amp2s_err'].append(t2.fit_params['amp2'].stderr)
    if var!=None:
        t2_fits['vars'].append(var)
    if fit_type == 'exp_decay_sine':
        plt.figure(fig_num)
        plt.clf()
        if t2_fits['vars']==[]:       
            plt.subplot(211).axis(xmin=-len(t2_fits['t2s'])*0.10, xmax=len(t2_fits['t2s'])*1.10, ymin= min(t2_fits['t2s'])*0.7, ymax=max(t2_fits['t2s'])*1.3)
            plt.errorbar(range(len(t2_fits['t2s'])),t2_fits['t2s'],t2_fits['t2s_err'],fmt='rs')
        else:
            xmin=min(t2_fits['vars'])
            xmax=max(t2_fits['vars'])    
            plt.subplot(211).axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax), ymin= min(t2_fits['t2s'])*0.7, ymax=max(t2_fits['t2s'])*1.3)
            plt.errorbar(t2_fits['vars'],t2_fits['t2s'],t2_fits['t2s_err'],fmt='rs')
        plt.xlabel("Measurement iterations")
        plt.ylabel("T2(us)")
        if t2_fits['vars']==[]: 
            plt.subplot(212).axis(xmin=-len(t2_fits['t2freqs'])*0.10, xmax=len(t2_fits['t2freqs'])*1.10, ymin=min(t2_fits['t2freqs'])-0.02, ymax=max(t2_fits['t2freqs'])+0.02)
            plt.errorbar(range(len(t2_fits['t2freqs'])),t2_fits['t2freqs'],t2_fits['t2freqs_err'],fmt='b^')
        else:
            xmin=min(t2_fits['vars'])
            xmax=max(t2_fits['vars'])
            plt.subplot(212).axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax), ymin=min(t2_fits['t2freqs'])-0.02, ymax=max(t2_fits['t2freqs'])+0.02)
            plt.errorbar(t2_fits['vars'],t2_fits['t2freqs'],t2_fits['t2freqs_err'],fmt='b^')   
        plt.xlabel("Measurement iterations")
        plt.ylabel("Ramsey Freq.(MHz) (= Actual Qubit Freq. - Drive Freq.)")


    # if fit_type == 'exp_decay_sine':
    #     plt.figure(fig_num)
    #     plt.clf()
    #     plt.subplot(311).axis(xmin=-len(t2_fits['t2s'])*0.10, xmax=len(t2_fits['t2s'])*1.10, ymin= min(t2_fits['t2s'])*0.7, ymax=max(t2_fits['t22s'])*1.3)
    #     plt.errorbar(range(len(t2_fits['t2s'])),t2_fits['t2s'],t2_fits['t2s_err'],fmt='rs')
    #     plt.errorbar(range(len(t2_fits['t22s'])),t2_fits['t22s'],t2_fits['t22s_err'],fmt='b^')
    #     plt.ylabel("T2(us)")
        # plt.subplot(312).axis(xmin=-len(t2_fits['t2freqs'])*0.10, xmax=len(t2_fits['t2freqs'])*1.10,ymin= min(min(t2_fits['t2freqs']),min(t2_fits['t22freqs']))-0.02, ymax=max(max(t2_fits['t2freqs']), max(t2_fits['t22freqs']))+0.02)
        # plt.errorbar(range(len(t2_fits['t2freqs'])),t2_fits['t2freqs'],t2_fits['t2freqs_err'],fmt='rs')
        # plt.errorbar(range(len(t2_fits['t22freqs'])),t2_fits['t22freqs'],t2_fits['t22freqs_err'],fmt='b^')
        # plt.ylabel("Ramsey Freq.(MHz) (= Actual Qubit Freq. - Drive Freq.)")
        # plt.subplot(313).axis(xmin=-len(t2_fits['amps'])*0.10, xmax=len(t2_fits['amps'])*1.10,ymin= min(t2_fits['amp2s'])*0.8, ymax=max(t2_fits['amps'])*1.2)
        # plt.errorbar(range(len(t2_fits['amps'])),t2_fits['amps'],t2_fits['amps_err'],fmt='rs')
        # plt.errorbar(range(len(t2_fits['amp2s'])),t2_fits['amp2s'],t2_fits['amp2s_err'],fmt='b^')
        # plt.xlabel("Measurement iterations")
        # plt.ylabel("Amplitudes (AU)")

#    plt.semilogy()

def do_T2echo(qubit_info, delays, detune, fix_freq=None, fit_type='exp_decay_sine'):
    # t2e = T2measurement.T2Measurement(qubit_info, delays, detune, echotype=T2measurement.ECHO_HAHN, title='T2 Echo')
    from scripts.single_qubit import T2measurement    
    t2e = T2measurement.T2Measurement(qubit_info, delays, detune, echotype=T2measurement.ECHO_CPMG, fix_freq = fix_freq, fit_type = fit_type, title='T2 Echo')
    t2e.data.set_attrs(field_current=field)
    t2e.data.set_attrs(temperature=temp)
#    t2e.data.set_attrs(laser_power=voltage)
    t2e.measure()
    plt.close()
    return t2e

def do_T2echo_plot(qubit_info, n_avg, delays, detune, t2E_fits, fig_num, fix_freq=None, fit_type='exp_decay_sine', var=None):
    alazar.set_naverages(n_avg)
    t2e = do_T2echo(qubit_info, delays, detune, fix_freq, fit_type)
    if fit_type == 'gaussian_decay':
        tname = 'sigma'
    else:
        tname = 'tau'

    if t2e!=None:
        t2E_fits['t2es'].append(t2e.fit_params[tname].value)
        t2E_fits['t2es_err'].append(t2e.fit_params[tname].stderr)
    if var!=None:
        t2E_fits['vars'].append(var)

    plt.figure(fig_num)
    plt.clf()
    if t2E_fits['vars']==[]:   
        plt.axis(xmin=-len(t2E_fits['t2es'])*0.10, xmax=len(t2E_fits['t2es'])*1.10, ymin= min(t2E_fits['t2es'])*0.8, ymax=max(t2E_fits['t2es'])*1.2)
        plt.errorbar(range(len(t2E_fits['t2es'])),t2E_fits['t2es'],t2E_fits['t2es_err'],fmt='mv') # magenta color and v-shape markers
    else:
        xmin=min(t2E_fits['vars'])
        xmax=max(t2E_fits['vars'])
        plt.axis(xmin=xmin-0.1*abs(xmin), xmax=xmax+0.1*abs(xmax), ymin= min(t2E_fits['t2es'])*0.8, ymax=max(t2E_fits['t2es'])*1.2)
        plt.errorbar(t2E_fits['vars'],t2E_fits['t2es'],t2E_fits['t2es_err'],fmt='mv') # magenta color and v-shape markers
    plt.xlabel("Measurement iterations")
    plt.ylabel("T2Echo(us)")

def smart_T1_delays(T1_int=90e3, QPT1=1.5e6, half_decay_point=1e6, eff_T1_delay=800.0, probe_point=0.5, meas_per_QPinj=30, meas_per_reptime=5):
    """
    T1_int = 90e3                  # Intrinsic T1 of the qubit
    QPT1 = 1.5e6                    # Guess the lifetime of the quasiparticles
    half_decay_point = 1e6        # The QP_delay time that would make qubit relax halfway to ground state with T1_delay=0, i.e. relax during readout pulse
    eff_T1_delay = 800.0            # The effective T1_delay due to the finite length of the readout pulse, usually taken as readout pulse length/2
    """
#    rep_time = 1.0e9/fg.get_frequency()
#    T1_QPref = 1/(np.log(2)/eff_T1_delay-1/T1_int)      # T1 at half decay point = effective readout delay/ln(2), excluding intrinsic part giving the T1 due to quasiparticles
#    n_delayless = int(half_decay_point/rep_time)           # Number of points with T1_delay = 0
#
##    QP_times_s = np.linspace(rep_time, half_decay_point, n_delayless)
#    T1_delays_s = np.linspace(0, 0, n_delayless)
#    QP_times_l = np.linspace(half_decay_point+rep_time, meas_per_QPinj*rep_time, meas_per_QPinj-n_delayless)
#    T1_delays_l = np.log(2)/(1/T1_int+1/T1_QPref*np.exp(-(QP_times_l-half_decay_point)/QPT1))-eff_T1_delay
##    QP_times = np.concatenate((QP_times_s, QP_times_l))
#    T1_delays = np.concatenate((T1_delays_s, T1_delays_l))

    rep_time = 1.0e9/fg.get_frequency()
    n_points = meas_per_QPinj * meas_per_reptime
    step_time = rep_time / meas_per_reptime
    T1_QPref = 1/(np.log(2)/eff_T1_delay-1/T1_int)      # T1 at half decay point = effective readout delay/ln(2), excluding intrinsic part giving the T1 due to quasiparticles

    QP_times = np.linspace(0, (n_points-1)*step_time, n_points)
    T1_est = 1/(1/T1_int+1/T1_QPref*np.exp(-(QP_times-half_decay_point)/QPT1))
    T1_delays = -np.log(probe_point)*T1_est-eff_T1_delay
    for j, delay in enumerate(T1_delays):
        if delay < 0:
            T1_delays[j]=0.0
    return T1_delays

def do_QPdecay(qubit_info, T1_delay, **kwargs):
    rep_time = 1e9/fg.get_frequency()
    qpd = QPdecay.QPdecay(qubit_info, T1_delay, rep_time, **kwargs)
    qpd.data.set_attrs(field_current=field)
    qpd.data.set_attrs(temperature=temp)
#    qpd.data.set_attrs(T1_delay=T1_delay)
    qpd.data.set_attrs(inj_power=ag3.get_power())
#    qpd.data.set_attrs(laser_voltage=laser_info.get_DCOffset())
#    qpd.measure()
#    plt.close()
    return qpd

def do_QPdecay_plot(qubit_info, n_avg, T1_delay, qpd_fits, fig_num, **kwargs):
    alz.set_naverages(n_avg)
    ag3.set_rf_on(True)
    qpd = do_QPdecay(qubit_info, T1_delay, **kwargs)
    qpd.measure()
    plt.close()
    if qpd!=None:
        qpd_fits['qpt1s'].append(qpd.fit_params['tau'].value/1000.0)
        qpd_fits['qpt1s_err'].append(qpd.fit_params['tau'].stderr/1000.0)
        qpd_fits['qpofs'].append(qpd.fit_params['ofs'].value)
        qpd_fits['qpofs_err'].append(qpd.fit_params['ofs'].stderr)
#        qpd_fits['amps'].append(qpd.fit_params['amplitude'].value)
    qpofs_array = np.array(qpd_fits['qpofs'])
    qpofs_err_array = np.array(qpd_fits['qpofs_err'])
    plt.figure(fig_num)
    plt.clf()
    plt.subplot(211).axis(xmin=-len(qpd_fits['qpt1s'])*0.10, xmax=len(qpd_fits['qpt1s'])*1.10)#, ymin=0, ymax=1)
    plt.errorbar(range(len(qpd_fits['qpt1s'])),qpd_fits['qpt1s'],qpd_fits['qpt1s_err'],fmt='go')
    plt.ylabel("Tau QP(ms)")
    plt.subplot(212).axis(xmin=-len(np.array(qpd_fits['qpofs']))*0.10, xmax=len(np.array(qpd_fits['qpofs']))*1.10)#, ymin=10, ymax=30)
    plt.errorbar(range(len(qpofs_array)), 1/qpofs_array, qpofs_err_array/qpofs_array/qpofs_array, fmt='b^')
    plt.xlabel("Measurement iterations")
    plt.ylabel("Qubit T1-floor(us)")
    ag3.set_rf_on(False)
    return qpd



def do_FT1(qubit_info, ef_info, delays):
    ft1 = FT1measurement.FT1Measurement(qubit_info, ef_info, delays)
    ft1.data.set_attrs(field_current=field)
    ft1.data.set_attrs(temperature=temp)
    ft1.measure()
    plt.close()
    return ft1

def do_FT1_plot(qubit_info, ef_info, n_avg, delays, ft1_fits, fig_num):
    alz.set_naverages(n_avg)
    brick1.set_rf_on(True)
    ft1 = do_FT1(qubit_info, ef_info, delays)
    if ft1!=None:
        ft1_fits['ft1s'].append(ft1.fit_params['tau'].value/1000.0)
        ft1_fits['ft1s_err'].append(ft1.fit_params['tau'].stderr/1000.0)
        ft1_fits['ofs'].append(ft1.fit_params['ofs'].value)
        ft1_fits['amps'].append(ft1.fit_params['amplitude'].value)
    plt.figure(fig_num)
    plt.clf()
    plt.axis(xmin=-len(ft1_fits['ft1s'])*0.10, xmax=len(ft1_fits['ft1s'])*1.10, ymin= min(ft1_fits['ft1s'])*0.8, ymax=max(ft1_fits['ft1s'])*1.2)
    plt.errorbar(range(len(ft1_fits['ft1s'])),ft1_fits['ft1s'],ft1_fits['ft1s_err'],fmt='go')
    plt.xlabel("Measurement iterations")
    plt.ylabel("FT1(us)")
    brick1.set_rf_on(False)

def do_EFT2(qubit_info, ef_info, delays, detune, double_freq=False, QP_injection_delay=None, QP_injection_length=10e3):
    eft2 = EFT2measurement.EFT2Measurement(qubit_info, ef_info, delays, detune=detune, double_freq=double_freq)
    eft2.data.set_attrs(field_current=field)
    eft2.data.set_attrs(temperature=temp)
    eft2.measure()
    plt.close()
    return eft2

def do_EFT2_plot(qubit_info, ef_info, n_avg, delays, detune, ft2_fits, fig_num, double_freq=False, QP_injection_delay=None, QP_injection_length=10e3, laser_power = None):
    alz.set_naverages(n_avg)
    brick1.set_rf_on(True)
    eft2 = do_EFT2(qubit_info, ef_info, delays, detune, double_freq, QP_injection_delay, QP_injection_length)
    if (eft2!=None):
        ft2_fits['eft2s'].append(eft2.fit_params['tau'].value/1000)
        ft2_fits['eft2s_err'].append(eft2.fit_params['tau'].stderr/1000.0)
        ft2_fits['eft2freqs'].append(eft2.fit_params['freq'].value*1000 - detune/1e6)
        ft2_fits['eft2freqs_err'].append(eft2.fit_params['freq'].stderr*1000.0)
        ft2_fits['eft2amps'].append(eft2.fit_params['amp'].value)
        ft2_fits['eft2amps_err'].append(eft2.fit_params['amp'].stderr)
        if double_freq == True:
            ft2_fits['eft22s'].append(eft2.fit_params['tau2'].value/1000)
            ft2_fits['eft22s_err'].append(eft2.fit_params['tau2'].stderr/1000.0)
            ft2_fits['eft22freqs'].append(eft2.fit_params['freq2'].value*1000 -detune/1e6)
            ft2_fits['eft22freqs_err'].append(eft2.fit_params['freq2'].stderr*1000.0)
            ft2_fits['eft2amp2s'].append(eft2.fit_params['amp2'].value)
            ft2_fits['eft2amp2s_err'].append(eft2.fit_params['amp2'].stderr)
        if QP_injection_delay is not None:
            ft2_fits['eft2s_QP'].append(eft2.fit_params['tau'].value/1000)
            ft2_fits['eft2s_QP_err'].append(eft2.fit_params['tau'].stderr/1000.0)
            ft2_fits['eft2freqs_QP'].append(eft2.fit_params['freq'].value*1000 -detune/1e6)
            ft2_fits['eft2freqs_QP_err'].append(eft2.fit_params['freq'].stderr*1000.0)

    if double_freq == False and QP_injection_delay is None:
        plt.figure(fig_num)
        plt.clf()
        plt.subplot(211).axis(xmin=-len(ft2_fits['eft2s'])*0.10, xmax=len(ft2_fits['eft2s'])*1.10, ymin= min(ft2_fits['eft2s'])*0.7, ymax=max(ft2_fits['eft2s'])*1.3)
        plt.errorbar(range(len(ft2_fits['eft2s'])),ft2_fits['eft2s'],ft2_fits['eft2s_err'],fmt='rs')
        plt.ylabel("EFT2(us)")
        plt.subplot(212).axis(xmin=-len(ft2_fits['eft2freqs'])*0.10, xmax=len(ft2_fits['eft2freqs'])*1.10, ymin=min(ft2_fits['eft2freqs'])-0.02, ymax=max(ft2_fits['eft2freqs'])+0.02)
        plt.errorbar(range(len(ft2_fits['eft2freqs'])),ft2_fits['eft2freqs'],ft2_fits['eft2freqs_err'],fmt='b^')
        plt.xlabel("Measurement iterations")
        plt.ylabel("Ramsey Freq.(MHz) (= Actual Qubit Freq. - Drive Freq.)")
    if double_freq == False and QP_injection_delay is not None:
        plt.figure(fig_num)
        plt.clf()
        plt.subplot(211).axis(xmin=-len(ft2_fits['eft2s_QP'])*0.10, xmax=len(ft2_fits['eft2s_QP'])*1.10, ymin= min(ft2_fits['eft2s_QP'])*0.7, ymax=max(ft2_fits['eft2s_QP'])*1.3)
        plt.errorbar(range(len(ft2_fits['eft2s_QP'])),ft2_fits['eft2s_QP'],ft2_fits['eft2s_QP_err'],fmt='rs')
        plt.ylabel("EFT2 with QP injection (us)")
        plt.subplot(212).axis(xmin=-len(ft2_fits['eft2freqs_QP'])*0.10, xmax=len(ft2_fits['eft2freqs_QP'])*1.10, ymin=min(ft2_fits['eft2freqs_QP'])-0.02, ymax=max(ft2_fits['eft2freqs_QP'])+0.02)
        plt.errorbar(range(len(ft2_fits['eft2freqs_QP'])),ft2_fits['eft2freqs_QP'],ft2_fits['eft2freqs_QP_err'],fmt='b^')
        plt.xlabel("Measurement iterations")
        plt.ylabel("Ramsey Freq.(MHz) (= Actual Qubit Freq. - Drive Freq.)")
    if double_freq is True:
        plt.figure(fig_num)
        plt.clf()
        plt.subplot(311).axis(xmin=-len(ft2_fits['eft2s'])*0.10, xmax=len(ft2_fits['eft2s'])*1.10, ymin= min(ft2_fits['eft2s'])*0.7, ymax=max(ft2_fits['eft22s'])*1.3)
        plt.errorbar(range(len(ft2_fits['eft2s'])),ft2_fits['eft2s'],ft2_fits['eft2s_err'],fmt='rs')
        plt.errorbar(range(len(ft2_fits['eft22s'])),ft2_fits['eft22s'],ft2_fits['eft22s_err'],fmt='b^')
        plt.ylabel("EFT2(us)")
        plt.subplot(312).axis(xmin=-len(ft2_fits['eft2freqs'])*0.10, xmax=len(ft2_fits['eft2freqs'])*1.10,ymin= min(min(ft2_fits['eft2freqs']),min(ft2_fits['eft22freqs']))-0.02, ymax=max(max(ft2_fits['eft2freqs']), max(ft2_fits['eft22freqs']))+0.02)
        plt.errorbar(range(len(ft2_fits['eft2freqs'])),ft2_fits['eft2freqs'],ft2_fits['eft2freqs_err'],fmt='rs')
        plt.errorbar(range(len(ft2_fits['eft22freqs'])),ft2_fits['eft22freqs'],ft2_fits['eft22freqs_err'],fmt='b^')
        plt.ylabel("Ramsey Freq.(MHz) (= Actual Qubit Freq. - Drive Freq.)")
        plt.subplot(313).axis(xmin=-len(ft2_fits['eft2amps'])*0.10, xmax=len(ft2_fits['eft2amps'])*1.10,ymin= min(ft2_fits['eft2amp2s'])*0.8, ymax=max(ft2_fits['eft2amps'])*1.2)
        plt.errorbar(range(len(ft2_fits['eft2amps'])),ft2_fits['eft2amps'],ft2_fits['eft2amps_err'],fmt='rs')
        plt.errorbar(range(len(ft2_fits['eft2amp2s'])),ft2_fits['eft2amp2s'],ft2_fits['eft2amp2s_err'],fmt='b^')
        plt.xlabel("Measurement iterations")
        plt.ylabel("Amplitudes (AU)")
    brick1.set_rf_on(False)

def do_EFT2echo(qubit_info, ef_info, delays, detune, laser_power = None):
    eft2e = EFT2measurement.EFT2Measurement(qubit_info, ef_info, delays, detune, echotype=EFT2measurement.ECHO_HAHN, title='EFT2 Echo')
    eft2e.data.set_attrs(field_current=field)
    eft2e.data.set_attrs(temperature=temp)
#    t2e.data.set_attrs(laser_power=voltage)
    eft2e.measure()
    plt.close()
    return eft2e

def do_EFT2echo_plot(qubit_info, ef_info, n_avg, delays, detune, t2E_fits, fig_num, laser_power = None):
    alz.set_naverages(n_avg)
    brick1.set_rf_on(True)
    eft2e = do_EFT2echo(qubit_info, ef_info, delays, detune, laser_power = laser_power)
    if eft2e!=None:
        t2E_fits['eft2es'].append(eft2e.fit_params['tau'].value/1000)
        t2E_fits['eft2es_err'].append(eft2e.fit_params['tau'].stderr/1000)
    plt.figure(fig_num)
    plt.clf()
    plt.axis(xmin=-len(t2E_fits['eft2es'])*0.10, xmax=len(t2E_fits['eft2es'])*1.10, ymin= min(t2E_fits['eft2es'])*0.8, ymax=max(t2E_fits['eft2es'])*1.2)
    plt.errorbar(range(len(t2E_fits['eft2es'])),t2E_fits['eft2es'],t2E_fits['eft2es_err'],fmt='mv') # magenta color and v-shape markers
    plt.xlabel("Measurement iterations")
    plt.ylabel("EFT2Echo(us)")
    brick1.set_rf_on(False)

def do_GFT2(qubit_info, ef_info, delays, detune, double_freq=False, QP_injection_delay=None, QP_injection_length=10e3):
    gft2 = GFT2measurement.GFT2Measurement(qubit_info, ef_info, delays, detune=detune, double_freq=double_freq)
    gft2.data.set_attrs(field_current=field)
    gft2.data.set_attrs(temperature=temp)
    gft2.measure()
    plt.close()
    return gft2

def do_GFT2_plot(qubit_info, ef_info, n_avg, delays, detune, ft2_fits, fig_num, double_freq=False, QP_injection_delay=None, QP_injection_length=10e3, laser_power = None):
    alz.set_naverages(n_avg)
    brick1.set_rf_on(True)
    gft2 = do_GFT2(qubit_info, ef_info, delays, detune, double_freq, QP_injection_delay, QP_injection_length)
    if (gft2!=None):
        ft2_fits['gft2s'].append(gft2.fit_params['tau'].value/1000)
        ft2_fits['gft2s_err'].append(gft2.fit_params['tau'].stderr/1000.0)
        ft2_fits['gft2freqs'].append(gft2.fit_params['freq'].value*1000 - detune/1e6)
        ft2_fits['gft2freqs_err'].append(gft2.fit_params['freq'].stderr*1000.0)
        ft2_fits['gft2amps'].append(gft2.fit_params['amp'].value)
        ft2_fits['gft2amps_err'].append(gft2.fit_params['amp'].stderr)
        if double_freq == True:
            ft2_fits['gft22s'].append(gft2.fit_params['tau2'].value/1000)
            ft2_fits['gft22s_err'].append(gft2.fit_params['tau2'].stderr/1000.0)
            ft2_fits['gft22freqs'].append(gft2.fit_params['freq2'].value*1000 -detune/1e6)
            ft2_fits['gft22freqs_err'].append(gft2.fit_params['freq2'].stderr*1000.0)
            ft2_fits['gft2amp2s'].append(gft2.fit_params['amp2'].value)
            ft2_fits['gft2amp2s_err'].append(gft2.fit_params['amp2'].stderr)
        if QP_injection_delay is not None:
            ft2_fits['gft2s_QP'].append(gft2.fit_params['tau'].value/1000)
            ft2_fits['gft2s_QP_err'].append(gft2.fit_params['tau'].stderr/1000.0)
            ft2_fits['gft2freqs_QP'].append(gft2.fit_params['freq'].value*1000 -detune/1e6)
            ft2_fits['gft2freqs_QP_err'].append(gft2.fit_params['freq'].stderr*1000.0)

    if double_freq == False and QP_injection_delay is None:
        plt.figure(fig_num)
        plt.clf()
        plt.subplot(211).axis(xmin=-len(ft2_fits['gft2s'])*0.10, xmax=len(ft2_fits['gft2s'])*1.10, ymin= min(ft2_fits['gft2s'])*0.7, ymax=max(ft2_fits['gft2s'])*1.3)
        plt.errorbar(range(len(ft2_fits['gft2s'])),ft2_fits['gft2s'],ft2_fits['gft2s_err'],fmt='ks')
        plt.ylabel("GFT2(us)")
        plt.subplot(212).axis(xmin=-len(ft2_fits['gft2freqs'])*0.10, xmax=len(ft2_fits['gft2freqs'])*1.10, ymin=min(ft2_fits['gft2freqs'])-0.02, ymax=max(ft2_fits['gft2freqs'])+0.02)
        plt.errorbar(range(len(ft2_fits['gft2freqs'])),ft2_fits['gft2freqs'],ft2_fits['gft2freqs_err'],fmt='c^')
        plt.xlabel("Measurement iterations")
        plt.ylabel("Ramsey Freq.(MHz) (= Actual Qubit Freq. - Drive Freq.)")
    if double_freq == False and QP_injection_delay is not None:
        plt.figure(fig_num)
        plt.clf()
        plt.subplot(211).axis(xmin=-len(ft2_fits['gft2s_QP'])*0.10, xmax=len(ft2_fits['gft2s_QP'])*1.10, ymin= min(ft2_fits['gft2s_QP'])*0.7, ymax=max(ft2_fits['gft2s_QP'])*1.3)
        plt.errorbar(range(len(ft2_fits['gft2s_QP'])),ft2_fits['gft2s_QP'],ft2_fits['gft2s_QP_err'],fmt='ks')
        plt.ylabel("GFT2 with QP injection (us)")
        plt.subplot(212).axis(xmin=-len(ft2_fits['gft2freqs_QP'])*0.10, xmax=len(ft2_fits['gft2freqs_QP'])*1.10, ymin=min(ft2_fits['gft2freqs_QP'])-0.02, ymax=max(ft2_fits['gft2freqs_QP'])+0.02)
        plt.errorbar(range(len(ft2_fits['gft2freqs_QP'])),ft2_fits['gft2freqs_QP'],ft2_fits['gft2freqs_QP_err'],fmt='c^')
        plt.xlabel("Measurement iterations")
        plt.ylabel("Ramsey Freq.(MHz) (= Actual Qubit Freq. - Drive Freq.)")
    if double_freq is True:
        plt.figure(fig_num)
        plt.clf()
        plt.subplot(311).axis(xmin=-len(ft2_fits['gft2s'])*0.10, xmax=len(ft2_fits['gft2s'])*1.10, ymin= min(ft2_fits['gft2s'])*0.7, ymax=max(ft2_fits['gft22s'])*1.3)
        plt.errorbar(range(len(ft2_fits['gft2s'])),ft2_fits['gft2s'],ft2_fits['gft2s_err'],fmt='ks')
        plt.errorbar(range(len(ft2_fits['gft22s'])),ft2_fits['gft22s'],ft2_fits['gft22s_err'],fmt='c^')
        plt.ylabel("GFT2(us)")
        plt.subplot(312).axis(xmin=-len(ft2_fits['gft2freqs'])*0.10, xmax=len(ft2_fits['gft2freqs'])*1.10,ymin= min(min(ft2_fits['gft2freqs']),min(ft2_fits['gft22freqs']))-0.02, ymax=max(max(ft2_fits['gft2freqs']), max(ft2_fits['gft22freqs']))+0.02)
        plt.errorbar(range(len(ft2_fits['gft2freqs'])),ft2_fits['gft2freqs'],ft2_fits['gft2freqs_err'],fmt='ks')
        plt.errorbar(range(len(ft2_fits['gft22freqs'])),ft2_fits['gft22freqs'],ft2_fits['gft22freqs_err'],fmt='c^')
        plt.ylabel("Ramsey Freq.(MHz) (= Actual Qubit Freq. - Drive Freq.)")
        plt.subplot(313).axis(xmin=-len(ft2_fits['gft2amps'])*0.10, xmax=len(ft2_fits['gft2amps'])*1.10,ymin= min(ft2_fits['gft2amp2s'])*0.8, ymax=max(ft2_fits['gft2amps'])*1.2)
        plt.errorbar(range(len(ft2_fits['gft2amps'])),ft2_fits['gft2amps'],ft2_fits['gft2amps_err'],fmt='ks')
        plt.errorbar(range(len(ft2_fits['gft2amp2s'])),ft2_fits['gft2amp2s'],ft2_fits['gft2amp2s_err'],fmt='c^')
        plt.xlabel("Measurement iterations")
        plt.ylabel("Amplitudes (AU)")
    brick1.set_rf_on(False)

def do_GFT2echo(qubit_info, ef_info, delays, detune, laser_power = None):
    gft2e = GFT2measurement.GFT2Measurement(qubit_info, ef_info, delays, detune, echotype=EFT2measurement.ECHO_HAHN, title='GFT2 Echo')
    gft2e.data.set_attrs(field_current=field)
    gft2e.data.set_attrs(temperature=temp)
#    t2e.data.set_attrs(laser_power=voltage)
    gft2e.measure()
    plt.close()
    return gft2e

def do_GFT2echo_plot(qubit_info, ef_info, n_avg, delays, detune, t2E_fits, fig_num, laser_power = None):
    alz.set_naverages(n_avg)
    brick1.set_rf_on(True)
    gft2e = do_GFT2echo(qubit_info, ef_info, delays, detune, laser_power = laser_power)
    if gft2e!=None:
        t2E_fits['gft2es'].append(gft2e.fit_params['tau'].value/1000)
        t2E_fits['gft2es_err'].append(gft2e.fit_params['tau'].stderr/1000)
    plt.figure(fig_num)
    plt.clf()
    plt.axis(xmin=-len(t2E_fits['gft2es'])*0.10, xmax=len(t2E_fits['gft2es'])*1.10, ymin= min(t2E_fits['gft2es'])*0.8, ymax=max(t2E_fits['gft2es'])*1.2)
    plt.errorbar(range(len(t2E_fits['gft2es'])),t2E_fits['gft2es'],t2E_fits['gft2es_err'],fmt='yv') # yellow color and v-shape markers
    plt.xlabel("Measurement iterations")
    plt.ylabel("GFT2Echo(us)")
    brick1.set_rf_on(False)

def do_FT2echo_plot(qubit_info, ef_info, n_avg, delays, detune, t2E_fits, fig_num, laser_power = None):
    alz.set_naverages(n_avg)
    brick1.set_rf_on(True)
    eft2e = do_EFT2echo(qubit_info, ef_info, delays, detune, laser_power = laser_power)
    if eft2e!=None:
        t2E_fits['eft2es'].append(eft2e.fit_params['tau'].value/1000)
        t2E_fits['eft2es_err'].append(eft2e.fit_params['tau'].stderr/1000)
    plt.figure(fig_num)
    plt.clf()
    plt.axis(xmin=-len(t2E_fits['eft2es'])*0.10, xmax=len(t2E_fits['eft2es'])*1.10, ymin= min(t2E_fits['eft2es'])*0.8, ymax=max(t2E_fits['eft2es'])*1.2)
    plt.errorbar(range(len(t2E_fits['eft2es'])),t2E_fits['eft2es'],t2E_fits['eft2es_err'],fmt='mv', label='EFT2echo') # magenta color and v-shape markers
    plt.errorbar(range(len(t2E_fits['gft2es'])),t2E_fits['gft2es'],t2E_fits['gft2es_err'],fmt='yv', label='GFT2echo') # yellow color and v-shape markers
    plt.xlabel("Measurement iterations")
    plt.ylabel("FT2Echo(us)")

    gft2e = do_GFT2echo(qubit_info, ef_info, delays, detune, laser_power = laser_power)
    if gft2e!=None:
        t2E_fits['gft2es'].append(gft2e.fit_params['tau'].value/1000)
        t2E_fits['gft2es_err'].append(gft2e.fit_params['tau'].stderr/1000)
    plt.figure(fig_num)
    plt.clf()
    plt.axis(xmin=-len(t2E_fits['gft2es'])*0.10, xmax=len(t2E_fits['gft2es'])*1.10, ymin= min(t2E_fits['eft2es'])*0.8, ymax=max(t2E_fits['gft2es'])*1.2)
    plt.errorbar(range(len(t2E_fits['eft2es'])),t2E_fits['eft2es'],t2E_fits['eft2es_err'],fmt='mv', label='EFT2echo') # magenta color and v-shape markers
    plt.errorbar(range(len(t2E_fits['gft2es'])),t2E_fits['gft2es'],t2E_fits['gft2es_err'],fmt='yv', label='GFT2echo') # yellow color and v-shape markers
    plt.xlabel("Measurement iterations")
    plt.ylabel("FT2Echo(us)")
    brick1.set_rf_on(False)


def do_rabiup(qubit_info, ef_info, amps, QP_injection_delay=None, laser_power= None):
    if QP_injection_delay == None:
        rabiup = efrabi.EFRabi(qubit_info, ef_info, amps, laser_power = laser_power)
    else:
        rabiup = efrabi_QP.EFRabi_QP(qubit_info, ef_info, amps, QP_injection_delay, laser_power = laser_power)
        rabiup.data.set_attrs(QP_delay=QP_injection_delay)
    rabiup.data.set_attrs(field_current=field)
    rabiup.data.set_attrs(temperature=temp)
    rabiup.data.set_attrs(laser_power=laser_power)
    rabiup.measure()
    plt.close()
    return rabiup

def do_rabinoup(qubit_info, ef_info, amps, force_period, QP_injection_delay=None, laser_power=None):
    if QP_injection_delay == None:
        rabinoup = efrabi.EFRabi(qubit_info, ef_info, amps, first_pi=False, force_period=force_period,laser_power = laser_power)
    else:
        rabinoup = efrabi_QP.EFRabi_QP(qubit_info, ef_info, amps, first_pi=False, force_period=force_period, QP_delay=QP_injection_delay)
        rabinoup.data.set_attrs(QP_delay=QP_injection_delay)
    rabinoup.data.set_attrs(field_current=field)
    rabinoup.data.set_attrs(temperature=temp)
    rabinoup.data.set_attrs(laser_power=laser_power)
    rabinoup.measure()
    #population = 100*rabinoup.fit_params['amp'].value/(rabiup.fit_params['amp'].value+rabinoup.fit_params['amp'].value)
    plt.close()
    return rabinoup

def do_population_plot(qubit_info, ef_info, n_avg_rabiup, n_avg_rabinoup, amps, pops_fits, fig_num, QP_injection_delay=None, laser_power = None):
    brick1.set_rf_on(True)
    alz.set_naverages(n_avg_rabiup)
    rabiup = do_rabiup(qubit_info, ef_info, amps, QP_injection_delay, laser_power = laser_power)
    if rabiup!=None:
        pops_fits['rabiupAmp'].append(abs(rabiup.fit_params['amp'].value))
        pops_fits['rabiupAmp_err'].append(rabiup.fit_params['amp'].stderr)
    plt.figure(fig_num).show()
#    plt.clf()
    plt.subplot(211).axis(xmin=-len(pops_fits['rabiupAmp'])*0.10, xmax=len(pops_fits['rabiupAmp'])*1.10, ymin=min(pops_fits['rabiupAmp'])*0.7, ymax=max(pops_fits['rabiupAmp'])*1.3)
    plt.errorbar(range(len(pops_fits['rabiupAmp'])),pops_fits['rabiupAmp'],pops_fits['rabiupAmp_err'],fmt='b^')
    #plt.xlabel("Measurement iterations")
    plt.ylabel("Rabiup")

    alz.set_naverages(n_avg_rabinoup)
    rabinoup = do_rabinoup(qubit_info, ef_info, amps, force_period=rabiup.fit_params['period'].value, QP_injection_delay=QP_injection_delay, laser_power = laser_power)
    if rabinoup!=None:
        pops_fits['rabinoupAmp'].append(abs(rabinoup.fit_params['amp'].value))
        pops_fits['rabinoupAmp_err'].append(rabinoup.fit_params['amp'].stderr)
        #population.append(population)
    plt.figure(fig_num).show()
    plt.subplot(212).axis(xmin=-len(pops_fits['rabinoupAmp'])*0.10, xmax=len(pops_fits['rabinoupAmp'])*1.10, ymin=0.0, ymax=max(pops_fits['rabinoupAmp'])*2.0)
    plt.errorbar(range(len(pops_fits['rabinoupAmp'])),pops_fits['rabinoupAmp'],pops_fits['rabinoupAmp_err'],fmt='go')
    plt.xlabel("Measurement iterations")
    plt.ylabel("Rabinoup")
    brick1.set_rf_on(False)

'''
def do_qubitSSBspec()
   from scripts.single_qubit import ssbspec
   qubitSSBspec = ssbspec.SSBSpec(qubit_info, np.linspace(-3e6, 3e6, 51), plot_seqs=False)
   qubitSSBspec.measure()
   return qubitSSBspec
'''
