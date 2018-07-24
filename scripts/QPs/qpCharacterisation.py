import mclient
reload(mclient)
import numpy as np
import matplotlib.pyplot as plt
from pulseseq import sequencer, pulselib
import matplotlib as mpl
#from t1t2_plotting import smart_T1_delays
import math as math
from scripts.QPs.single_qubit import T1measurement_CW
from t1t2_plotting import do_T1_plot, do_T2_plot, do_T2echo_plot
from t1t2_plotting import do_FT1_plot, do_GFT2_plot, do_EFT2_plot, do_EFT2echo_plot, do_GFT2echo_plot, do_FT2echo_plot
from t1t2_plotting import do_QPdecay_plot, do_population_plot, smart_T1_delays, calibrate_IQ

#mpl.rcParams['figure.figsize']=[5,3.5]
#mpl.rcParams['axes.color_cycle'] = ['b', 'g', 'r', 'c', 'm', 'k']
alz = mclient.instruments['alazar']
fg = mclient.instruments['funcgen']
#laserfg = mclient.instruments['laserfg']

# Load old settings.
if 0:
    toload = ['AWG1','ag1','ag2', 'ag3' 'alazar', 'qFC14#1', 'eFC14#1','qubit_DO13#3', 'ef_DO13#3', 'qubit_DO13#4', 'ef_DO13#4']
    mclient.load_settings_from_file(r'c:\_data\settings\20131214\165409.set', toload)    # Last time-Rabi callibration
    bla

#qubits = mclient.get_qubits()
qubit_info = mclient.get_qubit_info('qubit_info_1')
#ef_info = mclient.get_qubit_info('qubit_ef_info')
#cavity_info = mclient.get_qubit_info('cavity0')

if 1: # T1_QP
    from scripts.QPs.single_qubit import T1measurement_QP
    for i in range(1):
        for l in [210e3]:
            for delay in [100e3]:
    #        delay = 0.02e6#np.array([0.1e3,2.5e3,40e3])#
    #            t1times = np.concatenate((np.linspace(0, 1e3,11), np.linspace(1e3,10e3, 13)))
    #            t1times = np.logspace(1, np.log10(10e3), 3 )
                alz.set_naverages(14e3)
                t1times = np.array([0, 100, 200, 500, 1e3, 10e3])
                t1 = T1measurement_QP.T1Measurement_QP(qubit_info, t1times, QP_delay=delay, inj_len=l, injection_marker='1m2')
            
                t1.measure()
    bla       
    
#if 0: # T1 with CW laser
#    alz.set_naverages(1500)
#    laserfg.set_output_on(0)    
#    laserfg.set_function('DC')
#    laserfg.set_output_on(0)
#    laserV=2.67
#    '''record the attenuation!'''
#    atten = 55
#    laserfg.set_DCOffset(laserV)    
#    laserfg.set_output_on(1)
#    delays =  np.linspace(0, 3e3, 101)
#    for i in range(1):
#        t1 = T1measurement_CW.T1Measurement_CW(qubit_info, delays, laserV, atten=atten)
#        t1.measure()
#    laserfg.set_output_on(0)
##    laserfg.set_Vlow(0)
##    laserfg.set_Vhigh(laserV)
#
#    bla
#    
     
        
if 0: #pulsed laser T1_QP
    from scripts.QPs.single_qubit import T1measurement_laser
    injection_length = 110e3
    QP_delay = 0.25e6
    delays = np.concatenate((np.linspace(0,1e3, 41), np.linspace(1e3, 8e3, 41)))
    '''record the attenuation! in dB'''
    atten = 48
    
    laserfg.set_output_on(False)
    alz.set_naverages(2000)      #averages pulled from alazar settings - plot updates every 100
    rep_rate = 500#1/(injection_length*1e-9 + QP_delay*1e-9 + )
    fg.set_frequency(int(rep_rate))  #rep rate 
    
    lvoltage = 2.67
    laserfg.set_Vlow(0.0)
    laserfg.set_Vhigh(lvoltage)
    laserfg.set_function('PULSE')
    laserfg.set_burst_on('True')
    rise_time = 3e3

    laserfg.set_output_on(True)
    #set delay to be negative to measure during the pulse
    t1 = T1measurement_laser.T1Measurement_laser(qubit_info, delays, inj_len = injection_length, QP_delay = QP_delay,
                                                edgewidth = rise_time, laser_voltage = lvoltage, atten = atten, double_exp=False)
    lfg_freq = 1.0/(t1.vpulse_len + 3*rise_time + max(0, QP_delay) + max(delays))*1e9
    laserfg.set_frequency(int(lfg_freq))

    laserfg.set_pulsewidth(int(t1.vpulse_len)*1e-9)
    laserfg.set_edgetime(rise_time*1e-9)
    t1.measure()
    
    laserfg.set_output_on(False)
    bla
    

if 0: #T2_QP
    from scripts.QPs.single_qubit import T2measurement_QP
    #    for i in range(1):
    alz.set_naverages(1000)
    for l in [110e3]:
        for delay in [0.5e6]:
#        delay = 0.02e6#np.array([0.1e3,2.5e3,40e3])#
            t2times = np.linspace(0, 2e3,101)
            t2 = T2measurement_QP.T2Measurement_QP(qubit_info, t2times, QP_delay=delay, detune=2400e3, double_freq=False, inj_len=l)# ,echotype = T2measurement_QP.ECHO_HAHN)
            t2.measure()
    bla            
    
   
    bla

#EF rabi_QP and laser in CW
if 0:
    from scripts.QPs.single_qubit import efrabi_QP
#    from scripts.QPs.single_qubit import efrabi_laser
#    laser_info = mclient.instruments['laserfg']
#    laser_info.set_function('PULS')
#    laser_info.set_Vhigh(1.5)
#    laser_info.set_Vlow(0)
#    laser_info.burst_mode()
#    laser_info.set_output_on(True)
#    pulse = 500e-6
#    plen = pulse
#    laser_info.set_pulsewidth(plen)
#    edge = 1e-6
#    laser_info.set_edgetime(edge)

#    for delay in np.concatenate((np.linspace(0.3e6,plen*1e9, 3), np.linspace(plen*1e9,plen*1e9+400e3, 4),np.linspace(plen*1e9+400e3, plen*1e9+3e6, 4))):
    length = 110e3    
    for delay in [0.4e6, 0.4e6, 0.8e6, 1e6, 2e6, 5e6]:
        if delay < 1.5e6:
            alz.set_naverages(1500)#1500)
            fg.set_frequency(400)
        else:
            alz.set_naverages(1000)
            fg.set_frequency(200)

#        efr = efrabi_laser.EFRabi_laser(qubit_info, ef_info, np.linspace(0, 1, 81), laser_delay= delay)
        efr = efrabi_QP.EFRabi_QP(qubit_info, ef_info, np.linspace(-0.3, 0.3, 81), QP_delay= delay, inj_len = length)
        efr.measure()
        period = efr.fit_params['period'].value

        if delay < 1.5e6:
            alz.set_naverages(40000)
            fg.set_frequency(400)
        else:
            alz.set_naverages(40000)
            fg.set_frequency(200)

        efr = efrabi_QP.EFRabi_QP(qubit_info, ef_info, np.linspace(-0.3, 0.3, 81), QP_delay=delay, first_pi=False, inj_len = length, force_period= period)
#        efr = efrabi_laser.EFRabi_laser(qubit_info, ef_info, np.linspace(0, 1, 81), laser_delay=delay, first_pi=False, force_period= period)
        efr.measure()


"""
Quasi-particle Decay
"""
if 0:
#    from scripts.single_qubit import T1measurement
    from scripts.single_qubit import QPdecay
    eff_T1_delay = 500.0
    meas_per_QPinj = 80
    meas_per_reptime = 2
    fg = mclient.instruments['funcgen']
    rep_time=1.0e9/fg.get_frequency()

    T1_delays = smart_T1_delays(T1_int=15.0e3, QPT1=10.0e6, half_decay_point = 6.0e6, eff_T1_delay=eff_T1_delay, probe_point=0.5, meas_per_QPinj=meas_per_QPinj, meas_per_reptime=meas_per_reptime)
    for i in range(5):
        alz.set_naverages(4000)
        qpd = QPdecay.QPdecay(qubit_info, T1_delays, rep_time, meas_per_reptime, meas_per_QPinj, fit_start=5, vg=0.2, ve=7.0, eff_T1_delay=eff_T1_delay, inj_len=360e3)
        qpd.measure()
        ag2 = mclient.instruments['ag2']
        qpd.data.set_attrs(inj_power=ag2.get_power())

        T1_delays = (T1_delays -np.log(0.5)*1000.0/qpd.invT1 - eff_T1_delay)/2.0
        for j, delay in enumerate(T1_delays):
            if delay < 0:
                T1_delays[j]=0.0

#        alz.set_naverages(2000)
#        t1 = T1measurement.T1Measurement(qubit_info, np.concatenate((np.linspace(0, 100e3, 81), np.linspace(100e3, 200e3, 81))), double_exp=False)
#        t1.measure()
'''quantum jumps'''
if 0:
    def moving_average(a, n=20) :
        ret = np.cumsum(a)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    '''we need to play something in the AWG before running this. e.g I can play a pi-pulse and then a long readout pulse.'''
    alz.setup_channels()
    alz.setup_clock()
    alz.setup_trigger()

    alz.setup_shots(1)
    '''this sets the period of the filter. i.e how long we are integrating over for each point. Default is 20ns. I need this to be ~1.5us'''
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

#for number splitting and cavity lifetime using a laser injection
if 0:
    from scripts.single_qubit import ssbspec
#    laser_info.set_output_on(True)
#    laser_delay = 50e3
#    plen = 10-6
#    laser_info.set_pulsewidth(plen)
#    edge = 5e-6
#    laser_info.set_edgetime(edge)

#    seq = sequencer.Join([sequencer.Trigger(250), sequencer.Constant(250, 1, '1m2'), sequencer.Delay(laser_delay)])

    spec = ssbspec.SSBSpec(qubit_info, np.linspace(-6e6, 3e6, 121),
                           extra_info=cavity_info, plot_seqs=False)
    spec.measure()
#    laser_info.set_output_on(False)


if 0:
#for i in range(20):
    tau=[]
    tau_err=[]
    delays=[]
    freq=[]
    freq_err=[]
    T2=[]
    T2_err=[]
    from scripts.single_qubit import T1measurement_laser
#    from scripts.single_qubit import T2measurement_laser
    laser_info = mclient.instruments['laserfg']
    laser_info.set_function('PULS')
    laser_info.set_Vhigh(1.5)
    laser_info.set_Vlow(0)
    laser_info.burst_mode()
    laser_info.set_output_on(True)
#    for pulse in[10e-6, 50e-6, 100e-6, 250e-6, 400e-6]:
    for j in range(3):
        pulse = 500e-6
        plen = pulse
        laser_info.set_pulsewidth(plen)
        edge = 1e-6
        laser_info.set_edgetime(edge)
        for add_delay in np.concatenate((np.linspace(plen*1e9,plen*1e9+400e3, 4),np.linspace(plen*1e9+400e3, plen*1e9+3e6, 4))):
#        for add_delay in  np.linspace(plen*1e9,plen*1e9+100e3, 26):
    #    for add_delay in [5e3, 10e3, 50e3, 100e3, 250e3, 500e3, 750e3, 900e3, 1e6, 1.2e6, 1.5e6, 2.5e6, 5e6]:
            delays.append(add_delay)

#            rep_rate = 1/(plen  + 3e-3)
#            rep_rate=100.0*floor(rep_rate/100.0)
    #        rep_rate = 1/(10e-3)
    #        fg.set_frequency(500)
    #        add_delay = 10e3
#
            if add_delay < 50e3:
                fg.set_frequency(250)
                alz.set_naverages(1000)
                t1_range = np.concatenate((np.linspace(0, 10e3, 41), np.linspace(11e3, 20e3, 41)))
#                t1_range = np.concatenate((np.linspace(0, 4e3, 41), np.linspace(4.5e3, 10e3, 41)))
            elif add_delay < 100e3:
                fg.set_frequency(250)
                alz.set_naverages(1000)
                t1_range = np.linspace(0, 10e3, 41)
            elif add_delay < plen*1e9:
                fg.set_frequency(250)
                alz.set_naverages(1000)
                t1_range = np.linspace(0, 2e3, 81)
#                t1_range = np.concatenate((np.linspace(0, 5e3, 41), np.linspace(6e3, 10e3, 41)))
            elif add_delay < plen*1e9 + 200e3:
                fg.set_frequency(250)
                alz.set_naverages(1000)
                t1_range = np.linspace(0, 3e3, 81)
            elif add_delay < plen*1e9 + 400e3:
                fg.set_frequency(200)
                alz.set_naverages(1000)
                t1_range = np.linspace(0, 6e3, 81)

            elif add_delay < plen*1e9 + 1.1e6:
                fg.set_frequency(200)
                alz.set_naverages(1000)
                t1_range = np.linspace(0, 40e3, 81)
            else:
                fg.set_frequency(200)
                alz.set_naverages(400)
                t1_range = np.concatenate((np.linspace(0, 10e3, 41), np.linspace(11e3, 20e3, 41)))

            #t1 = T1measurement_laser.T1Measurement_laser(qubit_info, t1_range, double_exp=False, laser_plen=250, laser_delay = (plen*1e9 + 1.7*edge*1e9 + add_delay))
            t1 = T1measurement_laser.T1Measurement_laser(qubit_info, t1_range, double_exp=False, laser_plen=250, laser_delay = add_delay)
            tau_new, tau_new_err = t1.measure()
            t1.data.set_attrs(post_inj_delay = add_delay)
            t1.data.set_attrs(injection_length = plen)
            plt.close()
            tau.append(tau_new)
            tau_err.append(tau_new_err)
            plt.figure(1)
            plt.clf()
#            plt.errorbar(np.array(delays)/1000.0, np.array(tau)/1000.0, np.array(tau_err)/1000.0, fmt ='mo')
            plt.errorbar(np.array(range(len(tau))),np.array(tau)/1000.0, np.array(tau_err)/1000.0, fmt ='mo')
#            plt.axis(xmin=min(delays)*0.9/1000.0, xmax=max(delays)*1.10/1000.0)
            #plt.semilogx()
            plt.title('QP Decay After Optical Injection')
            plt.xlabel('Iterations - from 0 to 100 us after inj end')
            plt.ylabel('T1 (us)')

#            t2 = T2measurement_laser.T2Measurement_laser(qubit_info, t1_range/2.0, detune= 20.0e9/max(t1_range), laser_plen=250, laser_delay = (plen*1e9 + 1.7*edge*1e9 + add_delay))
#            t2 = T2measurement_laser.T2Measurement_laser(qubit_info, t1_range/8.0, detune= 80.0e9/max(t1_range), laser_plen=250, laser_delay = (plen*1e9 + 1.7*edge*1e9 + add_delay))
#            freq_new, freq_new_err, T2_new, T2_new_err = t2.measure()
#            t2.data.set_attrs(post_inj_delay = add_delay)
#            t2.data.set_attrs(injection_length = plen)
#            plt.close()
#            T2.append(T2_new)
#            T2_err.append(T2_new_err)
#            plt.figure(2)
#            plt.clf()
#            plt.errorbar(np.array(delays)/1000.0, np.array(T2)/1000.0, np.array(T2_err)/1000.0, fmt ='b^')
#            plt.axis(xmin=min(delays)*0.9/1000.0, xmax=max(delays)*1.10/1000.0)
#            plt.figure(3)
#            plt.clf()
##            delta_freq = freq_new*1e6 - 20e9/max(t1_range)/1e3 #delta_freq is in kHz
#            delta_freq = freq_new*1e6 - 80e9/max(t1_range)/1e3
#            freq.append(delta_freq)
#            freq_err.append(freq_new_err*1e6)
#            plt.errorbar(np.array(delays)/1000.0, np.array(freq), np.array(freq_err), fmt ='g^')
#            plt.axis(xmin=min(delays)*0.9/1000.0, xmax=max(delays)*1.10/1000.0)

    laser_info.set_output_on(False)


if 0:#QP decay with laser:
    from scripts.QPs.single_qubit import QPdecay_laser
    laser_info = mclient.instruments['laserfg']
    laser_info.set_output_on(False)
    lvoltage = 2.67
    laserfg.set_Vlow(0.0)
    laserfg.set_Vhigh(lvoltage)
    laserfg.set_function('PULSE')
    laserfg.set_burst_on('True')
    rise_time = 3e3
    laserfg.set_output_on(True)
    
    eff_T1_delay = 400.0
    meas_per_QPinj = 4
    meas_per_reptime = 40
    T1_int = 20.0e3
    QPT1=0.26e6
    half_decay_point=0.2e6
    rep_time=1e9/1000
    fg.set_frequency(rep_rate)
    laser_inj_len = 10e-6
    laser_info.set_pulsewidth(laser_inj_len)
    edge = 5e-6
    laser_info.set_edgetime(edge)
    T1_delays = smart_T1_delays(T1_int=T1_int, QPT1=QPT1, half_decay_point=half_decay_point, eff_T1_delay=eff_T1_delay, probe_point=0.5, meas_per_QPinj=meas_per_QPinj, meas_per_reptime=meas_per_reptime)

    for i in range(5):
        alz.set_naverages(2000)
        qpd = QPdecay_laser.QPdecay_laser(qubit_info, T1_delays, rep_time, meas_per_reptime=meas_per_reptime, meas_per_QPinj=meas_per_QPinj, fit_start=5, vg=0.0, ve=6.75, eff_T1_delay=eff_T1_delay, inj_len= laser_inj_len*1e9)
        qpd.measure()
#        ag3 = mclient.instruments['ag3']
#        qpd.data.set_attrs(inj_power=ag3.get_power())

        T1_delays = (T1_delays -np.log(0.5)*1000.0/qpd.invT1 - eff_T1_delay)/2.0
        for j, delay in enumerate(T1_delays):
            if delay < 0:
                T1_delays[j]=0.0

    laser_info.set_output_on(False)


if 0:
    from scripts.single_qubit import T2measurement
    from scripts.single_qubit import QPdecayRamsey
    fg = mclient.instruments['funcgen']
    rep_time=1.0e9/fg.get_frequency()
    T2_delay = 22500 #ns
    for i in range(5):
        alz.set_naverages(25000)
        qpd = QPdecayRamsey.QPdecayRamsey(qubit_info, T2_delay=T2_delay, detune=200e3, rep_time=rep_time, meas_per_reptime=1, meas_per_QPinj=400, fit_start=5, vg=0, ve=4.25, inj_len=100e3)
        qpd.measure()
        alz.set_naverages(8000)
        t2e = T2measurement.T2Measurement(qubit_info, np.linspace(0.3e3, 40e3, 100), detune=200e3, echotype = T2measurement.ECHO_HAHN)
        t2e.measure()
