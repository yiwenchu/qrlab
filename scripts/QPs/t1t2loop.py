import mclient
reload(mclient)
import numpy as np
import matplotlib as mpl
from t1t2_plotting import do_T1_plot, do_T2_plot, do_T2echo_plot
from t1t2_plotting import do_FT1_plot, do_GFT2_plot, do_EFT2_plot, do_EFT2echo_plot, do_GFT2echo_plot, do_FT2echo_plot
from t1t2_plotting import do_QPdecay_plot, do_population_plot, smarter_T1_delays, smart_T1_delays, calibrate_IQ, do_QPdecay_plot_laser
import time
import math
#from automati on_helper import auto_set_fg_freq, estimate_T1

################################################################################################################################################
mpl.rcParams['figure.figsize']=[7,5]
mpl.rcParams['axes.color_cycle'] = ['b', 'g', 'r', 'c', 'm', 'k']

# Load old settings.
if 0:
    toload = ['AWG1']
    mclient.load_settings_from_file(r'c:\_data\settings\20131119\094145.set', toload)    # Last time-Rabi callibration

awg1 = mclient.instruments['AWG1']
fg = mclient.instruments['funcgen']
#laserfg = mclient.instruments['laserfg']
alz = mclient.instruments['alazar']
ag1 = mclient.instruments['ag1']
LO = mclient.instruments['LO_brick']
ag2 = mclient.instruments['ag2']
#qubit_brick = mclient.instruments['qubit_brick']
#brick2 = mclient.instruments['brick2']


#qGap = mclient.get_qubit_info('qGap')
#eGap = mclient.get_qubit_info('eGap')
qubit_info_1 = mclient.get_qubit_info('qubit_info_1')
qubit_info_2 = mclient.get_qubit_info('qubit_info_2')
#qubit_ef_info = mclient.get_qubit_info('qubit_ef_info')
#qJ15_2 = mclient.get_qubit_info('qJ15#2')
#eJ15_2 = mclient.get_qubit_info('qJ15#2')
#qI15_7 = mclient.get_qubit_info('qI15#7')
#eI15_7 = mclient.get_qubit_info('eI15#7')
#qI15_8 = mclient.get_qubit_info('qI15#8')
#eI15_8 = mclient.get_qubit_info('eI15#8')
#qGZ14_3 = mclient.get_qubit_info('qGZ14#3')
#eGZ14_3 = mclient.get_qubit_info('qGZ14#3')
#qGZ14_4 = mclient.get_qubit_info('qGZ14#4')
#eGZ14_4 = mclient.get_qubit_info('qGZ14#4')
#qB = mclient.get_qubit_info('qB')
#eB = mclient.get_qubit_info('eB')
#qAG13_4 = mclient.get_qubit_info('qAG13#4')
#eAG13_4= mclient.get_qubit_info('qAG13#4')

readout_info = mclient.instruments['readout']

################################################################################################################################################

Q1_t1s = {'t1s':[], 't1s_err':[], 'ofs':[], 'amps':[], 'qpt1s':[],'qpt1s_err':[], 'qpofs':[], 'qpofs_err':[],'t1s_QP':[], 't1s_QP_err':[]}
Q2_t1s = {'t1s':[], 't1s_err':[], 'ofs':[], 'amps':[], 'qpt1s':[],'qpt1s_err':[], 'qpofs':[], 'qpofs_err':[],'t1s_QP':[], 't1s_QP_err':[]}
Q1_t2s = {'t2s':[], 't2s_err':[], 't2freqs':[], 't2freqs_err':[], 'amps':[], 'amps_err':[], 't22s':[], 't22s_err':[], 't22freqs':[], 't22freqs_err':[], 'amp2s':[], 'amp2s_err':[],'t2s_QP':[], 't2s_QP_err':[], 't2freqs_QP':[], 't2freqs_QP_err':[]}
Q1_t2Es = {'t2es':[], 't2es_err':[]}
I15_1_ft1s = {'ft1s':[], 'ft1s_err':[], 'ofs':[], 'amps':[]}
I15_1_pops = {'rabiupAmp':[], 'rabiupAmp_err':[], 'rabinoupAmp':[], 'rabinoupAmp_err':[]}

################################################################################################################################################

def auto_set_fg_freq(seq_len, max_freq=10000):
    fg_freqs = [20000, 16000, 12500, 10000, 8000, 6250, 5000, 4000, 3200, 2500, 2000, 1600, 1250, 1000, 800, 625, 500, 400, 320, 250, 200, 160, 125, 100]
    for freq in fg_freqs:
        if (freq <= max_freq) and (seq_len < 1.0e9/freq):
            fg.set_frequency(freq)
            return freq
    print "Warning: auto_set_fg_frequency failed!"
    return

def estimate_T1(QP_delay, T1_int=90e3, tau_QP=1.5e6, half_decay_point=1e6, eff_T1_delay=500.0):
    T1_QPref = 1/(np.log(2)/eff_T1_delay-1/T1_int)      # T1 at half decay point = effective readout delay/ln(2), excluding intrinsic part giving the T1 due to quasiparticles
    return 1/(1/T1_int+1/T1_QPref*np.exp(-(QP_delay-half_decay_point)/tau_QP))
#    return T1_int/(1+ np.exp((half_decay_point - QP_delay)/tau_QP))

def exp_estimate_T1(QP_delay, T1_floor, tau, known):
    (known_delay, known_T1) = known#(delay, T1)
    A =  (1.0/known_T1 - 1.0/T1_floor)*np.exp(known_delay/tau)
    guess = A*np.exp(-QP_delay/tau)+1.0/T1_floor
    return 1.0/guess
    
def exp_estimate_T1_rise(QP_delay, T1_CW, tau, known):#for negative dlays for laser injection
    (known_delay, known_T1) = known#(delay, T1)
    A =  (1.0/T1_CW - 1.0/known_T1)
    guess = 1.0/T1_CW - A*np.exp(-(QP_delay-known_delay)/tau)
    return 1.0/guess

def exp_tanh_T1(QP_delay, T1_0, T1_CW, known):
    (known_delay, known_T1) = known#(delay, T1)
    A =  1.0/T1_CW - 1.0/T1_0
    tau = known_delay/np.arctanh((1.0/known_T1 - 1.0/T1_0)/A)
    guess = A*np.tanh(QP_delay/tau)+1.0/T1_0
    return 1.0/guess
    
#
def switch_to_qubit(qubit):

    if qubit == 1:
        meas_per_QPinj = 30
        meas_per_reptime = 4

        T1_int = 50.0e3
        QPT1 = 2e6
        half_decay_point = 4e6
        if T1_delays_1 != None:
            T1_delays = T1_delays_1
        else:
            T1_delays = smarter_T1_delays(T1_int=T1_int, QPT1=QPT1, decade_point=half_decay_point, probe_point=0.5, meas_per_QPinj=meas_per_QPinj, meas_per_reptime=meas_per_reptime)

        filename = 'c:/Data/20170623/CK2Q1.h5'
        mclient.datafile = mclient.datasrv.get_file(filename)
#        return T1_delays
    elif qubit == 2:
        meas_per_QPinj = 35
        meas_per_reptime = 4
        
        T1_int = 20.0e3
        QPT1 = 4e6
        half_decay_point = 4e6
        if T1_delays_2 != None:
            T1_delays = T1_delays_2
        else:
            T1_delays = smarter_T1_delays(T1_int=T1_int, QPT1=QPT1, decade_point=half_decay_point, probe_point=0.5, meas_per_QPinj=meas_per_QPinj, meas_per_reptime=meas_per_reptime)

        filename = 'c:/Data/20170623/CK2Q2.h5'
        mclient.datafile = mclient.datasrv.get_file(filename)
        
    return T1_int, T1_delays, meas_per_QPinj, meas_per_reptime

################################################################################################################################################
#for qubits in C2
qubits = [qubit_info_1, qubit_info_2]#, qubit_ef_info]
Q_t1s = [Q1_t1s, Q2_t1s]
T1_delays_1, T1_delays_2 = None, None
#qubit = [qI15_7, eI15_7]
#switch_to_qubit(qubit[0])
#eff_T1_delay = 1000.0
rep_rates =[1000]#, 3000, 1000]# [500, 1000, 2000, 3000, 4000, 5000]
inj_power = 13
inj_lengths = [210e3]
#meas_per_QPinj = 3
#meas_per_reptime = 6

#T1_int = 4e3
#QPT1 = 3e6 
#half_decay_point = 0.5e6

for i in range(300):
##
#    if i is 0:
##        T1_delays = smart_T1_delays(T1_int=T1_int, QPT1=QPT1, half_decay_point=half_decay_point, eff_T1_delay=eff_T1_delay, probe_point=0.5, meas_per_QPinj=meas_per_QPinj, meas_per_reptime=meas_per_reptime)
#        T1_delays = smarter_T1_delays(T1_int=T1_int, QPT1=QPT1, decade_point=half_decay_point, probe_point=0.5, meas_per_QPinj=meas_per_QPinj, meas_per_reptime=meas_per_reptime)
         
    if 0:
        mpl.rcParams['figure.figsize']=[5.5,4]
        for qubit_j, qubit in enumerate(qubits[0:]):
            for rep_rate in rep_rates:
                fg.set_frequency(rep_rate)
                T1_int, _, _, _ = switch_to_qubit(qubit_j+1)
                fig_num = 300+qubit_j
#                delays = np.array([0, 3e3, 80e3])
#                do_T1_plot(qubit[0], 12000, delays, Q1_t1s, 300, double_exp =False)
#                delays = np.concatenate((np.linspace(0, 70e3, 41), np.linspace(70e3,400e3, 31)))
                delays = np.logspace(np.log10(150), np.log10(T1_int*14), 40)
                T1_vals = Q_t1s[qubit_j]
                
                do_T1_plot(qubit, 1200, delays, T1_vals, fig_num, double_exp =False)
#                do_T2_plot(qubit[0], 1500, np.linspace(0,80e3, 81), 100e3, Q1_t2s, 301, double_freq=False)
#                do_T2echo_plot(qubit[0], 1500, np.linspace(0e3, 200e3, 101), 40e3, Q1_t2Es, 302)
#            if 1:
    #            do_population_plot(qubit[0], qubit[1], 1000, 5000, np.linspace(0,1,61), qLuke_pops, 206)

   
    if 1:#Dispersive repeated measurement
        
        qubit_j = 1
        n_avgs = 3000
        _, T1_delays, _, meas_per_reptime = switch_to_qubit(qubit_j+1)
        T1_vals = Q_t1s[qubit_j]
        fig_num = 306+qubit_j
        
        for inj_len in inj_lengths:
            ag2.set_rf_on(True)
            freq = 1e3
            ag2.set_power(inj_power)
                            
            fg.set_frequency(freq)
            for i in range(1):
                qpd=do_QPdecay_plot(qubits[qubit_j], n_avgs, T1_delays, T1_vals, fig_num, meas_per_reptime = meas_per_reptime, meas_per_QPinj=None, fit_start= 3, fit_end=None, inj_len=inj_len, 
                                    injection_marker='1m2')
                
#                T1_delays = (T1_delays -np.log(0.5)*1000.0/qpd.invT1 - eff_T1_delay)/2 # Update the new T1_delay array to be the average of the existing one and the newly measured values
                T1_delays = (T1_delays -np.log(0.5)*1000.0/qpd.invT1 )/2 # sUpdate the new T1_delay array to be the average of the existing one and the newly measured values
 
                for k, delay in enumerate(T1_delays):
                    if math.isnan(delay):
                        T1_delays[k] = 100.0
                    if delay < 0:
                        T1_delays[k] = 100.0
                if qubit_j == 0:
                    T1_delays_1 = T1_delays
                else:
                    T1_delays_2 = T1_delays
                       
        ag2.set_rf_on(False)   
   
   
    if 0:
       laserfg.set_output_on(0)    
       laserfg.set_function('DC')
       laserfg.set_output_on(0)
       laserV=2.67
       '''record the attenuation!'''
       atten = 60
       laserfg.set_DCOffset(laserV)    
       laserfg.set_output_on(1)
       alz.set_naverages(1500)
       #            delays =  np.concatenate((np.linspace(0, 30e3, 41), np.linspace(30e3, 160e3, 81)))
       for j in range(10):
           do_T1_plot(qubit[0], 2500, np.concatenate((np.linspace(0, 0.8e3, 41), np.linspace(0.8e3, 4e3,40))), Q1_t1s, 300, laserV=laserV, atten=atten)
#                    t1 = T1measurement_CW.T1Measurement_CW(qubit_info, delays, laserV, atten=atten)
#                    t1.measure()
        
        
    if 0:
        ag1.set_frequency(9046.3e6)
        LO.set_frequency(9096.3e6)
        ag1.set_power(-30.0)
        LO.set_rf_on(True)
        alz.set_ch1_range('200mV')
        alz.set_ch2_range('40mV')
        readout_info.set_rotype('Dispersive')
        
        
    if 0:
        ag1.set_frequency(9022.97e6)
        LO.set_frequency(9072.97e6)
        ag1.set_power(7)
        LO.set_rf_on(True)
        readout_info.set_pulse_len(1000)
        alz.set_ch1_range('2V')
        alz.set_ch2_range('1V')
        readout_info.set_rotype('High-power')





    if 0:#Full T1_QP curves
#        QP_delays =np.array([3e6])#np.array([1.5e6,2e3, 2.5e6,3e6])
        QP_delays = np.linspace(2e6,10e6,5)#[0.1e6, 0.15e6, 0.2e6, 0.25e6, 0.3e6, 0.4e6]
#        QP_delays = np.concatenate((np.linspace(0.3e6,0.9e6,7),np.linspace(1e6,5.0e6,9)))#,np.linspace(6e6,10e6,15)))
    
        inj_lengths = [110e3] 
        ag2.set_rf_on(True)
        for il, length in enumerate(inj_lengths):
            QP_injection_length = length
#            inj_pwr = 23.0
    #        for inj_pwr in inj_powers:
            ag2.set_power(inj_power)
            
            T1_floor = 30e3
            tau_qp= 8e6
            known1 = (4e6, 1200)#delay, T1
            
            T2_floor = 20e3
            known2 = (0.75e6, 2600)#delay, T2


            for QP_delay in QP_delays:
                auto_set_fg_freq((QP_delay + 1000e3), max_freq=500) #
#               
                T1guess = exp_estimate_T1(QP_delay, T1_floor, tau_qp, known1)
                T2guess = exp_estimate_T1(QP_delay, T2_floor, tau_qp, known2)
                if T1guess < 1.0e3:
                    avg =  3000
                    T1end = T1guess*7
                    T2end = T2guess*1.2
                else:#if QP_delay < 1.1e6:
                    avg =  2500
                    T1end = T1guess*8
                    T2end = T2guess*1.2
    #          
                T1_delays = np.concatenate((np.linspace(0, T1end/3.0, 40), np.linspace(T1end/3.0, T1end, 41)))
                do_T1_plot(qubit[0], avg,  T1_delays, Q1_t1s, 107, QP_injection_delay = QP_delay, QP_injection_length=QP_injection_length,)#) lv=voltage, atten=attenuation)
                
                
#                detune = 6.0e9/T2end
#                T2_delays =np.linspace(0, T2end, 101)
#                do_T2_plot(qubit[0], avg, T2_delays, detune, Q1_t2s, 310, QP_injection_delay=QP_delay, QP_injection_length=QP_injection_length)

        #ag2.set_rf_on(False)


    if 0:#T2 QP
        
        QP_delays = np.linspace(0.5e6,1.5e6,21)
      
        inj_lengths = [110e3] 
        ag2.set_rf_on(True)
        for il, length in enumerate(inj_lengths):
            QP_injection_length = length
            inj_pwr = 20.0
            ag2.set_power(inj_pwr)
            
            T2_floor = 20e3
            tau_qp= 0.26e6
            known = (0.75e6, 2600)#delay, T2
            

            for QP_delay in QP_delays:
                auto_set_fg_freq((QP_delay + 1000e3), max_freq=500) #
                T2guess = exp_estimate_T1(QP_delay, T2_floor, tau_qp, known)
 
                if T1guess < 2e3:
                    avg =  2500 
                    T2end = T2guess*1.2
                else:#if QP_delay < 1.1e6:
                    avg =  2000

                    T2end = T2guess*1.2

                detune = 6.0e9/T2end
                
                T2_delays =np.linspace(0, T2end, 101)
                do_T2_plot(qubit[0], avg, T2_delays, detune, Q1_t2s, 310, QP_injection_delay=QP_delay, QP_injection_length=length)

    if 0:#pulesed laser injection
#        for rep in rep_rate:
        atten = 48
        lvoltage = 2.67
        inj_len = 110e3
        
        QP_delays = np.linspace(0.0e6, 1.25e6, 11)
#        QP_delays = np.concatenate(( np.linspace(0.25e6, 1.0e6, 4), np.linspace(5e6,9e6, 5)))

#        T1_0 = 2.9e3
#        T1_CW = 269
#        known = (100e3, 1.24e3)
        
        T1_floor = 16e3
        tau_qp= 0.5e6
        known = (0.25e6, 1500)#delay, T1
        
        #for negative delays, guess T1
        T1_CW = 500
        tau_rise= 0.1e6
        known_rise = (-0.1e6, 350)
            
        
        laserfg.set_output_on(False)
        laserfg.set_Vlow(0.0)
        laserfg.set_Vhigh(lvoltage)
        laserfg.set_function('PULSE')
        laserfg.set_burst_on('True')
        rise_time = 3e3
        laserfg.set_edgetime(rise_time*1e-9)
        laserfg.set_output_on(True)
        
        for QP_delay in QP_delays:
#            rep = 400
#            fg.set_frequency(rep)
            auto_set_fg_freq((QP_delay + inj_len), max_freq=1000) 
#            T1_guess = exp_tanh_T1(inj_len + QP_delay, T1_0, T1_CW, known)
            if QP_delay >= 0.0:
                T1_guess = max(200, exp_estimate_T1(QP_delay, T1_floor, tau_qp, known))
            else:
                T1_guess = max(200, exp_estimate_T1_rise(QP_delay, T1_CW, tau_rise, known_rise))
            T1_end = 7*T1_guess
            T1_delays = np.concatenate((np.linspace(0, T1_end/3.0, 41),np.linspace(T1_end/3.0, T1_end, 61)))#np.concatenate((np.linspace(0, 6e3, 51), np.linspace(6e3, 30e3, 31)))
            for j in range(1):
                if T1_guess < 400:
                    avg = 2500
                elif T1_guess < 1e3:
                    avg = 2500
                elif T1_guess < 2e3:
                    avg = 2000
                else:
                    avg = 1500
                #avg = 1500
                do_T1_plot(qubit[0], avg, T1_delays, Q1_t1s, 100,
                           QP_injection_delay=QP_delay, QP_injection_length= inj_len, laserV = lvoltage, atten = atten, pulsed_laser=True)
                
#         
        laserfg.set_output_on(False)


    if 0:#Laser repeated masurement
        '''record the attenuation'''
        atten = 48
        freq = 1000
        
        laser_info = mclient.instruments['laserfg']
        laser_info.set_output_on(False)
        lvoltage = 2.67
        laserfg.set_Vlow(0.0)
        laserfg.set_Vhigh(lvoltage)
        laserfg.set_function('PULSE')
        laserfg.set_burst_on('True')

        laserfg.set_output_on(True)
        rise_time = 3e3
        laserfg.set_edgetime(rise_time*1e-9)
        
        laser_inj_len = 110e-6
        laser_info.set_pulsewidth(laser_inj_len)
        
#        for length in inj_lengths:
        for power in inj_power:
            inj_len = 110e3
            ag2.set_power(power)
                
            fg.set_frequency(4000)
            ve, vg = calibrate_IQ(qubit[0], 50000)
            
            fg.set_frequency(freq)
            for i in range(4):
                qpd=do_QPdecay_plot_laser(qubit[0], 2500, T1_delays, Q1_t1s, 306, meas_per_reptime = meas_per_reptime, meas_per_QPinj=None, fit_start= 1, fit_end=None, vg=vg, ve=ve+0.1, eff_T1_delay=eff_T1_delay, inj_len=inj_len, atten=atten, laser_voltage=lvoltage, edgewidth = rise_time)
                                
                T1_delays = (T1_delays -np.log(0.5)*1000.0/qpd.invT1 - eff_T1_delay)/2 # Update the new T1_delay array to be the average of the existing one and the newly measured values
                for k, delay in enumerate(T1_delays):
                    if delay < 0:
                        T1_delays[k]=0.0
                       
        ag2.set_rf_on(False)

