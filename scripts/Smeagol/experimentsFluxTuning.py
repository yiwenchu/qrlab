import mclient
from mclient import instruments
import time
import numpy as np
from pulseseq import sequencer
from pulseseq import pulselib
from matplotlib import gridspec
import matplotlib.pyplot as plt
import logging
from scripts.single_qubit import spectroscopy as spectroscopy
from scripts.single_qubit import tracked_spectroscopy_yoko_SG as spec_yoko
from scripts.single_qubit import tuned_qubit_characterization as tuned_qubit
from scipy.interpolate import interp1d
#from scripts.jk.single_qubit import temperature as temperature

ag1 = instruments['ag1']
ag2 = instruments['ag2']
#ag3 = instruments['ag3']
qubit_info = mclient.get_qubit_info('qubit_info')
qubit_ef_info = mclient.get_qubit_info('qubit_ef_info')
vspec = instruments['vspec']
awg1 = instruments['AWG1']
qubit_brick = instruments['qubit_brick']
#qubit_brick = instruments['ag2']
#qubit_brick = instruments['ag2']
qubit_ef_brick = instruments['qubit_ef_brick']
va_lo = instruments['va_lo_2']
funcgen = instruments['funcgen']
alazar = instruments['alazar']
spec_brick = instruments['spec_brick']
spec_info = mclient.get_qubit_info('spec_info')
LO_brick = instruments['LO_brick']
LO_info = mclient.get_qubit_info('LO_info')
#cavity_info = mclient.get_qubit_info('cavity0')
yoko1=instruments['yoko1']
yoko2=instruments['yoko2']

#trigger for experiments
#funcgen.sync_on(1)

if 0:
#    rofreq=9.006e9
    rofreq=9.146836e9
#    ag3.set_frequency(9197e6)

    ag1.set_frequency(rofreq)
    LO_brick.set_frequency(rofreq+50e6)
    ro_pow= -13
    ag1.set_power(ro_pow)
    
    yoko_currs = np.linspace(0.56e-3, 0.70e-3, 53)
    qubit_freq = 6771.4e6
    freq_range = 2e6
    init_freqs = np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 501)
    spec_params = (qubit_brick, -30)
        
    sy=spec_yoko.Tracked_Spectroscopy(qubit_info, yoko1, 'CURR', yoko_currs, init_freqs, spec_params, ro_pow, rofreq,
                                      ro_freq_tune=-1, ro_pwr_tune=-1, ro_pwr_tune_interval=300e6, plen=50000, amp=0.01, 
                                      freq_step=0, 
                                      width_min = 0.005e6, width_max = 6e6,
                                      plot_seqs=False, subtraction = False, )
    sy.measure()
    
    blah
    

if 0:
#    rofreq=9.006e9
#    rofreq=f2(6.2e-3)
###    ag3.set_frequency(9197e6)
##
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ro_pow=-21
#    ag1.set_power(ro_pow)
    
#    yoko_currs = np.linspace(0e-3, 0.66e-3, 19)
    y = np.linspace (1, 0, 100)
#    y2 = np.linspace (y[15], 0, 300-15)
    yoko_currs = np.arccos(y)*0.34e-3  
    qubit_freq = 6748.39e6
    qubit_brick.set_frequency(qubit_freq) 
    freq_range = 3e6
    init_freqs = np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 401)
    amps = amps = np.linspace(-1, 1, 81)
    T1_delays = np.linspace(0, 30e3, 201)
    qubit_rf_pwr = 13
        
    sy=tuned_qubit.Tuned_Qubit('qubit_info', qubit_brick, yoko2, funcgen, alazar, 'CURR', yoko_currs, 
                               ['ROFREQ', 'SPEC', 'RABI', 'T1', 'ROPWR'],

                               init_ro_freq = 9.15028e9,
                               ro_freq_tune = -1,
                               ro_freq_tune_interval = 5e6,
                               
                               init_ro_power = -12,
                               ropwr_qubit_rf_pwr = qubit_rf_pwr,
                               ro_range = 0, #will search += xdbm, 0 if not doing power tuning 
                               ro_step = 1,
                               ro_pwr_tune = -1,
                               ro_pwr_tune_interval = 5e6, #tune ro pwr if the qubit frequency has changed by this much since last tuning, -1 if using steps only
                               ro_pwr_pi = True,                                   
                              
                               init_freqs = init_freqs, plen=50000, amp=0.001, spec_avgs = 2000,
                               init_spec_power = qubit_rf_pwr,
                               freq_step=0.00000001e6, 
                               width_min = 0.0005e6, width_max = 5e6, subtraction = False, 
                               use_IQge=True, use_weight=True,

#                               ro_freq_fxn = f2E,
#                               qubit_freq_fxn = f1,
                               rabi_rf_pwr = qubit_rf_pwr, rabi_avgs = 500, init_rabi_amps = amps, 
                               
                               T1_rf_pwr = qubit_rf_pwr, T1_funcgen_freq = 5000, 
                               T1_avgs = 2000, init_T1_delays = T1_delays, T1_update_delays = False,
                               )
                               
    sy.measure()
    
    blah    
 

if 1:
#    rofreq=9.006e9
#    rofreq=f2(6.2e-3)
###    ag3.set_frequency(9197e6)
##
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ro_pow=-21
#    ag1.set_power(ro_pow)
    
#    y = np.linspace (1, 0, 201)
#    y = np.linspace (0.824, 0.679, 80)
    yoko_currs = np.linspace(0.93, 1.33, 401)*1e-3
#    yoko_currs = yoko_currs[28:-28]
#    yoko_currs = np.arccos(np.linspace(1, np.sqrt(np.cos(1)), 401)**2)*1.5e-3
#    yoko_currs = yoko_currs[35:]
#    y = np.linspace (1, 0, 150)
#    y = np.linspace (0.75, 0.62, 100)
#    y2 = np.linspace (0.399, 0.257, 200)
#    yoko_currs = np.arccos(y)*1e-3
#    yoko_currs = yoko_currs[56:]
#    qubit_freq = 6748.39e6
#    stark_pwrs=np.linspace(1e-4, 4e-3, 20)
#    stark_pwrs=10*np.log10(stark_pwrs/1e-3)
    qubit_freq = 6340e6-50e6
    qubit_brick.set_frequency(qubit_freq+50e6) 
    freq_range = 3e6
    init_freqs = np.linspace(-freq_range, freq_range, 201)
#    init_freqs = init_freqs[:200]
#    init_freqs = all_freqs[201:]
    amps = 0.2*np.linspace(-1, 1, 81)
    T1_delays = np.linspace(0, 20e3, 200)#np.concatenate((np.linspace(0, 20e3, 150), np.linspace(21e3, 40e3, 50)))
    qubit_rf_pwr = 13
        
    sy=tuned_qubit.Tuned_Qubit('qubit_info', qubit_brick, funcgen, alazar, 'CURR', yoko_currs, 
                               ['SSBSPEC','RABI', 'T1'],#'ROFREQ', 'RABI', 'T1', 'ROPWR'
#                               stark_rfsource = ag2,
                               qubit_yoko=yoko2,
                               
                               init_ro_freq = 9.080240e9,
                               ro_freq_tune = -1,
                               ro_freq_tune_interval = 1000e6,
                               
                               init_ro_power = -25,
                               ropwr_qubit_rf_pwr = qubit_rf_pwr,
                               ro_range = 0, #will search += xdbm, 0 if not doing power tuning 
                               ro_step = 1,
                               ro_pwr_tune = -1,
                               ro_pwr_tune_interval = 500e6, #tune ro pwr if the qubit frequency has changed by this much since last tuning, -1 if using steps only
                               ro_pwr_pi = True,                                   
                              
                               init_freqs = init_freqs, plen=2000, amp=0.5e-3, spec_avgs = 5000,
                               spec_funcgen_freq = 10000,
                               init_spec_power = qubit_rf_pwr,
                               freq_step=-0.001,
                               width_min = 0.0005e6, width_max = 5e6, subtraction = False, 
                               use_IQge=True, use_weight=True, spec_bgcor = True,
#                               spec_generate = True,

#                               ro_freq_fxn = f2E,
#                               qubit_freq_fxn = f1,
                               rabi_funcgen_freq = 10000, rabi_rf_pwr = qubit_rf_pwr, 
                               rabi_avgs = 1000, init_rabi_amps = amps, 
                               
                               T1_rf_pwr = qubit_rf_pwr, T1_funcgen_freq = 5000, 
                               T1_avgs = 5000, init_T1_delays = T1_delays, T1_update_delays = False, T1_bgcor = True,
                               )
                               
    sy.measure()
#
#    plt.figure(301);
#    plt.savefig('C:\Data\images\\20160804\\first_half.png')
#
#    qubit_freq = 6744.950e6
#    qubit_brick.set_frequency(qubit_freq+50e6) 
#    freq_range = 1.99e6
#    init_freqs = np.linspace(-freq_range, freq_range, 200)
#    
#    sy=tuned_qubit.Tuned_Qubit('qubit_info', qubit_brick, funcgen, alazar, 'CURR', yoko_currs, 
#                               ['SSBSPEC',],#'ROFREQ', 'RABI', 'T1', 'ROPWR'
##                               stark_rfsource = ag2,
#                               qubit_yoko=yoko2,
#                               
#                               init_ro_freq = 9.1502e9,
#                               ro_freq_tune = -1,
#                               ro_freq_tune_interval = 5e6,
#                               
#                               init_ro_power = -12,
#                               ropwr_qubit_rf_pwr = qubit_rf_pwr,
#                               ro_range = 0, #will search += xdbm, 0 if not doing power tuning 
#                               ro_step = 1,
#                               ro_pwr_tune = -1,
#                               ro_pwr_tune_interval = 5e6, #tune ro pwr if the qubit frequency has changed by this much since last tuning, -1 if using steps only
#                               ro_pwr_pi = True,                                   
#                              
#                               init_freqs = init_freqs, plen=2000, amp=1.434e-3, spec_avgs = 20000,
#                               spec_funcgen_freq = 50000,
#                               init_spec_power = qubit_rf_pwr,
#                               freq_step=0,
#                               width_min = 0.0005e6, width_max = 5e6, subtraction = False, 
#                               use_IQge=True, use_weight=True,
#
##                               ro_freq_fxn = f2E,
##                               qubit_freq_fxn = f1,
#                               rabi_funcgen_freq = 50000, rabi_rf_pwr = qubit_rf_pwr, 
#                               rabi_avgs = 1000, init_rabi_amps = amps, 
#                               
#                               T1_rf_pwr = qubit_rf_pwr, T1_funcgen_freq = 5000, 
#                               T1_avgs = 2000, init_T1_delays = T1_delays, T1_update_delays = False,
#                               )
#                               
#    sy.measure()
#    
    blah       

#import processed data from earlier runs that contains RO and qubit freq data
#flux_data_path='C:\\labpython\\analysis\\Smeagol_analysis\\AlNQubit20160623_outputs\\flux_data.txt'
#currents = np.loadtxt(flux_data_path, skiprows=1, usecols = [0])
#centers = np.loadtxt(flux_data_path, skiprows=1, usecols = [1])
#
#f1= interp1d(currents, centers-50e6, kind='linear')

##import processed data from earlier runs that contains RO and qubit freq data
#flux_data_pathE='C:\\labpython\\analysis\\Smeagol_analysis\\flux_tuning_05122015_outputs\\flux_data_rof_qf_etched.txt'
#currentsE = np.loadtxt(flux_data_pathE, skiprows=1, usecols = [0])
#centersE = np.loadtxt(flux_data_pathE, skiprows=1, usecols = [1])
#RO_freqsE = np.loadtxt(flux_data_pathE, skiprows=1, usecols = [2])
#
#f1E= interp1d(currentsE, centersE, kind='linear')
#f2E= interp1d(currentsE, RO_freqsE, kind='linear')

if 0:
#    rofreq=9.006e9
#    rofreq=f2(0e-3)
##    ag3.set_frequency(9197e6)
#
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ro_pow=-21
#    ag1.set_power(ro_pow)
    
    yoko_currs = np.linspace(0.565e-3, 0.695e-3, 27)
    qubit_freq = 6771.4e6
    qubit_brick.set_frequency(qubit_freq) 
    freq_range = 2e6
    init_freqs = np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 301)  
    amps = np.linspace(-1, 1, 81)
    T1_delays = np.linspace(0, 30e3, 201)
        
    sy=tuned_qubit.Tuned_Qubit('qubit_info', qubit_brick, yoko1, funcgen, alazar, 'CURR', yoko_currs, 
                               ['SPEC', 'RABI', 'T1',],
                               init_freqs = init_freqs, plen=50000, amp=0.01, 
                               init_spec_power = -30,
                               freq_step=0, 
                               width_min = 0.001e6, width_max = 5e6,
                               use_IQge=True, use_weight=True,
                               init_ro_power = -13,
                               init_ro_freq = 9.146836e9,
                               ro_freq_tune = -1,
                               ro_range = -1, 
                               ro_pwr_tune = -1, 
#                               ro_pwr_tune_interval = 100e6, ro_pwr_pi = True,
#                               ro_freq_fxn = f2,
#                               qubit_freq_fxn = f1,
                               rabi_rf_pwr = -30, rabi_avgs = 1000, init_rabi_amps = amps, 
                               T1_rf_pwr = -30, T1_funcgen_freq = 5000, T1_avgs = 5000, 
                               init_T1_delays = T1_delays, T1_update_delays = False,
                               )
                               
    sy.measure()
    
    blah

if 0:
#    rofreq=9.006e9
#    rofreq=f2(0e-3)
##    ag3.set_frequency(9197e6)
#
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ro_pow=-21
#    ag1.set_power(ro_pow)
    
    y = np.linspace (1, 0, 500)
#    y2 = np.linspace (y[15], 0, 300-15)
    yoko_currs = np.arccos(y)*1.5e-3   
#    yoko_currs = np.linspace(0, 0.695e-3, 27)
    qubit_freq = 6.354660e9
#    qubit_freq = 6777e6
    qubit_brick.set_frequency(qubit_freq) 
    freq_range = 3e6
    init_freqs = np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 101)  
    amps = np.linspace(-1, 1, 81)
#    T1_delays = np.linspace(0, 30e3, 201)
        
    sy=tuned_qubit.Tuned_Qubit('qubit_info', qubit_brick, yoko2, funcgen, alazar, 'CURR', yoko_currs, 
                               ['ROFREQ', 'SPEC',],
                                
                               ro_freq_tune = -1,
                               init_ro_freq = 9.146535e9,
                               ro_freq_tune_interval = 5e6,
                               
                               init_freqs = init_freqs, plen=50000, amp=0.01, 
                               init_spec_power = -25,
                               freq_step=0.0004e6, 
                               width_min = 0.0005e6, width_max = 5e6,
                               use_IQge=True, use_weight=True,
                               
                               init_ro_power = -13,
                               
                               ro_pwr_tune_interval = 200e6,
                               ro_range = -1, 
                               ro_pwr_tune = -1, 
#                               ro_pwr_tune_interval = 100e6, ro_pwr_pi = True,
#                               ro_freq_fxn = f2,
#                               qubit_freq_fxn = f1,
#                               rabi_rf_pwr = -30, rabi_avgs = 1000, init_rabi_amps = amps, 
#                               T1_rf_pwr = -30, T1_funcgen_freq = 5000, T1_avgs = 5000, 
#                               init_T1_delays = T1_delays, T1_update_delays = False,
                               )
                               
    sy.measure()
    
    blah


#Does Spec, Rabi, and T1 measurements assuming ro frequencies are known. Recalibrates readout

if 0:
#    rofreq=9.006e9
#    rofreq=f2(6.2e-3)
###    ag3.set_frequency(9197e6)
##
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ro_pow=-21
#    ag1.set_power(ro_pow)
    
    yoko_currs = np.linspace(7.7e-3, 9e-3, 66)
    qubit_freq = f1E(yoko_currs[0])
    qubit_brick.set_frequency(qubit_freq) 
    freq_range = 5e6
    init_freqs = np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 51)
    amps = amps = np.linspace(-0.2, 0.2, 81)
    T1_delays = np.linspace(0, 200e3, 51)
        
    sy=tuned_qubit.Tuned_Qubit('qubit_info', qubit_brick, yoko1, funcgen, alazar, 'CURR', yoko_currs, 
                               ['SPEC', 'RABI', 'T1', 'ROPWR'],
                               init_freqs = init_freqs, plen=50000, amp=0.002, spec_avgs = 1000,
                               init_spec_power = -2,
                               freq_step=f1E(yoko_currs[1])-f1E(yoko_currs[0]), 
                               width_min = 0.1e6, width_max = 5e6, subtraction = True, 
                               init_ro_power = -21, ro_range = 1, ro_pwr_tune = -1, 
                               ro_pwr_tune_interval = 100e6, ro_pwr_pi = True,
                               ro_freq_fxn = f2E,
#                               qubit_freq_fxn = f1,
                               rabi_avgs = 500, init_rabi_amps = amps, 
                               T1_funcgen_freq = 1000, T1_avgs = 1000, init_T1_delays = T1_delays,
                               )
                               
    sy.measure()
    
    blah    
    
if 0:
#    rofreq=9.006e9
#    rofreq=f2(6.2e-3)
###    ag3.set_frequency(9197e6)
##
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ro_pow=-21
#    ag1.set_power(ro_pow)
    
    yoko_currs = np.linspace(4.4e-3, 10e-3, 29)
#    qubit_freq = f1(yoko_currs[0])
#    qubit_brick.set_frequency(qubit_freq) 
    qubit_freq = 5.303e9
    freq_range = 10e6
    init_freqs = np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 51)
#    init_freqs = np.linspace(6205e6, 6225e6, 51)
    amps = np.linspace(-1, 1, 81)
    T1_delays = np.concatenate((np.linspace(0,30e3, 51), np.linspace(31e3,100e3,41)))
    T2_delays = np.linspace(0, 10e3, 101)
    T2E_delays = np.linspace(0, 50e3, 101)
        
    sy=tuned_qubit.Tuned_Qubit('qubit_info', qubit_brick, yoko2, funcgen, alazar, 'CURR', yoko_currs, 
                               ['ROFREQ','SPEC', 'RABI', 'T1', 'ROPWR', 'T2', 'T2Echo'],
                               init_ro_freq = 9.118359e9, 
#                               init_ro_freq = 9.120560e9,
                               init_freqs = init_freqs, plen=50000, amp=0.06, spec_avgs = 2000,
                               init_spec_power = -20,
                               freq_step=-50e6, 
                               width_min = 0.1e6, width_max = 5e6, subtraction = False, 
                               init_ro_power = -14, ro_spec_range = 10e6, ro_range = 1, ro_pwr_tune = -1, 
                               ro_pwr_tune_interval = 100e6, ro_pwr_pi = True,
#                               ro_freq_fxn = f2E,
#                               qubit_freq_fxn = f1,
                               rabi_avgs = 1000, init_rabi_amps = amps, rabi_rf_pwr = -20,
                               T1_funcgen_freq = 1000, T1_avgs = 2000, init_T1_delays = T1_delays, T1_rf_pwr = -20,
                               T2_funcgen_freq = 1000, T2_avgs = 2000, init_T2_delays = T2_delays, 
                               T2_delta = 800e3, T2_rf_pwr = -20, T2_set_f = True,
                               T2E_funcgen_freq = 1000, T2E_avgs = 2000, init_T2E_delays = T2E_delays, T2E_rf_pwr = -20,
#                               TE_delta= 300e3
                               )
                               
    sy.measure()
    
    blah
    
if 0:
#    rofreq=9.006e9
#    rofreq=f2(6.2e-3)
###    ag3.set_frequency(9197e6)
##
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ro_pow=-21
#    ag1.set_power(ro_pow)
    
    yoko_currs = np.linspace(1e-3, 10e-3, 91)
#    qubit_freq = f1(yoko_currs[0])
#    qubit_brick.set_frequency(qubit_freq) 
    qubit_freq = 6.223e9
    freq_range = 10e6
    init_freqs = np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 51)
#    init_freqs = np.linspace(6205e6, 6225e6, 51)
    amps = np.linspace(-1, 1, 81)
    T1_delays = np.linspace(0, 30e3, 51)
    T2_delays = np.linspace(0, 20e3, 101)
    T2E_delays = np.linspace(0, 50e3, 101)
        
    sy=tuned_qubit.Tuned_Qubit('qubit_info', qubit_brick, yoko2, funcgen, alazar, 'CURR', yoko_currs, 
                               ['ROFREQ','SPEC', 'RABI', 'T1', 'ROPWR', 'T2', 'T2Echo'],
#                               init_ro_freq = 9.03100e9, 
                               init_ro_freq = 9.0304e9,
                               init_freqs = init_freqs, plen=50000, amp=0.01, spec_avgs = 3000,
                               init_spec_power = -10,
                               freq_step = -1e6, 
                               width_min = 0.1e6, width_max = 5e6, subtraction = False, 
                               init_ro_power = -21, ro_spec_range = 10e6, ro_range = 1, ro_pwr_tune = -1, 
                               ro_pwr_tune_interval = 100e6, ro_pwr_pi = True,
#                               ro_freq_fxn = f2E,
#                               qubit_freq_fxn = f1,
                               rabi_avgs = 1000, init_rabi_amps = amps, rabi_rf_pwr = -20,
                               T1_funcgen_freq = 1000, T1_avgs = 1000, init_T1_delays = T1_delays, T1_rf_pwr = -20,
                               T2_funcgen_freq = 1000, T2_avgs = 1000, init_T2_delays = T2_delays, 
                               T2_delta = 500e3, T2_rf_pwr = -20, T2_set_f = True,
                               T2E_funcgen_freq = 1000, T2E_avgs = 1000, init_T2E_delays = T2E_delays, T2E_rf_pwr = -20,
                               )
                               
    sy.measure()
    
    blah
 
if 1:
#    rofreq=9.006e9
#    rofreq=f2(6.2e-3)
###    ag3.set_frequency(9197e6)
##
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ro_pow=-21
#    ag1.set_power(ro_pow)
    
    yoko_currs = np.linspace(0.6e-3, 0.8e-3, 11)
#    qubit_freq = f1(yoko_currs[0])
#    qubit_brick.set_frequency(qubit_freq) 
    qubit_freq = 6290e6
    freq_range = 2e6
    init_freqs = np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 51)
#    init_freqs = np.linspace(6153e6, 6207e6, 151)
        
    sy=tuned_qubit.Tuned_Qubit('qubit_info', qubit_brick, yoko1, funcgen, alazar, 'CURR', yoko_currs, 
                               ['SPEC',],
#                               init_ro_freq = f2E(yoko_currs[0]), 
                               init_freqs = init_freqs, plen=50000, amp=0.003, spec_avgs = 3000,
                               init_spec_power = 13,
                               freq_step=0, 
                               width_min = 0.1e6, width_max = 5e6, subtraction = False, 
                               init_ro_power = -22, init_ro_freq = 9.017600e9, ro_range = 1, ro_pwr_tune = -1, 
                               ro_pwr_tune_interval = -1, ro_pwr_pi = True,
#                               ro_freq_fxn = f2E,
#                               qubit_freq_fxn = f1,
                               )
                               
    sy.measure()
    
    blah   
    
    
if 0:
#    rofreq=9.006e9
#    rofreq=f2(6.2e-3)
###    ag3.set_frequency(9197e6)
##
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ro_pow=-21
#    ag1.set_power(ro_pow)
    
    yoko_currs = np.linspace(1.01e-3, 10e-3, 51)
#    qubit_freq = f1(yoko_currs[0])
#    qubit_brick.set_frequency(qubit_freq) 
    qubit_freq = 6.243e9
    freq_range = 10e6
    init_freqs = np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 51)
#    init_freqs = np.linspace(6205e6, 6225e6, 51)
    amps = np.linspace(-1, 1, 81)
    T1_delays = np.linspace(0, 50e3, 51)
    T2_delays = np.linspace(0, 70e3, 101)
    T2E_delays = np.linspace(0, 50e3, 101)
        
    sy=tuned_qubit.Tuned_Qubit('qubit_info', qubit_brick, yoko2, funcgen, alazar, 'CURR', yoko_currs, 
                               ['SPEC', 'RABI', 'T1', 'ROPWR', 'T2', 'T2Echo'],
#                               init_ro_freq = 9.03100e9, 
                               init_ro_freq = 9.0304e9,
                               init_freqs = init_freqs, plen=50000, amp=0.01, spec_avgs = 3000,
                               init_spec_power = -10,
                               freq_step = -1e6, 
                               width_min = 0.1e6, width_max = 5e6, subtraction = False, 
                               init_ro_power = -22, ro_spec_range = 10e6, ro_range = 1, ro_pwr_tune = -1, 
                               ro_pwr_tune_interval = 100e6, ro_pwr_pi = True,
                               ro_freq_fxn = fRU,
#                               qubit_freq_fxn = f1,
                               rabi_avgs = 1000, init_rabi_amps = amps, rabi_rf_pwr = -25,
                               T1_funcgen_freq = 1000, T1_avgs = 1000, init_T1_delays = T1_delays, T1_rf_pwr = -25,
                               T2_funcgen_freq = 1000, T2_avgs = 1000, init_T2_delays = T2_delays, 
                               T2_delta = 300e3, T2_rf_pwr = -25, T2_set_f = True,
                               T2E_funcgen_freq = 1000, T2E_avgs = 1000, init_T2E_delays = T2E_delays, T2E_rf_pwr = -25,
                               )
                               
    sy.measure()
    
    blah   
    
    
    
    
    
    
    
    
    
    