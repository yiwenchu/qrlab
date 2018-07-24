import mclient
from mclient import instruments
from datetime import datetime
import time
import numpy as np
from pulseseq import sequencer
from pulseseq import pulselib
from matplotlib import gridspec
import matplotlib.pyplot as plt
import logging
from scripts.single_qubit import spectroscopy as spectroscopy
from scripts.jk.single_qubit import temperature as temperature
from scripts.single_qubit import ssbspec, ssbspec_fit, stark_swap 
from scripts.single_qubit import rocavspectroscopyPhononSwap as rospecPS
from scripts.single_qubit import tuned_qubit_characterization as tuned_qubit


ag1 = instruments['ag1']
ag2 = instruments['ag2']
#ag3 = instruments['ag3']
qubit_info = mclient.get_qubit_info('qubit_info')
phonon1_info = mclient.get_qubit_info('phonon1_info')
qubit_ef_info = mclient.get_qubit_info('qubit_ef_info')
vspec = instruments['vspec']
awg1 = instruments['AWG1']
va_lo_5_10 = instruments['va_lo_5_10'] 
cavity_brick = instruments['cavity_brick'] #4-8 GHz brick
qubit_brick = instruments['qubit_brick'] #vector generator
qubit_ef_brick = instruments['qubit_ef_brick']
#va_lo = instruments['va_lo']
va_lo_4_8 = instruments['va_lo']
funcgen = instruments['funcgen']
alazar = instruments['alazar']
spec_brick = instruments['spec_brick']
spec_info = mclient.get_qubit_info('spec_info')
LO_brick = instruments['LO_brick']
LO_info = mclient.get_qubit_info('LO_info')
cavity_info = mclient.get_qubit_info('cavity_info')
yoko1=instruments['yoko1']
yoko2=instruments['yoko2']
#ro = mclient.get_readout_info('readout')


precool = sequencer.Sequence([sequencer.Trigger(250), pulselib.GaussSquare(0.660e3, 0.056, 10, chan = 3)])
precool2 = sequencer.Sequence([pulselib.GaussSquare(0.660e3, 0.056, 10, chan = 3)])

#r = qubit_info.rotate
#        
#
#calibSeq = sequencer.Sequence([sequencer.Trigger(250)])
#calibSeq.append(sequencer.Combined([
#                pulselib.Constant(ro.pulse_len, 1, chan=ro.readout_chan),
#                pulselib.Constant(ro.pulse_len, 1, chan=ro.acq_chan),
#                ]))            
#calibSeq.append(sequencer.Trigger(250))
#calibSeq.append(r(np.pi, 0))
#calibSeq.append(sequencer.Combined([
#                pulselib.Constant(ro.pulse_len, 1, chan=ro.readout_chan),
#                pulselib.Constant(ro.pulse_len, 1, chan=ro.acq_chan),
#                ]))
#calibSeq.append(sequencer.Trigger(250))

if 0:# Number splitting w/ SSBspec
    
#    seq = sequencer.Trigger(250)
#    rofreq=9046.3e6
#    ro_pow=-30
#    rofreq = 9.02297e9
#    ro_pow = 
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ag1.set_power(ro_pow)
#    cav_drive = sequencer.Sequence([sequencer.Trigger(250), cavity_info.rotate(np.pi, 0), sequencer.Delay(10)])
#    delayseq = sequencer.Delay(1000)
    
    #patchmon:
#    seq = sequencer.Sequence([sequencer.Trigger(250), sequencer.Delay(200)])
#    postseq = sequencer.Delay(-160)
    
#    qubit_ef_brick.set_rf_on(True)
#    qubit_brick.set_rf_on(False)
    alazar.set_naverages(5000)
    
    stark_chan = 3
    w = 100
    simulseq = sequencer.Sequence(pulselib.GaussSquare(qubit_info.w_selective*4+2*w, 0.1, w, chan = stark_chan))
#    simulseq = simulseq.append(pulselib.GaussSquare(qubit_info.w_selective*4, 0.1, 10, chan = stark_chan))
#    simulseq.add_marker('3m1', stark_chan, ofs = -85, bufwidth = 5)

    spec = ssbspec_fit.SSBSpec_fit(qubit_info, np.linspace(-3e6, 3e6, 201), 
#                           seq=seq, 
                           simulseq = simulseq,
#                           postseq = postseq,
                           plot_seqs=False,
#                           extra_info = cavity_info
                            )
    spec.measure()
    
    
#    qubit_ef_brick.set_rf_on(False)
    bla 
    
#tuned experiments
if 0:
#    rofreq=9.006e9
#    rofreq=f2(6.2e-3)
###    ag3.set_frequency(9197e6)
##
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ro_pow=-21
#    ag1.set_power(ro_pow)
    
#    yoko_currs = np.linspace(0.006890, 0.006912, 150)
#    y = np.linspace (1, 0, 150)
#    y = np.linspace (0.75, 0.62, 100)
#    y2 = np.linspace (y[15], 0, 300-15)
#    yoko_currs = np.arccos(y)*1e-3
#    yoko_currs = yoko_currs[26:]
#    qubit_freq = 6748.39e6
#    stark_pwrs=np.linspace(1e-4, 4e-3, 20)
#    stark_pwrs=10*np.log10(stark_pwrs/1e-3)
    stark_pwrs = np.linspace(-0.08, 0, 401)
#    stark_pwrs = np.sqrt(stark_pwrs)
    qubit_freq = 6337.438e6-50e6
    qubit_brick.set_frequency(qubit_freq+50e6) 
#    freq_range = 5e6
#    init_freqs = np.linspace(-freq_range, freq_range, 201)
    init_freqs = np.linspace(-6e6, 3e6, 201)
    amps = amps = np.linspace(-0.5, 0.5, 81)
    T1_delays = np.linspace(0, 40e3, 201)
    qubit_rf_pwr = 13
        
    sy=tuned_qubit.Tuned_Qubit('qubit_info', qubit_brick, funcgen, alazar, 'STARK', stark_pwrs, 
                               ['SSBSPEC', 'RABI', 'T1',],#'ROFREQ', 'RABI', 'T1', 'ROPWR'
                               stark_rfsource = qubit_ef_brick,
                               
                               init_ro_freq = 9.080170e9,
                               ro_freq_tune = -1,
                               ro_freq_tune_interval = 5e6,
                               
                               init_ro_power = -19,
                               ropwr_qubit_rf_pwr = qubit_rf_pwr,
                               ro_range = 0, #will search += xdbm, 0 if not doing power tuning 
                               ro_step = 1,
                               ro_pwr_tune = -1,
                               ro_pwr_tune_interval = 5e6, #tune ro pwr if the qubit frequency has changed by this much since last tuning, -1 if using steps only
                               ro_pwr_pi = True,                                   
                              
                               init_freqs = init_freqs, plen=2000, amp=1.434e-3, spec_avgs = 10000,
                               spec_funcgen_freq = 10000,
                               init_spec_power = qubit_rf_pwr,
                               freq_step=0, 
                               width_min = 0.0005e6, width_max = 5e6, subtraction = False, 
                               use_IQge=True, use_weight=True,
                               

#                               ro_freq_fxn = f2E,
#                               qubit_freq_fxn = f1,
                               rabi_funcgen_freq = 50000, rabi_rf_pwr = qubit_rf_pwr, 
                               rabi_avgs = 1000, init_rabi_amps = amps, 
                               
                               T1_rf_pwr = qubit_rf_pwr, T1_funcgen_freq = 5000, 
                               T1_avgs = 2000, init_T1_delays = T1_delays, T1_update_delays = False,
                               )
                               
    sy.measure()
    
    blah  
  
#tuned experiments
if 0:
#    rofreq=9.006e9
#    rofreq=f2(6.2e-3)
###    ag3.set_frequency(9197e6)
##
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ro_pow=-21
#    ag1.set_power(ro_pow)
    
#    yoko_currs = np.linspace(0.006890, 0.006912, 150)
#    y = np.linspace (1, 0, 150)
#    y = np.linspace (0.75, 0.62, 100)
#    y2 = np.linspace (y[15], 0, 300-15)
#    yoko_currs = np.arccos(y)*1e-3
#    yoko_currs = yoko_currs[26:]
#    qubit_freq = 6748.39e6
#    stark_pwrs=np.linspace(1e-4, 4e-3, 20)
#    stark_pwrs=10*np.log10(stark_pwrs/1e-3)
    stark_V = np.linspace(0.16, 0.200, 81)
#    stark_pwrs = np.sqrt(stark_pwrs)
    qubit_freq = 6335.170120e6-50e6
    qubit_brick.set_frequency(qubit_freq+50e6) 
#    freq_range = 5e6
#    init_freqs = np.linspace(-freq_range, freq_range, 201)
    init_freqs = np.linspace(-3e6, 3e6, 201)
    amps = np.linspace(-0.2, 0.2, 81)
    T1_delays = np.concatenate((np.linspace(0, 20e3, 100), np.linspace(21e3, 50e3, 51)))
    PhT1_delays = np.concatenate((np.linspace(0, 50e3, 101),
                                 np.linspace(51e3, 150e3, 50), 
                                 np.linspace(152e3, 300e3, 51)))
    qubit_rf_pwr = 13
        
    sy=tuned_qubit.Tuned_Qubit('qubit_info', qubit_brick, funcgen, alazar, 'STARK_V', stark_V, 
                               ['SSBSPEC', 'RABI', 'T1', 'PhT1'],#'ROFREQ', 'RABI', 'T1', 'ROPWR'
                               stark_rfsource = awg1,
                               
                               init_ro_freq = 9.080100e9,
                               
                               init_ro_power = -25,
                               ropwr_qubit_rf_pwr = qubit_rf_pwr,                                  
                              
                               init_freqs = init_freqs, spec_avgs = 1000,
                               spec_funcgen_freq = 10000,
                               init_spec_power = qubit_rf_pwr,
                               freq_step=-0.00001,
                               width_min = 0.0005e6, width_max = 50e6, subtraction = False, 
                               use_IQge=True, use_weight=True,
                               
                               rabi_funcgen_freq = 10000, rabi_rf_pwr = qubit_rf_pwr, 
                               rabi_avgs = 1000, init_rabi_amps = amps,
                               
                               T1_rf_pwr = qubit_rf_pwr, T1_funcgen_freq = 5000, 
                               T1_avgs = 2000, init_T1_delays = T1_delays, T1_update_delays = False,
                               
                               PhT1_funcgen_freq = 2000, 
                               PhT1_avgs = 2000, init_PhT1_delays = PhT1_delays, Ph_amp = 0.131, 
                               Ph_piLength = 0.600e3, Ph_sigma = 10,

                               )
                               
    sy.measure()
    
    blah    
#
if 1: #vacuum rabi
    baseAmp = 0.185
#    amps = np.sqrt(np.linspace(0.06**2, 0.165**2, 101))-baseAmp
#    amps = np.array([0.1985])-baseAmp
#    amps = np.linspace(0.128, 0.120, 9)-baseAmp
    amps = [0.125-baseAmp]
#    amps = np.array([0.29])-baseAmp #this is the difference between the target amplitude and the base (off-resonant) amplitude
#    for amp in amps:
    delays =np.concatenate((np.linspace(0, 10e3, 100), np.linspace(10e3, 20e3, 21)))
#    delays = np.linspace(0, 4e3, 201)
    p = phonon1_info.rotate_selective
    seq = sequencer.Sequence([p(0, 0, amp = 0.01)]) 
    
    for amp in amps:
#    for amp in np.linspace(-0.034, -0.032, 10):
        spec = stark_swap.stark_swap(qubit_info, delays, amp = amp,
                                     sigma =10, calib = 1, #qubit_pi = False,
#                                     seq2= seq,#precool, 
#                                     seq = precool,
    #                           simulseq = simulseq,
    #                           postseq = postseq,
    #                           plot_seqs=False,
                               extra_info = phonon1_info
                               )
        spec.measure()   
#        plt.close()
    
    bla
    
if 0: #vacuum rabi
    baseAmp = 0.185
#    amps = np.sqrt(np.linspace(0.06**2, 0.165**2, 101))-baseAmp
#    amps = np.array([0.1985])-baseAmp
#    amps = np.linspace(0.128, 0.120, 9)-baseAmp
    toAmps = [0.085, 0.125, 0.241, 0.245]
#    amps = toAmps - baseAmp
#    amps = np.array([0.29])-baseAmp #this is the difference between the target amplitude and the base (off-resonant) amplitude
#    for amp in amps:
#    delays =np.concatenate((np.linspace(0, 10e3, 100), np.linspace(10e3, 20e3, 21)))
#    delays = np.linspace(0, 4e3, 201)
    delays = np.linspace(0, 20e3, 201)
    p = phonon1_info.rotate_selective
    seq = sequencer.Sequence([p(0, 0, amp = 0.01)]) 
    
    centers = []    
    
    for amp in toAmps:
#    for amp in np.linspace(-0.034, -0.032, 10):
        awg1.set_ch3_offset(amp)
        alazar.set_naverages(1000)
        funcgen.set_frequency(10000)
        spec = ssbspec_fit.SSBSpec_fit(qubit_info, np.linspace(-7e6, 10e6, 201), 
                               plot_seqs=False,
#                               bgcor = True,
                                )
        spec.measure()
        centers = np.append(centers, spec.fit_params['x0'].value)
        
        awg1.set_ch3_offset(baseAmp)
        alazar.set_naverages(1000)
        funcgen.set_frequency(5000)       
        spec = stark_swap.stark_swap(qubit_info, delays, amp = amp-baseAmp,
                                     sigma =10, calib = 1, #qubit_pi = False,
#                                     seq2= seq,#precool, 
#                                     seq = precool,
    #                           simulseq = simulseq,
    #                           postseq = postseq,
    #                           plot_seqs=False,
                               extra_info = phonon1_info
                               )
        spec.measure()   
#        plt.close()

    data=mclient.datafile.create_group('starkSwpCenters')
    data.create_dataset('centers', data=centers)
    
    bla
    

if 0: #phonon T1
    #swap values for different stark amplitudes, from ipython notebook:
#    swapLengths = np.loadtxt(r'Z:\Yiwen\Acoustics\swapLengths20171204.txt')
    swapLengths = [0.660e3]
#    baseAmp = 0.1575
#    amps = np.sqrt(np.linspace(0.114**2, 0.144**2, 51))-baseAmp
    baseAmp = 0.185
#    amps = np.sqrt(np.linspace(0.084**2, 0.140**2, 61))-baseAmp
#    amps = amps[21:46]
    amps = np.array([0.125])-baseAmp
    funcgen.set_frequency(5000)
    delays = np.concatenate((np.linspace(0, 50e3, 101),
                             np.linspace(51e3, 150e3, 50), 
                             np.linspace(152e3, 300e3, 51)))
#    delays = np.concatenate((np.linspace(0, 8e3, 61),
#                             np.linspace(8e3, 20e3, 20)))
#    delays = np.linspace(0, 20e3, 81)
    for n_swaps in [1]:# range(1,8):
        for amp, piLength in zip(amps, swapLengths):
    #    piLength = 0.672e3
    #    for amp in amps:

    #        delays = np.linspace(0, 350e3, 201)
            spec = stark_swap.phonon_T1(qubit_info, 
                                    delays, phonon_pi = piLength, amp = amp,
                                    sigma = 10,
#                                    n_swaps = n_swaps,
                                    calib = 1,
        #                           seq=seq, 
        #                           simulseq = simulseq,
        #                           postseq = postseq,
        #                           plot_seqs=False,
        #                           extra_info = cavity_info
                                   )
            spec.measure()
#            plt.close()
    
    bla
    
if 0: #phonon and qubit T1
    #swap values for different stark amplitudes, from ipython notebook:
#    swapLengths = np.loadtxt(r'Z:\Yiwen\Acoustics\swapLengths20171130.txt')
#    baseAmp = 0.1575
#    amps = np.sqrt(np.linspace(0.114**2, 0.144**2, 51))-baseAmp

    ch3Vals = np.linspace(0.125, 0.200, 51)
    brickFreqs = 3.2999*(ch3Vals-0.084)**2/(0.165-0.084)**2-3.2999
    piLength = 0.725e3
        
    for ch3Val, brickf in zip(ch3Vals, brickFreqs):
        awg1.set_ch3_offset(ch3Val)
        qubit_brick.set_frequency(6135.306650e6-brickf*1e6)
        amp = 0.103-ch3Val
    
    
#    for amp, piLength in zip(amps[15:], swapLengths[15:]):

#    for amp in amps:
#        delays = np.concatenate((np.linspace(1, 50e3, 100),
#                                 np.linspace(50e3, 100e3, 51), 
#                                 np.linspace(100e3, 150e3, 21),
#                                 np.linspace(150e3, 200e3, 10)))
        delays = np.linspace(0, 300e3, 201)
        spec = stark_swap.phonon_T1(qubit_info, 
                                delays, phonon_pi = piLength, amp = amp,
                                sigma = 10,
    #                           seq=seq, 
    #                           simulseq = simulseq,
    #                           postseq = postseq,
    #                           plot_seqs=False,
    #                           extra_info = cavity_info
                               )
        spec.measure()
        plt.close()
    
    bla

#repeated phonon T1's
if 0:
    from scripts.single_qubit.t1t2_plotting import do_T1_plot, do_T1_phonon_plot
    Qubit_t1s = {'t1s':[], 't1s_err':[], 'ofs':[], 'ofs_err':[], 'amps':[], 'amps_err':[], 'vars':[],}
    Phonon_t1s = {'t1s':[], 't1s_err':[], 'ofs':[], 'ofs_err':[], 'amps':[], 'amps_err':[], 'vars':[],}
    
    delays = np.concatenate((np.linspace(0, 100e3, 101),
                                 np.linspace(102e3, 200e3, 50), 
                                 np.linspace(203e3, 350e3, 51)))
    st = datetime.now()
    for i in range(1000):
        time.sleep(1)
        do_T1_plot(qubit_info, 2000, np.concatenate((np.linspace(0, 20e3, 51), np.linspace(21e3, 50e3, 51))), 
                   Qubit_t1s, 300, var=(datetime.now()-st).total_seconds()/60)
                  
        do_T1_phonon_plot(qubit_info, 2000, delays, 0.103-0.165, 0.725e3,
                          Phonon_t1s, 301, var=(datetime.now()-st).total_seconds()/60, sigma = 10)
    
if 0: #phonon T2
#    for amp in -np.sqrt(np.linspace(0.25**2, 0.1**2, 21)):
    swapLength = 0.660e3
    baseAmp = 0.185
    amps = np.array([0.125])-baseAmp
    
    for amp in amps:
#        delays = np.concatenate((np.linspace(1, 5e3, 150), np.linspace(5e3, 10e3, 100),
#                                 np.linspace(10e3, 20e3, 61), 
#                                 np.linspace(20e3, 30e3, 21)))
        delays = np.linspace(0, 200e3, 201)
        spec = stark_swap.phonon_T2(qubit_info, 
                                delays, detune = 6.28743828E9 - (6332.492620e6-50e6)+50e3, phonon_pi = swapLength, amp = amp,
                                sigma = 10,
    #                           seq=seq, 
    #                           simulseq = simulseq,
    #                           postseq = postseq,
    #                           plot_seqs=False,
    #                           extra_info = cavity_info
                               )
        spec.measure()    
    
    bla
    
if 0: #phonon T2 echo
#    for amp in -np.sqrt(np.linspace(0.25**2, 0.1**2, 21)):
    swapLength = 0.615e3
    baseAmp = 0.184
    amps = np.array([0.131])-baseAmp
    for amp in amps:
#        delays = np.concatenate((np.linspace(1, 5e3, 150), np.linspace(5e3, 10e3, 100),
#                                 np.linspace(10e3, 20e3, 61), 
#                                 np.linspace(20e3, 30e3, 21)))
        delays = np.linspace(0, 200e3, 151)
        spec = stark_swap.phonon_T2(qubit_info, 
                                delays, detune = 50e3, phonon_pi = swapLength, amp = amp,
                                sigma = 10,
                                necho = 1,
    #                           seq=seq, 
    #                           simulseq = simulseq,
    #                           postseq = postseq,
    #                           plot_seqs=False,
    #                           extra_info = cavity_info
                               )
        spec.measure()    
    
    bla
    
if 0:#swap Rabi (qubit Rabi after swap with phonon)
#    for amp in -np.sqrt(np.linspace(0.25**2, 0.1**2, 21)):
    for amp in [-0.08]:
#        for phonon_pi in np.linspace(0, 2*0.786e3, 21):
        for phonon_pi in [0.7e3]:
            for delay in [0]:#np.concatenate((np.linspace(0, 10e3, 5), np.linspace(15e3, 30e3, 4))):
                rabi_amps = np.linspace(-1, 1, 101)*0.5
                spec = stark_swap.swap_Rabi(qubit_info, 
                                        rabi_amps, phonon_pi = phonon_pi, amp = amp,
                                        sigma = 5,
                                        delay = delay,
            #                           seq=seq, 
            #                           simulseq = simulseq,
            #                           postseq = postseq,
            #                           plot_seqs=False,
            #                           extra_info = cavity_info
                                       )
                spec.measure()    
    
    bla
    
if 0:
#    for amp in -np.sqrt(np.linspace(0.25**2, 0.1**2, 21)):
    for amp in [-0]:
#        delays = np.concatenate((np.linspace(1, 5e3, 150), np.linspace(5e3, 10e3, 100),
#                                 np.linspace(10e3, 20e3, 61), 
#                                 np.linspace(20e3, 30e3, 21)))
        ro_powers = [-15]
        rofreq=9.150e9
#        rofreq=6.8e9
        freq_range=10e6
        freqs =  np.linspace(rofreq-freq_range, rofreq+freq_range, 101)        

        spec = rospecPS.ROCavSpectroscopyPS(qubit_info, ro_powers, freqs, 
                                phonon_pi = 0.786e3, amp = amp,
                                sigma = 50,
                                qubit_pulse = np.pi/2,
    #                           seq=seq, 
    #                           simulseq = simulseq,
    #                           postseq = postseq,
    #                           plot_seqs=False,
    #                           extra_info = cavity_info
                               )
        spec.measure()    
    
    bla
    
if 0: #swap temperature
#    for amp in -np.sqrt(np.linspace(0.25**2, 0.1**2, 21)):
    baseAmp = 0.184
#    amps = np.sqrt(np.linspace(0.06**2, 0.165**2, 101))-baseAmp
#    amps = np.array([0.1985])-baseAmp
#    amps = np.linspace(0.1986, 0.1998, 13)-baseAmp
    
    p = phonon1_info.rotate_selective
    seq = sequencer.Sequence([sequencer.Trigger(250), p(0, 0, amp = 0.01)]) 
    
    amps = [0]#[0.1995-baseAmp]
    for amp in amps:
        for phonon_pi in [0]:#[2.9e3]:
#        delays = np.concatenate((np.linspace(1, 5e3, 150), np.linspace(5e3, 10e3, 100),
#                                 np.linspace(10e3, 20e3, 61), 
#                                 np.linspace(20e3, 30e3, 21)))
            amps = np.linspace(-1, 1, 101)*0.2     
    
            spec = stark_swap.phonon_swap_temperature(qubit_ef_info, qubit_info, amps, 
                                    phonon_pi = phonon_pi, amp = amp,
                                    sigma = 10,
                                    seq=seq, 
        #                           simulseq = simulseq,
        #                           postseq = postseq,
        #                           plot_seqs=False,
                                   extra_info = phonon1_info
                                   )
            spec.measure()    
    
    bla
    
if 0: #phonon fock
    navg = 5000
    alazar.set_naverages(navg)
    funcgen.set_frequency(2000)
#    for amp in -np.sqrt(np.linspace(0.25**2, 0.1**2, 21)):
    sig = 10
    baseAmp = 0.185
    amps = [0.125-baseAmp]    
    
    for amp in amps:#-0.183 np.linspace(-0.194-0.05, -0.194+0.05, 10)
#        for phonon_pi in np.linspace(0, 2.2*0.786e3, 23):
        for phonon_pi in [0.660e3]:
#        delays = np.concatenate((np.linspace(1, 5e3, 150), np.linspace(5e3, 10e3, 100),
#                                 np.linspace(10e3, 20e3, 61), 
#                                 np.linspace(20e3, 30e3, 21)))
#            amps = np.linspace(-1, 1, 101)*0.25 
#            delays = np.concatenate((np.linspace(1, 5e3, 90), np.linspace(5e3, 10e3, 50),
#                                 np.linspace(10e3, 20e3, 20), 
#                                 np.linspace(20e3, 30e3, 5)))
#            delays = np.concatenate((np.linspace(1, 5e3, 140), np.linspace(5e3, 10e3, 70)))
#            delays1 = np.linspace(0, 6e3, 151) #breaking up into parts because AWG can't load so many delays for long sequences
#            delays2 = np.linspace(6e3+40, 12e3, 150)
#
#
#            
#            for n_swaps in [5]:#[5, 4, 3, 2, 1, 0]:
#                spec = stark_swap.phonon_fock(qubit_info, delays1, n_swaps = n_swaps, 
#                                        phonon_pi = phonon_pi, amp = amp,
#                                        sigma = sig,
#                                        seq=precool, 
#            #                           simulseq = simulseq,
#            #                           postseq = postseq,
#            #                           plot_seqs=False,
#            #                           extra_info = cavity_info
#                                       )
#                spec.measure()  
#                spec = stark_swap.phonon_fock(qubit_info, delays2, n_swaps = n_swaps, 
#                                        phonon_pi = phonon_pi, amp = amp,
#                                        sigma = sig,
#            #                           seq=seq, 
#            #                           simulseq = simulseq,
#            #                           postseq = postseq,
#            #                           plot_seqs=False,
#            #                           extra_info = cavity_info
#                                       )
#                spec.measure()  

            delays1 = np.linspace(0, 4e3, 101) #breaking up into parts because AWG can't load so many delays for long sequences
            delays2 = np.linspace(4e3+40, 8e3, 100)
            delays3 = np.linspace(8e3+40, 12e3, 100)
            delaysArr = [delays1, delays2, delays3]
#            delays = np.linspace(0, 12e3, 301)

            
#            for n_swaps in [7, 6, 5, 4, 3, 2, 1, 0]:
#                for delays in [delays1, delays2, delays3]:
#                    spec = stark_swap.phonon_fock(qubit_info, delays, n_swaps = n_swaps, 
#                                            phonon_pi = phonon_pi, amp = amp,
#                                            sigma = sig,
#                                            seq=precool,
#                                            calib = 1,
#                #                           simulseq = simulseq,
#                #                           postseq = postseq,
#                #                           plot_seqs=False,
#                #                           extra_info = cavity_info
#                                           )
#                    spec.measure()  
#                    plt.close()
                    
            for n_swaps in [7, 6, 5, 4, 3, 2, 1, 0]:
                for delays in delaysArr:#[delays1, delays2, delays3]:
                    spec = stark_swap.phonon_fock(qubit_info, delays, n_swaps = n_swaps, 
                                            phonon_pi = phonon_pi, amp = amp,
                                            sigma = sig,
                                            seq=precool,
                                            calib = 1,
                                            take_shots = True,
                                            shots_avg = navg/50,
                #                           simulseq = simulseq,
                #                           postseq = postseq,
                #                           plot_seqs=False,
                #                           extra_info = cavity_info
                                           )
                    spec.measure()  
                    plt.close()
    bla
    
if 0:
#    for amp in -np.sqrt(np.linspace(0.25**2, 0.1**2, 21)):
    for amp in [-0.0445]:
#        delays = np.concatenate((np.linspace(1, 5e3, 150), np.linspace(5e3, 10e3, 100),
#                                 np.linspace(10e3, 20e3, 61), 
#                                 np.linspace(20e3, 30e3, 21)))
        drive_amps = [-35, -30, -25, -20]
        spec_params = [spec_brick, spec_info]
        spec = stark_swap.phonon_drive(qubit_info, spec_params,
                                drive_amps, phonon_pi = 0.5846e3, amp = amp,
                                sigma = 50,
    #                           seq=seq, 
    #                           simulseq = simulseq,
    #                           postseq = postseq,
    #                           plot_seqs=False,
    #                           extra_info = cavity_info
                               )
        spec.measure()    
    
    bla
    
if 0:# SSB Spec
    
    r = qubit_info.rotate
    seq = sequencer.Sequence() #[sequencer.Trigger(250)]
    seq.append(r(np.pi, 0))
    seq.append(pulselib.GaussSquare(0.580e3, 0.124-0.184, 10, chan = 3))

#    postseq = sequencer.Delay(-160)
    
    alazar.set_naverages(500000)

    spec = ssbspec_fit.SSBSpec_fit(qubit_info, np.linspace(-0.2e6, 0.4e6, 151), 
                           seq2=seq, 
#                           postseq = postseq,
#                           plot_seqs=False,
                           bgcor = True,
#                           extra_info = cavity_info
                            )
    spec.measure()
    
    
#    qubit_ef_brick.set_rf_on(False)
    bla 


if 0:# SSB Spec with phonon drive, changing drive amplitude
    
#    ch3_offsets = np.linspace(-0.027, 0.027, 11)
    alazar.set_naverages(10000)
    phonon_pi = 0.660e3
    amp = 0.125-0.185
    
#    freqs = np.concatenate((np.linspace(-0.5e6, 0.5e6, 81), np.linspace(0.5e6, 5.6e6, 151)))
    freqs = np.linspace(-0.3e6, 0.3e6, 101)
    seq = sequencer.Sequence([sequencer.Trigger(250), # prepend pi pulse
                qubit_info.rotate(np.pi,0)])
                
    simulseq = qubit_info.rotate_selective(np.pi,0)

    for p_drive_amp in [0.001]:# 10**np.linspace(np.log10(0.002), np.log10(0.05), 20):
#    for p_drive_amp in [0.01]:#0.0006
        for ch3_offset in [0.0]:
#            awg1.set_ch3_offset(0.342+ch3_offset)
    
            #with swap pulse after drive (so measuring phonon excitation)
            spec = stark_swap.phonon_SSB_spec(phonon1_info, freqs, drive_amp = p_drive_amp,
                                               phonon_pi = phonon_pi, amp = amp-ch3_offset,
                                               sigma = 10, calib = 1,
#                                               simul_drive_amp = 0.6e-3,#0.5e-3,
                                               qubit_info=qubit_info,
#                                               seq2=seq, 
#                                               simulseq = simulseq,
                    #                           postseq = postseq,
                                               plot_seqs=False,
                                               extra_info = qubit_info
                                                )
            spec.measure()
            
#            #without swap pulse after drive (so measuring qubit excitation)
#            spec = stark_swap.phonon_SSB_spec(phonon1_info, freqs, drive_amp = p_drive_amp,
#                                               phonon_pi = phonon_pi, amp = 0,#amp-ch3_offset,
#                                               sigma = 10, calib = 1,
##                                               simul_drive_amp = 0.6e-3,#0.5e-3,
#                                               qubit_info=qubit_info,
##                                               seq=seq, 
##                                               simulseq = simulseq,
#                    #                           postseq = postseq,
#                                               plot_seqs=False,
#                                               extra_info = qubit_info
#                                                )
#            spec.measure()
#            
        
    
#    qubit_ef_brick.set_rf_on(False)
    bla 
        

if 0:# SSB Spec with phonon drive
    
    ch3_offsets = np.linspace(-0.027, 0.027, 11)
    alazar.set_naverages(10000)
    phonon_pi = 0.700e3
    amp = -0.077
    
#    freqs = np.concatenate((np.linspace(-0.5e6, 0.5e6, 81), np.linspace(0.5e6, 5e6, 101)))
    freqs = np.linspace(-4e6, 4e6, 151)
    seq = sequencer.Sequence([sequencer.Trigger(250), # prepend pi pulse
                qubit_info.rotate(np.pi,0)])
                
    simulseq = qubit_info.rotate_selective(np.pi,0)

#    for p_drive_amp in 10**np.linspace(-3, -2, 11):
    for p_drive_amp in [0.002]:#0.0006
        for ch3_offset in [0.01]:
            awg1.set_ch3_offset(0.168+ch3_offset)
    
            spec = stark_swap.phonon_SSB_spec(phonon1_info, freqs, drive_amp = p_drive_amp,
                                               phonon_pi = phonon_pi, amp = amp-ch3_offset,
                                               sigma = 50,
#                                               simul_drive_amp = 0.6e-3,#0.5e-3,
#                                               qubit_info=qubit_info,
#                                               seq=seq, 
#                                               simulseq = simulseq,
                    #                           postseq = postseq,
                                               plot_seqs=False,
                                               extra_info = qubit_info
                                                )
            spec.measure()
            
        
    
#    qubit_ef_brick.set_rf_on(False)
    bla 
    
if 0: #phonon displacement followed by phonon-qubit interaction for readout
    alazar.set_naverages(30000)
    funcgen.set_frequency(2000)
#    for amp in -np.sqrt(np.linspace(0.25**2, 0.1**2, 21)):
    delays1 = np.linspace(0, 6e3, 151)
    delays2 = np.linspace(6e3+40, 12e3, 150)
#    delays1 = np.linspace(0, 500, 10)
#    p_drive_amps = np.linspace(0.0005, 0.015, 30)[:20] #for w=4us
    p_drive_amps = np.linspace(0.0525, 0.08, 12)#np.linspace(0.00, 0.05, 21) #for w = 0.5 us
    sig = 10
    baseAmp = 0.184
    amps = [0.124-baseAmp]  
    
    for amp in amps:#-0.183 np.linspace(-0.194-0.05, -0.194+0.05, 10)
#        for phonon_pi in np.linspace(0, 2.2*0.786e3, 23):
        for phonon_pi in [0.580e3]:
#            delays = np.concatenate((np.linspace(1, 5e3, 140), np.linspace(5e3, 10e3, 70)))
            
            
            for drive_amp in p_drive_amps:
#            for drive_amp in [np.pi]:
                spec = stark_swap.phonon_displacement_SSB(phonon1_info, delays1, drive_amp = drive_amp, 
                                        phonon_pi = phonon_pi, amp = amp,
                                        sigma = 10,
                                        calib = 1,
                                        qubit_info = qubit_info,
#                                        seq2=precool2, 
            #                           simulseq = simulseq,
            #                           postseq = postseq,
            #                           plot_seqs=False,
            #                           extra_info = cavity_info
                                       )
                spec.measure()  
                
                spec = stark_swap.phonon_displacement_SSB(phonon1_info, delays2, drive_amp = drive_amp, 
                                        phonon_pi = phonon_pi, amp = amp,
                                        sigma = 10,
                                        calib = 1,
                                        qubit_info = qubit_info,
#                                        seq2=precool2, 
            #                           simulseq = simulseq,
            #                           postseq = postseq,
            #                           plot_seqs=False,
            #                           extra_info = cavity_info
                                       )
                spec.measure()  
               
#                plt.close
    bla
    
if 1: #Phonon Wigner (started 20180331105600)
    navg = 10000
    alazar.set_naverages(navg)
    funcgen.set_frequency(2000)
#    for amp in -np.sqrt(np.linspace(0.25**2, 0.1**2, 21)):
#    delay_array = [np.linspace(0, 4e3, 101), np.linspace(4e3+40, 8e3, 100), np.linspace(8e3+40, 12e3, 100)]
    delay_array = [np.linspace(0, 12e3, 101)]
#    delay_array = [np.linspace(0, 250e3, 51)]
    dx = np.linspace(-2, 2, 17)#np.linspace(0, 2, 6)
#    delta = (dx[1] - dx[0]) / 2
    dy = 1j * dx
    XS, YS = np.meshgrid(dx, dy)
    displacements = -(XS + YS)
    displacements = -0.6*np.exp(1j*2*np.pi*np.linspace(0, 1, 16, endpoint = False))
#    displacements = np.array([-0.4+0j, 0+0.4j, 0.4+0j, 0-0.4j])
    baseAmp = 0.185
    amp = 0.125-baseAmp
    phonon_pi = 0.660e3
    N = 2
    sig = 10
    alphaScale = 1/83.#2.9e-3/1.3*2.*1.2 #
    
    r = qubit_info.rotate
    seq = sequencer.Sequence()
    
    for j in range(N):
#        seq.append(r(np.pi/2, 0))
        seq.append(r(np.pi, 0))
        seq.append(pulselib.GaussSquare(phonon_pi/np.sqrt(j+1), amp, sig, chan = 3))
            
    for ind, d in enumerate(alphaScale*displacements.flatten()):#[136:]:#np.array([1.5+1.5j]):
#    for d in displacements[0]:
        if 1:
            alazar.set_naverages(2000)
            funcgen.set_frequency(10000)
            spec = ssbspec_fit.SSBSpec_fit(qubit_info, np.linspace(-2e6, 2e6, 101), 
                               plot_seqs=False,
                               bgcor = True,
                               keep_data = False,
                                )
            spec.measure()  
#            plt.close()
            curCur = yoko2.do_get_current()
            if spec.fit_params['x0'].value > 50e3:
                yoko2.do_set_current(curCur-0.001e-3)
                print 'setting current to %0.6f A\n' % (curCur-0.001e-3)
                
            elif spec.fit_params['x0'].value < -50e3:
                yoko2.do_set_current(curCur+0.001e-3)
                print 'setting current to %0.6f A\n' % (curCur+0.001e-3)
        
        
        alazar.set_naverages(navg)
        funcgen.set_frequency(2000)
        for delays in delay_array:
#            for drive_amp in [np.pi]:
            spec = stark_swap.phonon_wigner(phonon1_info, delays, 
                                    drive = d, 
                                    phonon_pi = phonon_pi, amp = amp,
                                    sigma = sig,
                                    statePrep=seq, 
                                    seq = precool,
                                    qubit_info=qubit_info,
                                    calib=1,
#                                    post_cooling = True,
                                    take_shots = True,
                                    shots_avg = navg/50,
        #                           simulseq = simulseq,
        #                           postseq = postseq,
        #                           plot_seqs=False,
                                   extra_info = [qubit_info, phonon1_info]
                                   )
            spec.measure()  
            plt.close('all')

    bla    
    
if 0: #qubit cooling
    #swap values for different stark amplitudes, from ipython notebook:
#    swapLengths = np.loadtxt(r'Z:\Yiwen\Acoustics\swapLengths20171023.txt')
#    baseAmp = 0.342
#    amps = np.sqrt(np.linspace(0.255**2, 0.282**2, 21))-baseAmp
    amps=[-0.08]
#    for amp, piLength in zip(amps, swapLengths):
    piLength = 0.700e3
    for amp in amps:
        delays = np.linspace(0, 80e3, 201)
#        delays = np.linspace(0, 80e3, 201)
        spec = stark_swap.phonon_T1(qubit_info, 
                                delays, phonon_pi = piLength, amp = amp,
                                sigma = 10,
                                qubit_pi = True,
                                second_swap = False,
    #                           seq=seq, 
    #                           simulseq = simulseq,
    #                           postseq = postseq,
    #                           plot_seqs=False,
    #                           extra_info = cavity_info
                               )
        spec.measure()
#        plt.close()
    
    bla