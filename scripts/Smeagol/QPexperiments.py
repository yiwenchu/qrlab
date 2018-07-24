import mclient
from mclient import instruments
import time
import numpy as np
from pulseseq import sequencer
from pulseseq import pulselib
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
import logging
from scripts.single_qubit import spectroscopy as spectroscopy
from scripts.jk.single_qubit import temperature as temperature

mpl.rcParams['figure.figsize']=[6,4]

n_qubit = 1

ag1 = instruments['ag1']
ag2 = instruments['ag2']
#ag3 = instruments['ag3']
qubit_info = mclient.get_qubit_info('qubit_info')#_{}'.format(n_qubit))
qubit_brick = instruments['qubit_brick']#_{}'.format(n_qubit)]
qubit_ef_info = mclient.get_qubit_info('qubit_ef_info')#_{}'.format(n_qubit))

filename = 'c:/Data/20170623/CK2Q{}.h5'.format(n_qubit)
mclient.datafile = mclient.datasrv.get_file(filename)

#cavity_info = mclient.get_qubit_info('cavity_info')
#spec_sh = instruments['spec']
awg1 = instruments['AWG1']

#qubit_ef_brick = instruments['qubit_brick']

cavity_brick = instruments['cavity_brick']
#qubit_ef_brick = cavity_brick
#va_lo = instruments['va_lo']
#va_lo_5_10 = instruments['va_lo_5_10'] 
funcgen = instruments['funcgen']
alz = alazar = instruments['alazar']
spec_brick = instruments['spec_brick']
spec_info = mclient.get_qubit_info('spec_info')
LO_brick = instruments['LO_brick']
#LO_info = mclient.get_qubit_info('LO_info')
cavity_info = mclient.get_qubit_info('cavity_info')

#laserfg = mclient.instruments['laserfg']
#trigger for experiments
#funcgen.sync_on(1);
readout_info = mclient.instruments['readout']



if 0:#switch qubits
    qi = mclient.instruments['qubit_info']
    efi = mclient.instruments['qubit_ef_info']
    
    if 1:        
        #LC qubit 1
        qubit_brick.set_frequency(5453.198e6+50e6)
        awg1.set_ch1_amplitude(2)
        awg1.set_ch1_offset(-0.042)
        awg1.set_ch2_amplitude(1.956)
        awg1.set_ch2_offset(-0.035)
        qi.set_sideband_phase(-0.019478)
        qi.set_pi_amp(0.2170)
        qi.set_w(20)
#        qubit_ef_brick.set_frequency(5815.9e6+50e6)
#        awg1.set_ch3_amplitude(2)
#        awg1.set_ch3_offset(0.078)
#        awg1.set_ch4_amplitude(2.87)
#        awg1.set_ch4_offset(0.227)
#        efi.set_sideband_phase(0.059690)
#        efi.set_pi_amp(0.247)
        filename = 'c:/Data/20160210/LCQ1.h5'
        mclient.datafile = mclient.datasrv.get_file(filename)
        readout_info.set_IQe(-3.23521545657-0.611922014166j) 
        readout_info.set_IQg(-5.32504395516-15.7363640546j) 

 
        
    if 0:        
        #Lc qubit 2
        qubit_brick.set_frequency(5775.011e6)
        awg1.set_ch1_amplitude(2)
        awg1.set_ch1_offset(-0.045)
        awg1.set_ch2_amplitude(1.970)
        awg1.set_ch2_offset(-0.042)
        qi.set_sideband_phase(0.033929)
        qi.set_pi_amp(0.1908)
        qi.set_w(20)
        
        qubit_ef_brick.set_frequency(5431e6-50e6)
        awg1.set_ch3_amplitude(2)
        awg1.set_ch3_offset(-0.047)
        awg1.set_ch4_amplitude(1.954)
        awg1.set_ch4_offset(-0.051)
        efi.set_sideband_phase(-0.052779)
        efi.set_pi_amp(0.142)
        
        filename = 'c:/Data/20160210/LCQ2.h5'
        mclient.datafile = mclient.datasrv.get_file(filename)
        readout_info.set_IQe(-3.31964134163-0.757886916155j)
        readout_info.set_IQg(-5.18031659483-15.7383967438j)
    
    bla



if 0:# Cavity spectroscopy 
    from scripts.single_cavity import rocavspectroscopy
    if 0: # sweep frequency, want to use a long readout tone
        f0 = ag1.get_frequency()
        p0 = ag1.get_power()

        ro_powers = [-15, -13, -11, -9]
        rofreq = 9142.226e6#9168.3e6#9170.471e6
        alazar.set_naverages(1e3)
        freq_range = 1e6
        freqs =  np.linspace(rofreq-freq_range, rofreq+freq_range, 41)
        funcgen.set_frequency(5e3)
        
#        seq =sequencer.Sequence()
#        seq.append(sequencer.Trigger(250))
#        seq.append(qubit_info.rotate(np.pi, 0))
#        seq.append(qubit2_info.rotate(np.pi, 0))
#        seq.append(sequencer.Constant(110e3, 1, chan="3m1"))
#        seq.append(sequencer.Delay(500e3))
        
        rospec = rocavspectroscopy.ROCavSpectroscopy(qubit_info, ro_powers, freqs, qubit_pulse=False, seq=None)
        rospec.measure()
#        rospec = rocavspectroscopy.ROCavSpectroscopy(qubit_info, ro_powers, freqs, qubit_pulse=False, seq=seq,
#                                                     extra_info = qubit2_info
#                                                     ) #qubit_pulse=np.pi/2
#        rospec.measure()
        
#        rospec = rocavspectroscopy.ROCavSpectroscopy(qubit_info, ro_powers, freqs, qubit_pulse=np.pi/2, seq=seq)
#        rospec.measure()
#        

#        
        ag1.set_frequency(f0)
        ag1.set_power(p0)
        LO_brick.set_frequency(f0+50e6)

    
    if 1: # sweep power 
        f0 = ag1.get_frequency()
        p0 = ag1.get_power()
        
        ro_powers = np.linspace(-20, 20, 21)
        rofreq= 9.1290e9
        freqs = np.array([rofreq,])
        rospec = rocavspectroscopy.ROCavSpectroscopy(qubit_info, ro_powers, freqs, qubit_pulse=0)
        rospec.measure()
        
        
#        rospec = rocavspectroscopy.ROCavSpectroscopy(qubit_info, ro_powers, freqs, qubit_pulse=np.pi)
#        rospec.measure()
        
        ag1.set_frequency(f0)
        ag1.set_power(p0)
        LO_brick.set_frequency(f0+50e6)
        
    if 0: #special power sweep for ability in reverse direction
        #set ag to step sweep over some powers manually. and type in here.  
        #trigger ag with fungen
        #trigger alazar with ag's trigger out
        from scripts.single_cavity import rocavspectroscopy_reverse
        ro_powers = np.linspace(-15, -8, 61)
#        ro_powers = np.linspace(-10, -8, 61)
        rofreq = 9.26355e9
#        rofreq = 9.2637e9
        freqs = np.array([rofreq,])
        rospec = rocavspectroscopy_reverse.ROCavSpectroscopy(qubit_info, ro_powers, freqs, plot_type = POWER, qubit_pulse=0)
        rospec.measure()
        
    if 0: #special freq sweep for ability in reverse directiond
        #You set ag to step sweep over some freqs manually. and type in here.  
        #remeber to set BOTH LO and RF gen sweeps IF appart
        #trigger LO gen with fungen (and set sweeptrig=freerun, pointtrig=extpos)
        #trigger RF gen with LO gen's trigger out (and set sweeptrig=extpos, pointtrig=extpos)
        #trigger alazar with RF gen's trigger out
        #to run: 
            #press RF "single sweep" to arm the sweep.
            #run this script,
            #press RF single sweep
        from scripts.single_cavity import rocavspectroscopy_reverse
#        rofreq = 9.26355e9
#        freq_range = 8.0e6
#        freqs = np.linspace(9.261e9, 9.265e9, 401)   #up
        freqs = np.linspace(9.265e9, 9.261e9, 401)   #down
        ro_power = -10
#        rofreq = 9.2637e9
        ro_powers = np.array([ro_power,])
        rospec = rocavspectroscopy_reverse.ROCavSpectroscopy(qubit_info, ro_powers, freqs, plot_type = SPEC, qubit_pulse=0)
        rospec.measure()
        
    blah
    
if 0: 
    from scripts.single_cavity import cavspectroscopy 
    freqs = np.linspace(8.84e9, 8.87e9, 101)   #down

    cavspec = cavspectroscopy.CavSpectroscopy(qubit_ef_brick, qubit_info, cavity_info, [0.1*np.pi], freqs)
    cavspec.measure()
    
if 0:  #old school RO cal
    from scripts.calibration import ropower_calibration as ropcal
#    specfreq=7.895e9 
    specfreq=9.2637e9
    specpower=5
        
    spec_brick.set_rf_on(1)
    spec_brick.set_pulse_on(1) 
    spec_brick.set_frequency(specfreq)
    spec_brick.set_power(specpower)
    
    powers = np.linspace(-15, 10, 26)
    # powers = np.linspace(-30, -10, 11)
    # ag_vector.set_frequency(9125.49e6)
    # ag_vector.set_power(30)`
    
    # ropcal.SAT_PULSE ropcal.PI_PULSE
    r = ropcal.ROPower_Calibration(
        spec_info, powers, qubit_pulse=ropcal.SAT_PULSE,
        pulse_len=100e3)
    outs = r.measure()

# qubit spectroscopy with no sidebands
if 0:  # qubit spectroscopy with no sidebands
    qubit_freq = 6572e6
    freq_range = 0.2e6
    n_points = 81
    freqs = np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, n_points)
 
    ro_powers=[-5]
    drive_powers=[-10]

    qubit_brick.set_rf_on(0)
    spec_brick.set_rf_on(1)
    spec_brick.set_pulse_on(1)
    for ro_pow in ro_powers:
#        ag1.set_power(ro_pow)
        for drive_power in drive_powers:
            spec_params = (spec_brick, drive_power)
            s = spectroscopy.Spectroscopy(spec_info, freqs, spec_params,
                     plen=100e3, freq_delay=0.25,
                     seq=None, postseq=None,
                     use_weight=False, use_IQge=False,
                     use_marker = True,
                     subtraction = False,)
            s.measure()
    spec_brick.set_rf_on(0)
    bla



if 0:# sideband modulated spectroscopy
    qubit_brick.set_rf_on(1)
    spec_brick.set_rf_on(0)
    qubit_freq = 5735.6e6 # 50 MHz above qubit mode
    freq_range = 2e6
    npoints = 81
    alz.set_naverages(1000)

    spec_params = (qubit_brick, None) # brick name, power    
    s = spectroscopy.Spectroscopy(qubit_info,
                                      np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, npoints),
                                      spec_params, 
                                      plen=100e3, amp=0.01
                                      , plot_seqs=False,subtraction = False,
                                      use_IQge=True) #1=1ns
    s.measure()
    
    
#    qubit_brick.set_rf_on(0)
#    qubit_ef_brick.set_rf_on(1)
##    spec_brick.set_rf_on(0)
#    qubit_freq = 5823.8e6 #7257e6
#    freq_range = 2e6
#    
#
#    spec_params = (qubit_ef_brick, None) # brick name, power    
#    s = spectroscopy.Spectroscopy(qubit_ef_info,
#                                      np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 61),
#                                      spec_params, 
#                                      plen=20000, amp=0.002, plot_seqs=False,subtraction = False) #1=1ns
#    s.measure()    
    
#    s = spectroscopy.Spectroscopy(qubit_info,
#                                      np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 1201),
#                                      spec_params, 
#                                      plen=10000, amp=0.00, plot_seqs=False,subtraction = False) #1=1ns
#    s.measure()
    blah
    
#    ag3_pwrs=np.linspace(-10, 12, 12)
#    ag3_pwrs=10*np.log10([2])   
#    for pwr in ag3_pwrs:
#        ag3.set_power(pwr)
#        s = spectroscopy.Spectroscopy(qubit_info,
#                                         np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 71), spec_params,
#                                         plen=50000, amp=0.01, plot_seqs=False,subtraction = False) #1=1ns5
#        s.measure()
#    qubit_brick.set_rf_on(0)

                                     


#"""Power Rabi -- Pi pulse calibration"""
if 0: # Calibrate pi pulse
    funcgen.set_frequency(5e3)
    alz.set_naverages(2000)   
    amps = np.linspace(-1,1,101)*0.001
    for i in range(1):
        from scripts.single_qubit import rabi
        tr = rabi.Rabi(qubit_info, amps,
                       seq = None, postseq = None,
                       selective = True,
                       plot_seqs=False,
                       update=True)

#        from scripts.single_qubit import rabi_IQ
#        tr = rabi_IQ.Rabi(qubit_info, np.linspace(0, 0.5, 101), plot_seqs=False, real_signals=False)
        tr.measure()
#    bla
    funcgen.set_frequency(1e3)
    

if 0:# calibrate readout
    from scripts.calibration import ro_power_cal
#
#    rofreq= 9.151220e9
#
#    
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ag2.set_frequency(rofreq+50e6)
#    ro_pow=np.linspace(-40, -30, 6)
    ro_pow = [-23]
    cal = ro_power_cal.Optimal_Readout_Power(qubit_info, ro_pow, plen=None, amp=0.005, shots=8e3, 
                                             update_readout=True, 
#                                             hist_steps=25,
                                             )    
    cal.measure()
    bla    
    



if 0: # spectroscopy for EF transition
    from scripts.single_qubit import spectroscopy as spectroscopy
    qubit_ef_freq = 5898.7e6#5239.790e6-50e6-175e6
    freq_range = 10e6
    freqs = np.linspace(qubit_ef_freq-freq_range, qubit_ef_freq+freq_range, 101)
    drive_power = -33
    
    qubit_brick.set_rf_on(1)
    spec_brick.set_rf_on(1)
    spec_brick.set_pulse_on(1)
    seq = sequencer.Sequence([sequencer.Trigger(250), # prepend pi pulse
                qubit_info.rotate(np.pi,0),
                sequencer.Delay(10)])
                
    postseq = sequencer.Sequence(qubit_info.rotate(np.pi, 0)) # postpend pi pulse
#    for drive_power in drive_powers:
    spec_params = (spec_brick, drive_power)
    spec_brick.set_power(drive_power)
#    spec_params = (qubit_ef_brick, None)
    s = spectroscopy.Spectroscopy(spec_info, freqs, spec_params, use_marker=True,
                                  plen = 5e3,amp = 0.1, freq_delay=0.25,
                                  seq=seq, # this and the next line adds pi pulses before and after spec pulse, for e-f spec
                                  postseq=postseq,
                                  extra_info = qubit_info)
    s.measure()
    spec_brick.set_rf_on(0)
    bla

if 0: #sideband EF spec  w/ stepped generator
    funcgen.set_frequency(5000)
    alz.set_naverages(1000)
    qubit_ef_brick.set_rf_on(1)
    qubit_ef_brick.set_pulse_on(0)
#    cavity_brick.set_rf_on(1)    
#    cavity_brick.set_pulse_on(0)
    qubit_ef_freq = 5949.12e6 #4758e6
#    qubit_freq =  5197.589e6
#   freqs = np.linspace(5960, 6000, 101) * 1e6
    freqs = np.linspace(qubit_ef_freq - 4e6, qubit_ef_freq + 4e6, 41)
    
#    rofreq=9022.97e6
#    ro_pow=7
    
#    rofreq=9.039020e9
#    ro_pow=-20
    
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)

#    ag1.set_power(ro_pow)
#
    seq = sequencer.Sequence([sequencer.Trigger(250), # prepend pi pulse
                qubit_info.rotate(np.pi,0),
                sequencer.Delay(10)])
                
    postseq = sequencer.Sequence(qubit_info.rotate(np.pi, 0)) # postpend pi pulse

    spec_params = (qubit_ef_brick, None)
#    spec_params = (cavity_brick, None)
    s = spectroscopy.Spectroscopy(qubit_ef_info, freqs, spec_params,
                                 plen=2000, amp=0.005, 
                                 seq=seq, postseq=postseq, 
                                 freq_delay=0.25,
                                 extra_info=[qubit_info,],
                                 plot_seqs=False, subtraction = False,                                 
                                 use_IQge=False)
#    s = spectroscopy.Spectroscopy(cavity_info, freqs, spec_params,
#                                 plen=2000, amp=0.005, 
#                                 seq=seq, postseq=postseq, 
#                                 freq_delay=0.25,
#                                 extra_info=[qubit_info,],
#                                 plot_seqs=False, subtraction = False,                                 
#                                 use_IQge=True)    
#    s.generate()
    s.measure()
    bla
    
 
if 1: #ef spec with swept ssb freq
    from scripts.single_qubit import ssbspec

    pi_pulse_ge = qubit_info.rotate(np.pi, 0)
    pre_seq = sequencer.Sequence([sequencer.Trigger(250), pi_pulse_ge])
    post_seq = pi_pulse_ge

#    delayseq = sequencer.Delay(3000)
    alz.set_naverages(1000)
#    while True:
                
    spec = ssbspec.SSBSpec(qubit_ef_info, np.linspace(-4e6, 4e6, 161), 
                           seq=pre_seq, 
                           postseq = post_seq,
                           plot_seqs=False,
                           selective = True,
                           extra_info = qubit_info,
                           )
    spec.measure()
    
    
    bla 

 

if 0:  #***COPY (to be deleted) of cavity displacement calibration
    alz.set_naverages(2000)
#    delayseq = sequencer.Delay(3000)
    from scripts.single_cavity import cavdisp
    dispcal = cavdisp.CavDisp(qubit_info, cavity_info, np.linspace(0.05, 2.55, 51), proj_num = 0, 
                     seq=None, delay=0, bgcor=True, update=True, fit_type='poisson')
#                     extra_info=None, postseq = None, plot_seqs=False)
    dispcal.measure() 
 
if 0: # Number splitting w/ SSBspec
    from scripts.single_qubit import ssbspec
#    seq = sequencer.Trigger(250)
    cav_drive = sequencer.Sequence([sequencer.Trigger(250), cavity_info.rotate(np.pi, 0), sequencer.Delay(10)])
#    cav_drive = sequencer.Sequence([sequencer.Trigger(250), cavity_info.rotate.displace(1.14, 0),qubit_info.rotate_selective(2*np.pi,0),cavity_info.rotate.displace(0.58, np.pi), sequencer.Delay(10)])
#    cav_drive = sequencer.Sequence([sequencer.Trigger(250), cavity_info.rotate.displace(0.56, 0),qubit_info.rotate_selective(2*np.pi,0),cavity_info.rotate.displace(0.24, np.pi), sequencer.Delay(10)])

#    delayseq = sequencer.Delay(3000)
    alz.set_naverages(2000)
                
    spec = ssbspec.SSBSpec(qubit_info, np.linspace(-0.25e6, 7.5e5, 51), 
                           seq=cav_drive, 
#                           postseq = delayseq,
                           plot_seqs=False,
                           selective = True,
                           extra_info = cavity_info,
                           )
    spec.measure()
    
    
#    bla 





#EF rabi with and without pi pulse to extract temperature. Alternatively, just run script in block after this    
if 1: #EF rabi
    for i in range(1):
        from scripts.single_qubit import efrabi

        alazar.set_naverages(2000)
        efr = efrabi.EFRabi(qubit_info, qubit_ef_info, np.linspace(-1, 1, 81)*0.05, plot_seqs=False)
        efr.measure()
        
#        period = efr.fit_params['period'].value
#        alazar.set_naverages(50000)
#        efr = efrabi.EFRabi(qubit_info, qubit_ef_info, np.linspace(-0.4, 0.4, 81), first_pi=False, force_period= period)
#        efr.measure()
    bla
 
if 0: #EF rabi with qp injection 
    from scripts.QPs.single_qubit import efrabi_QP
    inj_powers = [25]
    delays =  [0.3e6, 1.0e6, 1.5e6]
    for power in inj_powers: 
        ag2.set_power(power)
        for dt in delays:
            qp_delay = dt
            if dt == 1.5e6:
                funcgen.set_frequency(200)
            else:
                funcgen.set_frequency(400)
                
            alazar.set_naverages(4000)
            efr = efrabi_QP.EFRabi_QP(qubit_info, qubit_ef_info, np.linspace(-0.75, 0.75, 81), qp_delay, inj_len =360e3, plot_seqs=False)
            efr.measure()
            
            period = efr.fit_params['period'].value
            alazar.set_naverages(50000)
            efr = efrabi_QP.EFRabi_QP(qubit_info, qubit_ef_info, np.linspace(-0.75, 0.75, 81), qp_delay, inj_len =360e3, first_pi=False, force_period= period)
            efr.measure()
    bla
       
       
if 0: # Temperature measurement to determine excited state population
    # use tuned sidebands for both GE and EF
#    for EFfreq in [5865.9e6,]: # set to frequency for sideband modulation
#        from scripts.single_qubit import temperature
    amps = np.linspace(-1, 1, 51)*0.8
    alazar.set_naverages(5e3)
#        qubit_ef_brick.set_frequency(EFfreq) # consider -50 MHz sideband
        # qubitsrc.set_frequency(5606e6) # stable at -46.5 MHz
        # want to change deltaf for qubit_ef
    for i in range(3):
        tempext = temperature.Temperature(qubit_ef_info, qubit_info, amps)
    #    tempext = temperature.Temperature(cavity_info, qubit_info, amps)
        tempext.measure()
    bla
    
    
if 0: # T1
    from scripts.single_qubit import T1measurement
       
    alazar.set_naverages(1500)      #averages pulled from alazar settings - plot updates every 100
#    delays =np.concatenate((np.linspace(0,20e3, 41), np.linspace(20e3,150e3,21)))
    delays = np.logspace(np.log10(60), np.log10(600e3),40)
#    delays = np.array([0, 30e3, 400e3])
#    delays = np.linspace(0,300e3, 61)
    funcgen.set_frequency(1000)  #rep rate 
    for i in range(2):
        t1 = T1measurement.T1Measurement(qubit_info, delays, 
                                         double_exp=False)
        t1.measure()
#    bla
#    

    

if 0: # T2
    from scripts.single_qubit import T2measurement
    delays = np.linspace(0e3, 80e3, 81)
    delta = 100e3
    alazar.set_naverages(500)
    for j in range(2):
#    for j in [1000]:
#        fg.set_frequency(j)
        t2 = T2measurement.T2Measurement(qubit_info, delays,
                                         detune=delta)#, fit_type='exp_decay_double_sine')
        t2.measure()
    bla

if 0: # T2echo
    delays = np.linspace(0, 100e3, 81)
    delta = 100e3
    alazar.set_naverages(1500)
    from scripts.single_qubit import T2measurement
    for j in range(2):
        t2 = T2measurement.T2Measurement(qubit_info, delays, 
                                         detune= delta,# double_freq=False,
                                         echotype = T2measurement.ECHO_HAHN)
        t2.measure()
    bla
    
if 0: # T1 with CW laser
    from scripts.QPs.single_qubit import T1measurement_CW
    funcgen.set_frequency(5000)
    alazar.set_naverages(2500)
    laserfg.set_output_on(0)    
    laserfg.set_function('DC')
    laserfg.set_output_on(1)
    laserV=2.67
    '''record the attenuation!'''
    atten = 55
    laserfg.set_DCOffset(laserV)    
    laserfg.set_output_on(1)
    delays =np.concatenate((np.linspace(0,2e3, 61), np.linspace(2e3,20e3,41)))
    for i in range(4):
        t1 = T1measurement_CW.T1Measurement_CW(qubit_info, delays, laserV, atten=atten)
        t1.measure()
    laserfg.set_output_on(0)

    bla
    
if 0: # T2 with CW laser
    from scripts.QPs.single_qubit import T2measurement_CW
    funcgen.set_frequency(5000)
    alazar.set_naverages(3000)
    laserfg.set_output_on(0)    
    laserfg.set_function('DC')
    laserfg.set_output_on(1)
    laserV=2.67
    '''record the attenuation!'''
    atten = 50
    laserfg.set_DCOffset(laserV)    
    laserfg.set_output_on(1)
    t_end = 1.2e3
    delays =np.linspace(0,t_end, 101)
    delta = 6e9/t_end
    for i in range(3):
        t2 = T2measurement_CW.T2Measurement_CW(qubit_info, delays, laserV, atten=atten,
                                         detune= delta,# double_freq=False,
                                         )
        t2.measure()
    laserfg.set_output_on(0)

    bla



if 0:  # cav sat spectroscopy with no sidebands
    cav_freq = 5478.297e6#6572.12e6
    freq_range =0.25e6
    n_points = 101
    freqs = np.linspace(cav_freq-freq_range, cav_freq+freq_range, n_points)
 
    ro_powers=[-5]
    drive_powers=[-33]#-33]

    qubit_brick.set_rf_on(1)
    cavity_brick.set_rf_on(0)
    spec_brick.set_rf_on(1)
    spec_brick.set_pulse_on(1)
    
    selective_pi_pulse = qubit_info.rotate_selective(np.pi, 0)
    for ro_pow in ro_powers:
#        ag1.set_power(ro_pow)
        for drive_power in drive_powers:
            spec_params = (spec_brick, drive_power)
            s = spectroscopy.Spectroscopy(spec_info, freqs, spec_params,
                     plen=25e3, freq_delay=0.25,
                     seq=None, postseq=selective_pi_pulse,
                     use_weight=False, use_IQge=True,
                     use_marker = True,
                     subtraction = False,
                     extra_info = qubit_info)
            s.measure()
    spec_brick.set_rf_on(0)
    bla

if 0:#storage cavity spec
    from scripts.single_qubit import ssbspec
#   
    alz.set_naverages(4000)
    seq = sequencer.Sequence([sequencer.Trigger(250), # prepend pi pulse
                qubit_info.rotate_selective(np.pi,0),
                sequencer.Delay(10)])
    pi_pulse = qubit_info.rotate_selective(np.pi, 0)   
    spec = ssbspec.SSBSpec(cavity_info, np.linspace(-5e6, 5e6, 101), 
#                           seq=seq, # for EF SSB spec 
                           postseq = pi_pulse,
                           plot_seqs=False,
                           selective = True,
                           extra_info = qubit_info
                           )
    spec.measure()
    
    bla
    
    
    
if 0:  #cavity lifetime 
#           - To use SNAP to generate a single photon, set displacement to 0.
    alz.set_naverages(2000)
#    delayseq = sequencer.Delay(3000)
    from scripts.single_cavity import cavT1SNAP
    t1 = cavT1SNAP.CavT1SNAP(qubit_info, cavity_info, 0, np.linspace(1.0e3, 8001.0e3, 26), proj_num = 0, 
                     seq=None, extra_info=None, bgcor=True, postseq = None,
                     plot_seqs=False)
    t1.measure()
#    bla

if 0:  #cavity coherence (T2 with coherent state, attempt 1)
    alz.set_naverages(4000)
#    delayseq = sequencer.Delay(3000)
    from scripts.single_cavity import cavT2
    t2 = cavT2.CavT2(qubit_info, cavity_info, 0.2, np.linspace(1.0e3, 1501.0e3, 301), proj_num=0, 
                     detune =2e4,seq=None, bgcor=False,extra_info=None, postseq = None,
                     plot_seqs=False)
    t2.measure()
    bla

if 0:  #cavity displacement calibration
    alz.set_naverages(2000)
#    delayseq = sequencer.Delay(3000)
    from scripts.single_cavity import cavdisp
    dispcal = cavdisp.CavDisp(qubit_info, cavity_info, np.linspace(0.05, 2.55, 51), proj_num = 0, 
                     seq=None, delay=0, bgcor=True, update=True, fit_type='poisson')
#                     extra_info=None, postseq = None, plot_seqs=False)
    dispcal.measure()
    bla



if 0:  #cavity coherence (T2 with SNAP - assumes calibrated amplitude!)
    alz.set_naverages(4000)
#    delayseq = sequencer.Delay(3000)
    from scripts.single_cavity import cavT2SNAP2
    t2 = cavT2SNAP2.CavT2SNAP2(qubit_info, cavity_info, 0, np.linspace(1.0e3, 750.0e3, 151), proj_num=0, 
                     detune =2e4,seq=None, bgcor=False,extra_info=None, postseq = None,
                     plot_seqs=False)
    t2.measure()
    bla



if 0: # Detuning ramsey
    from scripts.single_qubit import T2measurement
    funcgen.set_frequency(2e3)
    delta = 1000e3
    
    alazar.set_naverages(3000)
    real_freq = 6008.125e6 + 50e6
    phys_detunings = np.linspace(-400e3, 400e3, 9)
    phases = {}
    for detuning in phys_detunings:
        qubit_brick.set_frequency(real_freq + detuning)
        
        delays = np.linspace(0e3, 2/(delta - detuning)*1e9, 31)
        t2 = T2measurement.T2Measurement(qubit_info, delays,
                                         detune=delta,)
#                                         fit_type='quadratic')# 
        t2.measure()
        
        phases[detuning] = t2.fit_params['dphi'].value/(2*np.pi) -0.25       
        
    qubit_brick.set_frequency(real_freq)    
    fig, ax = plt.subplots()
    ax.plot(phases.keys(), phases.values(),'o')
    bla

if 0: #Amp tuneup
    from scripts.jk.pulse_tuning import amplitude_tuneup
    
    amptune = amplitude_tuneup.Amplitude_Tuneup(qubit_info, update_ins=False,
                                                relative_range = np.linspace(0.7,1.3,81))
    
    amptune.measure()
    bla

# repeated measurements cycling through T1, T2 and T2 echo
#if 0:
#    from scripts.single_qubit.t1t2_plotting import do_T1_plot, do_T2_plot, do_T2echo_plot
#    C2Q2_t1s = {'t1s':[], 't1s_err':[], 'ofs':[], 'ofs_err':[], 'amps':[], 'amps_err':[], 'vars':[],}
#    C2Q2_t2s = {'t2s':[], 't2s_err':[], 't2freqs':[], 't2freqs_err':[], 'amps':[], 'amps_err':[], 't22s':[], 't22s_err':[], 't22freqs':[], 't22freqs_err':[], 'amp2s':[], 'amp2s_err':[],'vars':[],}
#    C2Q2_t2Es = {'t2es':[], 't2es_err':[]}
#    rep_rates = [500]
#    for i in range(1000): #set number of repetitions.
#        if 1:
#            for rep_rate in rep_rates:
#                funcgen.set_frequency(rep_rate)
#                do_T1_plot(qubit_info, 1000, np.concatenate((np.linspace(0, 20e3, 21), np.linspace(21e3, 100e3, 40))), C2Q2_t1s, 300)
#                do_T2_plot(qubit_info, 1000, np.linspace(0e3, 40e3, 201), 500e3, C2Q2_t2s, 301, double_freq=False)
#                do_T2echo_plot(qubit_info, 1000, np.linspace(0, 50e3, 101), 300e3, C2Q2_t2Es, 302)

# measurements of spec, T1, etc with ag3 power
if 0:
    from scripts.single_qubit.t1t2_plotting import do_spec_plot, do_ROspec_plot, do_T1_plot, do_T2_plot, do_T2echo_plot
#    C2Q2_freqs = {'x0s':[], 'x0s_err':[], 'ofs':[], 'ofs_err':[], 'ws':[], 'ws_err':[], 'vars':[],}
#    C2Q2_ros = {'x0s':[], 'x0s_err':[], 'As':[], 'As_err':[], 'ws':[], 'ws_err':[], 'vars':[],}
    C2Q2_t1s = {'t1s':[], 't1s_err':[], 'ofs':[], 'ofs_err':[], 'amps':[], 'amps_err':[], 'vars':[],}
    C2Q2_t2s = {'t2s':[], 't2s_err':[], 't2freqs':[], 't2freqs_err':[], 'amps':[], 'amps_err':[], 't22s':[], 't22s_err':[], 't22freqs':[], 't22freqs_err':[], 'amp2s':[], 'amp2s_err':[], 'vars':[],}
    C2Q2_t2Es = {'t2es':[], 't2es_err':[], 'vars':[],}
#    rofreq = 9136.6e6
#    ro_range = 6e6
#    ro_pwrs = [-32]
#    qubit_freq = 6769e6
#    freq_range = 15e6
    spec_params = (qubit_brick, 13)
    ag3_pwrs=[0.01]
#    ag3_pwrs=10*np.log10(ag3_pwrs)
    for i in range(1000):
        for pwr in ag3_pwrs:
#            ag3.set_power(10*np.log10(pwr))
            if 1:
#                funcgen.set_frequency(5000)    
# Take readout cavity spec. and reset frequency according to fit                                
#                ro=do_ROspec_plot(qubit_info, 2000, np.linspace(rofreq-ro_range, rofreq+ro_range, 25), 
#                             ro_pwrs, C2Q2_ros, 304,  var=pwr)
#                cur_ro=ro.fit_params[0][2]
#                ag1.set_frequency(cur_ro)
#                ag2.set_frequency(cur_ro+50e6)
# Do qubit spec and reset frequency according fit                
#                s=do_spec_plot(qubit_info, 1000, np.linspace(qubit_freq-freq_range, qubit_freq+5e6, 151), 
#                             spec_params, C2Q2_freqs, 303, plen=50000, amp=0.01, var=pwr)
#                cur_freq=s.fit_params['x0'].value*1e6
#                cur_freq = (-6.8851*pwr+6769.29)*1e6
#                qubit_brick.set_frequency(cur_freq)
                
                funcgen.set_frequency(500)
                do_T1_plot(qubit_info, 5000, np.concatenate((np.linspace(0, 50e3, 25), np.linspace(51e3, 200e3, 50))), C2Q2_t1s, 300)
                do_T2_plot(qubit_info, 5000, np.linspace(0e3, 50e3, 101), 300e3, C2Q2_t2s, 301, double_freq=False)
                do_T2echo_plot(qubit_info, 5000, np.linspace(0, 100e3, 101), 200e3, C2Q2_t2Es, 302)

#save data from repeated measurements. TODO: consolidate with repetition code
if 0:
    ts = time.localtime()
    tstr = time.strftime('%Y%m%d/%H%M%S', ts)
    groupname='%s_%s'  % (tstr, 'Repetition_T1')
    repData=mclient.datafile.create_group(groupname)
    for key, value in C2Q2_t1s.items():
        repData.create_dataset(key, data=value)
    groupname='%s_%s'  % (tstr, 'Repetition_T2')
    repData=mclient.datafile.create_group(groupname)
    for key, value in C2Q2_t2s.items():
        repData.create_dataset(key, data=value)
    groupname='%s_%s'  % (tstr, 'Repetition_T2E')
    repData=mclient.datafile.create_group(groupname)
    for key, value in C2Q2_t2Es.items():
        repData.create_dataset(key, data=value)
#    groupname='%s_%s'  % (tstr, 'Repetition_freq')
#    repData=mclient.datafile.create_group(groupname)
#    for key, value in C2Q2_freqs.items():
#        repData.create_dataset(key, data=value)
#    groupname='%s_%s'  % (tstr, 'Repetition_RO')
#    repData=mclient.datafile.create_group(groupname)
#    for key, value in C2Q2_ros.items():
#        repData.create_dataset(key, data=value)
    


if 0: ### calibration routines

#    if 0: # find vspec if freq
#        vspec = instruments['vspec']
#        
#        base_df0s = vspec.get_df0()
#        df0s = np.linspace(-1, 1, 101)*1e6
#        freq = 7050e6
##        qubit_brick.set_frequency(freq)
#        powers = []        
#        for df0 in df0s:
#            va_lo.set_frequency(freq+base_df0s+df0)
##            vspec.set_df0(base_df0s + df0)
##            vspec.set_frequency(freq)
#            time.sleep(0.1)
#            p = vspec.get_power()
#            powers.append(p)
#            print df0, p
#        plt.figure()
#        plt.plot(df0s, powers, 'rs-')

    if 1:   #IQ mixer leakage Calibration
        from scripts.calibration import mixer_calibration
        reload(mixer_calibration)
        from scripts.calibration.mixer_calibration \
                import Mixer_Calibration as mixer_cal
#        from scripts.calibration.mixer_calibration_fmin \
#                import Mixer_Calibration as mixer_cal


        #############################################################
#        vspec.set_rfsource('va_lo')
#        vspec.set_rfsource('va_lo_2')

        qubit_freq = 4985.7e6 #this is the actual frequency of the qubit
        qubit_brick.set_frequency(qubit_freq-qubit_info.deltaf)
        qubit_brick.set_rf_on(1)
        qubit_cal = mixer_cal('qubit_info_1', qubit_freq, 
#                              spec='vspec',
                              spec='spec',
                              verbose=True,
                              base_amplitude=2,)
                              
                              
#        qubit_freq = 5665.6e6 #this is the actual frequency of the qubit
#        qubit_brick.set_frequency(qubit_freq-qubit_info.deltaf)
#        qubit_brick.set_rf_on(1)
#        qubit_cal = mixer_cal('qubit_info_2', qubit_freq, 
##                              spec='vspec',
#                              spec='spec',
#                              verbose=True,
#                              base_amplitude=2,)

                              
#        qubit_ef_freq = 4758.51e6-50e6
#        qubit_ef_brick.set_frequency(qubit_ef_freq-qubit_ef_info.deltaf)
#        qubit_ef_brick.set_rf_on(1)
#        qubit_ef_brick.set_frequency(qubit_ef_freq-qubit_ef_info.deltaf)
#        qubit_cal = mixer_cal('qubit_ef_info', qubit_ef_freq, 
#                              spec='spec',
#                              verbose=True,
#                              base_amplitude=2)


####                              
#        cav_freq = 5528.297e6-50e6 #6622.1215e6 - 50e6
#        cavity_brick.set_frequency(cav_freq-cavity_info.deltaf)
#        cavity_brick.set_rf_on(1)
#        spec_brick.set_pulse_on(False)
#        qubit_cal = mixer_cal('cavity_info', cav_freq, 
#                              spec='spec',
#                              verbose=True,
#                              base_amplitude=2,)
#
###
        cal = qubit_cal
#        


        ###### using fmin routine  make sure using mixer_calibration_fmin
        if 0:
            cal.prep_instruments(reset_offsets=True, reset_ampskew=True)
            cal.tune_lo()
            cal.tune_osb()
        
        ###### BRUTE FORCE--- make sure using mixer_calibration
        if 1:
            if 0:
                cal.prep_instruments(reset_offsets=True, reset_ampskew=True)
                cal.tune_lo(mode=(0.6, 2, 2))
                cal.tune_osb(mode=(0.5, 1000, 2, 2))
                cal.tune_lo(mode='fine') # useful if using 10 dB attenuation;
                                    # LO leakage may creep up during osb tuning
            if 1:
                cal.prep_instruments(reset_offsets=False, reset_ampskew=False)
                cal.tune_osb(mode=(0.2, 400, 2, 2))#'fine')#(0.2, 2000, 1, 2))
                cal.tune_lo(mode='fine') # useful if using 10 dB attenuation;
                                    # LO leakage may creep up during osb tuning
            if 0:#very fine
                cal.prep_instruments(reset_offsets=False, reset_ampskew=False)
                cal.tune_osb(mode=(0.05, 100, 3, 3))#'fine')#(0.2, 2000, 1, 2))
                cal.tune_lo(mode=(0.05, 2, 3)) # useful if using 10 dB attenuation;
  
        # this function will set the correct qubit_info sideband phase for use in experiments
        #    i.e. combines the AWG skew with the current sideband phase offset
        cal.set_tuning_parameters(set_sideband_phase=True)
        cal.load_test_waveform()
        cal.print_tuning_parameters()




    
#sideband tuning only for ef transition 9when using same brick for eg and ef)
    
    if 0:   #IQ mixer leakage Calibration
        from scripts.calibration import mixer_calibration
        reload(mixer_calibration)
        from scripts.calibration.mixer_calibration \
                import Mixer_Calibration as mixer_cal


        #############################################################


        qubit_ef_freq = 4697.24e6 #this is the actual frequency of the ef transition
        qubit_brick.set_rf_on(1)
        qubit_ef_cal = mixer_cal('qubit_ef_info_1', qubit_ef_freq, 
                              spec='spec',
                              verbose=True,
                              base_amplitude=2,)
                              
                              

        cal = qubit_ef_cal

        
        if 1:
            if 1:
                cal.prep_instruments(reset_offsets=False, reset_ampskew=True)
                cal.tune_osb(mode=(0.5, 1000, 2, 2))
                
            if 1:
                cal.prep_instruments(reset_offsets=False, reset_ampskew=False)
                cal.tune_osb(mode=(0.2, 400, 2, 2))#'fine')#(0.2, 2000, 1, 2))
                
            if 0:#very fine
                cal.prep_instruments(reset_offsets=False, reset_ampskew=False)
                cal.tune_osb(mode=(0.05, 100, 3, 3))#'fine')#(0.2, 2000, 1, 2))
  
        # this function will set the correct qubit_info sideband phase for use in experiments
        #    i.e. combines the AWG skew with the current sideband phase offset
        cal.set_tuning_parameters(set_sideband_phase=True)
        cal.load_test_waveform()
        cal.print_tuning_parameters()

    
    
    