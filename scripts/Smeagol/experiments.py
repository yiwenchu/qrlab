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
from scripts.single_qubit import ssbspec, ssbspec_fit


ag1 = instruments['ag1']
ag2 = instruments['ag2']
#ag3 = instruments['ag3']
qubit_info = mclient.get_qubit_info('qubit_info')
qi = mclient.instruments['qubit_info']
phonon1_info = mclient.get_qubit_info('phonon1_info')
qubit_ef_info = mclient.get_qubit_info('qubit_ef_info')
#vspec = instruments['vspec']
#spec = instruments['spec']
awg1 = instruments['AWG1']
#va_lo_5_10 = instruments['va_lo_5_10'] 
qubit_brick = instruments['qubit_brick'] #4-8 GHz brick
#qubit_brick = instruments['ag2'] #vector generator
qubit_ef_brick = instruments['qubit_ef_brick']
#va_lo = instruments['va_lo']
#va_lo_4_8 = instruments['va_lo_2']
funcgen = instruments['funcgen']
alazar = instruments['alazar']
spec_brick = instruments['spec_brick']
spec_info = mclient.get_qubit_info('spec_info')
LO_brick = instruments['LO_brick']
LO_info = mclient.get_qubit_info('LO_info')
cavity_info = mclient.get_qubit_info('cavity_info')
cavity_brick = instruments['cavity_brick']
yoko1=instruments['yoko1']
yoko2=instruments['yoko2']
PS1 = instruments['AgPS1']


#trigger for experiments
#funcgen.sync_on(1);



if 0: # Cavity spectroscopy 
    funcgen.set_frequency(10000)
    from scripts.single_cavity import rocavspectroscopy
    if 1: # sweep frequency, want to use a long readout tone
#        f0 = ag1.get_frequency()
#        p0 = ag1.get_power()
#        ro_powers = [-26, -23,]
#        ro_powers=[p0]
        ro_powers=[-35]
#        ro_powers = [-38]
#        rofreq = 9.1118e9
#        rofreq=9.26355e9
#        rofreq=  6955.89e6
        rofreq=9.135e9
#        rofreq=6.8e9
        freq_range=5e6
        freqs =  np.linspace(rofreq-freq_range, rofreq+freq_range, 100)
        
#        p = phonon1_info.rotate_selective
#        seq = sequencer.Sequence([sequencer.Trigger(250), p(0, 0, amp = 0.01)])         
        
        rospec = rocavspectroscopy.ROCavSpectroscopy(qubit_info, ro_powers, freqs, qubit_pulse = None, 
#                                                     seq = seq, extra_info=phonon1_info
                                                     )#np.pi
        rospec.measure()
        
#        ag1.set_frequency(f0)
#        LO_brick.set_frequency(f0+50e6)
#        ag1.set_power(p0)
#        
        
    
    if 0: # sweep power 
        ro_powers = np.linspace(-40, -25, 20)
#        rofreq=9.2637e9
        rofreq=6955.89e6
        freqs = np.array([rofreq,])
        rospec = rocavspectroscopy.ROCavSpectroscopy(qubit_info, ro_powers, freqs, qubit_pulse=0)
        rospec.measure()
        
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
#    specfreq=8088.82e6
    specfreq=9170e6
    specfreq=8086.6e6  
    specfreq=8130e6
    specfreq=7925e6
    specfreq=9300e6
    specpower=0
    
    rofreq=6.9608e9
    rofreq=6.9600e9
    rofreq=6.9595e9
    rofreq=6.9595e9
    rofreq=6.9601e9
    rofreq=6.9610e9
    rofreq=6.9600e9

    ag1.set_frequency(rofreq)
    LO_brick.set_frequency(rofreq+50e6)
        
    spec_brick.set_rf_on(1)
    spec_brick.set_pulse_on(1) 
    spec_brick.set_frequency(specfreq)
    spec_brick.set_power(specpower)
    
    powers = np.linspace(-22, -27, 12)
    # powers = np.linspace(-30, -10, 11)
    # ag_vector.set_frequency(9125.49e6)
    # ag_vector.set_power(30)`
    
    # ropcal.SAT_PULSE ropcal.PI_PULSE
    r = ropcal.ROPower_Calibration(
        spec_info, powers, qubit_pulse=ropcal.SAT_PULSE, 
        pulse_len=100e3, simuldrive=True
        )
    outs = r.measure()
    
    bla

# qubit spectroscopy with no sidebands

if 0: #Saturation spec (no sidebands). MAKE SIGNIFICANT MODS IN NEXT BLOCK
#                   
    ############################

#    qubit_freq=6332.492620e6-50e6
#    qubit_freq=6180e6
    qubit_freq = 7343e6
#    freq_range = 4000e6
#    freq_range = 500e6
    
#    qubit_freq=5000e6 
    freq_range= 10e6      
#    freq_range = 300e6 #151 pts
#    freqs = np.linspace(qubit_freq, qubit_freq+freq_range, 4001)
    freqs = np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 101)
#    freqs = np.linspace(qubit_freq, qubit_freq+freq_range, 501)
#    freqs = np.linspace(5900e6,6320e6,421)
#    freqs = np.linspace(8500e6,8800e6,61)
#    
#    freqs = np.linspace(7e9,9.5e9,501)
#    freqs = np.linspace(7840e6,7980e6,700)
#    drive_powers=[-10,-7]`
    drive_powers=[-35]
     
##    rofreq=9.2637e9
##    ro_pow=-22

#    rofreq=9.146900e9
    ro_powers=[-35]
##
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
##    ag2.set_frequency(rofreq+50e6)
    

    qubit_brick.set_rf_on(0)
#    qubit_ef_brick.set_rf_on(0)
    spec_brick.set_rf_on(1)
    spec_brick.set_pulse_on(1)   #Spec CW or pulsed
        
    
    for ro_pow in ro_powers:
#        ag1.set_power(ro_pow)
        for drive_power in drive_powers:
            spec_params = (spec_brick, drive_power)
            s = spectroscopy.Spectroscopy(spec_info, freqs, spec_params,
                     plen=50e3, freq_delay=0.25,
                     seq=None, postseq=None,
                     use_weight=False, use_IQge=False,
                     subtraction = False,)
            s.measure()
    spec_brick.set_rf_on(0)

    bla

if 0: #Saturation spec (no sidebands)
#                   
    ############################

    qubit_freq=6748.3822e6
#    qubit_freq = 9300e6
    freq_range = 5e6
    
#    qubit_freq=9170e6       
#    freq_range = 300e6 #151 pts
    freqs = np.linspace(qubit_freq, qubit_freq+freq_range, 101)
#    freqs = np.linspace(8700e6,9700e6,201)
#    freqs = np.linspace(8500e6,8800e6,61)
#    
#    freqs = np.linspace(7e9,9.5e9,501)
#    freqs = np.linspace(7840e6,7980e6,700)
#    drive_powers=[-10,-7]
    drive_powers=[-15]
     
##    rofreq=9.2637e9
##    ro_pow=-22

    rofreq=9120.6e6
    ro_powers=[-30,]
    
    u_powers=[-2,2,4,6,8]  ######################

##
    ag1.set_frequency(rofreq)
    LO_brick.set_frequency(rofreq+50e6)
##    ag2.set_frequency(rofreq+50e6)
    

    qubit_brick.set_rf_on(0)
    qubit_ef_brick.set_rf_on(0)
    spec_brick.set_rf_on(1)
    spec_brick.set_pulse_on(0)   #Spec CW or pulsed
        
    
    for ro_pow in ro_powers:
        ag1.set_power(ro_pow)
        for drive_power in drive_powers:
            for u_pow in u_powers: ################
                ag2.set_power(u_pow) #################
                spec_params = (spec_brick, drive_power)
                s = spectroscopy.Spectroscopy(spec_info, freqs, spec_params,
                         plen=100e3, freq_delay=0.25,
                         seq=None, postseq=None,
                         use_weight=False, use_IQge=False,
                         subtraction = False,)
                s.measure()
    spec_brick.set_rf_on(0)
    
    ag2.set_power(-40)  ##############
    
    bla
    




if 0: # sideband modulated spectroscopy
#    rofreq=9.0241e9
#    rofreq=f2(8e-3)
##    ag3.set_frequency(9197e6)
#
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ag2.set_frequency(rofreq+50e6)
    
#    qubit_brick.set_rf_on(1)
#    qubit_ef_brick.set_rf_on(0)
#    spec_brick.set_rf_on(0)
#    qubit_freq = 5.185e9
#    qubit_freq = f1(0)+50e6
#    qubit_freq = f1(8e-3)+50e6
#    qubit_freq = 6000e6
#    freq_range = 2000e6
    qubit_freq = 6027e6+50e6
    freq_range = 100e6
#    freq_range = 10e6
#   freqs = np.linspace(5960, 6000, 101) * 1e6
    freqs = np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 201)
    spec_params = (qubit_brick, 13) # brick name, power

    
    s = spectroscopy.Spectroscopy(qubit_info,
                                      freqs, spec_params, 
                                      plen=50000, amp=0.001, plot_seqs=False,subtraction = False,
                                      use_weight=False, use_IQge=False,                                      
                                      ) #1=1ns5
    s.measure()
    
#    ag3_pwrs=np.linspace(-10, 12, 12)
#    ag3_pwrs=10*np.log10([2])   
#    for pwr in ag3_pwrs:
#        ag3.set_power(pwr)
#        s = spectroscopy.Spectroscopy(qubit_info,
#                                         np.linspace(qubit_freq-freq_range, qubit_freq+freq_range, 71), spec_params,
#                                         plen=50000, amp=0.01, plot_seqs=False,subtraction = False) #1=1ns5
#        s.measure()
#    qubit_brick.set_rf_on(0)
    bla
                                     
if 0:
#    rofreq=9.0241e9
#    rofreq=f2(8e-3)
##    ag3.set_frequency(9197e6)
#
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ag2.set_frequency(rofreq+50e6)
    
    qubit_brick.set_rf_on(1)
    spec_brick.set_rf_on(0)
    for current in np.linspace(0, 2e-3, 11):
        yoko1.do_set_current(current)
        qubit_freq = 6320e6
        freq_range = 100e6
        spec_params = (qubit_brick, 13) # brick name, power
        
        s = spectroscopy.Spectroscopy(qubit_info,
                                          np.linspace(qubit_freq-freq_range, qubit_freq, 101), spec_params, 
                                          plen=50000, amp=0.01, plot_seqs=False,subtraction = False) #1=1ns5
        s.measure()
    
    bla



#"""Power Rabi -- Pi pulse calibration"""
if 1: # Calibrate pi pulse
#    rofreq=6.955890e9
#    ro_pow=-27  
    
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)

#    qubit_brick_2.set_power(13)
#    qubit_brick_2.set_rf_on(1)
    spec_brick.set_rf_on(0)
    
    amps = np.linspace(-1, 1, 101)*0.3
#    for axis in np.linspace(0, np.pi*2, 9):
#    seq = sequencer.Sequence([sequencer.Trigger(250), pulselib.GaussSquare(2.9e3, 0.0155, 10, chan = 3)])
#    p = phonon1_info.rotate_selective
#    seq = sequencer.Sequence([sequencer.Trigger(250), p(0, 0, amp = 0.01)]) 
#    seq2 = sequencer.Sequence([p(0, 0, amp = 0.01)]) #, sequencer.Delay(50e3)
#    postseq = sequencer.Delay(-160)
#    seq=None
#    postseq=None
#    axes = np.linspace(0, 1, 100)
#    axes = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
#    for axis in axes:
    from scripts.single_qubit import rabi
    tr = rabi.Rabi(qubit_info, amps, #r_axis = axis,
#                   seq2 = seq2,
                   #postseq = postseq,
                   plot_seqs=False,
#                   extra_info = phonon1_info,
#                   take_shots = True,
#                   bgcor = True,
                   )

#        from scripts.single_qubit import rabi_IQ
#        tr = rabi_IQ.Rabi(qubit_info, np.linspace(0, 0.5, 101), plot_seqs=False, real_signals=False)
    tr.measure()
    bla
    

    
if 0: # Calibrate pi pulse (repeated Rabi)
#    rofreq=6.955890e9
#    ro_pow=-27  
    
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)

#    qubit_brick_2.set_power(13)
#    qubit_brick_2.set_rf_on(1)
    spec_brick.set_rf_on(0)
    
    amps = np.linspace(-1, 1, 101)*0.25
#    for axis in np.linspace(0, np.pi*2, 9):
#    seq = sequencer.Sequence([sequencer.Trigger(250), sequencer.Delay(200)])
#    postseq = sequencer.Delay(-160)
#    seq=None
#    postseq=None
    axes = np.arange(1000)
#    axes = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    rabi_amps = []
    rabi_ampserr = []
    rabi_ofs = []
    rabi_ofserr = []
    for axis in axes:
        from scripts.single_qubit import rabi
        tr = rabi.Rabi(qubit_info, amps, #r_axis = axis,
#                       seq = seq, postseq = postseq,
                       plot_seqs=False,
                       update=False,
                       )

#        from scripts.single_qubit import rabi_IQ
#        tr = rabi_IQ.Rabi(qubit_info, np.linspace(0, 0.5, 101), plot_seqs=False, real_signals=False)
        tr.measure()
        plt.close()
        rabi_amps.append(tr.fit_params['A'].value)
        rabi_ampserr.append(tr.fit_params['A'].stderr)
        rabi_ofs.append(tr.fit_params['ofs'].value)
        rabi_ofserr.append(tr.fit_params['ofs'].stderr)
        plt.figure(400)
        plt.clf()
        plt.subplot(211).errorbar(range(len(rabi_amps)), rabi_amps, rabi_ampserr)
        plt.subplot(212).errorbar(range(len(rabi_ofs)), rabi_ofs, rabi_ofserr)
    bla
    
if 0: # Check histogramming on GE

#    avg = alazar.get_naverages()
    alazar.set_naverages(50000)
    from scripts.single_qubit import rabi

    p = phonon1_info.rotate_selective
    seq = sequencer.Sequence([sequencer.Trigger(250), p(0, 0, amp = 0.01)]) 
    
    tr = rabi.Rabi(qubit_info, [qubit_info.pi_amp,], histogram=True, title='|e>',
                   seq = seq, 
                   extra_info = phonon1_info,)
    tr.measure()

#    seq = sequencer.Join([sequencer.Trigger(250),sequencer.Combined([cavity_info1A.rotate(1.5, 0), cavity_info1B.rotate(1.5, 0)])])
    tr = rabi.Rabi(qubit_info, [0.00,], histogram=True, title='|g>',
                   seq = seq, 
                   extra_info = phonon1_info,)
    tr.measure()

#    alz.set_naverages(avg)
    bla

    
#Readout calibration (takes histograms and calculates weighting function)
if 0: # calibrate readout
    from scripts.calibration import ro_power_cal

#    rofreq=9.025e9 
#    
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ag2.set_frequency(rofreq+50e6)
#    ro_pow=np.linspace(-40, -30, 6)
    ro_pow =[-35]
    cal = ro_power_cal.Optimal_Readout_Power(qubit_info, ro_pow, 
#                                             plen=50e3, amp=0.005, #comment out if using pi pulse
                                             shots=1e4, 
                                             update_readout=True, 
#                                             hist_steps=25,
                                             )    
    cal.measure()
    bla    
    


# spectroscopy for EF transition with spec brick
if 0: # EF spec
    qubit_freq = 6332.492620e6-50e6
    freq_range = 300e6
#    freq_range = 10e6
#   freqs = np.linspace(5960, 6000, 101) * 1e6
    freqs = np.linspace(qubit_freq-400e6, qubit_freq+10e6, 411)
#    qubit_freq = 6775.567e6-50e6

    drive_powers = [-35]
    
    #LP readout
#    rofreq = 9.006e9#8.769640e9
#    ro_pow = 5#2

#    #HP readout
#    rofreq=9129.58e6
#    ro_pow=-8  
    
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ag2.set_frequency(rofreq+50e6)
#    ag1.set_power(ro_pow)
    qubit_brick.set_rf_on(1)
    spec_brick.set_rf_on(1)
    spec_brick.set_pulse_on(1)
    seq = sequencer.Sequence([sequencer.Trigger(250), # prepend pi pulse
                qubit_info.rotate(np.pi,0),
                sequencer.Delay(10)])
#    seq = sequencer.Sequence([sequencer.Trigger(250), qubit_info.rotate(np.pi,0)])
                
    postseq = sequencer.Sequence(qubit_info.rotate(np.pi, 0)) # postpend pi pulse
#    postseq = [qubit_info.rotate(np.pi, 0), sequencer.Delay(-320)]
    for drive_power in drive_powers:
        spec_params = (spec_brick, drive_power)
        s = spectroscopy.Spectroscopy(spec_info, freqs, spec_params,
                 plen=1e3, #freq_delay=0.25,
                 seq=seq, # this and the next line adds pi pulses before and after spec pulse, for e-f spec
                 postseq=postseq,
                 use_weight=True, use_IQge=True,
                 subtraction = False, 
                 extra_info = qubit_info)
        s.measure()
#    spec_brick.set_rf_on(0)
    bla

if 0: # EF spec with ef brick
    
#    qubit_freq =  6699.394e6-50e6-300e6
    freq_range = 400e6
#    freq_range = 10e6
#   freqs = np.linspace(5960, 6000, 101) * 1e6
    freqs = np.linspace(qubit_freq, qubit_freq+freq_range, 401)
    
    #LP readout
#    rofreq = 9.006e9#8.769640e9
#    ro_pow = 5#2

#    #HP readout
#    rofreq=9129.58e6
#    ro_pow=-8  
    
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ag2.set_frequency(rofreq+50e6)
#    ag1.set_power(ro_pow)
    qubit_brick.set_rf_on(1)
    qubit_ef_brick.set_rf_on(1)
    
    
    seq = sequencer.Sequence([sequencer.Trigger(250), # prepend pi pulse
                qubit_info.rotate(np.pi,0),
                sequencer.Delay(10)])
                
    postseq = sequencer.Sequence(qubit_info.rotate(np.pi, 0)) # postpend pi pulse
    
    spec_params = (qubit_ef_brick, 13)
    s = spectroscopy.Spectroscopy(qubit_ef_info, freqs, spec_params,
             plen=1e3, amp=0.005, #freq_delay=0.25,
             seq=seq, # this and the next line adds pi pulses before and after spec pulse, for e-f spec
             postseq=postseq,
             use_weight=False, use_IQge=True,
             subtraction = False, 
             extra_info = qubit_info)
    s.measure()
#    spec_brick.set_rf_on(0)
    bla



    
if 0:# Number splitting w/ SSBspec
    
#    seq = sequencer.Trigger(250)
#    rofreq=9046.3e6
#    ro_pow=-30
#    rofreq = 9.02297e9
#    ro_pow = 7
#    ag1.set_frequency(rofreq)
#    LO_brick.set_frequency(rofreq+50e6)
#    ag1.set_power(ro_pow)
#    cav_drive = sequencer.Sequence([sequencer.Trigger(250), cavity_info.rotate(np.pi, 0), sequencer.Delay(10)])
    alazar.set_naverages(10000)
    funcgen.set_frequency(10000)
       
#    p = phonon1_info.rotate_selective
#    seq = sequencer.Sequence([sequencer.Trigger(250), p(0, 0, amp = 0.01)]) 
#    delayseq = sequencer.Delay(1000)

    #patchmon:
#    seq = sequencer.Sequence([sequencer.Trigger(250), sequencer.Delay(200)])
#    postseq = sequencer.Delay(-160)
    
#    qubit_ef_brick.set_rf_on(True)
#    qubit_brick.set_rf_on(False)

    for detuning in [0]:#np.linspace(-0.1e6, 0.1e6, 11):
#        amp = 0.0025
#        instruments['phonon1_info'].do_set_sideband_period(1e9/(-44.944020e6+detuning))
#        phonon1_info = mclient.get_qubit_info('phonon1_info')
#        p = phonon1_info.rotate_selective
#        seq2 = sequencer.Sequence([sequencer.Trigger(250), p(0, 0, amp = amp)]) 
        
        spec = ssbspec_fit.SSBSpec_fit(qubit_info, np.linspace(-2e6, 2e6, 201), 
#                               seq2=seq2, 
    #                           postseq = postseq,
                               plot_seqs=False,
#                               bgcor = True,
#                               keep_data = False,
    #                           generate=False
#                               extra_info = phonon1_info,
#                               txt = 'phonon detuning = %0.5f MHz\n phonon amp = %0.5f\n' % (detuning/1e6, amp)
                                )
        spec.measure()
    
    
#    qubit_ef_brick.set_rf_on(False)
    bla 
    
if 0:# Repeated number splitting w/ SSBspec
    alazar.set_naverages(5000)
    funcgen.set_frequency(5000)
    fs = []
    fserr = []
    for n in np.arange(1000):#np.linspace(-0.1e6, 0.1e6, 11):     
        spec = ssbspec_fit.SSBSpec_fit(qubit_info, np.linspace(-2e6, 2e6, 101), 
#                               seq2=seq2, 
    #                           postseq = postseq,
                               plot_seqs=False,
                               bgcor = True,
    #                           generate=False
#                               extra_info = phonon1_info,
#                               txt = 'phonon detuning = %0.5f MHz\n phonon amp = %0.5f\n' % (detuning/1e6, amp)
                                )
        spec.measure()
        plt.close()
        fs.append(spec.fit_params['x0'].value)
        fserr.append(spec.fit_params['x0'].stderr)
        plt.figure(400)
        plt.clf()
        plt.subplot(111).errorbar(range(len(fs)), fs, fserr)
    
#    qubit_ef_brick.set_rf_on(False)
    bla 
    
if 0:# ef spec with SSB
    
#    seq = sequencer.Trigger(250)
#    rofreq=9046.3e6
#    ro_pow=-30
#    rofreq = 9.02297e9
#    ro_pow = 7
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
#    alazar.set_naverages(5000)
    
    seq = sequencer.Sequence([sequencer.Trigger(250), # prepend pi pulse
                qubit_info.rotate(np.pi,0),
                sequencer.Delay(10)])
                
    postseq = sequencer.Sequence(qubit_info.rotate(np.pi, 0)) # postpend pi pulse

    spec = ssbspec_fit.SSBSpec_fit(qubit_ef_info, np.linspace(-10e6, 10e6, 101), 
                           seq=seq, 
                           postseq = postseq,
                           plot_seqs=False,
#                           generate=False
                           extra_info = qubit_info
                            )
    spec.measure()
    
    
#    qubit_ef_brick.set_rf_on(False)
    bla 
    
#EF rabi with and without pi pulse to extract temperature. Alternatively, just run script in block after this    
if 0: #EF rabi
    from scripts.single_qubit import efrabi
#    blah = mclient.instruments['eFC14#2']
    alazar.set_naverages(2000)
    efr = efrabi.EFRabi(qubit_info, qubit_ef_info, np.linspace(-0.3, 0.3, 81), plot_seqs=False)
    efr.measure()
    period = efr.fit_params['period'].value
    alazar.set_naverages(50000)
    efr = efrabi.EFRabi(qubit_info, qubit_ef_info, np.linspace(-0.3, 0.3, 81), first_pi=False, force_period= period)
    efr.measure()
    bla
    
    
if 0: # Temperature measurement to determine excited state population
    # use tuned sidebands for both GE and EF

    
    for EFfreq in [6456e6]: # set to frequency for sideband modulation
#        from scripts.single_qubit import temperature
        amps = np.linspace(-1, 1, 101)*0.1
#        alazar.set_naverages(30000)
#        qubit_ef_brick.set_frequency(EFfreq) # consider -50 MHz sideband
        
#        seq = sequencer.Sequence([sequencer.Trigger(250), sequencer.Delay(200)])
#        postseq = sequencer.Delay(-160)
        tempext = temperature.Temperature(qubit_ef_info, qubit_info, amps)#, seq=seq, postseq=postseq
        tempext.measure()
    bla
    
    
    
if 0: # T1
    from scripts.single_qubit import T1measurement
    alazar.set_naverages(10000)      #averages pulled from alazar settings - plot updates every 100
    for j in [1000,]:
        funcgen.set_frequency(j)  #rep rate    
#        delays = np.linspace(0,300,101)
#        delays = np.concatenate((np.linspace(-100, 100,50), np.linspace(100,300,100),np.linspace(300,500,50)))
#        delays = np.concatenate((np.linspace(-160, 200,100), np.linspace(200,800,50)))
#        delays = np.concatenate((np.linspace(-200,200,101), np.linspace(201,1000,100)))
#        delays =np.concatenate((np.linspace(0, 30e3, 100), np.linspace(31e3, 50e3, 51)))
#        delays =np.concatenate((np.linspace(0, 80e3, 51), np.linspace(81e3, 260e3, 51)))
        delays = np.concatenate((np.linspace(0, 70e3, 100), np.linspace(71e3, 150e3, 50)))
#        delays =np.linspace(1, 81e3, 101)
        
        
#        seq = sequencer.Sequence([sequencer.Trigger(250), sequencer.Delay(200)])
#        postseq = sequencer.Delay(-160)
        
#        seq = sequencer.Sequence([sequencer.Trigger(250), sequencer.Delay(1000)])
#        postseq = sequencer.Delay(-960)

        
#        seq = sequencer.Sequence([sequencer.Trigger(250), sequencer.Delay(200)])  
#        postseq = sequencer.Sequence([sequencer.Delay(-300), # prepend pi pulse
#                qubit_info.rotate(np.pi,0),
#                sequencer.Delay(-160)])
#
#        postseq =sequencer.Sequence()
#        postseq.append(sequencer.Constant(10e3, 1, chan="3m1"))

        t1 = T1measurement.T1Measurement(qubit_info, delays, 
#                                         seq = seq, 
#                                         postseq=postseq,
#                                         bgcor = True,
                                         double_exp=False, 
#                                         extra_info = qubit_info
                                         )
        t1.measure()
    bla

if 0: # T2
    from scripts.single_qubit import T2measurement
    delays = np.linspace(0e3, 20e3, 101)
#    delays = 1e3*np.ones(101)
#    delays = np.array([9.1919e3, 10.026e3, 10.860e3]) #three points on Ramsey curve ~T2
#    delays = np.linspace(2.5e3, 3.5e3, 5) #points near max of Ramsey curve
    delta = 500e3
    
    #for doing physical detuning.
#    qubit_freq = 5.591289e9
#    drive_freq = qubit_freq-224.2647e3
#    qubit_brick.set_frequency(drive_freq)
#    
    alazar.set_naverages(10000)
#    alazar.set_naverages(1000)
#    for j in range(1):
#    for j in [5000]:
#    seq = sequencer.Sequence([sequencer.Trigger(250), sequencer.Delay(200)])
#    postseq = sequencer.Delay(-160)
    for j in [5000]:
        funcgen.set_frequency(j)
        t2 = T2measurement.T2Measurement(qubit_info, delays,
                                         detune=delta, 
#                                         seq=seq,
#                                         postseq=postseq,
#                                         double_freq=False,
#                                         take_shots = True,
#                                         shots_avg = 1000,
                                         )
        t2.measure()
    
#    qubit_brick.set_frequency(qubit_freq)
    bla

if 0: # T2echo HAHN

#    qubit_freq = 5.591289e9
#    drive_freq = qubit_freq-0
##    drive_freq = 5.591314e9 #set to exact qubit frequency using T2 measurement
#    qubit_brick.set_frequency(drive_freq)
    
    delays = np.linspace(0e3, 30e3, 101)
#    delays = np.linspace(1, 801, 100)
    delta =400e3
    alazar.set_naverages(5000)
    from scripts.single_qubit import T2measurement
    
#    seq = sequencer.Sequence([sequencer.Trigger(250), sequencer.Delay(200)])
#    postseq = sequencer.Delay(-160)
    
    for j in range(1):
        t2 = T2measurement.T2Measurement(qubit_info, delays, 
                                         detune= delta, 
#                                         seq=seq,
#                                         postseq=postseq,
#                                         double_freq=False,
                                         echotype = T2measurement.ECHO_HAHN)
        t2.measure()
    bla

if 0: # T2echo CPMG

#    qubit_freq = 5.591289e9
#    drive_freq = qubit_freq-0
##    drive_freq = 5.591314e9 #set to exact qubit frequency using T2 measurement
#    qubit_brick.set_frequency(drive_freq)
    
#    delays = np.linspace(0.5e3, 50e3, 151)
    delta = 0e3
    alazar.set_naverages(5000)
    from scripts.single_qubit import T2measurement
#    Ns=[1]
    Ns=[1, 2, 4, 8, 14, 20, 30, 60, 80, 80, 100, 100]
    delayEnd = [80, 90, 100, 110, 110, 110, 110, 110, 50, 110, 50, 110]
    delayStart = [0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 50]
    delayN= [51, 51, 51, 51, 51, 51, 51, 31, 31, 15, 31, 15]
#    Ns=[80, 80, 100, 100]
#    delayEnd = [50, 110, 50, 110]
#    delayStart = [0, 50, 0, 50]
#    delayN= [31, 15, 31, 15]
#    Ns = [80, 100]
#    delayEnd = [120, 120]
    for ds, de, dN, j in zip(delayStart, delayEnd, delayN, Ns):
#    for de, j in zip(delayEnd, Ns):
#        delays = np.concatenate((np.linspace(0, 20e3, 21), np.linspace(21e3, 80e3, 30)))
#        delays = np.array([1e3])
        delays = np.linspace(ds*1e3, de*1e3, dN)
#        delays = np.linspace(0e3, de*1e3, 51)   
        t2 = T2measurement.T2Measurement(qubit_info, delays, 
                                         detune= delta, 
#                                         double_freq=False,
                                         echotype = T2measurement.ECHO_CPMG, 
#                                         fix_freq = True, 
                                         fit_type = 'exp_decay',
#                                         fit_type = 'gaussian_decay',
                                         necho = j, title = 'CPMG N=%d' % (j))
        t2.measure()

    bla
    

    
if 0:  #cavity lifetime
    from scripts.single_cavity import cavT1
    t1 = cavT1.CavT1(qubit_info, cavity_info, 2, np.linspace(0.01e3, 20e3, 81), proj_num =0, 
                     seq=None, extra_info=None, bgcor=False,
                     plot_seqs=False)
    t1.measure()
    blah

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


if 1: # measurements of spec, T1, etc with ag3 power
    from scripts.single_qubit.t1t2_plotting import do_spec_plot, do_ROspec_plot, do_T1_plot, do_T2_plot, do_T2echo_plot
#    C2Q2_freqs = {'x0s':[], 'x0s_err':[], 'ofs':[], 'ofs_err':[], 'ws':[], 'ws_err':[], 'vars':[],}
#    C2Q2_ros = {'x0s':[], 'x0s_err':[], 'As':[], 'As_err':[], 'ws':[], 'ws_err':[], 'vars':[],}
    if 1: #don't run this if appending to previous data
        C2Q1_t1s = {'t1s':[], 't1s_err':[], 'ofs':[], 'ofs_err':[], 'amps':[], 'amps_err':[], 'vars':[],}
        C2Q1_t2s = {'t2s':[], 't2s_err':[], 't2freqs':[], 't2freqs_err':[], 'amps':[], 'amps_err':[], 't22s':[], 't22s_err':[], 't22freqs':[], 't22freqs_err':[], 'amp2s':[], 'amp2s_err':[], 'vars':[],}
        C2Q1_t2Es = {'t2es':[], 't2es_err':[], 'vars':[],}
    
        C2Q2_t1s = {'t1s':[], 't1s_err':[], 'ofs':[], 'ofs_err':[], 'amps':[], 'amps_err':[], 'vars':[],}
        C2Q2_t2s = {'t2s':[], 't2s_err':[], 't2freqs':[], 't2freqs_err':[], 'amps':[], 'amps_err':[], 't22s':[], 't22s_err':[], 't22freqs':[], 't22freqs_err':[], 'amp2s':[], 'amp2s_err':[], 'vars':[],}
        C2Q2_t2Es = {'t2es':[], 't2es_err':[], 'vars':[],}
#    
#    C2Q3_t1s = {'t1s':[], 't1s_err':[], 'ofs':[], 'ofs_err':[], 'amps':[], 'amps_err':[], 'vars':[],}
#    C2Q3_t2s = {'t2s':[], 't2s_err':[], 't2freqs':[], 't2freqs_err':[], 'amps':[], 'amps_err':[], 't22s':[], 't22s_err':[], 't22freqs':[], 't22freqs_err':[], 'amp2s':[], 'amp2s_err':[], 'vars':[],}
#    C2Q3_t2Es = {'t2es':[], 't2es_err':[], 'vars':[],}
#    rofreq = 9136.6e6
#    ro_range = 6e6
#    ro_pwrs = [-32]
#    qubit_freq = 6769e6
#    freq_range = 15e6
#    spec_params = (qubit_brick, 13)
    ag3_pwrs=[0]
#    ag3_pwrs=10*np.log10(ag3_pwrs)
    st = datetime.now()
    funcgen.set_frequency(1000)
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
                
                
                
                
#                mclient.load_settings_from_file('C:\Data\settings\\20160201\\225702.set', ['AWG1', 'qubit_brick_2', 'qubit_info'])
#                mclient.restore_instruments()
               
#                time.sleep(1)
                mclient.load_settings_from_file(r'C:\Data\settings\20180511\103349.set', ['AWG1', 'qubit_brick', 'qubit_info'])
#                mclient.restore_instruments('C:\Data\settings\\20170516\\104443.set')    
#                qi.set_pi_amp(0.133700)
#                qubit_brick.set_frequency(7370.238314e6)
                time.sleep(1)
#                qubit_brick.get_frequency() 
                qubit_info = mclient.get_qubit_info('qubit_info')
                do_T1_plot(qubit_info, 3000, np.concatenate((np.linspace(0, 70e3, 100), np.linspace(71e3, 150e3, 50))), 
                           C2Q1_t1s, 303, var=(datetime.now()-st).total_seconds()/60)
#                do_T2_plot(qubit_info, 10000, np.linspace(0e3, 30e3, 101), 250e3, 
#                           C2Q1_t2s, 304, var=(datetime.now()-st).total_seconds()/60)
#                do_T2echo_plot(qubit_info, 10000, 
#                               np.linspace(0e3, 50e3, 101),
#                               150e3, 
#                               C2Q1_t2Es, 305, 
#                               fit_type = 'exp_decay_sine', 
#                               var=(datetime.now()-st).total_seconds()/60)
                               
#            
################                               
                mclient.load_settings_from_file('C:\Data\settings\\20180511\\114327.set', ['AWG1', 'qubit_brick', 'qubit_info'])
##                qi.set_pi_amp(0.3244)
##                qubit_brick.set_frequency(6529.615000e6)
##                mclient.restore_instruments('C:\Data\settings\\20170516\\151904.set')
##                funcgen.set_frequency(1000)
#                time.sleep(1)
##                amps = np.linspace(-1, 1, 101)*0.5
##                from scripts.single_qubit import rabi
##                tr = rabi.Rabi(qubit_2_info, amps, #r_axis = axis,
##        #                       seq = seq, postseq = postseq,
##                               plot_seqs=False,
##                               update=False,
##                               )
##                tr.measure()
##                qubit_brick.get_frequency()
                qubit_info = mclient.get_qubit_info('qubit_info')
                do_T1_plot(qubit_info, 3000, np.concatenate((np.linspace(0, 70e3, 100), np.linspace(71e3, 150e3, 50))), 
                           C2Q2_t1s, 203, var=(datetime.now()-st).total_seconds()/60)
#                do_T2_plot(qubit_info, 5000, np.linspace(0e3, 30e3, 101), 250e3, 
#                           C2Q2_t2s, 204, var=(datetime.now()-st).total_seconds()/60)
#                do_T2echo_plot(qubit_info, 5000, 
#                               np.linspace(0e3, 50e3, 101),
#                               150e3, 
#                               C2Q2_t2Es, 205, 
#                               fit_type = 'exp_decay_sine', 
#                               var=(datetime.now()-st).total_seconds()/60)  
###############                               
#                mclient.load_settings_from_file('C:\Data\settings\\20160201\\225813.set', ['AWG1', 'qubit_brick_2', 'qubit_info'])
#                mclient.restore_instruments('C:\Data\settings\\20160201\\225813.set')
#                funcgen.set_frequency(1000)
#                time.sleep(1)

#                do_T1_plot(qubit_info, 200, np.concatenate((np.linspace(0, 50e3, 31), np.linspace(50e3, 150e3, 20))), 
#                           C2Q3_t1s, 306, var=(datetime.now()-st).total_seconds()/60)
#                do_T2_plot(qubit_info, 200, np.linspace(0e3, 30e3, 101), 300e3, 
#                           C2Q3_t2s, 307, var=(datetime.now()-st).total_seconds()/60)
#                do_T2echo_plot(qubit_info, 200, 
#                               np.linspace(0e3, 60e3, 101),
#                               200e3, 
#                               C2Q3_t2Es, 308, 
#                               fit_type = 'exp_decay_sine', 
#                               var=(datetime.now()-st).total_seconds()/60)   
                               

#save data from repeated measurements. TODO: consolidate with repetition code
if 0:
    ts = time.localtime()
    tstr = time.strftime('%Y%m%d/%H%M%S', ts)
    groupname='%s_%s'  % (tstr, 'Repetition_T1')
    repData=mclient.datafile.create_group(groupname)
    for key, value in C2Q1_t1s.items():
        repData.create_dataset(key, data=value)
    groupname='%s_%s'  % (tstr, 'Repetition_T2')
    repData=mclient.datafile.create_group(groupname)
    for key, value in C2Q1_t2s.items():
        repData.create_dataset(key, data=value)
    groupname='%s_%s'  % (tstr, 'Repetition_T2E')
    repData=mclient.datafile.create_group(groupname)
    for key, value in C2Q1_t2Es.items():
        repData.create_dataset(key, data=value)

    groupname='%s_%s'  % (tstr, 'Repetition_T1_2')
    repData=mclient.datafile.create_group(groupname)
    for key, value in C2Q2_t1s.items():
        repData.create_dataset(key, data=value)
    groupname='%s_%s'  % (tstr, 'Repetition_T2_2')
    repData=mclient.datafile.create_group(groupname)
    for key, value in C2Q2_t2s.items():
        repData.create_dataset(key, data=value)
    groupname='%s_%s'  % (tstr, 'Repetition_T2E_2')
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
    


if 0: #Amp tuneup
    from scripts.jk.pulse_tuning import amplitude_tuneup
    alazar.set_naverages(2e3)
    amptune = amplitude_tuneup.Amplitude_Tuneup(qubit_info, update_ins=True,
                                                relative_range = np.linspace(0.97,1.03,21))
    
    amptune.measure()
    bla
    
if 0:  #DRAG
    from scripts.jk.pulse_tuning import simple_drag_tuneup
    alazar.set_naverages(5e3)
    
    dragexp = simple_drag_tuneup.Simple_DRAG_Tuneup(qubit_info, update=True)
    
    dragexp.measure()   
    bla

if 0:  #AllXY
    from scripts.jk.pulse_tuning import allxy
    alazar.set_naverages(3e3)

    alexp = allxy.AllXY(qubit_info, )
    
    alexp.measure()
    bla
    



if 0: ### calibration routines

    if 0: # find vspec if freq
        vspec = instruments['vspec']
        
        base_df0s = vspec.get_df0()
        df0s = np.linspace(-1, 1, 101)*1e6
        freq = 5719.49e6
#        qubit_brick.set_frequency(freq)
        powers = []        
        for df0 in df0s:
            va_lo.set_frequency(freq+base_df0s+df0)
#            vspec.set_df0(base_df0s + df0)
#            vspec.set_frequency(freq)
            time.sleep(0.1)
            p = vspec.get_power()
            powers.append(p)
            print df0, p
        plt.figure()
        plt.plot(df0s, powers, 'rs-')

    if 1:   #IQ mixer leakage Calibration
        from scripts.calibration import mixer_calibration
        reload(mixer_calibration)
        from scripts.calibration.mixer_calibration \
                import Mixer_Calibration as mixer_cal
                
        #Turn off power supply for amp:
        PS1.set_output_state('OFF')
#        from scripts.calibration.mixer_calibration_fmin \
#                import Mixer_Calibration as mixer_cal


        #############################################################
        qubit_freq = 7347.6677e6 #this is the actual frequency of the qubit
        qubit_brick.set_frequency(qubit_freq-qubit_info.deltaf)
        qubit_brick.set_rf_on(1)
        qubit_cal = mixer_cal('qubit_info', qubit_freq, 
                              spec='spec',
                              verbose=True,
                              base_amplitude=2,
#                              va_lo='va_lo'
                              )

#        cavity_freq = 9.08063e9 #this is the actual frequency of the cavity. For Stark shift set deltaf to detuning
#        cavity_brick.set_frequency(cavity_freq-cavity_info.deltaf)
#        cavity_brick.set_rf_on(True)
##        spec_brick.set_pulse_on(False)
#        qubit_cal = mixer_cal('cavity_info', cavity_freq, 
#                              spec='spec',
#                              verbose=True,
#                              base_amplitude=2,)

#        qubit_ef_freq = 6332.492620e6-248e6 #this is the actual frequency of the qubit
##        qubit_ef_brick.set_frequency(qubit_ef_freq-qubit_ef_info.deltaf)
##        qubit_ef_brick.set_rf_on(True)
##        spec_brick.set_pulse_on(False)
#        qubit_cal = mixer_cal('qubit_ef_info', qubit_ef_freq, 
#                              spec='spec',
#                              verbose=True,
#                              base_amplitude=2,) 
                              
#        phonon_freq = 6.287438E9 #this is the actual frequency of the qubit
#        qubit_cal = mixer_cal('phonon1_info', phonon_freq, 
#                              spec='spec',
#                              verbose=True,
#                              base_amplitude=2,) 
                              
#        qubit_freq = 6649.394e6 #this is the actual frequency of the qubit
#        qubit_brick.set_frequency(qubit_freq-qubit_info.deltaf)
#        qubit_brick.set_rf_on(True)
#        qubit_ef_brick.set_rf_on(False)
##        spec_brick.set_pulse_on(False)
#        qubit_cal = mixer_cal('qubit_info', qubit_freq, 
#                              spec='vspec',
#                              verbose=True,
#                              base_amplitude=2,
#                              va_lo='va_lo')
                              
#for manual tuneup of ef sideband phase with qubit_brick (run this block only)       
#        qubit_ef_freq = 6327.895100e6 
##        qubit_brick.set_frequency(qubit_freq-qubit_ef_info.deltaf)
#        qubit_brick.set_rf_on(True)
#        qubit_ef_brick.set_rf_on(False)
##        spec_brick.set_pulse_on(False)
#        qubit_cal = mixer_cal('qubit_ef_info', qubit_ef_freq, 
#                              spec='vspec',
#                              verbose=True,
#                              base_amplitude=2,
#                              va_lo='va_lo')     
                              

                              
#        qubit_ef_freq = 7956.8e6
#        qubit_ef_freq = 7957e6
#        qubit_ef_freq = 7925e6
#        qubit_ef_freq = 9293e6
#        qubit_ef_brick.set_rf_on(1)
#        qubit_ef_brick.set_pulse_on(0)
#        qubit_ef_brick.set_frequency(qubit_ef_freq-qubit_ef_info.deltaf)
#        qubit_cal = mixer_cal('qubit_ef_info', qubit_ef_freq, 
#                              spec='vspec',
#                              verbose=True,
#                              base_amplitude=2,
##                              va_lo='va_lo_5_10'
#                              va_lo='va_lo'
#                              )
#                              
#        cav_freq = 9.157810e9
#        qubit_ef_brick.set_frequency(cav_freq-cavity_info.deltaf)
##        va_lo_5_10.set_pulse_on(False)
#        qubit_cal = mixer_cal('cavity0', cav_freq, 
#                              spec='vspec',
#                              verbose=True,
#                              base_amplitude=2,
#                              va_lo='va_lo')
##                              va_lo='va_lo')
#
        cal = qubit_cal
        


        ###### using fmin routine  make sure using mixer_calibration_fmin
        if 0:
            cal.prep_instruments(reset_offsets=True, reset_ampskew=True)
            cal.tune_lo()
            cal.tune_osb()
        
        ###### BRUTE FORCE--- make sure using mixer_calibration
        
        if 1:
            
            if 1:
                cal.prep_instruments(reset_offsets=False, reset_ampskew=False)
                cal.tune_lo(mode=('coarse'))  #'coarse'
                cal.tune_osb(mode=(0.5, 2000, 3, 5)) #'coarse'
                cal.tune_lo(mode='fine') # useful if using 10 dB attenuation;
#                                     LO leakage may creep up during osb tuning
            else:
                cal.prep_instruments(reset_offsets=False, reset_ampskew=True)
                cal.tune_osb(mode=(0.2, 2000, 1, 2))
                cal.tune_lo(mode='fine') # useful if using 10 dB attenuation;
                                    # LO leakage may creep up during osb tuning

        # this function will set the correct qubit_info sideband phase for use in experiments
        #    i.e. combines the AWG skew with the current sideband phase offset
        cal.set_tuning_parameters(set_sideband_phase=True)
        cal.load_test_waveform()
        cal.print_tuning_parameters()

