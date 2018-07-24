### LOTS OF USEFUL FUNCTIONS IN THIS ONE

import numpy as np
# import mclient
# from mclient import instruments

import time
import objectsharer as objsh
objsh.backend.start_server(addr='127.0.0.1')
objsh.backend.connect_to('tcp://127.0.0.1:55555')
instruments = objsh.find_object('instruments')

from time import sleep
from time import localtime
import time
from scripts.vna import VNA_functions
# from scripts import VNA_functions
current_milli_time = lambda: int(round(time.time() * 1000))
import os
from matplotlib import pyplot as plt
from datetime import datetime, time, timedelta
# from Autocopy import autocopy
# from MatrixSwitch import matrix_switch

# blue = instruments.create('fridge', 'sauron', host='sauron.central.yale.internal', port=33590, fridge='BL', base_ruox_channel=5)
# vna = instruments.create('VNA','Agilent_E5071C', address='TCPIP0::172.28.140.120::inst0::INSTR') # Galahad
smeagol = instruments.create('fridge', 'sauron', host='sauron.central.yale.internal', port=33590, fridge='SG', base_ruox_channel=8)
#vna = instruments.create('VNA','Agilent_E5071C', address='TCPIP0::172.28.143.85::inst0::INSTR') # QV2
# vna = instruments.create('VNA','Agilent_E5071C', address='TCPIP0::VNA2.central.yale.internal::inst0::INSTR') # VNA2

def autocopy():
    pass

def photons(Pin, Qint, Qcoup, fres):
    hbar = 1.054572e-34
    Pcav = Pin-63.0
    return (10**((Pcav/10)-3)*(2*Qint**2*Qcoup)/(Qint+Qcoup)**2)/(hbar*(2*np.pi*fres)**2)

def to_dB(x):
    return 20*np.log10(x)

def from_dB(x):
    return 10**(x/20.)

VNA_MAX_POINTS = 20001
vna = instruments['VNA']
#fridge = instruments['fridge']

# pnc2a =  (r'Z:\_Data\PNC_Blue\diagnostics\pnc2a_screw3_161217\\',r'C:\_Data\161217_pnc2a_screw3.h5')
# sm4 =  (r'Z:\_Data\PNC_Blue\diagnostics\161209_SM4_sm2c_longTerm\\',r'C:\_Data\161209_SM4_sm2c_longTerm.h5')
# pnc3b3 =  (r'Z:\_Data\PNC_Blue\diagnostics\pnc3b3_170120\\',r'C:\_Data\20170120_pnc3b3.h5')
# pnc3b1 =  (r'Z:\_Data\PNC_Blue\diagnostics\pnc3b1_170120\\',r'C:\_Data\20170120_pnc3b1.h5')
# pnc3b2 =  (r'Z:\_Data\PNC_Blue\diagnostics\pnc3b2_170120\\',r'C:\_Data\20170120_pnc3b2.h5')
#pnc3b4 =  (r'Z:\_Data\PNC_Blue\diagnostics\pnc3b4_170221\\',r'C:\_Data\20170221_pnc3b4.h5')
#c4 = (r'Z:\_Data\Smeagol\Data\vna\CD_170609\\C4\\',r'C:\Data\vna\CD_170609\C4.h5')/
c12 = (r'Z:\_Data\Smeagol\Data\vna\Parker\\C12\\',r'C:\Data\vna\Parker\C12.h5')
# diagA =  (r'Z:\_Data\PNC_Blue\diagnostics\A_2\\',r'C:\_Data\exp\VNA\20160723_diagA2.h5')
# diagB =  (r'Z:\_Data\PNC_Blue\diagnostics\B_2\\',r'C:\_Data\exp\VNA\20160723_diagB2.h5')
# reagors=(r'Z:\_Data\Coaxline\WaterIceRDC\ReagorWaterice_Storage_BakedInSitu\\',r'C:\_Data\201604_Storage_BakedInSitu_RDC.h5')
# reagorr=(r'Z:\_Data\Coaxline\WaterIceRDC\ReagorWaterice_Readout_BakedInSitu\\',r'C:\_Data\201604_Readout_BakedInSitu_RDC.h5')
# jk=     (r'Z:\_Data\Coaxline\JacobKevin_TieFighter_HP1\\',r'C:\_Data\201603_JacobKevin_TieFighter_HP1.h5')
# cm1=    (r'Z:\_Data\Coaxline\Caltech_CM1\\',r'C:\_Data\201610_Caltech_CM1_run4.h5')
atten = 0

def wait_start(runTime, action):
    startTime = runTime # time(*(map(int, runTime.split(':')))) # must be a datetime.datetime object
    while startTime > datetime.now():
        n = datetime.now()
        print 'The current time %s is not yet the go-time, %s. Waiting...'%(str(n),str(runTime))
        sleep(300) # 5 min interval
    print "It's go time! Executing requested function..."
    return action()

def matrix_alias(alias):
    pass

import winsound, sys
def beep(loop):
    sound = r'C:\Windows\Media\notify'
    for i in range(loop):
        winsound.PlaySound('%s.wav' % sound, winsound.SND_FILENAME)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

if 0: # delay start
    runTime = datetime(2016, 3, 21, 7, 00, 0)
    wait_start(runTime, lambda: autocopy())

if 0: # advanced scanning program
#    binning = 101
    binning = 101
    polyval = 3
    start_freq= 7.5e9
    stop_freq = 8.5e9
    span_size = 10e6
    step_size = span_size
    if_bw = 10000
    points = VNA_MAX_POINTS
    avgs = 5
    power = -5 # dB out INCLUDING attenuation
    #######################
    scanfreq = np.array([],dtype=float)
    scandata = np.array([],dtype=float)
    vna.set_span(span_size)
    vna.set_center_freq(start_freq+span_size/2)
    vna.set_if_bandwidth(if_bw)
    vna.set_points(points)
    sleep(1)
    st = avgs*vna.get_sweep_time()
    stp = (stop_freq-start_freq)/span_size
    overhead = 5.*stp/60.
    tottime = st*stp/60.+overhead
    print 'Total search time will be %.2f minutes'%(tottime,)
    perstep = (st+5.)/60.
    stepno = 0
    while vna.get_stop_freq()<=stop_freq:
        stepno += 1
        try:
            collect_data = VNA_functions.power_sweep(vna,np.array([power]),
                                                     avgs_start=avgs,avgs_stop=avgs,
                                                     IF_BANDWIDTH=if_bw,NUM_POINTS=points,
                                                     save=False,revert=False,display=False,atten=atten)
            ys = collect_data[2][0][0]
            xs = collect_data[1]
            ysavg = savitzky_golay(ys, binning, polyval)
            scanfreq = np.append(scanfreq, xs)
            scandata = np.append(scandata, ys-ysavg)
            vna.set_center_freq(vna.get_start_freq()+step_size+span_size/2)
            sleep(1)
            print "Remaining time: ",tottime-perstep*stepno
        except Exception, e:
            print "Quit unexpectedly:", e
            break
    plt.figure()
    plt.plot(scanfreq, scandata, '.')
    autocopy()
    vna.set_averaging_trigger(0)
    beep(3)


if 0: # set up save states
#    modes = [(7.327760e9, 200e3), (7.968630e9, 200e3), (8.317140e9, 400e3), (8.859758e9, 200e3), (9.224434e9, 600e3)]
#    modes = [(6.382222e9, 600e3), (6.977416e9, 400e3), (7.717476e9, 20e3), (8.070393e9, 600e3), (8.391166e9, 150e3)]
#    modes = [(7.395958e9,300e3), (7.672391e9,40e3),(8.351326e9, 100e3)]
    modes = [(7.391965e9, 400e3), (7.669116e9, 20e3), (8.033643e9,300e3), (8.349494e9,20e3), (8.557839e9, 500e3),(8.566614e9,30e3)]
    zerospanmodes = [(7.3919650e9, 0), (7.6691157e9, 0), (8.0336405e9,0), (8.349494e9,0),(8.557839e9,0),(8.5666143e9,0)]

    for mi, mode in enumerate(modes):
        vna.set_center_freq(mode[0])
        vna.set_span(mode[1])
        sleep(1)
        vna.autoscale()
        vna.set_instrument_state_file('zmode'+str(mi+1))
        vna.save_state()


if 0: # collect existing VNA data; plot Qi progression
    import liveplot # start with python -m liveplot
    plotter = liveplot.LivePlotClient()

    fit_this = True
    goodness_threshold = 0.2 # first stage check
    # plot_threshold = 0.5 # second stage check
    plt.close('all')
    manyres = []
    import VNA_functions
    folder, h5folder = sm3
    m = 'S21'

    vna.set_measurement(m)

    while True:
        title = 'sm3_r1_'+vna.get_measurement()+'_'
        # title = 'mode1_multiCapture1_PToff_'+vna.get_measurement()+'_'
        collect_data, fn = VNA_functions.collect_and_display(vna,fridge=lazarus,atten=atten,
                                                         folder=folder, # fmt='UPH',
                                                         save=True, fpre=title, h5file=h5folder,
                                                         display=(not fit_this), returnfile=True)

        temperature = (mclient.instruments['fridge'].get_temperature()*1000. if mclient.instruments['fridge'] else -1.)
        lbl = '\npwr = %.2f dBm\ntemp = %.3f mK'%(vna.get_power()+atten,temperature)

        if fit_this:
            import mag_fit_resonator
            rtdata_f, rtdata_db, _ = np.loadtxt(fn,unpack=True)
            params, _, _ = mag_fit_resonator.quick_hanger_fit(rtdata_f, rtdata_db, show=False, extra_info=lbl, returnparams=True)

            qi = params['qi'].value
            qierr = params['qi'].stderr
            if qierr==0.0:
                qierr=qierr # qi
            goodness = 1-qierr/np.abs(qi)
            goodness1 = 1-params['qcr'].stderr/np.abs(params['qcr'].value)
            goodness2 = 1-params['qci'].stderr/np.abs(params['qci'].value)
            goodness = goodness*goodness1*goodness2
            print 'qi: %.3e; goodness: %.3f, %.3f, %.3f'%(qi,goodness,goodness1,goodness2)
            if goodness>goodness_threshold:
                manyres.append((goodness,params))

            # autocopy()
            # plt.close('all')

            plotter.append_y('Qtrack_'+title, qi)

            sleep(vna.get_sweep_time()+2.)


if 0: # collect full span or current data
    # vna.set_start_freq(300e3)
    # vna.set_stop_freq(20e9)
    # sleep(5)
    fit_this = False
    goodness_threshold = 0.2 # first stage check
    # plot_threshold = 0.5 # second stage check
    # plt.close('all')
    manyres = []
    excise_start = 0
    import VNA_functions
    folder, h5folder = ck2

    for m in ['S21' for i in np.arange(1)]: # ['S11', 'S21', 'S22']:
        vna.set_measurement(m)
        title = 'ReadoutLLP_'+vna.get_measurement()+'_'
        # title = 'Mode3Low_'+vna.get_measurement()+'_'
        collect_data, fn = VNA_functions.collect_and_display(vna,fridge=smeagol,atten=atten,
                                                         folder=folder, # fmt='UPH',
                                                         save=True, fpre=title, h5file=h5folder,
                                                         display=(not fit_this), returnfile=True)

        temperature = (instruments['fridge'].get_temperature()*1000. if instruments['fridge'] else -1.)
        fridge_in_pwr = vna.get_power()+atten
        lbl = '\npwr = %.2f dBm\ntemp = %.3f mK'%(fridge_in_pwr,temperature)

        if fit_this:
            from lib.math import fitter
            f = fitter.Fitter('asymmetric_v_hanger')
            rtdata_f, rtdata_db, _ = np.loadtxt(fn,unpack=True)
            x = rtdata_f[excise_start:]
            ylin = from_dB(rtdata_db[excise_start:])
            result = f.perform_lmfit(x, ylin, print_report=True, plot=True)
            params = result.params

            qi = params['qi'].value
            qierr = params['qi'].stderr
            if qierr==0.0:
                qierr=qierr # qi
            goodness = 1-qierr/np.abs(qi)
            goodness1 = 1-params['qcr'].stderr/np.abs(params['qcr'].value)
            goodness2 = 1-params['qci'].stderr/np.abs(params['qci'].value)
            goodness = goodness*goodness1*goodness2
            print 'qi: %.3e; goodness: %.3f, %.3f, %.3f'%(np.abs(qi),goodness,goodness1,goodness2)
            if goodness>goodness_threshold:
                manyres.append((goodness,params))

            qc = np.sqrt(params['qcr']**2 + params['qci']**2)
            fr = params['f0']
            phots = photons(fridge_in_pwr, qi, qc, fr)
            print 'f0: %.6f GHz'%(params['f0'].value/1e9)
            print '%.1f photons at %.2f dBm into fridge'%(phots,fridge_in_pwr)
            print '(1 photon at %.2f dBm)'%(fridge_in_pwr-10*np.log10(phots),)

        autocopy()
        # plt.close('all')
        sleep(0.5)

    if len(manyres)>1:
        manyresarr = np.array(manyres)
        goodnesses = np.array(zip(*manyres)[0])
        mr = manyresarr[goodnesses > 0.7]
        if mr.size>0:
            qis = np.array([params['qi'].value for params in zip(*mr)[1]])
            stderrs = np.array([params['qi'].stderr/np.abs(params['qi'].value) for params in zip(*mr)[1]])
            plt.close('all')
            plt.figure()
            plt.hist(qis[qis<3e7])
            plt.xlabel('Qi')
            plt.ylabel('Counts')
            plt.title('Goodness Threshold: %.2f'%(plot_threshold))
            autocopy()

if 0: # histogram things
    goodness_threshold = 0.85


if 0: # simple temperature dependence
    fit_this = True
    import VNA_functions
    oldtemp = -2
    while temperature>50:
        for m in ['S21']:
            vna.set_measurement(m)
            title = 'ModeC_'+vna.get_measurement()+'_'
            temperature = (mclient.instruments['fridge'].get_temperature()*1000. if mclient.instruments['fridge'] else -1.)
            if temperature!=oldtemp:

                collect_data, fn = VNA_functions.collect_and_display(vna,fridge=lazarus,atten=0,
                                                                 folder=m3, # fmt='UPH',
                                                                 save=True, fpre=title, h5file=m3_h5,
                                                                 display=(not fit_this), returnfile=True)


                lbl = '\npwr = %.2f dBm\ntemp = %.3f mK'%(vna.get_power(),temperature)

                if fit_this:
                    import mag_fit_resonator
                    rtdata_f, rtdata_db, _ = np.loadtxt(fn,unpack=True)
                    mag_fit_resonator.quick_hanger_fit(rtdata_f, rtdata_db, show=True, extra_info=lbl)

                autocopy()
                plt.close('all')

            oldtemp = temperature

        sleep(10) # wait for temperature to be updated

if 0: # flux curve
    import matplotlib.cm as cm
    currents = np.linspace(-2.5e-3,2.5e-3,1001)
    mags = np.ndarray(shape=(currents.shape[0],vna.get_points()))
    phases = np.ndarray(shape=(currents.shape[0],vna.get_points()))
    pows = [vna.get_power(),]
    avgs = 1
    pts = 1601
    small_steps = 5e-6
    ifbw = vna.get_if_bandwidth()
    ### ramp this up slowly
    if yoko.get_output_state==0:
        yoko.set_current(0.)
    yoko.set_output_state(1)
    for cur in np.linspace(yoko.get_current(),currents[0],np.abs(np.round((currents[0]-current_now)/small_steps))):
        yoko.set_current(cur)
        sleep(0.25)
    for i, cur in enumerate(currents):
        yoko.set_current(cur)
        sleep(3)
        title = 'flux{:.2e}_'.format(cur)
        # collect_data, fn = VNA_functions.collect_and_display(vna,fridge=lazarus,atten=0,
                                                     # folder=lazcooldwn,
                                                     # save=True, fpre=title,
                                                     # display=False, returnfile=True)
        collect_data = VNA_functions.power_sweep(vna,pows,fridge=lazarus,folder=lazcooldwn,
                        avgs_start=avgs,avgs_stop=avgs,
                        IF_BANDWIDTH=ifbw,NUM_POINTS=pts,fpre=title,
                        save=True,revert=True,display=False)
        freqs = collect_data[1]
        mag, phase = collect_data[2][0]
        mags[i] = mag
        phases[i] = phase

    plt.figure()
    plt.pcolormesh(currents, freqs, mags.T)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Current (A)')
    plt.title('Magnitude')
    plt.figure()
    plt.pcolormesh(currents, freqs, phases.T)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Current (A)')
    plt.title('Phase')



    np.savetxt(lazcooldwn+'flux_sweep_currents.dat', currents)
    np.savetxt(lazcooldwn+'flux_sweep_freqs.dat', freqs)
    np.savetxt(lazcooldwn+'flux_sweep_mags.dat', mags.T)
    np.savetxt(lazcooldwn+'flux_sweep_phases.dat', phases.T)

    # ramp back down to 0
    for cur in np.linspace(yoko.get_current(),0.0,np.abs(np.round((0.0-current_now)/small_steps))):
        yoko.set_current(cur)
        sleep(0.25)

if 1: # collect current data
    title = 'test'
    myfile, h5file = c12
    VNA_functions.collect_and_display(vna,atten=atten, folder=myfile,
                                   save=True, fpre=title,
                                   display=True, fridge=smeagol,
                                   h5file = h5file,
                                   )

if 0: # standard power sweep
    title = 'powers_'
    powers = np.linspace(-75,-5,71)
    myfile, h5file = pnc2a
    power_sweep_data = VNA_functions.power_sweep(vna,powers,fridge=fridge,
                                                 folder=myfile,
                                                 PULSE_TUBE='ON',
                                                 avgs_start=400,avgs_stop=2,
                                                 NUM_POINTS=1601,atten=atten,
                                                 IF_BANDWIDTH=100,fpre=title,
                                                 save=True,
                                                 h5file=h5file,
                                                 fit=False)

if 0: # standard power sweep, from save state file, works with SEGM and regular
    if 1:
        modes = ['cavA', 'cavB']
        notes = '_'
        pt_status = 'ON'
        myfile, h5file = pnc2a
        for mode in modes:
            title = mode+notes
            vna.set_instrument_state_file(mode)
            vna.load_state()
            powers = np.linspace(-85,-5,33) # post-attenuation
            power_sweep_data = VNA_functions.power_sweep(vna,powers,fridge=blue,
                                                         folder=myfile,
                                                         PULSE_TUBE=pt_status,
                                                         avgs_start=25,avgs_stop=2,
                                                         NUM_POINTS=1601,atten=atten,
                                                         IF_BANDWIDTH=20,fpre=title,
                                                         save=True, h5file=h5file,
                                                         fit=True,sweep_type=vna.get_sweep_type())
            if (mode is not modes[-1]) and pt_status=='OFF':
                sleep(20*60)
            if mode is not modes[-1]:
                plt.close('all')

    if 0:
        modes = ['SM3_S2']
        notes = '_'
        pt_status = 'ON'
        myfile, h5file = cm1
        for mode in modes:
            title = mode+notes
            vna.set_instrument_state_file(mode)
            vna.load_state()
            powers = np.linspace(-85,-10,21) # post-attenuation
            power_sweep_data = VNA_functions.power_sweep(vna,powers,fridge=blue,
                                                         folder=myfile,
                                                         PULSE_TUBE=pt_status,
                                                         avgs_start=450,avgs_stop=2,
                                                         NUM_POINTS=400,atten=atten,
                                                         IF_BANDWIDTH=100,fpre=title,
                                                         save=True, h5file=h5file,
                                                         fit=True,sweep_type=vna.get_sweep_type())
            if (mode is not modes[-1]) and pt_status=='OFF':
                sleep(20*60)
            if mode is not modes[-1]:
                plt.close('all')

#
# TODO: get phase FFT in useful units
#

if 0: # collect and FFT data; only works for ZERO SPAN measurements
    import VNA_functions
    plt.close('all')
    mode = 'reagors'
    if 1:
        vna.set_instrument_state_file(mode)
        vna.load_state()
        sleep(1)
    vna.set_span(0.)
    save = True
    # vna.set_center_freq(5.89657e9)
    ifbw = 1e4 # 5e5 # 5e5 # sets max frequency
    points = VNA_MAX_POINTS # set resolution
    pt = 'ON'
    folsave = m7+'\\'+mode+'\\'
    fsave = folsave+'FFT_%d'%(ifbw,)+'IFBW_pt'+pt+'.dat'
    h5file = m47[1]
    collect_data = VNA_functions.power_sweep(vna,np.array([vna.get_power()]),fridge=lazarus,
                                                     save=False,
                                                     display=False,
                                                     revert=False,
                                                     fmt='SCOM',PULSE_TUBE=pt,
                                                     IF_BANDWIDTH=ifbw,
                                                     NUM_POINTS=points,h5file=h5file)
    sleep(1)
    phase_data = VNA_functions.power_sweep(vna,np.array([vna.get_power()]),fridge=lazarus,
                                                     save=False,
                                                     display=False,
                                                     revert=False,
                                                     fmt='PHAS',PULSE_TUBE=pt,
                                                     IF_BANDWIDTH=ifbw,
                                                     NUM_POINTS=points,h5file=h5file)
    samp_len = collect_data[1].size
    samp_rate = 1/(collect_data[1][1]-collect_data[1][0])
    zdata = collect_data[2][0][0] + np.complex(0,1) * collect_data[2][0][1]
    z = zdata/np.average(zdata)
    pdata = phase_data[2][0][0]
    p = pdata/np.average(pdata)
    ts = np.linspace(0,samp_len/samp_rate,samp_len)
    fs = np.fft.fftfreq(samp_len, 1/samp_rate) # number of samples * 2, sample rate
    fig = plt.figure()
    plt.plot(ts,pdata)
    plt.ylabel('Phase on resonance (deg)')
    plt.xlabel('Time (s)')
    plt.title(fsave)
    autocopy()
    posfreqs = np.size(fs)/2
    Fmag = np.abs(2*np.fft.fft(np.abs(z))/samp_len)**2
    Fpha = np.angle(2*np.fft.fft(z)/samp_len) # not unwrapped, rad
    Fdeg = np.abs(2*np.fft.fft(p)/samp_len) # unwrapped, deg
    # F_avg.append(F)
    fig = plt.figure()
    plt.plot(fs[:posfreqs], Fmag[:posfreqs], '.-')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT [V**2]')
    fig = plt.figure()
    plt.plot(fs[:posfreqs], Fdeg[:posfreqs], '.-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT [deg]')
    dsave = np.array([fs[:posfreqs], Fmag[:posfreqs], Fpha[:posfreqs], Fdeg[:posfreqs]]).T
    if save and fsave and not os.path.exists(folsave):
        os.makedirs(folsave)
    if save:
        np.savetxt(fsave, dsave, fmt='%.6e')

if 0: # collect and FFT data for multiple modes
    import VNA_functions
    # Only if very overcoupled and the dips go very low should we consider moving away from the maximum phase roll/resonant frequency.
    # plt.close('all')
#    modes = ['M6modeF1', 'M6modeF2', 'M6modeF3', 'M6modeF4', 'M6modeF5', 'M6modeR1', 'M6modeR2', 'M6modeR3', 'M6modeR4', 'M6modeR5']
    # modes = ['M10mode1','M10mode2', 'M10mode3', 'M10mode4', 'M10mode5']
    # modes = ['M8mode1', 'M8mode2', 'M8mode3', 'M8mode4']
    modes=['reagors']
    edelay = 47.7e-9
    ifbws = [1e3,1e4,1e5,5e5]
    myfile, h5file = reagors
    save = True
    points = VNA_MAX_POINTS # set resolution
    pt = 'ON'
    for ifidx, ifbw in enumerate(ifbws):
        for midx, mode in enumerate(modes):
            if 1:
                vna.set_instrument_state_file(mode)
                vna.load_state()
                sleep(1)
            vna.set_electrical_delay(edelay)
            vna.set_span(0.)
            folsave = myfile+'\\'+mode+'\\'
            fsave = folsave+'FFT_%d'%(ifbw,)+'IFBW_pt'+pt+'.dat'

            collect_data = VNA_functions.power_sweep(vna,np.array([vna.get_power()]),fridge=lazarus,
                                                             save=False,
                                                             display=False,
                                                             revert=False,
                                                             fmt='SCOM',PULSE_TUBE=pt,
                                                             IF_BANDWIDTH=ifbw,
                                                             NUM_POINTS=points,h5file=h5file)
            sleep(1)
            phase_data = VNA_functions.power_sweep(vna,np.array([vna.get_power()]),fridge=lazarus,
                                                             save=False,
                                                             display=False,
                                                             revert=False,
                                                             fmt='UPH',PULSE_TUBE=pt,
                                                             IF_BANDWIDTH=ifbw,
                                                             NUM_POINTS=points,h5file=h5file)
            samp_len = collect_data[1].size
            samp_rate = 1/(collect_data[1][1]-collect_data[1][0])
            zdata = collect_data[2][0][0] + np.complex(0,1) * collect_data[2][0][1]
            z = zdata/np.average(zdata)
            pdata = phase_data[2][0][0]
            p = pdata-np.average(pdata) # this was division before. now it is subtraction
            ts = np.linspace(0,samp_len/samp_rate,samp_len)
            fs = np.fft.fftfreq(samp_len, 1/samp_rate) # number of samples * 2, sample rate

            fig = plt.figure(ifidx)
            plt.plot(ts,pdata,label=mode)
            plt.ylabel('Phase on resonance (deg)')
            plt.xlabel('Time (s)')
            plt.title(fsave+'\n%.3e IFBW'%(ifbw,))
            plt.legend(loc='best')
            # autocopy()
            posfreqs = np.size(fs)/2
            Fmag = np.abs(2*np.fft.fft(np.abs(z))/samp_len)**2
            Fpha = np.angle(2*np.fft.fft(z)/samp_len) # not unwrapped, rad
            Fdeg = np.abs(2*np.fft.fft(p)/samp_len) # unwrapped, true deg
            # F_avg.append(F)

            from scipy import fftpack
            # compute PSD using simple FFT
            data = p
            dt = 1/samp_rate
            N = samp_len
            df = 1. / (N * dt)
            PSDp = abs(dt * fftpack.fft(data)[:N / 2]) ** 2 # deg / Hz
            PSDf = df * np.arange(N / 2)
            Fdeg = PSDp


            fig = plt.figure(ifidx+100)
            plt.plot(fs[:posfreqs], Fmag[:posfreqs], '.-', label=mode)
            plt.yscale('log')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('FFT [V**2]')
            plt.title(fsave+'\n%.3e IFBW'%(ifbw,))
            plt.legend(loc='best')

            fig = plt.figure(ifidx+200)
            plt.plot(fs[:posfreqs], Fdeg[:posfreqs], '.-', label=mode)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('FFT [deg]')
            plt.title(fsave+'\n%.3e IFBW'%(ifbw,))
            plt.legend(loc='best')

            dsave = np.array([fs[:posfreqs], Fmag[:posfreqs], Fpha[:posfreqs], Fdeg[:posfreqs]]).T
            if save and fsave and not os.path.exists(folsave):
                os.makedirs(folsave)
            if save:
                np.savetxt(fsave, dsave, fmt='%.6e')
    vna.set_electrical_delay(0)
    beep(3)

if 0:  #find electrical delay
    pows = [-30,-20,-10,0]
    avgs = [10,10]
    VNA_functions.tau_power_sweep(vna,pows,avgs,center=6e9,span=3e9,revert=True,display=True,atten=0)