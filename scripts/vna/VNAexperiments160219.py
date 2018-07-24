### LOTS OF USEFUL FUNCTIONS IN THIS ONE

import numpy as np
import mclient
from time import sleep
from time import localtime
import time
import VNA_functions
current_milli_time = lambda: int(round(time.time() * 1000))
import os

#lazarus = mclient.instruments['fridge']

VNA_MAX_POINTS = 10001
vna = mclient.instruments['VNA']
smeagol = mclient.instruments['fridge_SG']

#sm2=    (r'Z:\_Data\Coaxline\SeamMux2_sm2c_repkg_SM2\\',r'C:\_Data\201602_SeamMux2_sm2c_repkg_SM2.h5')
#m8a=    (r'Z:\_Data\Coaxline\qmsht8b_M8A\\',r'C:\_Data\201602_qmsht8b_M8A.h5')
#reagorr=(r'Z:\_Data\Coaxline\ReagorWaterice_Readout_Vented\\',r'C:\_Data\201602_ReagorWaterice_Readout_Vented.h5')
#reagors=(r'Z:\_Data\Coaxline\ReagorWaterice_Storage_Vented\\',r'C:\_Data\201602_ReagorWaterice_Storage_Vented.h5')
#teresa= (r'Z:\_Data\Si_uMachine\201602_InStriplines45-48\\',r'C:\_Data\201602_InStriplines45-48.h5')
teresa_nonmon=r'Z:\_Data\Si_uMachine\201602_Patchnonmon2\\'
teresa_mon=r'Z:\_Data\Si_uMachine\201602_Patchmon2\\'

atten = 0

def autocopy(figno=None,p=True):
    import PyQt4
    if figno:
        fig = plt.figure(figno)
    else:
        fig = plt.gcf()
    canvas = fig.canvas
    fc = fig.get_facecolor()
    fig.set_facecolor('white')
    canvas.draw()
    pixmap = PyQt4.QtGui.QPixmap.grabWidget(canvas)
    PyQt4.QtGui.QApplication.clipboard().setPixmap(pixmap)
    fig.set_facecolor(fc)
    canvas.draw()
    if p:
        print "Image copied to clipboard."

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

if 0: # advanced scanning program
#    binning = 101
    binning = 101
    polyval = 3
    start_freq= 13e9
    stop_freq = 14e9
    span_size = 250e6
    step_size = span_size
    if_bw = 10000
    points = VNA_MAX_POINTS
    avgs = 5
    power = 0 # dB out INCLUDING attenuation
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
    folder, h5folder = reagors
    m = 'S21'
    
    vna.set_measurement(m)

    while True:
        title = 'Storage_'+vna.get_measurement()+'_'
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
    folder = teresa_mon


#    for m in ['S21' for i in np.arange(1)]: # ['S11', 'S21', 'S22']:
#        vna.set_measurement(m)
#        vna.set_measurement('S21')
#        title = 'spanS21_'+ vna.get_measurement()+'_'
#    title = 'mode2_S21'
    title='mode1'
    collect_data, fn = VNA_functions.collect_and_display(vna,fridge=None,atten=atten,
                                                     folder=folder, # fmt='UPH',
                                                     save=True, fpre=title, h5file=None,
                                                     display=(not fit_this), returnfile=True)
                                    
    temperature = (mclient.instruments['fridge'].get_temperature()*1000. if mclient.instruments['fridge'] else -1.)
    lbl = '\npwr = %.2f dBm\ntemp = %.3f mK'%(vna.get_power()+atten,temperature)

    if fit_this:                   
        import mag_fit_resonator
        rtdata_f, rtdata_db, _ = np.loadtxt(fn,unpack=True)
        params, _, _ = mag_fit_resonator.quick_hanger_fit(rtdata_f[excise_start:], rtdata_db[excise_start:], show=True, extra_info=lbl, returnparams=True)
#
        
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
        
    # vna.set_measurement('S21')

if 0: # histogram things
    goodness_threshold = 0.85


if 0: # simple temperature dependence
    fit_this = True
    import VNA_functions
    oldtemp = -2
#    while temperature>50:
    while True:
#        vna.set_measurement('S21')
#            title = 'Mode4_'+vna.get_measurement()+'_'
        title = 'Mode4_'
#        temperature = (mclient.instruments['fridge_SG'].get_temperature()*1000. if mclient.instruments['fridge_SG'] else -1.)
        temperature = (smeagol.get_temperature()*1000. if smeagol else -1.)
        print(temperature)
        if temperature!=oldtemp:

            collect_data, fn = VNA_functions.collect_and_display(vna,fridge=smeagol,atten=atten,
                                                             folder=teresa_tempdep, # fmt='UPH',
                                                             save=True, fpre=title, h5file=False,
                                                             display=True, returnfile=True)
                                            
            
            lbl = '\npwr = %.2f dBm\ntemp = %.3f mK'%(vna.get_power(),temperature)
        
            if fit_this:                   
                import mag_fit_resonator
                rtdata_f, rtdata_db, _ = np.loadtxt(fn,unpack=True)
#                mag_fit_resonator.quick_hanger_fit(rtdata_f, rtdata_db, show=True, extra_info=lbl)
                mag_fit_resonator.quick_hanger_fit(rtdata_f, rtdata_db, show=True)
                
            autocopy()
            plt.close('all')

        oldtemp = temperature

        sleep(10) # wait for temperature to be updated

if 0: # collect current data
    title = 'test'
    VNA_functions.collect_and_display(vna,atten=atten, folder=teresa,
                                   save=True, fpre=title,
                                   display=True, 
#                                   returnfile=True)
                                   )


if 0: # standard power sweep
    title = 'mode2_'
    powers = np.linspace(5,-75,8)
#    powers = np.linspace(-15.0,-15.5,10)
    power_sweep_data = VNA_functions.power_sweep(vna,powers,fridge=None,
                                                 folder=teresa_mon, #sweep_type='SEGM',
#                                                 PULSE_TUBE='ON',
                                                 avgs_start=2,avgs_stop=300,
                                                 NUM_POINTS=401,atten=atten,
                                                 IF_BANDWIDTH=1000,fpre=title,
                                                 save=True, 
#                                                 h5file=teresa_h5, 
                                                 fit=False)

if 0: # standard power sweep, from save state file
    # modes = ['SM1modeR1', 'SM1modeR2', 'SM1modeR3', 'SM1modeR4', 'SM1modeR5', 'SM1modeF1', 'SM1modeF2', 'SM1modeF3', 'SM1modeF4']
    modes = ['Patchmon_1','Patchmon_2']
    # -85 is 10 photons for S1
    pt_status = 'ON'
#    myfile, h5file = teresa_nonmon
    folder = teresa_nonmon
    for mode in modes:
        title = mode+'_'
        vna.set_instrument_state_file(mode)
        vna.load_state()
        powers = np.linspace(5,-85,9) # post-attenuation
        power_sweep_data = VNA_functions.power_sweep(vna,powers,fridge=None,
                                                     folder=folder,
                                                     PULSE_TUBE=pt_status,
                                                     avgs_start=2,avgs_stop=500,
                                                     NUM_POINTS=400,atten=atten,
                                                     IF_BANDWIDTH=1e3,fpre=title,
                                                     save=True, 
#                                                     h5file=h5file,
                                                     fit=False,sweep_type=vna.get_sweep_type())
        if (mode is not modes[-1]) and pt_status=='OFF':
            sleep(20*60)
        if mode is not modes[-1]:
            plt.close('all')
    # beep(3)
    

#
# TODO: get phase FFT in useful units
#

if 0: # collect and FFT data; only works for ZERO SPAN measurements
    import VNA_functions
    plt.close('all')
    mode = 'M7mode1'
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
    modes=['SM2modeS1','SM2modeS2','SM2modeS3','SM2modeS4','SM2modeS5']
    edelay = 55.6116e-9
    ifbws = [2e4]
    myfile, h5file = sm2
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