import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from time import sleep
from PyQt4 import QtGui
import h5py
from time import strftime
#from mclient import instruments

DATA_LABEL = { 'PLOG': ('logmag','phase'),
              'SCOM': ('real', 'imag'),
                'UPH': ('phase', 'phase')}

def collect_and_display(vna,fridge=None,folder=None,save=True,display=True,fpost=".dat",fpre='',atten=0,fmt='PLOG',returnfile=False,h5file=None):
    
    if save and not os.path.exists(folder):
        os.makedirs(folder)
    
    xs = vna.do_get_xaxis()
    ys = vna.do_get_data(opc=False,fmt=fmt)
    power=float(vna.get_power()+atten)
    
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xs,ys[0],'bs-',label='%.2f dBm input'%(power,))
        ax.set_title('VNA Capture')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(DATA_LABEL[fmt][0])
        ax.legend(loc='best')
    
    try:
        temp=fridge.do_get_temperature()*1000
    except:
        temp=-1.0

    if save:
        fname="%s%s%2.2fdB_%3.2fmK%s"%(folder,fpre,power,temp,fpost)
        np.savetxt(fname, np.transpose((xs,ys[0],ys[1])), fmt='%.9f', delimiter='\t')
        print "Saved %s"%fname

    if h5file:
        h5f = h5py.File(h5file)
        cur_group = strftime('%y%m%d')
        _iteration_start = strftime('%H%M%S')
        datadir = _iteration_start+'//'   
        g = h5f.require_group(cur_group)

        dg = g.create_dataset(datadir+DATA_LABEL[fmt][0], data=ys[0])
        dg.attrs.create('run_time', _iteration_start)
        dg.attrs.create('power',vna.get_power())
        dg.attrs.create('attenuation',atten)
        dg.attrs.create('averages',vna.get_average_factor())
        dg.attrs.create('IFbandwidth',vna.get_if_bandwidth())
#        dg.attrs.create('measurement',str(vna.get_measurement()))
        dg.attrs.create('smoothing',vna.get_smoothing())
#        dg.attrs.create('electrical_delay',vna.get_electrical_delay())
        dg.attrs.create('format',fmt)
        dg.attrs.create('title',str(fpre))
        if temp>0:
            dg.attrs.create('fridge_temperature',temp)
        if folder:
            dg.attrs.create('datafolder',folder)
        
        dg1 = g.create_dataset(datadir+'frequencies', data=xs)
        dg1.attrs.create('run_time', _iteration_start)
        dg2 = g.create_dataset(datadir+DATA_LABEL[fmt][1], data=ys[1])
        dg2.attrs.create('run_time', _iteration_start)
        
        h5f.close()
        del(h5f)
        
    if not returnfile:
        return (xs,ys)
    else:
        if save:
            return (xs,ys), fname
        else:
            return (xs,ys)

def power_sweep(vna,pows,fridge=None,folder=None,avgs_start=1,avgs_stop=1,
                IF_BANDWIDTH=20,NUM_POINTS=501,fpre='',fpost=".dat",
                PULSE_TUBE='ON',save=True,revert=True,display=True,atten=0,
                reverse=False,sweep_type='LIN',plot_type='MAG',fmt='PLOG',
                printlog=True, h5file=None):
    
    # atten = e.g. -10 dB :: negative if attenuating; enter powers you actually want AFTER attenuation (not VNA output)

    VNA_timeout = None
    NUM_STEPS = np.size(pows)
    if reverse:
        pows = pows[::-1]
        AVGS = np.round(np.logspace(np.log10(avgs_stop),np.log10(avgs_start),NUM_STEPS))
    else:
        AVGS = np.round(np.logspace(np.log10(avgs_start),np.log10(avgs_stop),NUM_STEPS))
    VNA_pows = [0] * len(pows)
    
    if revert:
        _state = "python_temp"
        vna.set_instrument_state_data('CST') # settings and calibration
        vna.set_instrument_state_file(_state)
        vna.save_state()
    
    if save and folder and not os.path.exists(folder):
        os.makedirs(folder)
    elif save and not folder:
        print "Chose to save but no folder provided."
        return

    tot_traces = np.sum(AVGS)
    vna.do_enable_averaging()
    vna.set_points(NUM_POINTS)
    vna.set_if_bandwidth(IF_BANDWIDTH)
    vna.set_sweep_type(sweep_type)
    xs = vna.do_get_xaxis()
    
    if h5file:
        cur_group = strftime('%y%m%d')
        _iteration_start = strftime('%H%M%S')
        datadir = _iteration_start+'//'
                                
        h5f = h5py.File(h5file)
        g = h5f.require_group(cur_group)
        dg1 = g.create_dataset(datadir+'//frequencies', data=xs)
        dg1.attrs.create('run_time', _iteration_start)
        
        h5f.close()
    
    if printlog:
        print "Getting timing information and preparing trigger..."
    sleep(0.1) #why sleep? ask Chris...

    if vna.get_sweep_type()=='SEGM':
        sweep_time = float(vna.get_segment_sweep_time())
    else:
        sweep_time = float(vna.get_sweep_time())
    tot_time = tot_traces * sweep_time
    
    if printlog:
        print "Total duration of this experiment will be %.2f minutes."%(tot_time/60,)
        
    if fridge:
        fridge.do_set_pulse_tube(PULSE_TUBE)
        sleep(5)
        if printlog:
            print "Pulse tube is %s. Beginning measurement."%(fridge.do_get_pulse_tube(),)
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    try:
        ys = [None]*len(pows)
        for idx_power, power in enumerate(pows):
            vna.set_power(power-float(atten))
            this_pwr = vna.get_power()
            VNA_pows[idx_power] = this_pwr
            out_pwr = this_pwr+atten
            this_avgs = AVGS[idx_power]
            
            if VNA_timeout:
                this_time = (sweep_time * this_avgs * VNA_timeout) + np.size(pows)*NUM_POINTS*10 # ms
            else:
                this_time = (sweep_time * this_avgs * 1250.0) + np.size(pows)*NUM_POINTS*10 # ms
            
            # an approximation based on 10ms/pt for transfer, 25% overhead on waiting/capture
            vna.set_average_factor(this_avgs)
            ys[idx_power] = []
            this_time = np.max(np.array([this_time, 5000.])) # 5s timeout minimum
            if this_time < 30e3:   
                # print this_time
                vna.set_timeout(this_time)
                ys[idx_power] = vna.do_get_data(fmt=fmt, opc=True, timeout=this_time)
            else:
                if printlog:
                    print "[NOTICE] Triggering each average."
                if VNA_timeout:             
                    to = np.max(np.array([sweep_time * VNA_timeout, 5000.]))
                else:
                    to = np.max(np.array([sweep_time * 1250.0, 5000.]))
                vna.set_timeout(to)
                # print to
                ys[idx_power] = vna.do_get_data(fmt=fmt, opc=True, trig_each_avg=True, timeout=this_time)
            power=float(vna.get_power()+atten)
            
            try:
                temp=fridge.do_get_temperature()*1000
            except:
                temp=-1.0
                
            try:
                if display:
                    ax.plot(xs,ys[idx_power][0],label='%.2f dBm, %.2f mK'%(power,temp))
                    fig.canvas.draw()
                    QtGui.QApplication.processEvents()
            except:
                pass
            
            if save:
                fname="%s%s%2.2fdB_%3.2fmK%s"%(folder,fpre,power,temp,fpost)
                data = np.transpose((xs,ys[idx_power][0],ys[idx_power][1]))
                np.savetxt(fname, data, fmt='%.9f', delimiter='\t')
                if printlog:
                    print "Saved %s"%fname
                    print "%.2f minutes remain."%(np.sum(AVGS[idx_power+1:])*sweep_time/60,)
#==============================================================================
#             #Chris's way of a folder per power, good for Igor import
#             if h5file: 
#                 h5f = h5py.File(h5file)
#                 g = h5f.require_group(cur_group)
#         
#                 dg = g.create_dataset(datadir + '%.2f'%(out_pwr,) + '//' + DATA_LABEL[fmt][0], data=ys[idx_power][0])
#                 dg.attrs.create('run_time', _iteration_start)
#                 dg.attrs.create('VNA_power',this_pwr)
#                 dg.attrs.create('attenuation',atten)
#                 dg.attrs.create('averages',this_avgs)
#                 dg.attrs.create('IFbandwidth',vna.get_if_bandwidth())
#                 dg.attrs.create('measurement',str(vna.get_measurement()))
#                 dg.attrs.create('smoothing',vna.get_smoothing())
#                 dg.attrs.create('electrical_delay',vna.get_electrical_delay())
#                 dg.attrs.create('format',fmt)
#                 if temp>0:
#                     dg.attrs.create('fridge_temperature',temp)
#                 if folder:
#                     dg.attrs.create('datafolder',folder)
# 
#                 dg2 = g.create_dataset(datadir + '%.2f'%(out_pwr,) + '//' + DATA_LABEL[fmt][1], data=ys[idx_power][1])
#                 dg2.attrs.create('run_time', _iteration_start)
#                 
#==============================================================================
                #h5f.close()
                
        if display:
            ax.set_title('VNA Trace')
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(DATA_LABEL[fmt][0])
            ax.legend(loc='best')
        if h5file:
            h5f = h5py.File(h5file)
            g = h5f.require_group(cur_group)
            
            ys_arr = np.array(ys)
            dg = g.create_dataset(datadir + DATA_LABEL[fmt][0], data=ys_arr[:,0,:])
            dg.attrs.create('run_time', _iteration_start)
            dg.attrs.create('attenuation',atten)
            dg.attrs.create('IFbandwidth',vna.get_if_bandwidth())
            dg.attrs.create('measurement',str(vna.get_measurement()))
            dg.attrs.create('smoothing',vna.get_smoothing())
            dg.attrs.create('electrical_delay',vna.get_electrical_delay())
            dg.attrs.create('format',fmt)
            if temp>0:
                dg.attrs.create('fridge_temperature',temp)
            if folder:
                dg.attrs.create('datafolder',folder)

            dg2 = g.create_dataset(datadir + DATA_LABEL[fmt][1], data=ys_arr[:,1,:])
            dg2.attrs.create('run_time', _iteration_start)
            dg_aves = g.create_dataset(datadir + 'averages', data=AVGS)
            dg_aves.attrs.create('run_time',_iteration_start)
            dg_pwr = g.create_dataset(datadir + 'VNA_power', data=VNA_pows)
            dg_pwr.attrs.create('run_time', _iteration_start)
            
            h5f.close()
            
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        if fridge:
            fridge.do_set_pulse_tube('ON')
        if revert:
            try:
                vna.set_instrument_state_file(_state)
                vna.load_state() # restore original state
            except:
                print "[NOTICE] VNA failed to return to initial state"
                pass
        if h5file:
            del(h5f)
    return (pows,xs,ys)
    
def sweep_time(ifbw,pts,start_avgs,stop_avgs,steps):
    traces = np.sum(np.round(np.logspace(np.log10(start_avgs),np.log10(stop_avgs),steps)))
    print int(traces),"traces for",steps,"steps"
    print (traces*float(pts)/float(ifbw))/60,"minutes"
    print "(",(traces*float(pts)/float(ifbw))/3600,"hours )"
    
def tau_power_sweep(vna,pows,avgs,center=None,span=3e9,revert=True,display=True,atten=0):
    # input VNA parameters to determine electrical delay at different powers    
    if not center:
        center=vna.get_center_freq()
        sleep(1)
    
    vna.set_center_freq(center)
    sleep(1)
    vna.set_span(span)
    sleep(1)

    avg0 = avgs[0]
    avg1 = avgs[-1]
    
    pows, xs, ys = power_sweep(vna,pows,avgs_start=avg0,avgs_stop=avg1,save=False,display=display,
                      IF_BANDWIDTH=2000,NUM_POINTS=1000,revert=revert,sweep_type='LIN',fmt='UPH',
                      atten=atten)
                      
    taus = np.zeros(np.shape(pows))
    pows = np.array(pows)   
    
    # take derivative and average
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    for p in range(np.size(pows)):
        grad = np.diff(np.array([xs, ys[p][0]])) # derivative of fit
        scaledgrad = np.divide(grad[1],grad[0])
        taus[p] = np.mean(scaledgrad)*1e9/360
        if display:
            ax.plot(xs[:-1],scaledgrad,label='%.2f dBm'%(p,))
    
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(pows,taus)
        ax.set_title('Electrical Delays')
        ax.set_xlabel("Power (dB)")
        ax.set_ylabel("Delay (ns)")
        
    return pows, taus

def get_tau_from_file(fn):
    xs, _, ys = np.loadtxt(fn,unpack=True)
    plt.figure()
    plt.plot(xs,np.unwrap(ys,180))
    grad = np.diff(np.array([xs, np.unwrap(ys,180)])) # derivative of fit
    scaledgrad = np.divide(grad[1],grad[0])
    taus = np.mean(scaledgrad)*1e9/360
    plt.figure()
    plt.plot(xs[:-1],scaledgrad,label='%.2f dBm'%(-25,))
    print taus
    

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    # if vna:
#    h5filename = r'Z:\Data\JPC_2016\2016-04-15 Cooldown\mixer.hdf5'
#    yoko = instruments['Yoko']
#    vna = instruments['VNA']
#    data = collect_and_display(vna,save=False,h5file=h5filename,fmt='SCOM')
#    pows = [-80,-70,-60,-50,-40,-30,-20,-10]
    pows = np.linspace(-80, -29, 18)
#    data = power_sweep(vna,pows,folder='Z:/Data/JPC_2015/2015-11-18 Cooldown/EMR-AlN/2015-11-23/',
#                       IF_BANDWIDTH=3000,NUM_POINTS=1601,fpre='3_S21_psweep_',
#                       PULSE_TUBE='ON', avgs_start=999, avgs_stop=1, save=True,
#                       display=True, atten=0, revert=False, fmt='PLOG')
    if False: # power sweep
        data = power_sweep(vna,pows,IF_BANDWIDTH=3000,NUM_POINTS=1601,
                                      PULSE_TUBE='ON', avgs_start=200, avgs_stop=2, save=False,
                                      display=True, atten=0, revert=False, fmt='PLOG',
                                      h5file=h5filename)
    if True: # h5file collect and display
        data = collect_and_display(vna,save=False,h5file=h5filename,fmt='PLOG')
    if False: # txt file collect and display
        folder = 'Z:/Data/Andrew/Microwave_Switch/2016-07-29/'
        data = collect_and_display(vna, fridge=None, folder=folder, fpre='S21_switch_ON', fmt='PLOG')
    

    if False: # fit
        # fitting the last measurements data in h5 file
        h5file = h5py.File(h5filename)
        grp = h5file['/160620']
        meas = grp[grp.keys()[-1]]
        freqs = np.array(meas['frequencies'])
        #powers = meas['VNA_power']
        #aout_complex = np.array(meas['magnitude']) * np.exp(np.pi / 180 * 1j*np.array(meas['phase'])) # assumes data taken was in PLIN form (lin mag and phase)
        aout_complex = np.array(meas['real']) + 1j*np.array(meas['imag'])     # assumes data was taken in SCOM form (real and imag)
        #import sys
        #sys.path.insert(0, 'Z:/Data/JPC_2015/2015-08-27 Cooldown')
        cut = True
        if cut:
            cutoff = 400
            aout_complex = aout_complex[(cutoff):]
            aout_complex = aout_complex[:-1*(cutoff)]
            freqs = freqs[(cutoff):]
            freqs = freqs[:-1*(cutoff)]
        a_background = (sum(aout_complex[0:50]) + sum(aout_complex[-50:])) / 100.0
        import analysis_functions as af
        result = af.analyze_reflection(freqs, [-40], aout_complex, f_0 = 7.6320e9, kc=1e6,ki=1e6, a_in=a_background, T=1e-12)
        res0 = np.array(result[0][:])
        Qc = np.divide(res0, np.array(result[1][:]))
        Qi = np.divide(res0, np.array(result[2][:]))
        Qtot = np.divide(1, (np.divide(1,Qc) + np.divide(1,Qi)))
        # must save Q values into h5 file
        h5file.close()
        # del(h5f)

    
    
    
    # sweep_time(10,1979,1,3,33)


    # x,y = tau_power_sweep(vna, np.linspace(-55,-25,10), [1,1], atten=0)
    
    # data = power_sweep(vna,[-30,-40],IF_BANDWIDTH=2000,NUM_POINTS=501,
    #                                  PULSE_TUBE='ON',avgs_start=1,avgs_stop=1,save=False,
    #                                  display=True,atten=0,revert=False)



