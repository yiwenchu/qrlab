import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from time import sleep

def collect_and_display(vna,fridge=None,folder=None,save=True,display=True,fpost=".dat",fpre='',atten=0,fmt='PLOG'):
    
    if save and not os.path.exists(folder):
        os.makedirs(folder)
    
    xs = vna.do_get_xaxis()
    ys = vna.do_get_data(opc=False,fmt=fmt)
    power=float(vna.get_power()+atten)
    
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xs,ys[0],'b-',label='%.2f dBm input'%(power,))
        ax.set_title('VNA Capture')
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.legend(loc='best')
    
    try:
        temp=fridge.do_get_temperature()*1000
    except:
        temp=-1.0

    if save:
        #fname="%s%s%2.2fdB_%3.2fmK%s"%(folder,fpre,power,temp,fpost)
        fname="%s%s%2.2fdB%s"%(folder,fpre,power,fpost)
        #fname="%s%s"%(folder,fpost)
        np.savetxt(fname, np.transpose((xs,ys[0],ys[1])), fmt='%.6e', delimiter='\t')
        print "Saved %s"%fname
        
    return (xs,ys)

def power_sweep(vna,pows,fridge=None,folder=None,avgs_start=1,avgs_stop=1,
                IF_BANDWIDTH=20,NUM_POINTS=501,fpre='',fpost=".dat",
                PULSE_TUBE='ON',save=True,revert=True,display=True,atten=0,
                reverse=False,sweep_type='LIN',plot_type='MAG',fmt='PLOG'):
    
    # atten = e.g. -10 dB :: negative if attenuating; enter powers you actually want AFTER attenuation (not VNA output)

    NUM_STEPS = np.size(pows)
    if reverse:
        pows = pows[::-1]
        AVGS = np.round(np.logspace(np.log10(avgs_stop),np.log10(avgs_start),NUM_STEPS))
    else:
        AVGS = np.round(np.logspace(np.log10(avgs_start),np.log10(avgs_stop),NUM_STEPS))
    
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
    print "Getting timing information and preparing trigger..."
    sweep_time = float(vna.get_sweep_time())
    tot_time = tot_traces * sweep_time
    
    print "Total duration of this experiment will be %.2f minutes."%(tot_time/60,)
        
    if fridge:
        fridge.do_set_pulse_tube(PULSE_TUBE)
        sleep(5)
        print "Pulse tube is %s. Beginning measurement."%(fridge.do_get_pulse_tube(),)
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    try:
        ys = {}
        for idx_power, power in enumerate(pows):
            vna.set_power(power-float(atten))
            this_avgs = AVGS[idx_power]
            this_time = (sweep_time * this_avgs * 1250.0) + np.size(pows)*NUM_POINTS*10 # ms
            # an approximation based on 10ms/pt for transfer, 25% overhead on waiting/capture
            vna.set_average_factor(this_avgs)
            ys[idx_power] = []
            if this_time < 30e3:                
                vna.set_timeout(this_time)
                ys[idx_power] = vna.do_get_data(fmt=fmt, opc=True, timeout=this_time)
            else:
                print "[NOTICE] Triggering each average."
                vna.set_timeout(sweep_time * 1250.0)
                ys[idx_power] = vna.do_get_data(fmt=fmt, opc=True, trig_each_avg=True, timeout=this_time)
            power=float(vna.get_power()+atten)
            try:
                temp=fridge.do_get_temperature()*1000
            except:
                temp=-1.0
            try:
                if display:
                    ax.plot(xs,ys[idx_power][0],label='%.2f dBm, %.2f mK'%(power,temp))
            except:
                pass
            if save:
                fname="%s%s%2.2fdB_%3.2fmK%s"%(folder,fpre,power,temp,fpost)
                data = np.transpose((xs,ys[idx_power][0],ys[idx_power][1]))
                np.savetxt(fname, data, fmt='%6.3f', delimiter='\t')
                print "Saved %s"%fname
                print "%.2f minutes remain."%(np.sum(AVGS[idx_power+1:])*sweep_time/60,)
        if display:
            ax.set_title('VNA Trace')
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.legend(loc='best')
    except Exception as e:
        print "EXCEPTION", e
    finally:
        if fridge:
            fridge.do_set_pulse_tube('ON')
        if revert:
            vna.set_instrument_state_file(_state)
            vna.load_state() # restore original state
        
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
    
    vna.set_center_freq(center)    
    vna.set_span(span)

    avg0 = avgs[0]
    avg1 = avgs[-1]
    
    pows, xs, ys = power_sweep(vna,pows,avgs_start=avg0,avgs_stop=avg1,save=False,display=display,
                      IF_BANDWIDTH=2000,NUM_POINTS=20001,revert=revert,sweep_type='LIN',fmt='UPH',
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

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    # if vna:
    #     collect_and_display(vna,save=False)
    sweep_time(10,491,3,1,29)
    # from mclient import instruments
    # vna = instruments['VNA']
    # x,y = tau_power_sweep(vna, np.linspace(-45,-15,10), [1,1], atten=-20)
    # data = power_sweep(vna,[-30,-40],IF_BANDWIDTH=2000,NUM_POINTS=501,
    #                                  PULSE_TUBE='ON',avgs_start=1,avgs_stop=1,save=False,
    #                                  display=True,atten=0,revert=False)
