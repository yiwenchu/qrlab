# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 10:54:10 2015

@author: JPC-Acquisition - Nick Frattini

Implement a JPC style flux sweep with a VNA and a Yoko
"""

import numpy as np
#import os
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib.colorbar as plt_colorbar
from time import sleep
from PyQt4 import QtGui
import h5py
from time import strftime
import subprocess
#from mclient import instruments
#import analysis_functions as af

DATA_LABEL = { 'PLOG': ('logmag','phase'),
              'SCOM': ('real', 'imag')}

# set the Room temperature switch via executable dll
def set_rtSwitch(exe_path=r"C:\Users\JPC-Acquisition\Desktop\Evan\Switch_Control\simple_exe_managed\ConsoleApplication1\ConsoleApplication1\bin\Debug\SimpleMCSwitcher.exe",
                 serial_num="11404230002", switch="A", port="1"):
    # exe_path: path to executable that Evan wrote for controlling the switch with the dll
    # serial_num: serial number of the Mini-Circuits switch to communicate with
    # switch: the switch to be set (ie A, B, C, etc)
    # port: the port of that switch that you want the Common to be set to (0 for the port labeled '1', or 1 for the port labeled '2'. Thank Evan for numbering.)
    return subprocess.check_output([exe_path, serial_num, switch, port])

def set_rtSwitch_S_SS():
    set_rtSwitch(switch="A",port="1")
    set_rtSwitch(switch="B",port="0")
    set_rtSwitch(switch="E",port="1")
    set_rtSwitch(switch="F",port="0")

def set_rtSwitch_S_II():
    set_rtSwitch(switch="A",port="1")
    set_rtSwitch(switch="B",port="1")
    set_rtSwitch(switch="E",port="1")
    set_rtSwitch(switch="F",port="1")
    # take mixer out of loop
#    set_rtSwitch(switch='D', port='0')
#    set_rtSwitch(switch='H', port='0')

def set_rtSwitch_S_SI():
    set_rtSwitch(switch="A",port="1")
    set_rtSwitch(switch="B",port="1")
    set_rtSwitch(switch="E",port="1")
    set_rtSwitch(switch="F",port="0")
    # put mixer in loop
#    set_rtSwitch(switch='D', port='1')
#    set_rtSwitch(switch='H', port='1')

def set_rtSwitch_S_IS():
    set_rtSwitch(switch="A",port="1")
    set_rtSwitch(switch="B",port="0")
    set_rtSwitch(switch="E",port="1")
    set_rtSwitch(switch="F",port="1")

def set_rtSwitch_QswitchS():
    set_rtSwitch(switch="A",port="0")
    set_rtSwitch(switch="C",port="0") #7
    set_rtSwitch(switch="E",port="0")
    set_rtSwitch(switch="G",port="0") #B
    
def set_rtSwitch_QswitchI():
    set_rtSwitch(switch="A",port="0")
    set_rtSwitch(switch="C",port="1") #11
    set_rtSwitch(switch="E",port="0")
    set_rtSwitch(switch="G",port="1") #A

def flux_sweep(vna, yoko, curr_start=-2.5e-3, curr_stop=2.5e-3, curr_step=1e-6, averages=10, num_points=1601,
               fridge=None,folder=None,display=True, atten=0,vna_fmt='SCOM', vna_sweep_type='LIN',
               printlog=True, h5file=None):
    # vna: must be valid VNA object
    # yoko: must be valid yoko object from mclient.instruments
    # curr_start: current sweep start point [A]
    # curr_stop: current sweep end point inclusive [A]
    # curr_step: current sweep step size [A] such that the number of current points = (curr_stop - curr_start) / curr_step + 1
    # averages: number of averages at each current
    # num_points: number of points in VNA frequency sweep
    # fridge: valid fridge object, None otherwise
    # folder: folder for saving text data file, None if saving undesired
    # save: True is saving via text files desired, False otherwise
    # display: True to display, False otherwise
    # atten: attenuation in dB of input line, negative if attenuation (e.g. -60 for 60 dB atten down fridge)
    # vna_fmt: format of data out from VNA, SCOM for real and imaginary, PLOG for logmag and phase
    # printlog: True to print status messages to console. False to prevent printing
    # h5file: full h5file path for saving, None if h5file saving undesired
    numCurr = round(abs(curr_stop - curr_start) / curr_step + 1)
    currents = np.linspace(curr_start, curr_stop, numCurr)
    
    tot_traces = averages * numCurr #total number of traces taken
    vna.set_points(num_points)
    vna.set_sweep_type(vna_sweep_type)
    vna.set_average_factor(averages)
    xs = vna.do_get_xaxis()
    
    if h5file:
        cur_group = strftime('%y%m%d')
        _iteration_start = strftime('%H%M%S')
        datadir = _iteration_start+'//'
    if printlog:
        print "Getting timing information and preparing trigger..."
    sleep(0.1) # why are we sleeping here??? ask Chris...
    
    if vna.get_sweep_type()=='SEGM':
        sweep_time = float(vna.get_segment_sweep_time())
    else:
        sweep_time = float(vna.get_sweep_time())
    tot_time = tot_traces * sweep_time + 1*numCurr                              #total time estimate for all traces
    if printlog:
        print "Total duration of this experiment will be %.2f minutes."%(tot_time/60,)
    
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111) 
    
    try:
        ys = np.zeros((numCurr, 2, num_points))     #initializes matrix of results with indices [i_current, real/imag, i_frequency]
        slew = 10e-6  #yoko slew in [A/s]
        for i_current, current in enumerate(currents):
            # tell yoko to go to current
            wait_time = yoko.set_current_ramp(current, slew=slew)
            if wait_time is not None:            
                sleep(wait_time + 1) #1 second for GPIB overhead
#                print(wait_time+1)
            # VNA take averages and data
            vna.set_averaging_state(True)
            vna.set_averaging_trigger(False)
            this_time = (sweep_time * averages * 1250.0) + np.size(currents)*num_points*10 # ms
            this_time = np.max(np.array([this_time, 5000.])) # 5s timeout minimum
            if this_time < 30e3:   
                # print this_time
                vna.set_timeout(this_time)
                ys[i_current,:,:] = vna.do_get_data(fmt=vna_fmt, opc=True, timeout=this_time)
            else:
                if printlog:
                    print "[NOTICE] Triggering each average."
                to = np.max(np.array([sweep_time * 1250.0, 5000.]))
                vna.set_timeout(to)
                # print to
                ys[i_current,:,:] = vna.do_get_data(fmt=vna_fmt, opc=True, trig_each_avg=True, timeout=this_time)
            vna.set_averaging_state(False)
            # plotting
            try:
                if display:
                    ax.plot(xs,ys[i_current,0,:],label='%.3f mA'%(current*1e3))
                    fig.canvas.draw()
                    QtGui.QApplication.processEvents()
            except:
                pass
        #end for loop
        if display:
            ax.set_title('VNA Trace')
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(DATA_LABEL[vna_fmt][0])
            ax.legend(loc='best')
        
        if h5file:
            h5f = h5py.File(h5file)
            g = h5f.require_group(cur_group)
            dg1 = g.create_dataset(datadir+'frequencies', data=xs)
            dg1.attrs.create('run_time', _iteration_start)            
            
            dg = g.create_dataset(datadir + DATA_LABEL[vna_fmt][0], data=ys[:,0,:])
            dg.attrs.create('run_time', _iteration_start)
            dg.attrs.create('attenuation',atten)
            dg.attrs.create('IFbandwidth',vna.get_if_bandwidth())
            dg.attrs.create('averages',vna.get_average_factor())
            dg.attrs.create('VNA_power',float(vna.get_power()))
            dg.attrs.create('measurement',str(vna.get_measurement()))
            dg.attrs.create('smoothing',vna.get_smoothing())
            dg.attrs.create('electrical_delay',vna.get_electrical_delay())
            dg.attrs.create('format',vna_fmt)
            if folder:
                dg.attrs.create('datafolder',folder)

            dg2 = g.create_dataset(datadir + DATA_LABEL[vna_fmt][1], data=ys[:,1,:])
            dg2.attrs.create('run_time', _iteration_start)
            dg_currs = g.create_dataset(datadir + 'currents', data=currents)
            dg_currs.attrs.create('run_time',_iteration_start)
            
            h5f.close()
        
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        if h5file:
            del(h5file)
    
    return (currents,xs,ys)

    
# takes power sweep data for 1 resonator in reflection and fits it for internal Q and coupling Q
def power_sweep_Qfit(h5filename, dataDir=None):
    # h5filename: full file path and name of h5file with data to be fit
    # dataDir: relative directory inside h5file where data is store (eg '/150914/144652'), if None then the most recent data will be used 
    
    h5file = h5py.File(h5filename)
    if dataDir:
        meas = h5file[dataDir]
    else:
        # use most recent data
        grp = h5file[h5file.keys()[-1]]
        meas = grp[grp.keys()[-1]]
    freqs = np.array(meas['frequencies'])
    powers = meas['VNA_power']
    #aout_complex = np.array(meas['magnitude']) * np.exp(np.pi / 180 * 1j*np.array(meas['phase'])) # assumes data taken was in PLIN form (lin mag and phase)
    aout_complex = np.array(meas['real']) + 1j*np.array(meas['imag'])     # assumes data was taken in SCOM form (real and imag)
    #import sys
    #sys.path.insert(0, 'Z:/Data/JPC_2015/2015-08-27 Cooldown')
    cut = False
    if cut:
        cutoff = 400
        aout_complex = aout_complex[(cutoff):]
        aout_complex = aout_complex[:-1*(cutoff)]
        freqs = freqs[(cutoff):]
        freqs = freqs[:-1*(cutoff)]
    # a_background takes an average of points off resonance to get a resonable initial guess for a_in
    a_background = (sum(aout_complex[0,0:50]) + sum(aout_complex[0,-50:])) / 100.0
#    import analysis_functions as af
    result = af.analyze_reflection(freqs, powers, aout_complex, f_0 = 8.9023e9, kc=11e6,ki=1e6, a_in=a_background, T=1e-12)
    res0 = np.array(result[0][:])
    Qc = np.divide(res0, np.array(result[1][:]))
    Qi = np.divide(res0, np.array(result[2][:]))
    Qtot = np.divide(1, (np.divide(1,Qc) + np.divide(1,Qi)))
    # must save Q values into h5 file
    h5file.close()
    return (Qc,Qi,Qtot,result)


# takes flux sweep data for either S or I of JPC in reflection and fits it for internal Q and coupling Q
def flux_sweep_Qfit(h5filename, dataDir=None, numFitPoints=0):
    # h5filename: full file path and name of h5file with data to be fit
    # dataDir: relative directory inside h5file where data is store (eg '/150914/144652'), if None then the most recent data will be used 
    # numFitPoints: number of points used in the fit on both sides of the guessed res freq (eg numFitPoints=10 uses 20 points for each fit centered at the res freq).
    #               If 0, then will use all points.    
    
    h5file = h5py.File(h5filename)
    if dataDir:
        meas = h5file[dataDir]
    else:
        # use most recent data
        print('Using most recent data')
        grp = h5file[h5file.keys()[-1]]
        meas = grp[grp.keys()[-1]]
    freqs = np.array(meas['frequencies'])
    currents = np.array(meas['currents'])
    aout_real = np.array(meas['real'])
    aout_complex = aout_real + 1j*np.array(meas['imag'])     # assumes data was taken in SCOM form (real and imag)
    
    Qc = np.zeros_like(currents)
    Qi = np.zeros_like(currents)
    Qtot = np.zeros_like(currents)
    results = np.zeros((currents.shape[0], 6)) # 6 because analyze_reflection has 5 fit params, but 1 is complex so must be split to real and imag
    for ind_curr, current in enumerate(currents):
        f_0_ind = np.argmin(aout_real[ind_curr,:]) # using the minimum real value to make inital res freq guess
        f_0 = freqs[f_0_ind]
        
        # slice around resonant frequency, numFitPoints both + and - side
        freqs_slice = freqs
        aout_slice = aout_complex[ind_curr,:]
        if numFitPoints!=0:
            aout_slice = aout_slice[(f_0_ind-numFitPoints):(f_0_ind+numFitPoints)]
            freqs_slice = freqs_slice[(f_0_ind-numFitPoints):(f_0_ind+numFitPoints)]
        # a_background takes an average of points off resonance to get a resonable initial guess for a_in
        a_background = (sum(aout_slice[0:2]) + sum(aout_slice[-2:])) / 4
        result = af.analyze_reflection(freqs_slice, current, aout_slice, f_0 = f_0, kc=1e6,ki=1e6, a_in=a_background, T=1e-12)
        results[ind_curr,:] = np.array((result[0][0],result[1][0],result[2][0],np.real(result[3][0]), np.imag(result[3][0]), result[4][0]))
        Qc[ind_curr] = results[ind_curr,0] / results[ind_curr,1]
        Qi[ind_curr] = results[ind_curr,0] / results[ind_curr,2]
        Qtot[ind_curr] = 1 / (1/Qc[ind_curr] + 1/Qi[ind_curr])
    h5file.close()
    return (currents,Qc,Qi,Qtot,results)


# plot res freq and Q's versus flux (or power)
def plot_Qfit(xdata, f0, Qc, Qi, Qtot, xdataType='Power'):
    # xdata: 1D array of either currents (in A) or powers (in dBm)
    # xdataType: either 'Power' or 'Current'
    # f0: 1D array of resonant frequncy fits
    # Qc: 1D array of coupling Q fits
    # Qi: 1D array of internal Q fits
    # Qtot: 1D array of total Q fits
    if xdataType == 'Power':
        xlabel = 'Power (dBm)'
    else:
        xlabel = 'Current (mA)'
        xdata = xdata * 1e3
    fig, ax = plt.subplots(4)
    ax[0].plot(xdata, f0*1e-9)
    ax[0].set_ylabel('f0 (GHz)')
    ax[1].plot(xdata, Qc)
    ax[1].set_ylabel('Qc')
    ax[2].plot(xdata, Qi)
    ax[2].set_ylabel('Qi')
    ax[3].plot(xdata, Qtot)
    ax[3].set_ylabel('Qtotal')
    ax[3].set_xlabel(xlabel)

# load data from h5 file from flux sweep stored 'frequencies', 'currents', 'real' and 'imag' and return it as a dictionary
def load_fluxData(h5filename, dataDir=None):
    # h5filename: full file path and name of h5file with data to be fit
    # dataDir: relative directory inside h5file where data is stored, if None then use the most recent data
    h5file = h5py.File(h5filename)
    if dataDir:
        meas = h5file[dataDir]
    else:
        # use most recent data
        print('Using most recent data')
        grp = h5file[h5file.keys()[-1]]
        meas = grp[grp.keys()[-1]]
    data = {'frequencies':np.array(meas['frequencies']),
            'currents':np.array(meas['currents']),
            'a_complex':(np.array(meas['real']) + 1j*np.array(meas['imag']))}
    h5file.close()
    return data

# find index of nearest value in array
def find_nearest(arr, value, last=False):
    if last:
        arr_abs = np.abs(arr-value)[::-1]
        return arr_abs.shape[0] - 1 - arr_abs.argmin()
    return (np.abs(arr-value)).argmin()
    
# plot a line cut of a flux sweep at a given current
def lineCut_flux(h5filename, dataDir=None, current=0, verticalStack=True):
    # h5filename: full file path and name of h5file with data to be fit
    # dataDir: relative directory inside h5file where data is stored, if None then use most recent data
    # current: current (A) at which the line cut should be taken
    data = load_fluxData(h5filename, dataDir)
    curr_ind = find_nearest(data['currents'], current)
    a_abs = 20 * np.log10(np.abs((data['a_complex'])[curr_ind,:]))
    a_phase = np.angle((data['a_complex'])[curr_ind,:], deg=True)
    if verticalStack:
        fig, axarr = plt.subplots(2, sharex = True)
    else:
        fig, axarr = plt.subplots(1,2)
        axarr[0].set_ylabel(r'Phase (deg)')
        fig.subplots_adjust(wspace=0.5)
    axarr[0].plot(1e-9*data['frequencies'], a_abs, 'bo')
    axarr[0].set_title('S21 at {0}mA'.format(1e3*(data['currents'])[curr_ind]))
    axarr[0].set_ylabel(r'Log Mag (dB)')
    axarr[1].plot(1e-9*data['frequencies'], a_phase, 'bo')
    axarr[1].set_yticks([-180, -90, 0, 90, 180])
    axarr[1].set_ylabel(r'Phase (deg)')
    width = 2.0
    for k in xrange(2):
        axarr[k].tick_params(width=width)
        for axis in ['top', 'bottom','left','right']:
            axarr[k].spines[axis].set_linewidth(width)
    
# plot color map from h5file data
def colorMap_h5file(h5filename, dataDir=None, colorMap=('hot','phase')):
    # h5filename: full file path and name of h5file with data to be fit
    # dataDir: relative directory inside h5file where data is store (eg '/150914/144652'), if None then the most recent data will be used
    # colorMap: tuple of two strings for color maps, 0th('hot') is for a_abs, 1st ('hsv' is predefined, 'phase' is Qulab color scheme) is for phase    
    #               NOTE: Must run define_phaseColorMap() once before using 'phase' color scheme or plt does not recognize 'phase'
    data = load_fluxData(h5filename, dataDir)
    a_abs = 20 * np.log10(np.abs(data['a_complex']))
    a_phase = np.angle(data['a_complex'], deg=True)
    colorMap_flux(data['currents'],data['frequencies'], a_abs, a_phase, colorMap=colorMap)
    

# plot color map, phase and linear amp, of flux sweep
def colorMap_flux(currents, freqs, a_abs, a_phase, colorMap=('hot','phase')):
    # currents: yoko currents for flux sweep (A)
    # freqs: frequencies of VNA for flux sweep
    # a_abs: 2D array indexed (current, freq) of log magnitude of complex response
    # a_phase: 2D array indexed (current, freq) of phase (degrees) of complex response
    # colorMap: tuple of two strings which point to color maps, 0th ('hot') is for a_abs, 1st ('phase' or 'hsv') is for phase
    freqs = 1e-9 * freqs                  # switch from Hz to GHz
    currents = 1e3 * currents
    a_abs = np.transpose(a_abs)
    a_phase = np.transpose(a_phase)
    fig, ax = plt.subplots(2)
    p0 = ax[0].pcolormesh(currents, freqs, a_abs, cmap=colorMap[0])
    cb0 = fig.colorbar(p0,ax=ax[0], label=r'${\rm Log Mag \,(dB)}$')
    ax[0].set_ylabel(r'${\rm Frequency \,(GHz)}$')
    ax[0].set_title('20 log|a_out|')
    p1 = ax[1].pcolormesh(currents, freqs, a_phase, cmap=colorMap[1])
    cb1 = fig.colorbar(p1, ax=ax[1], ticks=[-180, -90, 0, 90, 180], label=r'${\rm Phase \,(deg)}$')
    ax[1].set_ylabel(r'${\rm Frequency \,(GHz)}$')
    ax[1].set_title('Phase of a_out (deg)')
    ax[1].set_xlabel(r'${\rm Current \,(mA)}$')

# saves phase Color Map for flux sweeps to name 'phase' in plt ColorMaps, standard Qulab color scheme for phase
def define_phaseColorMap():
    # all numbers from Igor wave 'phaseindex'
    # Igor colors take RGB from 0 to 65535
    rgb = np.zeros((360,3), dtype=np.float)
    rgb[0:90,0] = np.arange(0, 63000, 700)
    rgb[90:180, 0] = 63000 * np.ones(90)
    rgb[180:270, 0] = np.arange(63000, 0, -700)
    rgb[90:180, 1] = np.arange(0, 63000, 700)
    rgb[180:270, 1] = 63000 * np.ones(90)
    rgb[270:360, 1] = np.arange(63000, 0, -700)
    rgb = rgb  / 65535.0
    # ListedColormap takes an arry of RGB weights normalized to be in [0,1]
    phase_cmap = plt_colors.ListedColormap(rgb, name='phase')
    plt.register_cmap(name='phase', cmap=phase_cmap)
    
    

# plot Qc and coupling Cap versus gap size
def plot_Cap_v_gap(gap, Qc, Cc):
    # gap: gap size [um]
    # Qc: coupling Q
    # Cc: calculated coupling capacitor (F)
    fig, ax = plt.subplots(2)
    ax[0].plot(gap, Qc, marker='D')
    ax[0].set_ylabel('Qcoupling')
    ax[1].plot(gap, Cc*1e15, marker='D')
    ax[1].set_ylabel('Ccoupling (fF)')
    ax[1].set_xlabel('Capacitor gap size (um)')


# calculate coupling Capactitance based on Qc for symmetrically coupled loads--Flavius's thesis pg 98
def calcCap(f0, Qc, R_L=50, Z0=50):
    # f0: 1D array of resonant frequencies (Hz)
    # Qc: 1D array of coupling Q's for resonators
    # R_L: symmetically coupled loads of impedance R_L (Ohm)
    # Z0: transmission line impedance (Ohm)
    # return Cc which is a 1D array of calculated coupling Capacitances (F)
    return np.multiply(np.sqrt((np.pi / 4) * (float(R_L) / float(Z0)) * np.divide(1,Qc)),  np.divide(1, 2*np.pi*R_L*f0))

# take dynamic range trace at each pump power in pump power sweep for both S_SS and S_II
def take_DR_Ppower(vna, ag, cw_freq, ag_pow_start=-20, ag_pow_stop=-15, ag_pow_step=0.01,
                   vna_pow_start=-80, vna_pow_stop=-40, vna_pow_step=0.1, vna_states=('Signal','Idler'),
                    averages=(100,50), printlog=True, h5file=None):
    numPow = round(abs(ag_pow_stop - ag_pow_start) / ag_pow_step + 1)
    ag_pows = np.linspace(ag_pow_start, ag_pow_stop, numPow)
    tot_traces = np.sum(averages) * numPow
    num_vnaPow = round(abs(vna_pow_start - vna_pow_stop) / vna_pow_step + 1)
    if h5file:
        cur_group = strftime('%y%m%d')
        _iteration_start = strftime('%H%M%S')
        dataDir = _iteration_start+'//'
    if printlog:
        print "Getting timing information and preparing trigger..."
    sleep(0.1) # why are we sleeping here??? ask Chris...
    try:
        ag.set_power(ag_pow_start)
        ag.set_rf_on(True)
        data_S = np.zeros((numPow, num_vnaPow))
        data_I = np.zeros((numPow, num_vnaPow))
        for i_power, ag_power in enumerate(ag_pows):
            ag.set_power(ag_power)
            # set VNA to signal params and take DR trace
            set_rtSwitch_S_SS()
            vna.load_state(vna_states[0])
            vna.set_trigger_source('INT')
            sleep(0.05)
            (vna_pows, data_S[i_power,:]) = take_dynamic_range(vna, cw_freq[0], pow_start=vna_pow_start, pow_stop=vna_pow_stop,
                                                        pow_step=vna_pow_step,avgs=averages[0], display=False)
            # set VNA to Idler params and take DR
            set_rtSwitch_S_II()
            vna.load_state(vna_states[1])
            vna.set_trigger_source('INT')
            sleep(0.05)
            (vna_pows, data_I[i_power,:]) = take_dynamic_range(vna, cw_freq[1], pow_start=vna_pow_start, pow_stop=vna_pow_stop,
                                                        pow_step=vna_pow_step,avgs=averages[1], display=False)
        # end ag_power for loop
        ag.set_rf_on(False)
        if h5file:
            h5f = h5py.File(h5file)
            g = h5f.require_group(cur_group)
            dg1 = g.create_dataset(dataDir + 'ag_powers', data=ag_pows)
            dg1.attrs.create('run_time', _iteration_start)
            dg2 = g.create_dataset(dataDir + 'vna_powers', data=vna_pows)
            dg2.attrs.create('run_time', _iteration_start)
            dg3 = g.create_dataset(dataDir + 'DRtrace_S', data=data_S)
            dg3.attrs.create('run_time', _iteration_start)
            dg3.attrs.create('cw_freq', cw_freq[0])
            dg4 = g.create_dataset(dataDir + 'DRtrace_I', data=data_I)
            dg4.attrs.create('run_time', _iteration_start)
            dg4.attrs.create('cw_freq', cw_freq[1])
            h5f.close()
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        if h5file:
            del(h5file)
    return (ag_pows, vna_pows, data_S, data_I)

# load data from h5file for pump power sweep of dynamic range
def load_DR_Ppow_data(h5filename, dataDir=None):
    h5file = h5py.File(h5filename)
    if dataDir:
        meas = h5file[dataDir]
    else:
        # use most recent data
        print('Using most recent data')
        grp = h5file[h5file.keys()[-1]]
        meas = grp[grp.keys()[-1]]
    keys = meas.keys()
    data = {'ag_powers':np.array(meas['ag_powers']),
            'vna_powers':np.array(meas['vna_powers']),
            'DRtrace_S':np.array(meas['DRtrace_S']),
            'DRtrace_I':np.array(meas['DRtrace_I']),
            'cw_freq_S':np.array(meas['DRtrace_S'].attrs['cw_freq']),
            'cw_freq_I':np.array(meas['DRtrace_I'].attrs['cw_freq'])}
    if 'P_1dB_S' in keys and 'P_1dB_I' in keys:
        data['P_1dB_S'] = np.array(meas['P_1dB_S'])
        data['P_1dB_I'] = np.array(meas['P_1dB_I'])
    h5file.close()
    return data

#  calc P_-1dB from a DR trace: takes an average of first 1/3 points for low power value
def calc_P1dB(vna_pows, dr_trace, display=True):
    num_pows = vna_pows.shape[0]
    i_ave = int(num_pows / 3)
    dr_lowP = np.average(dr_trace[0:i_ave])
    i_p1dB = i_ave + find_nearest(dr_trace[i_ave:], dr_lowP - 1, last=True) #finds the index of the P_-1dB point
    p1dB = vna_pows[i_p1dB]
    if display:
        fig, ax = plt.subplots()
        ax.plot(vna_pows, dr_trace, color='b', marker='D', fillstyle='none', ls='None', label='P_-1dB = %.1f'%(p1dB))
        ax.plot(vna_pows[0:i_ave], dr_lowP * np.ones_like(vna_pows[0:i_ave]), linewidth=4.0, color='b')
        ax.set_xlabel('VNA Power (dBm)')
        ax.set_ylabel('Return Power (dB)')
        ax.set_title('Dynamic Range Trace')
        ax.legend(loc='best')
    return (p1dB, dr_lowP, i_ave)

# calc P-1dB vs gain and save to h5file
def calc_P1dB_gain(h5filename, dataDir=None, recalc=False, atten=0):
    data = load_DR_Ppow_data(h5filename, dataDir=dataDir)
    keys = data.keys()
    if recalc or 'P_1dB_S' not in keys or 'P_1dB_I' not in keys:
        P_1dB_S = np.zeros_like(data['ag_powers'])
        P_1dB_I = np.zeros_like(P_1dB_S)
        fig, ax = plt.subplots(2)
        for i_pow, ag_pow in enumerate(data['ag_powers']):
            (P_1dB_S[i_pow], dr_lowP_S, i_cutS) = calc_P1dB(data['vna_powers'], data['DRtrace_S'][i_pow,:], display=False)
            (P_1dB_I[i_pow], dr_lowP_I, i_cutI) = calc_P1dB(data['vna_powers'], data['DRtrace_I'][i_pow,:], display=False)
            ax[0].plot(data['vna_powers']+atten, data['DRtrace_S'][i_pow,:], marker='D', fillstyle='none', ls='None', label='Pump %.2f dBm, P_-1dB = %.1f'%(ag_pow, P_1dB_S[i_pow]))
            ax[0].plot(data['vna_powers'][0:i_cutS]+atten, dr_lowP_S*np.ones_like(data['vna_powers'][0:i_cutS]), linewidth=4.0)
            ax[1].plot(data['vna_powers']+atten, data['DRtrace_I'][i_pow,:], marker='D', fillstyle='none', ls='None', label='Pump %.2f dBm, P_-1dB = %.1f'%(ag_pow, P_1dB_I[i_pow]))
            ax[1].plot(data['vna_powers'][0:i_cutI]+atten, dr_lowP_I*np.ones_like(data['vna_powers'][0:i_cutI]), linewidth=4.0)
        # end ag power for loop
        ax[0].set_xlabel('Input power (dBm)')
        ax[1].set_xlabel('Input power (dBm)')
        ax[0].set_ylabel('Return Power (dB)')
        ax[0].set_title('Signal Dynamic Range Trace')
        ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[1].set_ylabel('Return Power (dB)')
        ax[1].set_title('Idler Dynamic Range Trace')
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #save in h5file
        h5file = h5py.File(h5filename)
        if dataDir:
            meas = h5file[dataDir]
        else:
            # use most recent data
            print('Using most recent data')
            grp = h5file[h5file.keys()[-1]]
            meas = grp[grp.keys()[-1]]
        measkeys = meas.keys()
        if 'P_1dB_S' in measkeys:
            meas['P_1dB_S'][:] = P_1dB_S
            meas['P_1dB_I'][:] = P_1dB_I
        else:
            dg1 = meas.create_dataset('P_1dB_S', data=P_1dB_S)
            dg2 = meas.create_dataset('P_1dB_I', data=P_1dB_I)
        h5file.close()
    else:
        P_1dB_S = data['P_1dB_S']
        P_1dB_I = data['P_1dB_I']
    # done recalculating
    return (data['ag_powers'], P_1dB_S, P_1dB_I)

# plot P_-1dB vs Gain, cut is the number of values at the beginning of the data set to ignore due to data which has no 1 dB compression point
def plot_P1dB_gain(h5filename, dataDir_gain, dataDir_P1dB, recalc=False, atten=0, cut=0, save=None):
    (ag_pows, P_1dB_S, P_1dB_I) = calc_P1dB_gain(h5filename, dataDir=dataDir_P1dB, recalc=recalc, atten=atten)
    h5file = h5py.File(h5filename)
    try:
        meas = h5file[dataDir_gain]
        gainS = np.array(meas['maxgain_S'])
        gainI = np.array(meas['maxgain_I'])
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        h5file.close()
    pS = np.polyfit(gainS[cut:], P_1dB_S[cut:] + atten, 1)
    pfit_S = np.poly1d(pS)
    pI = np.polyfit(gainI[cut:], P_1dB_I[cut:] + atten, 1)
    pfit_I = np.poly1d(pI)
    fig, ax = plt.subplots()
#    ax.plot(gainS[cut:], P_1dB_S[cut:] + atten, color='r', marker='D', fillstyle='none', mew = 2.0, ls='None', label='Signal')
    ax.plot(gainI[cut:], P_1dB_I[cut:] + atten, color='b', marker='D', fillstyle='none', mew = 2.0, ls='None', label='Idler')
#    ax.plot(gainS[cut:], pfit_S(gainS[cut:]), color='r', linewidth=4.0)
    ax.plot(gainI[cut:], pfit_I(gainI[cut:]), color='b', linewidth=4.0)
    width = 2.0
    ax.tick_params(width=width)
    for axis in ['top', 'bottom','left','right']:
        ax.spines[axis].set_linewidth(width)
    ax.set_xticks([5,10, 15, 20, 25, 30, 35])
    ax.set_xticklabels(['',10,'',20,'',30])
    ax.set_xlabel(r'${\rm Gain \, (dB)}$')
    ax.set_ylabel(r'$P_{-1{\rm dB}} \, {\rm(dBm)}$')
#    ax.set_title(r'$P_{-{\rm 1dB}} \, {\rm vs \, Gain}$')
    ax.legend(loc='lower left')
    if save:
        plt.rcParams.update({'font.size':18})
        for k in xrange(2):
            fig.set_size_inches(9,6)
            fig.subplots_adjust(bottom=0.15, left=0.15)
            fig.savefig(save + r'\fig_P1db_gain', dpi=600)
    print 'the slopes are Signal %.1f and Idler %.1f'%(pS[0], pI[0])

# plot dynamic range trace
def plot_dynamic_range(h5filename, dataDir=None, atten=0, save=None):
    (powers, logmag) = load_dynamic_range_data(h5filename, dataDir)
    fig, ax = plt.subplots()
    ax.plot(powers + atten, logmag, color='g', linewidth=2.0)
    width = 2.0
    ax.tick_params(width=width)
    for axis in ['top', 'bottom','left','right']:
        ax.spines[axis].set_linewidth(width)
    if save:
        ax.set_ylim([-2, 14])
#        ax.set_xticks([-160, -150, -140, -130, -120, -110])
        xticks = ax.get_xticks()
        xticks = ['%.0f'%(x) for x in xticks]
        xticks[::2] = ['' for x in xticks[::2]]
        ax.set_xticklabels(xticks)
        yticks = ax.get_yticks()
        yticks = ['%.0f'%(y) for y in yticks]
        yticks[::2] = ['' for y in yticks[::2]]
        ax.set_yticklabels(yticks)
        plt.rcParams.update({'font.size':18})
        fig.set_size_inches(3,2)
        fig.subplots_adjust(bottom=0.15, left=0.15)
        fig.savefig(save + r'\fig_DRtrace', dpi=600)

# load dynamic range trace data
def load_dynamic_range_data(h5filename, dataDir=None):
    h5file = h5py.File(h5filename)
    try:
        if dataDir:
            meas = h5file[dataDir]
        else:
            # use most recent data
            print('Using most recent data')
            grp = h5file[h5file.keys()[-1]]
            meas = grp[grp.keys()[-1]]
        powers = np.array(meas['powers'])
        logmag = np.array(meas['logmag'])
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        h5file.close()
        del(h5file)
    return (powers, logmag)
        
# take a dynamic range trace at a given CW frequency over given power span
def take_dynamic_range(vna, cw_freq, pow_start=-80, pow_stop=-30, pow_step=0.1, revert=True,
                       avgs=4,vna_fmt='PLOG', display=True, atten=0, h5file=None):
    # vna: must be valid VNA object
    # cw_freq: frequency [Hz] at which to do a power sweep
    # pow_start: lowest VNA power [dBm]
    # pow_stop: highest VNA power [dBm]
    # pow_step: power step [dBm]
    # revert: True to revert VNA to previous state, false otherwise
    # avgs: num of averages
    # vna_fmt: valid VNA format (usually 'PLOG') for log mag
    # display: True to plot at end, False otherwise
    # atten: attenuation in dB of input line, negative if attenuation (e.g. -60 for 60 dB atten down fridge)
    # h5file: full file path to h5file for data to be stored, None if no saving desired
    if revert:
        _state = 'python_temp'
        vna.set_instrument_state_data('CST') #settings and calibration
        vna.set_instrument_state_file(_state)
        vna.save_state()
        
    try:
        vna.set_sweep_type('POW')
        vna.set_cw_freq(cw_freq)
        vna.set_average_factor(avgs)
        num_pow = float(pow_stop - pow_start) / pow_step + 1
        pows = np.linspace(pow_start, pow_stop, num_pow)
        data = np.zeros_like(pows)    
        vna.set_stop_pow(pow_stop)    
        vna.set_start_pow(pow_start)
        sleep(0.1)
        sweep_time = float(vna.get_sweep_time())
        pow_index = 0
        while pow_stop > pows[pow_index]:
            vna.set_stop_pow(pow_stop)    
            vna.set_start_pow(pows[pow_index])
            sleep(0.1) # to let vna set params before asking for them, hopefully prevents timeout errors
            tmp_pow_stop = vna.get_stop_pow() # tmp_pow_stop will be <= pow_stop
            tmp_num_points = float(tmp_pow_stop - pows[pow_index]) / pow_step + 1
            vna.set_points(tmp_num_points)
            vna.set_averaging_state(True)
            vna.set_averaging_trigger(False)
            # wait for averaging
            this_time = (sweep_time*avgs*1250.0) + tmp_num_points*10 #ms
            this_time = np.max(np.array([this_time, 5000.])) # 5s timeout minimum
            to = np.max(np.array([sweep_time * 1250.0, 5000.]))
            vna.set_timeout(to)
            # get data chunck
            trace = vna.do_get_data(fmt=vna_fmt, opc=True, trig_each_avg=True, timeout=this_time)
            data[pow_index:(pow_index+tmp_num_points)] = trace[0]
            #update iterator        
            pow_index = pow_index + tmp_num_points - 1
        
        if h5file:
            h5f = h5py.File(h5file)
            cur_group = strftime('%y%m%d')
            _iteration_start = strftime('%H%M%S')
            datadir = _iteration_start+'//'   
            g = h5f.require_group(cur_group)
            dg = g.create_dataset(datadir+DATA_LABEL[vna_fmt][0], data=data)
            dg.attrs.create('run_time', _iteration_start)
            dg.attrs.create('averages',vna.get_average_factor())
            dg.attrs.create('IFbandwidth',vna.get_if_bandwidth())
            dg.attrs.create('smoothing',vna.get_smoothing())
            dg.attrs.create('electrical_delay',vna.get_electrical_delay())
            dg.attrs.create('format',vna_fmt)
            dg1 = g.create_dataset(datadir+'powers', data=pows)
            dg1.attrs.create('run_time', _iteration_start)
            h5f.close()
            del(h5f)
        if display:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(pows, data,'bs-', label='%.5f GHz CW frequency'%(1e-9*cw_freq))
            ax.set_title('Dynamic Range Trace')
            ax.set_xlabel('Input Power (dBm)')
            ax.set_ylabel('Output Power (dB)')
            ax.legend(loc='best')
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        if revert:
            try:
                vna.set_instrument_state_file(_state)
                vna.load_state() # restore original state
            except:
                print "[NOTICE] VNA failed to return to initial state"
                pass
    return (pows,data) # returns the input powers and output powers

# take 2D pump power and pump frequency sweeps
def take_2Dpump_sweep(vna, ag, spec=None, pow_start=-20, pow_stop=-15, pow_step= 0.01, averages=(100,50),
                      freq_start=14e9, freq_stop=14e9, freq_step=1e6, num_points=1601, vna_states=('Signal','Idler'),
                        change_rtSwitch=(set_rtSwitch_S_SS, set_rtSwitch_S_II), display=True, atten=0, vna_fmt='PLOG', h5file=None):
    numfreq = round(abs(freq_stop - freq_start) / freq_step + 1)
    pump_freqs = np.linspace(freq_start, freq_stop, numfreq)
    data = {'f_pump' : pump_freqs}
    for i_freq, freq in enumerate(pump_freqs):
        ag.set_frequency(freq)
        (pows, freqs_S, data_S, freqs_I, data_I) = take_Ppow_sweep(vna, ag, spec=spec, pow_start=pow_start, pow_stop=pow_stop, pow_step=pow_step,
                                                                    averages=averages, num_points=num_points, vna_states=vna_states, 
                                                                    change_rtSwitch=change_rtSwitch, display=display, atten=atten, vna_fmt=vna_fmt, h5file=h5file)
        data['pows'] = pows
        data[freq] = {'freqs_S': freqs_S,
                        'data_S': data_S,
                        'freqs_I': freqs_I,
                        'data_I': data_I }
    return data


# take pump power sweep, record Gain curves and NVR on both S_SS and S_II
def take_Ppow_sweep(vna, ag, spec=None, pow_start=-20, pow_stop=-15, pow_step=0.01, averages=(100,50),
                    num_points=1601, vna_states=('Signal','Idler'), change_rtSwitch=(set_rtSwitch_S_SS, set_rtSwitch_S_II),
                    display=True, atten=0, vna_fmt='PLOG', phase_save=False, printlog=True, h5file=None):
    # vna: must be valid VNA object
    # ag: must be valid agilent generator object
    # spec: must be valid spectrum analyzer object, if None then assume spectrum analyzer is not connected.
    # pow_start: power sweep start point [dBm] for Agilent generator
    # pow_stop: power sweep stop point [dBm]
    # pow_step: power sweep step size [dBm]
    # averages: number of VNA averages at each current
    # num_points: number of points in VNA sweep
    # vna_states: tuple of the names of VNA states to be loaded for Signal and Idler
    # display: True if sweep results are plotted, False otherwise
    # atten: attenuation in dB of input line, negative if attenuation (e.g. -60 dB)
    # vna_fmt: format of data out of VNA, PLOG for logmag and phase
    # phase_save: True to save phase data in h5file, False if not (False for normal gain curves, true for stark shift measurments)
    # printlog: True to print status messages to console. False to prevent printing
    # h5file: full h5file path for saving, None is h5file saving undesired
    ag_freq = ag.get_frequency()
    numPow = round(abs(pow_stop - pow_start) / pow_step + 1)
    pows = np.linspace(pow_start, pow_stop, numPow)
    tot_traces = np.sum(averages) * numPow #total num of traces taken
    
    change_rtSwitch[0]() # sets room temp switch to S_SS configuration
    vna.load_state(vna_states[0])
    freqs_S = vna.do_get_xaxis()
    vna.set_trigger_source('INT')
    f_S = freqs_S[round(freqs_S.shape[0] / 2)] #center freq for spectrum analyzer
    if spec:
        # get background traces with pump off
        ag.set_rf_on(False)
        spec_reg = {'S':0, 'I':1} # map to register number on spectrum analyzer where state will be stored
        AVG = 1000 # spec averages
        timeout = AVG * 60 # ms...gives 60 s timeout for 1000 averages. 
        spec.do_set_timeout(timeout)
        spec.set_span(100e6)
        spec.set_center_freq(f_S)
        spec.set_average_factor(AVG)
        spec.set_averaging_state(True)
        resBW_S = spec.get_if_bandwidth()
        spec.save_state(spec_reg['S'])
        sleep(2)
        spec.write('INIT:CONT 0') # selects single sweeping
        spec.single_meas(timeout=timeout)
        (spec_freqs_S, spec_pOff_S) = spec.do_get_data()
    change_rtSwitch[1]()
    vna.load_state(vna_states[1])
    freqs_I = vna.do_get_xaxis()
    vna.set_trigger_source('INT')
    f_I = freqs_I[round(freqs_I.shape[0] / 2)]
    if spec:
        spec.set_center_freq(f_I)
        sleep(2)
        spec.single_meas(timeout=timeout)
        (spec_freqs_I, spec_pOff_I) = spec.do_get_data()
        spec.write('INIT:CONT 1')
        resBW_I = spec.get_if_bandwidth()
        spec.save_state(spec_reg['I'])
        # initialize matrices for spec data with indices [i_pow, i_specFrequency]
        specdata_S = np.zeros((numPow, spec_pOff_S.shape[0]))
        specdata_I = np.zeros((numPow, spec_pOff_I.shape[0]))

    if h5file:
        cur_group = strftime('%y%m%d')
        _iteration_start = strftime('%H%M%S')
        datadir = _iteration_start+'//'
    if printlog:
        print "Getting timing information and preparing trigger..."
    sleep(0.1) # why are we sleeping here??? ask Chris...
    sweep_time = float(vna.get_sweep_time())
    tot_time = (tot_traces * sweep_time + 1*numPow)
    if printlog:
        print "Total duration of this experiment will be %.2f minutes."%(tot_time/60,)
    if display:
        fig, axarr = plt.subplots(2)
        if spec:
            fig2, axarr2 = plt.subplots(2)
    try:
        ag.set_power(pow_start)
        ag.set_rf_on(True)
        # initializes matrix of Signal and Idler results with indices [i_pow, logmag/phase, i_frequency]    
        data_S = np.zeros((numPow, 2, num_points))
        data_I = np.zeros((numPow, 2, num_points))
        for i_power, power in enumerate(pows):
            # Set Agilent generator to proper pump power
            ag.set_power(power)

            # set VNA to Signal params, average, take Signal trace
            change_rtSwitch[0]()
            vna.load_state(vna_states[0])
            vna.set_trigger_source('INT') # loading a state with memory included sets the trigger to Manual, so we set it back to Internal
            vna.set_points(num_points)
            vna.set_average_factor(averages[0])
            if spec:            
                spec.load_state(spec_reg['S']) # load state of spectrum analyzer
                spec.write('INIT:CONT 0') # selects single sweeping
                spec.write('INIT:IMM') # starts a single measurement
            sleep(0.1)
            vna.set_averaging_state(True)
            vna.set_averaging_trigger(False)
            this_time = (sweep_time * averages[0] * 1250.0) + np.size(pows)*num_points*10 # ms
            this_time = np.max(np.array([this_time, 5000.])) # 5s timeout minimum
            to = np.max(np.array([sweep_time*1250.0, 5000.]))
            vna.set_timeout(to)
            data_S[i_power,:,:] = vna.do_get_data(fmt=vna_fmt, opc=True, trig_each_avg=True, timeout=this_time)
            vna.set_averaging_state(False)
            if spec:
                spec.ask('*OPC?') # waits until spectrum single measurement operationi is complete
                specdata_S[i_power,:] = spec.do_get_yaxes()
                spec.write('INIT:CONT 1')
            
            # set VNA to Idler params, average, take trace
            change_rtSwitch[1]()
            vna.load_state(vna_states[1])
            vna.set_trigger_source('INT') # loading a state with memory included sets the trigger to Manual, so we set it back to Internal
            vna.set_points(num_points)
            vna.set_average_factor(averages[1])
            if spec:            
                spec.load_state(spec_reg['I']) # load state of spectrum analyzer
                spec.write('INIT:CONT 0') # selects single sweeping
                spec.write('INIT:IMM') # starts a single measurement
            sleep(0.1)
            vna.set_averaging_state(True)
            vna.set_averaging_trigger(False)
            this_time = (sweep_time * averages[1] * 1250.0) + np.size(pows)*num_points*10 # ms
            this_time = np.max(np.array([this_time, 5000.])) # 5s timeout minimum
            to = np.max(np.array([sweep_time*1250.0, 5000.]))
            vna.set_timeout(to)
            data_I[i_power,:,:] = vna.do_get_data(fmt=vna_fmt, opc=True, trig_each_avg=True, timeout=this_time)
            vna.set_averaging_state(False)
            if spec:
                spec.ask('*OPC?') # waits until spectrum single measurement operationi is complete
                specdata_I[i_power,:] = spec.do_get_yaxes()
                spec.write('INIT:CONT 1')
            
            # plotting
            try:
                if display:
                    axarr[0].plot(freqs_S*1e-9, data_S[i_power,0,:], label='%.2f dBm'%(power))
                    axarr[1].plot(freqs_I*1e-9, data_I[i_power,0,:], label='%.2f dBm'%(power))
                    if spec:
                        axarr2[0].plot(spec_freqs_S*1e-9, specdata_S[i_power,:], label='%.2f dBm'%(power))
                        axarr2[1].plot(spec_freqs_I*1e-9, specdata_I[i_power,:], label='%.2f dBm'%(power))
                    fig.canvas.draw()
                    QtGui.QApplication.processEvents()
            except:
                pass
        # end power for loop
        ag.set_rf_on(False)
        if h5file:
            h5f = h5py.File(h5file)
            g = h5f.require_group(cur_group)
            dg1 = g.create_dataset(datadir+vna_states[0]+'/frequencies', data=freqs_S)
            dg1.attrs.create('run_time', _iteration_start)
            dg3 = g.create_dataset(datadir+vna_states[1]+'/frequencies', data=freqs_I)
            dg3.attrs.create('run_time', _iteration_start)
            
            dg = g.create_dataset(datadir+vna_states[0]+'/'+ DATA_LABEL[vna_fmt][0], data=data_S[:,0,:])
            dg.attrs.create('run_time', _iteration_start)
            dg.attrs.create('attenuation',atten)
            dg.attrs.create('IFbandwidth',vna.get_if_bandwidth())
            dg.attrs.create('averages',averages[0])
            dg.attrs.create('VNA_power',float(vna.get_power()))
            dg.attrs.create('measurement',str(vna.get_measurement()))
            dg.attrs.create('smoothing',vna.get_smoothing())
            dg.attrs.create('format',vna_fmt)
            
            dg2 = g.create_dataset(datadir+vna_states[1]+'/'+ DATA_LABEL[vna_fmt][0], data=data_I[:,0,:])
            dg2.attrs.create('run_time', _iteration_start)
            dg2.attrs.create('averages',averages[1])
            
            dg_pows = g.create_dataset(datadir + 'powers', data=pows)
            dg_pows.attrs.create('run_time', _iteration_start)
            dg_pows.attrs.create('pump_frequency', ag_freq)
            
            if phase_save:
                dg_ph = g.create_dataset(datadir+vna_states[0]+'/'+ DATA_LABEL[vna_fmt][1], data=data_S[:,1,:])
                dg_ph.attrs.create('run_time', _iteration_start)
                dg_ph2 = g.create_dataset(datadir+vna_states[1]+'/'+ DATA_LABEL[vna_fmt][1], data=data_I[:,1,:])
                dg_ph2.attrs.create('run_time', _iteration_start)
            
            if spec:
                dg4 = g.create_dataset(datadir+vna_states[0]+'/NVRcurve', data=specdata_S)
                dg4.attrs.create('run_time', _iteration_start)
                dg4.attrs.create('spec_resBW', resBW_S)
                dg5 = g.create_dataset(datadir+vna_states[1]+'/NVRcurve', data=specdata_I)
                dg5.attrs.create('run_time', _iteration_start)
                dg5.attrs.create('spec_resBW', resBW_I)
                dg6 = g.create_dataset(datadir+vna_states[0]+'/NVRfreqs', data=spec_freqs_S)
                dg6.attrs.create('run_time', _iteration_start)
                dg7 = g.create_dataset(datadir+vna_states[1]+'/NVRfreqs', data=spec_freqs_I)
                dg7.attrs.create('run_time', _iteration_start)
                dg8 = g.create_dataset(datadir+vna_states[0]+'/NVRpoff', data=spec_pOff_S)
                dg8.attrs.create('run_time', _iteration_start)
                dg9 = g.create_dataset(datadir+vna_states[1]+'/NVRpoff', data=spec_pOff_I)
                dg9.attrs.create('run_time', _iteration_start)
            h5f.close()
        if display:
            axarr[0].set_title(vna_states[0] + ' Gain Curves, f_pump = %.5f'%(ag_freq*1e-9))
            axarr[0].set_xlabel('Probe Frequency (GHz)')
            axarr[0].set_ylabel(DATA_LABEL[vna_fmt][0])
            axarr[0].legend(loc='best')
            axarr[1].set_title(vna_states[1] + ' Gain Curves')
            axarr[1].set_xlabel('Probe Frequency (GHz)')
            axarr[1].set_ylabel(DATA_LABEL[vna_fmt][0])
            axarr[1].legend(loc='best')
            if spec:
                axarr2[0].plot(spec_freqs_S*1e-9, spec_pOff_S, label='Pump Off')
                axarr2[1].plot(spec_freqs_I*1e-9, spec_pOff_I, label='Pump Off')
                axarr2[0].set_title(vna_states[0] + ' NVR Curves')
                axarr2[0].set_xlabel('Probe Frequency (GHz)')
                axarr2[0].set_ylabel('NVR (dB)')
                axarr2[0].legend(loc='best')
                axarr2[1].set_title(vna_states[1] + ' NVR Curves')
                axarr2[1].set_xlabel('Probe Frequency (GHz)')
                axarr2[1].set_ylabel('NVR (dB)')
                axarr2[1].legend(loc='best')
                
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        if h5file:
            del(h5file)
    return (pows, freqs_S, data_S[:,0,:], freqs_I, data_I[:,0,:])

# takes a single set vna traces, record Gain curves and NVR on both S_SS and S_II


#plot the pump power sweep data as gain curves
def plot_Ppow_sweep_data(h5filename, dataDir=None, vna_states=('Signal', 'Idler'), nvr=True, skip=1, save=None):
    (vnadata, specdata) = load_Ppow_sweep_data(h5filename, dataDir=dataDir, vna_states=vna_states, nvr=nvr)
    colormap = plt.get_cmap('rainbow')
    powers = vnadata['powers'][::skip]
    num_pows = powers.size
    si = ('S','I')
    figS, axS = plt.subplots(2,1, sharex=nvr)
    if nvr:
        figS.subplots_adjust(hspace=0.0)
    axS[0].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_pows)])
    axS[1].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_pows)])
    figI, axI = plt.subplots(2,1, sharex=nvr)
    if nvr:
        figI.subplots_adjust(hspace=0.0)
    axI[0].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_pows)])
    axI[1].set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_pows)])
    figs = (figS, figI)
    axSI = (axS, axI)
    width=2.0
    for k in xrange(2):
        fig = figs[k]
        ax = axSI[k]
        ax[0].set_title(vna_states[k])
        f_g = vnadata['freqs_'+si[k]] * 1e-9
        gain = vnadata['gain_'+si[k]][::skip, :]
        if nvr:
            f_n = specdata['freqs_'+si[k]] * 1e-9
            nvr_data = specdata['p_on_'+si[k]][::skip, :]
            nvr_base = specdata['p_off_'+si[k]]
        for i_pow, power in enumerate(powers):
            ax[0].plot(f_g, gain[i_pow,:])
            if nvr:
                ax[1].plot(f_n, nvr_data[i_pow,:] - nvr_base)
        ax[0].tick_params(width=width)
        ax[1].tick_params(width=width)
        for axis in ['top', 'bottom','left','right']:
            ax[0].spines[axis].set_linewidth(width)
            ax[1].spines[axis].set_linewidth(width)
#        ax[0].set_ylim([0, 32])
#        ax[1].set_ylim([-1, 18])
#        ax[0].set_yticks(np.arange(4)*10)
#        ax[1].set_yticks(np.arange(4)*5)
#        fig.colorbar()
#    axI[1].set_xlim([7.65,7.74])
#    axI[1].set_xticks([7.66, 7.68, 7.70, 7.72, 7.74])
    cfig, cax = plt.subplots()
    norm = plt_colors.Normalize(vmin=powers[0], vmax=powers[-1])
    cb = plt_colorbar.ColorbarBase(cax, cmap=colormap, norm=norm, ticks=[5,6,7,8])
    cb.set_label(r'${\rm Pump\, power\, (dBm)}$')
    cax.tick_params(width=width)
    for axis in ['top', 'bottom','left','right']:
        cax.spines[axis].set_linewidth(width)
    if save:
        plt.rcParams.update({'font.size':18})
        for k in xrange(2):
            figs[k].set_size_inches(8,6)
            figs[k].subplots_adjust(bottom=0.15)
            figs[k].savefig(save + r'\fig_gainNVR_' + si[k], dpi=600)
        cfig.set_size_inches(1,3)
        cfig.subplots_adjust(right=0.5)
        cfig.savefig(save + r'\fig_gainNVR_colorbar', dpi=600)

# plot 2D pump power sweep as 2D heat maps, plots all sweeps in between and including dataDir1 and dataDirL
def plot_2DPpow_sweep_data_2D(h5filename, dataDir1, dataDirL, vna_states=('Signal', 'Idler'), nvr=True,
                              skip=1, cmap = ('hot', 'hot'), save=None):
    # dataDir1: first data directory/Ppow sweep in h5file to plot (eg '/160223/121212')
    # dataDirL: last data directory in h5file to plot
    dir1 = dataDir1.split('/')  # should be ['', '160223', '121212']
    dirL = dataDirL.split('/')
    dataDirs = []
    p_freqs = []
    h5f = h5py.File(h5filename)
    try:
        grp = h5f[dir1[1]]
        times = grp.keys()
        i_1 = times.index(dir1[2])
        if dir1[1] != dirL[1]: # when data was taken on the 2 days (overnight sweep)
            times = times[i_1:]
            dataDirs += ['/' + dir1[1] + '/' + time for time in times]
            p_freqs += [grp[time]['powers'].attrs['pump_frequency'] for time in times]
            grp = h5f[dirL[1]]
            times = grp.keys()
            i_1 = 0
        i_L = times.index(dirL[2])
        times = times[i_1 : i_L+1] # only select the times between the two specified by dir1 and dirL
        dataDirs += ['/' + dirL[1] + '/' + time for time in times]  #adds the full dataDir to the dataDirs list
        p_freqs += [grp[time]['powers'].attrs['pump_frequency'] for time in times]
    except Exception as e:
        print "EXCEPTION", e
        raise
    finally:
        h5f.close()
        del(h5f)
    # iterate through dataDirs and plot
    for i, dataDir in enumerate(dataDirs):
        print('\r ploting #' + str(i))
        plot_Ppow_sweep_data_2D(h5filename, dataDir=dataDir, vna_states=vna_states, nvr=nvr, skip=skip,
                                f_pump=p_freqs[i], colorMap=cmap, save=save)
        

# plot pump power sweep data as a 2D image  
def plot_Ppow_sweep_data_2D(h5filename, dataDir=None, vna_states=('Signal', 'Idler'), nvr=True, skip=1,
                            f_pump = 14.434e9, colorMap = ('rainbow', 'rainbow'), save=None, phase_save=False):
    (vnadata, specdata) = load_Ppow_sweep_data(h5filename, dataDir=dataDir, vna_states=vna_states, nvr=nvr, phase_save=phase_save)
    si = ('S','I') # (poor) naming scheme comes from the load_Ppow_sweep_data function, just separates btwn vna_states
    rb = ('r','b')
    powers = vnadata['powers'][::skip]
    num_pows = powers.size
    if nvr:
        figS, axS = plt.subplots(2,1, sharex=True)
        figI, axI = plt.subplots(2,1, sharex=True)
    else:
        figS, axS = plt.subplots(1,1)
        axS = (axS, None)
        figI, axI = plt.subplots(1,1)
        axI = (axI, None)
    if phase_save:
        fig_phS, ax_phS = plt.subplots(1,1)
        fig_phI, ax_phI = plt.subplots(1,1)
        figs_ph = (fig_phS, fig_phI)
        axs_ph = (ax_phS, ax_phI)
        fig_res, ax_resS = plt.subplots(1,1)
#        ax_resI = ax_resS.twinx()
        ax_res =(ax_resS, ax_resS)
        pSI = [0, 0]
    figs = (figS, figI)
    axSI = (axS, axI)
    width=2.0
    for k in xrange(2):
        fig = figs[k]
        ax = axSI[k]
        f_g = vnadata['freqs_'+si[k]] * 1e-9
        gain = vnadata['gain_'+si[k]][::skip, :]
        p0 = ax[0].pcolormesh(f_g, powers, gain, cmap=colorMap[0])
        cb0 = fig.colorbar(p0, ax=ax[0], label=r'${\rm Gain \,(dB)}$')
        ax[0].set_title(vna_states[k] + ' G/C, f_pump = %.5f GHz'%(f_pump*1e-9))
        ax[0].tick_params(width=width)
        if nvr:
            f_n = specdata['freqs_'+si[k]] * 1e-9
            nvr_data = specdata['p_on_'+si[k]][::skip, :]
            nvr_base = specdata['p_off_'+si[k]]
            nvr_tot = nvr_data - np.transpose(nvr_base)
            p1 = ax[1].pcolormesh(f_n, powers, nvr_tot, cmap=colorMap[1])
            cb1 = fig.colorbar(p1,ax=ax[1], label=r'$NVR \, {\rm (dB)}$')
            ax[1].tick_params(width=width)
        if phase_save:
            fig_ph = figs_ph[k]
            ax_ph = axs_ph[k]
            phase = vnadata['phase_'+si[k]][::skip,:]
            p1 = ax_ph.pcolormesh(f_g, powers, phase, cmap='phase')
            cb1 = fig_ph.colorbar(p1, ax=ax_ph, label=r'${\rm Phase \,(Deg)}$')
            ax_ph.set_title(vna_states[k] + ' G/C, f_pump = %.5f GHz'%(f_pump*1e-9))
            ax_ph.tick_params(width=width)
            resFreqs = vnadata['resFreqs_'+si[k]][::skip]
            starkShifts = (resFreqs - resFreqs[0])* 1e-6
            linPowers = np.power(10,powers/10)
            pcoeff = np.polyfit(linPowers, starkShifts, 1) #[num_pows/2:]
            pfit = np.poly1d(pcoeff)
            ax_res[k].plot(linPowers, starkShifts, color=rb[k], marker='D', fillstyle='none', mew = 2.0, ls='none', label=vna_states[k])
            ax_res[k].plot(linPowers, pfit(linPowers), color=rb[k], linewidth=3.0, label='Fit Slope %.4f MHz/mW'%(pcoeff[0]))
#            ax_res[k].set_ylabel(vna_states[k] + 'Frequency Shift (MHz)', color=rb[k])
            ax_res[k].tick_params(width=width)
        for axis in ['top', 'bottom','left','right']:
            ax[0].spines[axis].set_linewidth(width)
            if nvr:
                ax[1].spines[axis].set_linewidth(width)
            if phase_save:
                ax_ph.spines[axis].set_linewidth(width)
                ax_res[k].spines[axis].set_linewidth(width)
        if save:
            plt.rcParams.update({'font.size':18})
            figs[k].set_size_inches(8,6)
            figs[k].subplots_adjust(bottom=0.15)
            figs[k].savefig(save + r'\fig_gainNVR_' + si[k], dpi=600)
    if phase_save:
        ax_res[0].legend(loc='best')
        ax_res[0].set_ylabel('Frequency Shift (MHz)')
        ax_res[0].set_xlabel('Generator Power (mW)')
        ax_res[0].set_title('Stark Shifts, f_pump = %.5f GHz'%(f_pump*1e-9))

# load data from h5 file from a pump power sweep stored and return as dictionary
def load_Ppow_sweep_data(h5filename, dataDir=None, vna_states=('Signal','Idler'), nvr=True, phase_save=False):
    # h5filename: full file path and name of h5file with data to be fit
    # dataDir: relative directory inside h5file where data is stored, if None then use most recent data
    # vna_states: names of VNA states that serve as another folder underneath dataDir
    # spec: True if NVR data from spectrum analyzer should be loaded.
    # phase_save: True if phase is saved in h5file, same option as fo take_Ppow_sweep
    h5file = h5py.File(h5filename)
    if dataDir:
        meas = h5file[dataDir]
    else:
        # use most recent data
        print('Using most recent data')
        grp = h5file[h5file.keys()[-1]]
        meas = grp[grp.keys()[-1]]
    vnadata = {'freqs_S':np.array(meas[vna_states[0] +'/frequencies']),
               'freqs_I':np.array(meas[vna_states[1] +'/frequencies']),
                'gain_S':np.array(meas[vna_states[0] + '/logmag']),
                'gain_I':np.array(meas[vna_states[1] + '/logmag']),
                'powers':np.array(meas['powers'])}
#    'phase_S':np.array(meas[vna_states[0] + '/phase']),
#    'phase_I':np.array(meas[vna_states[1] + '/phase']),
    if phase_save:
        vnadata['phase_S'] = np.array(meas[vna_states[0] + '/phase'])
        vnadata['phase_I'] = np.array(meas[vna_states[1] + '/phase'])
        vnadata['resFreqs_S'] = calc_resShift(vnadata['phase_S'], vnadata['freqs_S'])
        vnadata['resFreqs_I'] = calc_resShift(vnadata['phase_I'], vnadata['freqs_I'])
    keys = meas.keys()
    if 'maxgain_S' in keys and 'f0_S' in keys and 'BW_S' in keys:
        vnadata['maxgain_S'] = np.array(meas['maxgain_S'])
        vnadata['f0_S'] = np.array(meas['f0_S'])
        vnadata['BW_S'] = np.array(meas['BW_S'])
    if 'maxgain_I' in keys and 'f0_I' in keys and 'BW_I' in keys:
        vnadata['maxgain_I'] = np.array(meas['maxgain_I'])
        vnadata['f0_I'] = np.array(meas['f0_I'])
        vnadata['BW_I'] = np.array(meas['BW_I'])
    specdata = None
    if nvr:
        specdata = {'freqs_S':np.array(meas[vna_states[0] + '/NVRfreqs']),
                    'freqs_I':np.array(meas[vna_states[1] + '/NVRfreqs']),
                    'p_on_S':np.array(meas[vna_states[0] + '/NVRcurve']),
                    'p_on_I':np.array(meas[vna_states[1] + '/NVRcurve']),
                    'p_off_S':np.array(meas[vna_states[0] + '/NVRpoff']),
                    'p_off_I':np.array(meas[vna_states[1] + '/NVRpoff'])}
        if 'NVR_S' in keys:
            specdata['NVR_S'] = np.array(meas['NVR_S'])
        if 'NVR_I' in keys:
            specdata['NVR_I'] = np.array(meas['NVR_I'])
    h5file.close()
    return (vnadata, specdata)

# extract resonator shift from phase of pump power sweep data
def calc_resShift(phase2D, freqs):
    # phase2D: 2D array of phase from pump power sweep [i_pow, i_freq]
    # freqs: 1D array of freqs from vna corresponding to phase2D
    resFreqs = np.zeros(phase2D.shape[0])
    for i_pow in xrange(resFreqs.size):
        i_min = np.argmin(np.abs(phase2D[i_pow,:]))
        resFreqs[i_pow] = freqs[i_min]
    return resFreqs

# fit a Lorentzian to extract gain and Bandwidth, input in log units
def fit_gain_curve(freqs, a_logMag, display=True):
    # freqs: array of frequency values to be fit (Hz)
    # a_logMag: S21 log mag
    a_lin = np.power(10.0, 0.1 * a_logMag) #switch to linear units
    freqsM = freqs * 1e-6 #switch to MHz to help out the numerical fittings
    popt = af.fit_lorentzian(freqsM, a_lin)
    if display:
        fit_lin = af.lorentzian(freqsM, popt[0], popt[1], popt[2], popt[3])
        fit_log = 10*np.log10(fit_lin)
        f, ax = plt.subplots()
        ax.plot(freqs*1e-9, a_logMag, color='r',marker='o', fillstyle='none', ls='None')
        ax.plot(freqs*1e-9, fit_log, linewidth=4.0, color='b')
        ax.set_xlabel('Probe Frequency (GHz)')
        ax.set_ylabel('Gain (dB)')
    f0 = popt[0] * 1e6 # center freq (Hz)
    bw = 2 * np.sqrt(popt[3]) * 1e6
    g_lin = popt[2]/popt[3]
    g_log = 10*np.log10(g_lin)
    return (f0, g_log, bw) #(Hz, dB, Hz)

# get NVR from spectrum analyzer data
def get_NVR(freqs, p_on, p_off, display = True):
    # freqs: np array of frequencies, corresponding to p_on and p_off
    # p_on: spectrum analyzer power trace with pump on (dBm)
    # p_off: spectrum analyzer power trace with pump off (background) (dBm)
    p_on_lin = np.power(10, 0.1 * p_on) # linear power in (mW)
    p_off_lin = np.power(10, 0.1 * p_off) # (mW)    
    delta_p = p_on_lin - p_off_lin
    freqsM = freqs * 1e-6 #switch to MHz to help out the numerical fittings
    popt = af.fit_lorentzian(freqsM, delta_p)
    if display:
        fit_lin = af.lorentzian(freqsM, popt[0],popt[1],popt[2],popt[3])
        f, ax = plt.subplots()
        ax.plot(freqs*1e-9, delta_p, color='r', marker='D', fillstyle='none', ls='None')
        ax.plot(freqs*1e-9, fit_lin, linewidth=4.0, color='b')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Pon - Poff (mW)')
    f0 = popt[0] * 1e6 # center freq(Hz)
    i_f0 = find_nearest(freqs, f0)
    p_off_lin_f0 = np.average(p_off_lin[i_f0-1:i_f0+2])
    p_on_lin_f0 = p_off_lin_f0 + popt[2] / popt[3]
    nvr_lin = p_on_lin_f0 / p_off_lin_f0
    nvr_log = 10 * np.log10(nvr_lin)
    return (nvr_log, nvr_lin, f0)
    
# get the gain and NVR as a function of pump power for a given pump power sweep data set
def get_gain_NVR(h5filename, dataDir=None, vna_states=('Signal','Idler'), nvr=True, recalc=False, display=True):
    # recalc: True will recalculate fits and save them to the h5 file
    # display: True will plot all of the fits if fitting happens, False will supress plotting
    (vnadata, specdata) = load_Ppow_sweep_data(h5filename, dataDir=dataDir, vna_states=vna_states, nvr=nvr)
    vnakeys = vnadata.keys()
    if recalc or 'maxgain_S' not in vnakeys or 'maxgain_I' not in vnakeys:
        # caculate max gain points, bandwidths, and center frequencies
        maxgain_S = np.zeros_like(vnadata['powers'])
        maxgain_I = np.zeros_like(maxgain_S)
        f0_S = np.zeros_like(maxgain_S)
        f0_I = np.zeros_like(maxgain_S)
        bw_S = np.zeros_like(maxgain_S)
        bw_I = np.zeros_like(maxgain_S)
        if nvr:
            nvr_S = np.zeros_like(maxgain_S)
            nvr_I = np.zeros_like(maxgain_S)
        for i_pow, power in enumerate(vnadata['powers']):
            (f0_S[i_pow], maxgain_S[i_pow], bw_S[i_pow]) = fit_gain_curve(vnadata['freqs_S'], vnadata['gain_S'][i_pow,:], display=display)
            (f0_I[i_pow], maxgain_I[i_pow], bw_I[i_pow]) = fit_gain_curve(vnadata['freqs_I'], vnadata['gain_I'][i_pow,:], display=display)
            if nvr:
                (nvr_S[i_pow], _, _) = get_NVR(specdata['freqs_S'], specdata['p_on_S'][i_pow,:], specdata['p_off_S'], display=display)
                (nvr_I[i_pow], _, _) = get_NVR(specdata['freqs_I'], specdata['p_on_I'][i_pow,:], specdata['p_off_I'], display=display)
        # end power for loop
        # save fitted data in h5file
        h5file = h5py.File(h5filename)
        if dataDir:
            meas = h5file[dataDir]
        else:
            # use most recent data
            print('Using most recent data')
            grp = h5file[h5file.keys()[-1]]
            meas = grp[grp.keys()[-1]]
        measkeys = meas.keys()
        if 'maxgain_S' in measkeys:
            meas['maxgain_S'][:] = maxgain_S
            meas['f0_S'][:] = f0_S
            meas['BW_S'][:] = bw_S
        else:
            dg1S = meas.create_dataset('maxgain_S', data=maxgain_S)
            dg2S = meas.create_dataset('f0_S', data=f0_S)
            dg3S = meas.create_dataset('BW_S', data=bw_S)
        if 'maxgain_I' in measkeys:
            meas['maxgain_I'][:] = maxgain_I
            meas['f0_I'][:] = f0_I
            meas['BW_I'][:] = bw_I
        else:
            dg1I = meas.create_dataset('maxgain_I', data=maxgain_I)
            dg2I = meas.create_dataset('f0_I', data=f0_I)
            dg3I = meas.create_dataset('BW_I', data=bw_I)
        vnadata['maxgain_S'] = maxgain_S
        vnadata['maxgain_I'] = maxgain_I
        vnadata['f0_S'] = f0_S
        vnadata['f0_I'] = f0_I
        vnadata['BW_S'] = bw_S
        vnadata['BW_I'] = bw_I
        if nvr:
            if 'NVR_S' in measkeys:
                meas['NVR_S'][:] = nvr_S
                meas['NVR_I'][:] = nvr_I
            else:
                dg4S = meas.create_dataset('NVR_S', data=nvr_S)
                dg4I = meas.create_dataset('NVR_I', data=nvr_I)
            specdata['NVR_S'] = nvr_S
            specdata['NVR_I'] = nvr_I
        h5file.close()
    # done recalculating
    if nvr:
        return (vnadata['powers'], vnadata['maxgain_S'], vnadata['maxgain_I'], specdata['NVR_S'], specdata['NVR_I'])
    return (vnadata['powers'], vnadata['maxgain_S'], vnadata['maxgain_I'])

# plot Gain - NVR (dB) vs. Gain which results from a pump power sweep
def plot_GmNVR_G(h5filename, dataDir=None, vna_states=('Signal','Idler'), recalc=False, displayFit=True, save=None):
    (powers, g_S, g_I, nvr_S, nvr_I) = get_gain_NVR(h5filename=h5filename, dataDir=dataDir, vna_states=vna_states, recalc=recalc, display=displayFit)
    g_S_lin = np.power(10, 0.1*g_S)
    g_I_lin = np.power(10, 0.1*g_I)
    g_nvr_S_log = g_S - nvr_S
    g_nvr_I_log = g_I - nvr_I
    g_nvr_S_lin = np.power(10, 0.1*g_nvr_S_log)
    g_nvr_I_lin = np.power(10, 0.1*g_nvr_I_log)
    poptS = af.fit_const_noise_model(g_S_lin, g_nvr_S_lin)
    poptI = af.fit_const_noise_model(g_I_lin, g_nvr_I_lin)
    fit_S = af.const_noise_model(g_S_lin, poptS[0])
    fit_I = af.const_noise_model(g_I_lin, poptI[0])
    fit_S = 10*np.log10(fit_S)
    fit_I = 10*np.log10(fit_I)
    #plot
    f, axarr = plt.subplots()
#    axarr.plot(g_S, g_nvr_S_log, color='r', marker='D', fillstyle='none', mew = 2.0, ls='None', label='Signal')
    axarr.plot(g_I, g_nvr_I_log, color='b', marker='D', fillstyle='none',mew = 2.0, ls='None', label='Idler data')
#    axarr.plot(g_S, fit_S, color='r', linewidth=4.0, label=r'$T_{sys} / T_{JPC} = %.1f $'%(poptS[0]))
    axarr.plot(g_I, fit_I, color='b', linewidth=4.0, label='Fit')
    width = 2.0    
    axarr.tick_params(width=width)
    for axis in ['top', 'bottom','left','right']:
        axarr.spines[axis].set_linewidth(width)
    axarr.set_xlabel(r'$G \,{\rm (dB)}$')
    axarr.set_ylabel('SNR Improvement (dB)')
#    axarr.set_title('SNR Improvement')
    axarr.legend(loc=4) #4=lower right
    if save:
        plt.rcParams.update({'font.size':18})
        axarr.set_ylim([4, 16])
        f.set_size_inches(8,6)
        f.subplots_adjust(bottom=0.15)
        f.savefig(save, dpi=600)
    return (poptS[0], poptI[0])


# plot Bandwidth vs. Gain
def plot_BW_G(h5filename, dataDir = None, vna_states=('Signal','Idler'), linB=(131e6,61.6e6)):
    (vnadata, specdata) = load_Ppow_sweep_data(h5filename, dataDir=dataDir, vna_states=vna_states, nvr=False)
    haraveB = 2* linB[0]*linB[1] / (linB[0] + linB[1]) * 1e-6 #harmonic average of linear bandwidth in MHz
    bw_S = 1e-6 * vnadata['BW_S'] # bandwidth in MHz
    bw_I = 1e-6 * vnadata['BW_I']
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(vnadata['maxgain_S'], bw_S, color='r', marker='D', ls='None', label='Signal')
    axarr[0].plot(vnadata['maxgain_I'], bw_I, color='b', marker='D', ls='None', label='Idler')
    axarr[0].set_ylabel(r'Bandwidth (MHz)')
    g_S = np.power(10, 0.1*vnadata['maxgain_S'])
    g_I = np.power(10, 0.1*vnadata['maxgain_I'])
    axarr[1].plot(vnadata['maxgain_S'], np.sqrt(g_S)*bw_S / haraveB, color='r', marker='D', ls='None', label='Signal')
    axarr[1].plot(vnadata['maxgain_I'], np.sqrt(g_I)*bw_I / haraveB, color='b', marker='D', ls='None', label='Idler')
    axarr[1].set_ylabel(r'$G^{1/2} BW / \kappa_0$')
    axarr[1].set_xlabel(r'Gain (dB)')
    axarr[0].set_title('Gain-Bandwidth Product')
    ylim = axarr[1].get_ylim()
    new_ylim = (0, ylim[1])
    axarr[1].set_ylim(new_ylim)
    axarr[1].set_xticks([5,10,15,20,25,30,35])
    axarr[1].set_xticklabels(['',10,'',20,'',30])
    axarr[0].legend(loc='best')
    axarr[1].legend(loc='best')
    width = 2.0
    for k in xrange(2):
        axarr[k].tick_params(width=width)
        for axis in ['top', 'bottom','left','right']:
            axarr[k].spines[axis].set_linewidth(width)

# plot summary charts of amplifier characteristics vs frequency
def plot_summary_charts(freqsS, freqsI, gainS, gainI, bwS, bwI, p1_S, p1_I, nvr_S=None, nvr_I=None):
    numRows = 3    
    if nvr_S is not None and nvr_I is not None:
        numRows = 4
    fig, axarr = plt.subplots(numRows, 2, sharex='col', sharey='row')
    axarr[0][1].plot(freqsS, gainS, color='r', marker='D', ls='None', label='Signal')
    axarr[0][1].plot(freqsI, gainI, color='b', marker='D', ls='None', label='Idler') #only  plotting for the legend label
    axarr[0][0].plot(freqsI, gainI, color='b', marker='D', ls='None', label='Idler')
    axarr[1][1].plot(freqsS, bwS, color='r', marker='D', ls='None', label='Signal')
    axarr[1][0].plot(freqsI, bwI, color='b', marker='D', ls='None', label='Idler')
    axarr[2][1].plot(freqsS, p1_S, color='r', marker='D', ls='None', label='Signal')
    axarr[2][0].plot(freqsI, p1_I, color='b', marker='D', ls='None', label='Idler')
    axarr[0][0].set_ylabel(r'${\rm Gain\, (dB)}$')
    axarr[1][0].set_ylabel(r'${\rm Bandwidth\, (MHz)}$')
    axarr[2][0].set_ylabel(r'$P_{-1{\rm dB}}\,{\rm(dBm)}$')
    axarr[-1][0].set_xlabel(r'${\rm Center\, frequency (GHz)}$')
    if nvr_S is not None and nvr_I is not None:
        axarr[3][0].plot(freqsI, nvr_I, color='b', marker='D', ls='None', label='Idler')
        axarr[3][1].plot(freqsS, nvr_S, color='r', marker='D', ls='None', label='Signal')
        axarr[3][0].set_ylim((0, 12))
        axarr[3][0].set_ylabel(r'$NVR \,{\rm (dB)}$')

    axarr[0][0].set_ylim((0, 30))
    axarr[1][0].set_ylim((0, 20))
    axarr[2][0].set_ylim((-150, -110))
    axarr[2][0].set_yticks((-150, -140, -130, -120, -110))
    axarr[-1][0].set_xlim([7.4,7.85])
    axarr[-1][0].set_xticks([7.4,7.6, 7.8])
    axarr[-1][1].set_xlim((8.5, 9.3))
    axarr[-1][1].set_xticks([8.6,8.8, 9.0, 9.2])
#    axarr[2].set_xticklabels([r'$hello$',r'$goodbye$'])
    axarr[0][0].set_yticks([0,10,20,30])
    #hide spines
    for k in xrange(numRows):
        width = 2.0
        axarr[k][0].spines['right'].set_visible(False)
        axarr[k][1].spines['left'].set_visible(False)
        axarr[k][0].yaxis.tick_left()
        axarr[k][0].tick_params(width=width)
        axarr[k][1].yaxis.tick_right()
        axarr[k][1].tick_params(labelright='off', width=width)
        for axis in ['top', 'bottom','left','right']:
            axarr[k][0].spines[axis].set_linewidth(width)
            axarr[k][1].spines[axis].set_linewidth(width)
        #make diagonal lines in axes coordinates
        d = 0.03 #diagonal length
        kwargs = dict(transform=axarr[k][0].transAxes, color='k', linewidth=width, clip_on=False)
        axarr[k][0].plot((1-d,1+d),(-d,+d), **kwargs)
        axarr[k][0].plot((1-d,1+d), (1-d,1+d), **kwargs)
        kwargs.update(transform=axarr[k][1].transAxes) #switch to bottom axes
        axarr[k][1].plot((-d,+d), (-d,+d), **kwargs)
        axarr[k][1].plot((-d,+d), (1-d,1+d), **kwargs)
    
    fig.subplots_adjust(wspace=0.05)
    axarr[0][1].legend(loc='lower right')

    
def take_circ_s_params(vna,fridge):
    vna.set_points(1601)
    vna.set_sweep_type('LIN')
    vna.set_average_factor(8)
    xs = vna.do_get_xaxis()
    
    stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.h5').format()    
    
    h5filename = r'Z:\Data\JPC_2016\2016-03-25 Cooldown\Mar28\new\%s' % stamp

    print 'Capturing to %s' % h5filename    
    
    h5file = h5py.File(h5filename)
    
    set_rtSwitch_S_II()
    (logmag,phase) = vna.do_get_data(fmt='PLOG', opc=True, trig_each_avg=True)
    h5file['II_lm'] = logmag; h5file['II_ph'] = phase
    
    set_rtSwitch_S_SS()
    (logmag,phase) = vna.do_get_data(fmt='PLOG', opc=True, trig_each_avg=True)
    h5file['SS_lm'] = logmag; h5file['SS_ph'] = phase

    set_rtSwitch_S_SI()
    (logmag,phase) = vna.do_get_data(fmt='PLOG', opc=True, trig_each_avg=True)
    h5file['SI_lm'] = logmag; h5file['SI_ph'] = phase
    
    set_rtSwitch_S_IS()
    (logmag,phase) = vna.do_get_data(fmt='PLOG', opc=True, trig_each_avg=True)
    h5file['IS_lm'] = logmag; h5file['IS_ph'] = phase
    
    #temps = array([fridge.get_ch1_temperature(), fridge.get_ch2_temperature()]) 

    #h5file['temps'] = temps
    
    h5file['f'] = xs
    
    h5file.close()

def do_temp_sweep_circs(vna,fridge):
    import time
    
    while True:
        take_circ_s_params(vna,fridge)
        time.sleep(60*5)

def take_spec_trace(spec, display=True, h5file=None):
    (freqs, specData) = spec.do_get_data()
    if h5file:
        cur_group = strftime('%y%m%d')
        _iteration_start = strftime('%H%M%S')
        datadir = _iteration_start+'//'
        h5f = h5py.File(h5file)
        g = h5f.require_group(cur_group)
        dg1 = g.create_dataset(datadir +'freqs', data=freqs)
        dg1.attrs.create('run_time', _iteration_start)
        dg2 = g.create_dataset(datadir + 'data', data=specData)
        dg2.attrs.create('run_time', _iteration_start)
        h5f.close()
    if display:
        fix, ax = plt.subplots(1)
        ax.plot(freqs*1e-6, specData)
    return (freqs, specData)

def calc_nbar_Pgen(f_pump, f_mode, atten, kappa_C, kappa_tot):
    # all input in SI (kappa_C/tot in Hz), atten in db (ie -80 dB), return nbar/Pgen(mW)
    h = 6.62607e-34
    linAtten = np.power(10.0, atten/10.0) #now in linear power units (for mW use of Pgen)
#    print linAtten
    delta = f_pump - f_mode
    return 2*np.pi * kappa_C/(delta**2 + (kappa_tot/2)**2) * linAtten / (h*f_pump)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    h5filename = r'Y:\Data\nickF\Data\2016-08-08 CooldownSmeagal\lines.hdf5'
#    h5filename = r'Z:\Data\JPC_2015\2015-11-18 Cooldown\jphf05.hdf5'
#    yoko = instruments['Yoko']
#    vna = instruments['VNA']
#    fridge = instruments['fridge']

    if False: # calc nbar calibration
        nbar_Pgen = calc_nbar_Pgen(18.3414e9, 10.6947e9, -80.0, 9.82e6, 10.69e6)
        print nbar_Pgen
    if False:
        data = take_spec_trace(spec, h5file=h5filename)
        
    do_take_circ_s_params = False
    if do_take_circ_s_params:
        take_circ_s_params(vna,fridge)
    
    if False: # sweep flux
        data = flux_sweep(vna, yoko, curr_start = -2.000e-3, curr_stop = 2.000e-3, curr_step=5e-6, averages=20, num_points=1601,
                   fridge=None,folder=None,display=False, atten=0,vna_fmt='SCOM', vna_sweep_type='LIN',
                   printlog=True, h5file=h5filename)

    if False: # plot flux sweep
        colorMap_h5file(h5filename, dataDir='/160620/183944', colorMap=('hot','phase'))
    
    if False: # linecut
        lineCut_flux(h5filename, dataDir='/151123/183335', current=0.375e-3, verticalStack=False)

    if False: # dynamic range trace
        data = take_dynamic_range(vna, cw_freq=7.4049e9,pow_start=-80, pow_stop=-30,pow_step=0.1, avgs=200, revert=True, h5file=h5filename)
    
    if False: # dynamic range Ppow sweep
        data = take_DR_Ppower(vna, ag1, (9.02418e9,7.69678e9), ag_pow_start=4.5, ag_pow_stop=9.0, ag_pow_step=0.1,
                   vna_pow_start=-80, vna_pow_stop=-30, vna_pow_step=0.1, vna_states=('Signal','Idler'),
                    averages=(200,100), printlog=True, h5file=h5filename)
    
    if False: #Pump power sweep
        data = take_Ppow_sweep(vna, ag2, spec=None, pow_start=-20.0, pow_stop=0.0, pow_step=1.0, averages=(200,100),
                    vna_states=('Signal','Idler'), change_rtSwitch=(set_rtSwitch_S_SS,set_rtSwitch_S_II), vna_fmt='PLOG', phase_save=True, h5file=h5filename)
                    
    if False: #2D pump power sweep
        data = take_2Dpump_sweep(vna, ag2, spec=None, pow_start=4.4, pow_stop=11.0, pow_step= 0.2, averages=(30,30),
                      freq_start=3.902e9, freq_stop=3.942e9, freq_step=1e6, num_points=1601, vna_states=('ref','trans'),
                        change_rtSwitch=(set_rtSwitch_S_II, set_rtSwitch_S_SI), display=True, atten=0, vna_fmt='PLOG', h5file=h5filename)
#        ag1.set_rf_on(False)
    
    flux_Qfit = False
    if flux_Qfit:
        fit_results = flux_sweep_Qfit(h5filename,dataDir='/150914/144652', numFitPoints = 0)
    
    pow_Qfit = False
    if pow_Qfit:
        pow_fit = power_sweep_Qfit(h5filename, dataDir='/150908/175906')
        
    summary_charts = False
    if summary_charts: #jphf05
        # center freq in GHz
        freqsS = np.array([8.58664, 8.75493, 8.90665, 8.9596, 9.02418, 9.0626, 9.1472, 9.1913])
        freqsI = np.array([7.429998, 7.51986, 7.6061, 7.64194, 7.69678, 7.7336, 7.7835, 7.8330])
        # gain in dB
        gainS = np.array([21.3, 22.1, 20.9, 20.6, 21.5, 23.0, 19.6, 25.0])
        gainI = np.array([20.0, 20.2, 20.1, 20.0, 20.5, 20.3, 20.4, 20.5])
        # bandwidth in MHz
        bwS = np.array([7.2, 5.2, 5.3, 7.6, 7.8, 4.0, 7.1, 4.7])
        bwI = np.array([7.7, 5.1, 5.9, 7.7, 7.0, 5.0, 7.4, 5.3])
        # P_-1dB in dBm at device input port
        p1_S = np.array([-132, -132, -134, -123, -123, -121, -129, -132])
        p1_I = np.array([-137, -137, -137, -127, -127, -126, -131, -137])
        # NVR in dB
        nvr_S = np.array([5.4, 8.0, 8.0, 6.5, 6.5, 7.7, 7.0, 10.0])
        nvr_I = np.array([7.2, 8.0, 8.2, 6.7, 7.4, 7.9, 7.0, 10.2])
        plot_summary_charts(freqsS, freqsI, gainS, gainI, bwS, bwI, p1_S, p1_I, nvr_S=nvr_S, nvr_I=nvr_I)
        
    plot_p1dB_vs_gain = False
    if plot_p1dB_vs_gain:
        plot_P1dB_gain(h5filename, '/151210/175328', '/151217/184325', recalc=False, atten=-80, cut=15,
                       save=r'Z:\Talks\Nick\APS 2016')
                    
    if False: # plot DR trace
        plot_dynamic_range(h5filename, dataDir='/151203/111652', atten=-80,
                           save=r'Z:\Talks\Nick\APS 2016')
        
    if False: # plot SNR improvement
        plot_GmNVR_G(h5filename, dataDir='/151210/175328', vna_states=('Signal','Idler'), recalc=False, displayFit=True,
                     save=r'Z:\Talks\Nick\APS 2016\fig_constNoiseModel2.png')
        
    if False: #plot pump power sweep traces 
        plot_Ppow_sweep_data(h5filename, dataDir='/160308/144215', vna_states=('ref', 'trans'), nvr=False, skip=1)
#                             save=r'Z:\Talks\Nick\APS 2016')
                    
    if False: #plot pump power sweep, 2D plot
        plot_Ppow_sweep_data_2D(h5filename, dataDir='/160608/182033', vna_states=('Signal', 'Idler'), nvr=False, skip=1,
                                f_pump = 7.648e9, colorMap = ('hot', 'hot'), save=None, phase_save=True)
    
    if False: # plot 2D pump sweep, 2D plot
        plot_2DPpow_sweep_data_2D(h5filename, '/160305/001026', '/160305/095958', vna_states=('ref', 'trans'), nvr=False,
                              skip=1, cmap = ('hot', 'hot'), save=None)
    
    plot_gain_BW_product = False
    if plot_gain_BW_product:
        plot_BW_G(h5filename, dataDir = '/151210/175328', vna_states=('Signal','Idler'), linB=(131e6,61.6e6))