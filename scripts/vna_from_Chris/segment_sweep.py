# -*- coding: utf-8 -*-
"""
Created on Mon Oct 06 14:15:29 2014

@author: rsl
"""

import mag_fit_resonator as fitter
import numpy as np

def form_segment_sweep(freq,db,vna=None,segments=200,num_points=1601,max_points=10001,fraction_outer=0.1,bw_multiplier=10.0,show=False,use_data=False):
    print "Be careful when you call this function (form_segment_sweep) that your input data is from a mostly LINEAR sweep!"
    # for highly asymmetric resonances, use data from real or imaginary instead of mag
    segments = np.clip(segments,3,200) # limitation of E5071C
    num_points = np.clip(num_points,6,20e3)
    fraction_outer = np.clip(fraction_outer,0.001,0.999)
    bw_multiplier = np.clip(bw_multiplier,1.1,100.0)
    
    (f0, qi, bw, grads) = fitter.quick_hanger_fit(freq,db,show=show,grads_are_data=use_data)
    if use_data: # just use raw data to guess if we suspect a bad fit, VERY BETA
        window_size = np.ceil(np.size(grads)/20)*2+1
        grads = savitzky_golay(grads,window_size,3)
        (f0, qi, bw, grads) = fitter.quick_lor_fit(freq,-1*grads,show=show)
        bw *= 2
    bw *= 3
    if show:
        print "Center frequency: %.6f GHz"%(f0/1e9,)
        print "BW: %.4f MHz"%(bw/1e6,)
    outer_points_each = np.clip(np.floor(num_points*fraction_outer/2),2,num_points)
    inner_points = np.clip(np.floor(num_points-2*outer_points_each),2,num_points)
    inner_segment_width = bw/(segments-2)
    
    # all the intermediate segments
    midstarts = np.linspace(f0-bw/2,f0+bw/2-inner_segment_width,segments-2)
    midcenters = midstarts + inner_segment_width/2
    midstops = midstarts+inner_segment_width
    weights = np.interp(midcenters,freq,grads)
    normweight = weights/np.sum(weights)
    # return (freq,grads)
    midpoints = np.clip(np.floor(normweight*inner_points),2,inner_points)

    starts = np.concatenate((np.array([f0-bw*bw_multiplier/2]),midstarts,np.array([f0+bw/2])))
    stops = np.concatenate((np.array([f0-bw/2]),midstops,np.array([f0+bw*bw_multiplier/2])))
    points = np.concatenate((np.array([outer_points_each]),midpoints,np.array([outer_points_each])))

    # check for too many points
    if vna:
        vna.set_points(20002)
        max_points = vna.get_points()
    actual_points = np.sum(points)
    while actual_points > max_points:
        outer_points_each = np.floor(outer_points_each/2)
        points = np.concatenate((np.array([outer_points_each]),midpoints,np.array([outer_points_each])))
        actual_points = np.sum(points)
        print "[WARNING] Too many points. Reducing outer band density to %d pts/side as a contingency measure. Try reducing num_points for future runs." % outer_points_each

    points.astype(int)
    print "Using %d segments with %d total points." % (np.size(starts),actual_points)
    extra = (freq,grads/np.max(grads))
    return (starts, stops, points, extra)
    
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
    
def load_segment_sweep(vna, sweep_params):
    vna.load_segment_table(sweep_params[0:3])

def test_segment_sweep(vna,wts): # tau must already be canceled
    vna.set_average_factor(1)
    vna.do_enable_averaging()
    use_segment_sweep(vna)
    xs = vna.do_get_xaxis()
    ys = vna.do_get_data(fmt='PLOG',opc=True)
    import matplotlib.pyplot as plt
    plt.figure()
    weights = np.interp(xs,wts[0],wts[1])
    plt.scatter(ys[0],ys[1],c=weights,s=100,cmap='jet')
    plt.jet()
    plt.show()
    return ys, wts

def use_segment_sweep(vna):
    vna.set_sweep_type('SEGM')

if __name__ == '__main__':
    # an example:
    import mclient
    vna = mclient.instruments['VNA']
    import VNA_functions
    (x, ys) = VNA_functions.collect_and_display(vna,save=False,fmt='PLOG',display=False)
    y0 = ys[0]
    # testfname = "Z:\\_Data\\3D\\LZ_140926\\CA\\1\\-20.00dB_26.00mK.dat"
    # x, y0, y1 = np.loadtxt(testfname, unpack=True, delimiter='\t')
    sweep_params = form_segment_sweep(x, y0, num_points=500, segments=50, use_data=False, show=True)
    load_segment_sweep(vna,sweep_params)
    use_segment_sweep(vna)
    # testys, testwts = test_segment_sweep(vna,sweep_params[3])
    # # sweep_params = form_segment_sweep(x, y0, num_points=500, segments=50, use_data=True, show=True)
    # # load_segment_sweep(vna,sweep_params)