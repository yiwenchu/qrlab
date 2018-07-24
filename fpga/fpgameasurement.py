import numpy as np
from pulseseq import sequencer, pulselib
import sys
import time
import fpgagenerator
from load_fpgadevelop import YngwieEncoding, YngwieDecoding
import fpgapulses
import gc
import os
import types
import mclient
import config
import plotconfig
import matplotlib.pyplot as plt
import matplotlib as mpl
from lib.math import fitter
from PyQt4 import QtGui

ins_yng = mclient.instruments['yngwie']

TABLES_DIR = os.path.join(config.datadir, 'tables')
if not os.path.exists(TABLES_DIR):
    os.makedirs(TABLES_DIR)
PRE_PREFIX = os.path.join(TABLES_DIR, 'seq_')

def get_marker_kwargs(mspec):
    if mspec is None or mspec == '':
        return dict(marker_chan=None)
    if len(mspec) < 3 or mspec[0] != 'm':
        raise Exception('Marker specification should be m<digital sequence id (1/2)><value (1/2), with optional +<marker pad>')
    d = dict(marker_chan=mspec[:2], marker_val=int(mspec[2]), marker_pad=0)
    if len(mspec) > 3 and mspec[3] == '+':
        d['marker_pad'] = int(mspec[3:])
    return d

def get_rotation(info, rotation, w, pi_amp, pi2_amp, drag=0, chop=4):
    if rotation == 'GAUSSIAN':
        Ipulse = pulselib.Gaussian(w, 0.7, chan=info.channels[0], chop=chop)
    elif rotation =='SQUARE':
        Ipulse = pulselib.Constant(w, 0.7, chan=info.channels[0])
    elif rotation =='TRIANGLE':
        Ipulse = pulselib.Triangle(w, 0.7, chan=info.channels[0])
    elif rotation =='SINC':
        Ipulse = pulselib.Sinc(w, 0.7, chan=info.channels[0])
    elif rotation =='HANNING':
        Ipulse = pulselib.Hanning(w, 0.7, chan=info.channels[0])
    elif rotation =='KAISER':
        Ipulse = pulselib.Kaiser(w, 0.7, chan=info.channels[0])
    elif rotation =='FLATTOP':
        Ipulse = pulselib.FlatTop(w, 0.7, chan=info.channels[0])
    else:
        raise Exception('Unknown rotation %s' % info.rotation)

    if drag != 0:
        Qpulse = pulselib.Pulse('%s_d'%Ipulse.name, info.drag * pulselib.derivative(Ipulse.data), chan=info.channels[1])
    else:
        Qpulse = None

    r = fpgapulses.FPGAAmplitudeRotation(Ipulse, Qpulse,
            pi_amp, pi2_amp=pi2_amp, chans=info.channels,
            **get_marker_kwargs(info.marker_channel))

    return r

def make_qubit(name, chop=4):
    info = mclient.get_qubit_info(name)
    info.ssb_kwargs = {'chan%d_ssb_reload'%info.channels[0]: True}
    info.load_ssb_kwargs = info.ssb_kwargs
    info.load_mixer_kwargs = {'chan%d_mixer_fetch'%info.channels[0]: True}
    info.use_mixer_kwargs = {
        'chan%d_mixer_mask'%info.channels[0]: (True, True),
        'chan%d_mixer_mask'%info.channels[1]: (True, True),
    }
    info.loaduse_mixer_kwargs = {
        'chan%d_mixer_fetch'%info.channels[0]: True,
        'chan%d_mixer_mask'%info.channels[0]: (True, True),
        'chan%d_mixer_mask'%info.channels[1]: (True, True),
    }
    info.rotate = get_rotation(info, info.rotation, info.w, info.pi_amp, info.pi2_amp, info.drag)
    info.rotate_selective = get_rotation(info, info.rotation_selective, info.w_selective, info.pi_amp_selective, info.pi2_amp_selective, info.drag_selective)
    return info

class FPGAMeasurement(object):

    def __init__(self, name='noname', seq=None, seqlen=None, bg=False,
                 se=True, iqrel=False, raw=False, rrec=False,
                 table_prefix=None,
                 block_size=None, xs=None, ys=None,
                 fit_func=None, fit_func_kwargs={},
                 multi_se=False, integrate_len=None, qubit_gaussian_chop=4,
                 fig=None, probabilities=True, **kwargs):

        self.name = name
        self.comment = kwargs.pop('comment', '')
        self.title = kwargs.pop('title', '')

        self.seq = seq
        self.seqlen = seqlen

        ts = time.localtime()
        self._timestamp_str = time.strftime('%Y%m%d/%H%M%S', ts)
        self.save_settings()

        self.readout = mclient.get_readout_info()
        inslist = mclient.instruments.list_instruments()
        if 'qubit0' in inslist:
            self.qubit = make_qubit('qubit0', chop=qubit_gaussian_chop)
        if 'qubit1' in inslist:
            self.qubit1 = make_qubit('qubit1', chop=qubit_gaussian_chop)
        if 'cavity' in inslist:
            self.cavity = make_qubit('cavity')
            self.cavity.displace = lambda alpha, **kwargs: self.cavity.rotate(np.abs(alpha)*np.pi, np.angle(alpha), **kwargs)
            self.cavity.displace_selective = lambda alpha, **kwargs: self.cavity.rotate_selective(np.abs(alpha)*np.pi, np.angle(alpha), **kwargs)

        self.se = se
        self.iqrel = iqrel
        self.raw = raw
        self.rrec = rrec
        self.xs = xs
        self.ys = ys
        self.block_size = block_size
        self.bg = bg
        if table_prefix is None:
            table_prefix = PRE_PREFIX + name
        self.table_prefix = table_prefix
        self.multi_se = multi_se
        self.integrate_len = integrate_len

        self.data_iqrel = None
        self.data_se = None

        self.fit_func = fit_func
        self.fit_func_kwargs = fit_func_kwargs
        self._generated = False

        self.fig = fig
        self.probabilities = probabilities

        self.setup_integration()

    def setup_integration(self):
        self.thresh0, self.thresh1 = 0, 0
        if self.readout.envelope is None or self.readout.envelope == '':
            envelope0 = np.ones(self.readout.acq_len/20, dtype=np.complex)
            envelope1 = envelope0
        else:
            try:
                amp = float(self.readout.envelope)
                envelope0 = amp * np.ones(self.readout.acq_len/20, dtype=np.complex)
                envelope1 = envelope0
            except:
                if os.path.exists(self.readout.envelope):
                    fdata = np.load(self.readout.envelope)
                    if 'envelope0' in fdata:
                        envelope0 = fdata['envelope0']
                        envelope1 = fdata['envelope1']
                    else:
                        envelope0 = fdata['envelope']
                        envelope1 = 1j * envelope0
                    if 'thresh0' in fdata:
                        self.thresh0 = fdata['thresh0']
                    if 'thresh1' in fdata:
                        self.thresh1 = fdata['thresh1']
                else:
                    raise ValueError('Envelope file %s does not exist' % self.readout.envelope)
        im = YngwieEncoding.IntegrationManager()
        if self.multi_se:
            boxIntegration = im.add({'length':self.integrate_len,'weights':[1,],'log':True,'multi_se':True},
                                    {'length':self.integrate_len,'weights':[-1j,],'log':True,'multi_se':True})
            self.integrate_log = boxIntegration
            self.integrate_nolog = boxIntegration
        else:
            boxIntegration = im.add({'length':len(envelope0),'weights':envelope0,'log':True, 'threshold': round(self.thresh0*0x7fff)},
                                     {'length':len(envelope1),'weights':envelope1,'log':True, 'threshold': round(self.thresh1*0x7fff)})
            boxIntegration2 = im.add({'length':len(envelope0),'weights':envelope0,'log':False, 'threshold': round(self.thresh0*0x7fff)},
                                     {'length':len(envelope1),'weights':envelope1,'log':False, 'threshold': round(self.thresh1*0x7fff)})
            self.integrate_log = boxIntegration
            self.integrate_nolog = boxIntegration2
        #boxIntegration = im.add({'length':1,'weights':[1,],'log':True}, {'length':1,'weights':[-1j,],'log':True})
        im.export(self.table_prefix)

    def load_data(self, fn, dtype=np.int16):
        f = open(fn, 'rb')
        return np.fromfile(f, dtype=dtype)

    def interact(self):
        objsh.helper.backend.main_loop(20)
        QtGui.QApplication.processEvents()

    def load_se_data(self, block=0):
        SE0 = self.load_data(self.get_fpgadata_fn(6, block), dtype=np.int16)
        SE1 = self.load_data(self.get_fpgadata_fn(7, block), dtype=np.int16)
        SE = SE0 / float(2**15) + 1j * SE1 / float(2**15)
        self.data_se = SE

    def get_fpgadata_fn(self, stream, block=0):
        dump_path = ins_yng.get_dump_path()
        return os.path.join(dump_path, 'Data_0x000%d[0x00]B0_#%04d.bin' % (stream, block))

    def get_rrec(self, nrec=None):
        d = self.load_data(self.get_fpgadata_fn(8), dtype=np.uint32)
        recs = np.reshape(d, (len(d)/8, 8))
        if nrec is None:
            return recs
        else:
            return recs[:min(recs.shape[0], nrec),:]

    def print_rrec(self, nrec=None):
        print YngwieDecoding.ResultRecord.title()
        recs = self.get_rrec(nrec)
        for i in range(len(recs)):
            print YngwieDecoding.ResultRecord(recs[i])

    def set_seq(self, seq, seqlen):
        self.seq = seq
        self.seqlen = seqlen

    def save_settings(self, fn=None):
        if fn is None and self._timestamp_str != '':
            fn = os.path.join(config.datadir, 'settings/%s.set'%self._timestamp_str)
        fdir = os.path.split(fn)[0]
        if not os.path.isdir(fdir):
            os.makedirs(fdir)
        mclient.save_instruments(fn)
        mclient.save_instruments()

    def save_attrs(self, **kwargs):
        self.data.set_attrs(**kwargs)

    def save_data(self, **kwargs):
        """
        save the data: creates a group via the data server, and stores
        the averaged data as returned by get_averaged. The name of the
        group is composed as '<datestamp>/<timestamp>_<measurement name>'.
        We store the x and y axes in 'xs' and 'ys' (y only if applicable),
        and the measured values in 'avg'.

        kwargs are treated as follows:
        - if a kwarg is a numpy array, it will be saved as data array.
        - else, it will be saved as attribute.

        some class attributes are stored as attributes automatically:
            title, comment, thresh0, bg, naverages
        """

        # data_arr_name = kwargs.pop('data_arr_name', 'fpgadata')
        self.datafile = mclient.datafile
        self.groupname = '%s_%s'  % (self._timestamp_str, self.name)
        self.data = self.datafile.create_group(self.groupname)

        # save the raw fpga data.
        if hasattr(config, 'save_raw_fpgadata') and \
            getattr(config, 'save_raw_fpgadata'):

            self.data.create_dataset('fpgadata',
                data = self.data_se)

        # save averaged data
        avg_data = self.get_averaged(ret_xs=True)
        if self.ys is not None:
            xs, ys, zs = avg_data
            avg = zs
            self.data.create_dataset('avg', data = zs)
            self.data.create_dataset('ys', data = ys)
        else:
            xs, ys = avg_data
            self.data.create_dataset('avg', data = ys)

        self.data.create_dataset('xs',
            data = xs)

        # save all arrays that have been supplied additionally
        for kw in kwargs:
            if type(kwargs[kw]) == np.ndarray:
                self.data.create_dataset(kw, data = kwargs.pop(kw))

        # remaining kwargs are attrs, save those too
        if 'title' not in kwargs:
            kwargs['title'] = self.title
        if 'comment' not in kwargs:
            kwargs['comment'] = self.comment

        # some more info about the measurement that should be saved in any case
        kwargs['thresh0'] = self.thresh0
        kwargs['bg'] = self.bg
        kwargs['naverages'] = self.readout.naverages

        self.data.set_attrs(**kwargs)

        return


    def save_fig(self, fn=None):
        if fn is None and self._timestamp_str != '':
            fn = os.path.join(os.path.join(config.datadir, 'images'), '%s_%s.png'%(self._timestamp_str, self.name))
        fdir = os.path.split(fn)[0]
        if not os.path.isdir(fdir):
            os.makedirs(fdir)
        plt.savefig(fn, dpi=300)

    def start_exp(self, infinite=False):
        if not self._generated:
            self.generate()

        self.save_settings()

        ins_yng.stop()
        ins_yng.update_modes()
        print 'Loading tables...'
        ins_yng.load_tables(self.table_prefix)

        block_size = self.block_size
        if block_size is None:
            block_size = self.readout.naverages * self.seqlen

        # Retry until we have no trigger problem
        while True:
            if self.iqrel:
                ins_yng.accept_stream('rel', bytes_needed=self.readout.naverages*self.seqlen*self.readout.acq_len/20*8)
            if self.raw:
                ins_yng.accept_stream('raw0', bytes_needed=self.readout.naverages*self.seqlen*self.readout.ref_len*2)
                ins_yng.accept_stream('raw1', bytes_needed=self.readout.naverages*self.seqlen*self.readout.acq_len*2)
            if self.rrec:
                ins_yng.accept_stream('rec', bytes_needed=32*self.seqlen*self.readout.naverages)
            ins_yng.accept_stream('se0', bytes_needed=self.readout.naverages*self.seqlen*2, file_size=block_size*2)
            ins_yng.accept_stream('se1', bytes_needed=self.readout.naverages*self.seqlen*2, file_size=block_size*2)
            ins_yng.start()
            time.sleep(0.05)
            if ins_yng.get_run_status() != 0:
                print 'Trigger failed, retrying...'
                ins_yng.stop()
                time.sleep(1)
            else:
                break

        if ins_yng.get_unlimited():
            return

        print 'Acquiring data...'
        while ins_yng.is_streaming():
            time.sleep(0.1)

        self.load_se_data(0)
        self.fit = self.perform_fit()

        if not self.iqrel:
            self.save_data()
            return self.se

        idx = (5,)
        for i in idx:
            d = self.load_data(self.get_fpgadata_fn(i,0), dtype=np.int32)
            I = ((d[::2] & 0x1ffff) - (d[::2] & 0x20000)).astype(np.float) / float(2**16)
            I = I.reshape((self.readout.naverages,self.readout.acq_len/20))
            Q = ((d[1::2] & 0x1ffff) - (d[1::2] & 0x20000)).astype(np.float) / float(2**16)
            Q = Q.reshape((self.readout.naverages,self.readout.acq_len/20))
            IQ = I + 1j*Q
            self.data_iqrel = IQ

        return self.se, IQ

    def stop_exp(self):
        print 'Stopping experiment'
        ins_yng.stop()

    def generate(self, plot=False):
        self._generated = True

        sequencer.Sequence.PRINT_EXTENDED = True
        sequencer.Pulse.RANGE_ACTION = sequencer.IGNORE
        s = sequencer.Sequencer(self.seq, minlen=0, ch_align=False)
        s.add_required_channel([0, 1, 2, 3, 'm0', 'master'])
        seqs = s.render(debug=False)
        if plot and 0:
            s.print_seqs(seqs)
            s.plot_seqs(seqs)

        # Demod and integration tables
        ssig = ins_yng.get_demod_scale_sig()
        sref = ins_yng.get_demod_scale_ref()
        YngwieEncoding.generate_demodulation_table(self.table_prefix, relative_phase=np.pi/180*90, amplitudes=[ssig,sref])

        fg = fpgagenerator.FPGAGenerator(self.table_prefix, integration=self.integrate_log, nmodes=ins_yng.get_nmodes(), nchannels=ins_yng.get_noutputs())
        fg.generate(seqs, plot=plot)

    def interact(self):
        gc.collect()
        QtGui.QApplication.processEvents()

    def plot_histogram(self, iqg=None, iqe=None, **kwargs):
        f = plt.figure()
        ax1 = f.add_subplot(211)
        ax1.hexbin(np.real(self.data_se), np.imag(self.data_se), cmap=mpl.cm.hot, **kwargs)
        ax2 = f.add_subplot(212)
        if iqg is not None:
            h, e = np.histogram(np.real(self.data_se * np.conjugate(iqg - iqe)), bins=51)
        else:
            h, e = np.histogram(np.real(self.data_se), bins=51)
        ax2.plot(np.cumsum(h))

    def analyze_histogram_blob(self, plot=False, nbins=41, fit_func_kwargs={}):
        '''
        Fit single shot histogram with a single Gaussian peak.
        '''
        hist, xs, ys = np.histogram2d(np.real(self.data_se), np.imag(self.data_se), bins=nbins)
        hist = hist.T
        XS, YS = np.meshgrid((xs[:-1]+xs[1:])/2, (ys[:-1]+ys[1:])/2)
        print 'Minmax xs: %s,%s' % (np.min(XS), np.max(XS))
        f = fitter.Fitter('gaussian2d')
        f.perform_lmfit(XS, YS, hist, plot=plot, plot_guess=False, **fit_func_kwargs)
        return f.fit_params

    def analyze_jpc_histogram(self, plot=False, nbins=41, fit_func_kwargs={}):
        '''
        Fit single shot histogram with two Gaussian peaks, convenient if a
        measurement is performed after a pi/2 pulse. Due to the arbitrariness
        of which blob represents |g> and |e>, it is probably better to do
        two experiments which result in |g> and |e> separately.
        '''
        hist, xs, ys = np.histogram2d(np.real(self.data_se), np.imag(self.data_se), bins=nbins)
        hist = hist.T
        XS, YS = np.meshgrid((xs[:-1]+xs[1:])/2, (ys[:-1]+ys[1:])/2)
        f = fitter.Fitter('double_gaussian_2dhist')
        f.perform_lmfit(XS, YS, hist, plot=plot, plot_guess=False, **fit_func_kwargs)
        return f.fit_params

    def get_binary(self, invert=False):
        shots = self.data_se
        mask = np.real(shots)<self.thresh0
        shots = np.zeros_like(shots, dtype=np.int8)
        if invert:
            mask = ~mask
        shots[mask] = 0
        shots[~mask] = 1
        return shots

    def get_averaged(self, ret_xs=False, thresh_fact=1):
        '''
        Return a properly shaped array of averaged shots
        If the sequence is 2d, xs is assumed to be the inner loop, ys the
        outer loop, so data will be available as ys[iy,ix]
        '''

        # Convert to probabilities if threshold specified
        shots = self.data_se
        if self.probabilities and self.thresh0 != 0:
            mask = np.real(shots)<(self.thresh0*thresh_fact)
            shots = np.zeros_like(shots, dtype=np.int8)
            shots[mask] = 0
            shots[~mask] = 1

        if self.bg:
            shots = shots[::2] - shots[1::2]
            shots = shots.reshape((self.readout.naverages, self.seqlen/2))
        else:
            shots = shots.reshape((self.readout.naverages, self.seqlen))

        zs = np.real(np.average(shots,0))
        if self.ys is not None:
            zs = zs.reshape([len(self.ys),len(self.xs)])

        if not ret_xs:
            return zs
        if self.ys is not None:
            return self.xs, self.ys, zs

        xs = self.xs
        if self.bg:
            if xs is None:
                xs = np.arange(self.seqlen/2)
        else:
            if xs is None:
                xs = np.arange(self.seqlen)
        return xs, zs

    def perform_fit(self):
        if self.fit_func is None:
            return
        xs, ys = self.get_averaged(ret_xs=True)
        f = fitter.Fitter(self.fit_func)
        ret = f.perform_lmfit(xs, ys, plot=False, **self.fit_func_kwargs)
        self.fit_params = ret.params
        self.fit = f
        return f

    def plot_se(self, fit_xs=None):
        """
        Plot the averaged data.
        if a fit has been performed with perform_fit(), the fit is also plotted.

        if an alternative x-array <fit_xs> is supplied, these values
        will be used for plotting the fit (for better looks when there's only
            few points in a region of interest)
        """

        xs, ys = self.get_averaged(ret_xs=True)

        if self.fig is not None:
            fig = self.fig
            ax1 = fig.axes[0]
            ax2 = fig.axes[0]
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

        ax2.set_ylabel('Residuals')

        if not self.probabilities:
            ax1.plot(xs, ys, 'o', label='data',
                mec='k', mfc='w')
        else:
            u_ys = (ys*(1.-ys)/self.readout.naverages)**.5
            ax1.errorbar(xs, ys, fmt='o', yerr=u_ys,
                label='data', mec='k', mfc='w')

        if self.probabilities and self.thresh0 != 0:
            ax1.set_ylabel('Probability')
        else:
            ax1.set_ylabel('Signal [mV]')

        if self.fit_func is not None and hasattr(self, 'fit'):
            ax1.plot(xs if fit_xs==None else fit_xs,
                self.fit.eval_func(xs if fit_xs==None else fit_xs),
                lw=2, label='fit')
            ax2.plot(xs, ys - self.fit.eval_func(xs), 'ks')
            ax1.legend(loc=0)

    def plot2d(self, swapax=False, ret_fig=True):
        zs = self.get_averaged()
        if swapax:
            xs, ys = self.ys, self.xs
            zs = np.swapaxes(zs, 0, 1)
        else:
            xs, ys = self.xs, self.ys
        dx = xs[1] - xs[0]
        plot_xs = np.concatenate([xs, [xs[-1]+dx]]) - dx/2
        dy = ys[1] - ys[0]
        plot_ys = np.concatenate([ys, [ys[-1]+dy]]) - dy/2
        fig,ax = plt.subplots(1,1)
        p = ax.pcolormesh(plot_xs, plot_ys, zs, cmap=plotconfig.pcolor_cmap)
        fig.colorbar(p)

        if ret_fig:
            return fig

    def plot_img(self, swapax=True):
        zs = self.get_averaged()
        if swapax:
            xs, ys = self.ys, self.xs
            zs = np.swapaxes(zs, 0, 1)
        else:
            xs, ys = self.xs, self.ys
        plt.figure()
        plt.imshow(zs, extent=(np.min(xs), np.max(xs), np.min(ys), np.max(ys)),
                   cmap=plotconfig.pcolor_cmap, interpolation='nearest')
        plt.colorbar()

    def plot_shot_trace(self, xscale=1, delta=False):
        shots = self.data_se
        if self.probabilities and self.thresh0 != 0:
            mask = np.real(shots)<self.thresh0
            shots = np.zeros_like(shots, dtype=np.int8)
            shots[mask] = 0
            shots[~mask] = 1
        if delta:
            shots = shots[1:] ^ shots[:-1]
        plt.figure()
        plt.plot(np.arange(len(shots))*xscale, shots)
