import sys
import os
import inspect
from lib import jsonext, misc

import localconfig

# Load config, copy from template if it doens't exist
try:
    import config
except Exception, e:
    if not misc.copy_template('templates/config.py', 'config.py'):
        raise(e)
    import config
reload(config)

# Make sure we have our objectsharer and pulse sequencer in the path
srcdir = os.path.split(os.path.abspath(inspect.getsourcefile(lambda _: None)))[0]
for modname in 'objectsharer', 'pulseseq':
    pathname = os.path.join(srcdir, modname)
    if pathname not in sys.path:
        sys.path.insert(0, pathname)

import time
import objectsharer as objsh
from pulseseq import sequencer, pulselib
reload(sequencer)
reload(pulselib)
import numpy as np
import types
import json
import matplotlib as mpl
import logging
from lib import jsonext
mpl.rcParams['legend.fontsize'] = 9

# Set long call timeout because some AWG functions are slow
objsh.DEFAULT_TIMEOUT = 120000

if hasattr(objsh, 'ZMQBackend'):
    if objsh.helper.backend is None:
        backend = objsh.ZMQBackend()
    else:
        backend = objsh.helper.backend
else:
    backend = objsh.backend
    
    
    
#settings for instrument server

isrvname = localconfig.instrument_server_alias
isrvaddr = localconfig.instrument_server_addr
isrvport = localconfig.instrument_server_port
isrv = 'tcp://{}:{}'.format(isrvaddr, isrvport)

dsrvname = 'dataserver'
dsrvaddr = '127.0.0.1'
dsrvport = 55556
dsrv = 'tcp://{}:{}'.format(dsrvaddr, dsrvport)

if backend.addr is None:
    backend.start_server(addr=isrvaddr)
    if dsrvaddr != isrvaddr:
        backend.start_server(addr=dsrvaddr)
    sys.path.append(os.getcwd())
    
logging.debug('Connecting to instrument/data server...')
# 55555 = instrument, 55556 = data
for addr in (isrv, dsrv):
    if not backend.connected_to_addr(addr):
        print 'Connecting to %s' % (addr,)
        backend.connect_to(addr)
        time.sleep(1)

instruments = objsh.find_object(isrvname)
datasrv = objsh.find_object(dsrvname)
datafile = datasrv.get_file(config.datafilename)

''' old local instrument server
if backend.addr is None:
    backend.start_server(addr='127.0.0.1')
    sys.path.append(os.getcwd())


logging.debug('Connecting to instrument/data server...')
# 55555 = instrument, 55556 = data
for addr in ('tcp://127.0.0.1:55555', 'tcp://127.0.0.1:55556'):
    if not backend.connected_to_addr(addr):
        print 'Connecting to %s' % (addr,)
        backend.connect_to(addr)
        time.sleep(1)

instruments = objsh.find_object('instruments')
datasrv = objsh.find_object('dataserver')
datafile = datasrv.get_file(config.datafilename)
'''

def parse_chans(chans):
    if chans == '':
        return None
    chans = chans.split(',')
    ret = []
    for chan in chans:
        chan = chan.replace(' ','')
        try:
            ret.append(int(chan))
        except:
            ret.append(chan)
    return ret

class Container(object):
    pass

def get_container_object(name):
    qins = instruments.get(name)
    vals = qins.get_parameter_values()
    ret = Container()
    for k, v in vals.iteritems():
        setattr(ret, k, v)
    return ret

def define_rotation(rotation_type, w, pi_amp, pi2_amp, sideband_channels):
    r = rotation_type
    if type(r) is types.StringType:
        r = r.upper()
    if r == 'GAUSSIAN':
        rotate = pulselib.AmplitudeRotation(pulselib.Gaussian, w, pi_amp, pi2_amp=pi2_amp, chans=sideband_channels)
    elif r == 'GAUSSIANSQUARE':
        rotate = pulselib.GSRotation(pi_amp, w, w, 0.0, 1.0, chans=sideband_channels)
    elif r == 'SQUARE':
        rotate = pulselib.AmplitudeRotation(pulselib.Square, w, pi_amp, pi2_amp=pi2_amp, chans=sideband_channels)
    elif r == 'TRIANGLE':
        rotate = pulselib.AmplitudeRotation(pulselib.Triangle, w, pi_amp, pi2_amp=pi2_amp, chans=sideband_channels)
    elif r == 'SINC':
        rotate = pulselib.AmplitudeRotation(pulselib.Sinc, w, pi_amp, pi2_amp=pi2_amp, chans=sideband_channels)
    elif r == 'HANNING':
        rotate = pulselib.AmplitudeRotation(pulselib.Hanning, w, pi_amp, pi2_amp=pi2_amp, chans=sideband_channels)
    elif r == 'KAISER':
        rotate = pulselib.AmplitudeRotation(pulselib.Kaiser, w, pi_amp, pi2_amp=pi2_amp, chans=sideband_channels)
    else:
        rotate = None
    if rotate is not None:
        rotate.displace = lambda amp, *args, **kwargs: rotate(np.pi*amp, *args, **kwargs)
    return rotate

def get_qubit_info(name, detune=None):
    '''
    This function can be used to get an object that represents a qubit for
    the sequencer.

    It has the following properties:
    - rotate(<alpha>, <axis>): a rotation <alpha> around <axis>
    - ssb, side-band modulation object, call ssb.modulate()
    '''
    ret = get_container_object(name)
    ret.insname = name
    ret.channels = parse_chans(ret.channels)
    ret.sideband_channels = parse_chans(ret.sideband_channels)
    if ret.sideband_channels is None:
        ret.sideband_channels = ret.channels

    # Setup channels for this element. If no sideband modulation is used
    # (i.e. <deltaf> = 0), render directly into <channels>, otherwise to
    # <sideband_channels>
    df = ret.deltaf
    if not df:
        df = 0
    if detune:
        df += detune
    if not df:
        ret.ssb = None
        ret.sideband_channels = ret.channels
    else:
        period = 1e9 / df
        if ret.sideband_channels == ret.channels:
            replace = True
        else:
            replace = False
        ret.ssb = sequencer.SSB(period, ret.sideband_channels, ret.sideband_phase, outchans=ret.channels, replace=replace)

    ret.rotate = define_rotation(ret.rotation, ret.w, ret.pi_amp, ret.pi2_amp, ret.sideband_channels)
    if type(ret.rotation_selective) is not None:
        ret.rotate_selective = define_rotation(ret.rotation_selective, ret.w_selective, ret.pi_amp_selective, ret.pi2_amp_selective, ret.sideband_channels)

    if ret.marker_channel is not None and ret.marker_channel != '':
        ret.marker = dict(channel=ret.marker_channel, ofs=ret.marker_ofs, bufwidth=ret.marker_bufwidth)
    else:
        ret.marker = None

    return ret

def get_qubits():
    qs = {}
    l = instruments.list_instruments()
    for name in l:
        if not name.startswith('qubit'):
            continue
        qname = name[5:]
        info = get_qubit_info(name)
        try:
            qs[int(qname)] = info
        except:
            qs[qname] = info
    return qs

def get_readout_info(readout='readout'):
    ret = get_container_object(readout)
    ret.rfsource1 = instruments.get(ret.rfsource1)
    ret.rfsource2 = instruments.get(ret.rfsource2)
    return ret

def load_settings_from_file(fn, inslist):
    '''
    Load instrument settings from file <fn>.
    inslist is a list of instrument names for which to apply the settings.
    If <inslist> is 'all' the settings for all instruments in the file will be
    loaded.
    '''

    f = open(fn)
    settings = jsonext.load(f)
    if inslist == 'all':
        inslist = settings.keys()
    for insname in inslist:
        print '%s:' % (insname,)
        if insname not in settings:
            print '    No settings available'
            continue
        ins = instruments[insname]
        if ins is None:
            print '    Instrument not present'
            continue
        for key, val in settings[insname].iteritems():
            print '    Setting %s to %s' % (key, val)
            if type(val) is types.UnicodeType:
                val = str(val)
            ins.set(str(key), val)

def get_temp_file():
    return datasrv.get_file(config.tempfilename)

def remove_temp_file():
    logging.info('Removing temp file from dataserver')
    try:
        tmp = get_temp_file()
        name = tmp.get_fullname()
        tmp.close()
        os.remove(config.tempfilename)
    except Exception, e:
        logging.warning('Failed to remove temporary file: %s' % str(e))
        pass

def save_instruments(fn=config.ins_store_fn):
    '''
    Save instrument settings for later use, by default in <ins_store_fn>.
    '''
    settings = instruments.get_all_parameters()
    with open(fn, "w") as sfile:
        jsonext.dump(settings, sfile)

def restore_instruments(fn=config.ins_store_fn):
    '''
    Restore previously saved instrument settings.
    '''
    load_settings_from_file(fn, 'all')

def save_fig(fig, name):
    '''
    Save figures beyond the built-in Measurement figures.
    '''
    ts = time.localtime()
    tstr = time.strftime('%Y%m%d/%H%M%S', ts)
    fn = os.path.join(config.datadir, 'images/%s_%s.%s'%(tstr, name, 'png'))
    fdir = os.path.split(fn)[0]
    if not os.path.isdir(fdir):
        os.makedirs(fdir)
    fig.savefig(fn, dpi=200)

def get_source_dir():
    '''
    Return the directory that contains the qrlab source.
    '''
    return srcdir
