# Instruments server
try:
    import localconfig
except:
    localconfig = None

import os
import logging
import types
import json

logging.getLogger().setLevel(logging.INFO)
import objectsharer as objsh
import time
import pythonprocess

_insdir = 'instrument_plugins'
_user_insdir = None

class WaitForInstrument(object):
    def __init__(self, instruments, name):
        self.instruments = instruments
        self.name = name
    def is_valid(self):
        return self.name in self.instruments._instruments

class Instruments(object):

    def __init__(self):
        self._create_parameters = {}
        self._instruments = {}

    def __getitem__(self, name):
        return self.get(name)

    def register_instrument(self, ins):
        uid = ins.os_get_uid()
        name = ins.get_name()
        server = ins.os_get_client()
        logging.info('New instrument connected: %s (UID %s) @ %s' % (name, uid, server))
        self._instruments[name] = ins
        self.emit('instrument-added', name)

    def list_instruments(self):
        return self._instruments.keys()

    def get(self, name):
        return self._instruments.get(name, None)

    def remove(self, name):
        if name not in self._instruments:
            return
        # Remove asynchronously so that it doesn't have to return
        self._instruments[name].remove(async=True)
        del self._instruments[name]
        self.emit('instrument-removed', name)

    def get_instrument_dirs(self):
        dirs = [_insdir,]
        if _user_insdir is not None:
            dirs.append(_user_insdir)
        return dirs

    def type_exists(self, typename):
        for d in self.get_instrument_dirs():
            driverfn = os.path.join(d, '%s.py' % typename)
            if os.path.exists(driverfn):
                return True
        return False

    def get_types(self):
        '''
        Return list of supported instrument types
        '''
        ret = []
        for d in self.get_instrument_dirs():
            filelist = os.listdir(d)
            for path_fn in filelist:
                path, fn = os.path.split(path_fn)
                name, ext = os.path.splitext(fn)
                if ext == '.py' and name != "__init__" and name[0] != '_' and ret.count(name) == 0:
                    ret.append(name)

        ret.sort()
        return ret

    def get_all_parameters(self):
        '''
        Retrieve parameters from all connected instruments and return as a
        dictionary of instrument -> parameter dictionary (name -> value).
        '''

        # Query all instruments asynchronously
        params = {}
        for name, ins in self._instruments.iteritems():
            params[name] = ins.get_parameter_values(async=True)

        # Wait for replies
        backend.main_loop(1000, wait_for=params.values())

        # Remove timed-out queries
        for name in params.keys():
            if not params[name].is_valid():
                del params[name]
            else:
                params[name] = params[name].val

        return params

    def start_instrument_instance(self, name, instype, **kwargs):
        logging.debug('Starting subprocess for instrument %s of type %s' % (name, instype))
        pid = pythonprocess.start_python_process('instrument_server.py',
                insname=name, instype=instype, kwargs=kwargs)

    def create(self, name, instype, waittime=5000, **kwargs):
        '''
        Create an instrument called 'name' of type 'type'.

        Input:  (1) name of the newly created instrument (string)
                (2) type of instrument (string)
                (3) optional: keyword arguments.
                    (1) tags, array of strings representing tags
                    (2) many instruments require address=<address>

        Output: Instrument object (Proxy)
        '''

        if not self.type_exists(instype):
            logging.error('Instrument type %s not supported', instype)
            return None

        if name in self._instruments:
            logging.warning('Instrument "%s" already exists', name)
            return self._instruments[name]

        # Store creation parameters for reloading
        self._create_parameters[name] = (instype, waittime, kwargs)

        self.start_instrument_instance(name, instype, **kwargs)
        if waittime == 0:
            return

        logging.debug('Waiting for instrument...')
        ins = WaitForInstrument(self, name)
        start = time.time()
        ret = backend.main_loop(waittime, wait_for=ins)
        end = time.time()
        if not ins.is_valid():
            raise Exception('Timed out')
        logging.debug('Should be created, returning: %s', self._instruments[name])
        return self._instruments.get(name, None)

    def reload(self, ins, **kwargs):
        '''
        Reload instrument <ins> using previous arguments to ''create'',
        optionally updating some keyword arguments.
        (<ins> can be the instrument name or the instrument object itself)
        '''

        if isinstance(ins, objsh.ObjectProxy):
            ins = ins.get_name()

        if ins not in self._create_parameters:
            raise Exception('Create parameters for instrument %s not available' % ins)
        instype, waittime, oldkwargs = self._create_parameters[ins]
        oldkwargs.update(kwargs)

        self.remove(ins)
        return self.create(ins, instype, waittime=waittime, **oldkwargs)

    def auto_load(self, driver):
        '''
        Automatically load all instruments detected by 'driver' (an
        instrument_plugin module). This works only if it is supported by the
        driver by implementing a detect_instruments() function.
        '''

        module = _get_driver_module(driver)
        if module is None:
            return False
        reload(module)

        if not hasattr(module, 'detect_instruments'):
            logging.warning('Driver does not support instrument detection')
            return False

        devs = self.get_instruments_by_type(driver)
        for dev in devs:
            dev.remove()

        try:
            module.detect_instruments()
            return True
        except Exception, e:
            logging.error('Failed to detect instruments: %s', str(e))
            return False

    def save_instruments(self, fn):
        '''
        Save instrument settings for later use, by default in <ins_store_fn>.
        '''
        settings = self.get_all_parameters()
        for ins_name, ins_settings in settings.items():
            ins_settings['_create_params'] = self._create_parameters.get(ins_name)
        with open(fn, "w") as sfile:
            json.dump(settings, sfile, indent=4, sort_keys=True)

    def restore_instruments(self, fn):
        '''
        Restore previously saved instrument settings.
        '''
        self.load_settings_from_file(fn, 'all')

    def load_settings_from_file(self, fn, inslist, create=False):
        '''
        Load instrument settings from file <fn>.
        inslist is a list of instrument names for which to apply the settings.
        If <inslist> is 'all' the settings for all instruments in the file will be
        loaded.
        '''

        f = open(fn)
        settings = json.load(f)
        if inslist == 'all':
            inslist = settings.keys()
        for insname in inslist:
            print '%s:' % (insname,)
            if insname not in settings:
                print '    No settings available'
                continue
            ins = self[insname]
            create_params = settings[insname].pop('_create_params', None)
            if ins is None:
                print '    Instrument not present'
                if create and create_params is not None:
                    print '    Create params found, creating'
                    instype, waittime, oldkwargs = create_params
                    ins = self.create(insname, instype, waittime=waittime, **oldkwargs)
                else:
                    continue
            for key, val in settings[insname].iteritems():
                print '    Setting %s to %s' % (key, val)
                if type(val) is types.UnicodeType:
                    val = str(val)
                ins.set(str(key), val)

if __name__ == '__main__':
    logging.info('Starting instruments server...')
    instruments = Instruments()

    try:
        alias = localconfig.instrument_server_alias
        addr = localconfig.instrument_server_addr
        port = localconfig.instrument_server_port
    except AttributeError:
        alias = 'instruments'
        addr = '127.0.0.1'
        port = 55555

    objsh.register(instruments, name=alias)

    if hasattr(objsh, 'ZMQBackend'):
        backend = objsh.ZMQBackend()
    else:
        backend = objsh.backend

    backend.start_server(addr=addr, port=port)
    backend.main_loop()

