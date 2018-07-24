import time
from instrument import Instrument
import types
import pyvisa
import objectsharer as objsh
import logging
logging.getLogger().setLevel(logging.INFO)

OLD_VISA = True
if hasattr(pyvisa, '__version__'):
    if pyvisa.__version__ >= '1.6':
        OLD_VISA = False

if OLD_VISA:
    print "Using deprecated version of pyvisa"
    print "Please upgrade to pyvisa >= 1.6 at your nearest convenience"
    import pyvisa.visa
    from pyvisa.visa_exceptions import VisaIOError
    from pyvisa.visa_messages import VI_ERROR_TMO
else:
    from pyvisa.errors import VisaIOError
    from pyvisa.constants import StatusCode

DEFAULT_TIMEOUT = 2000

class VisaInstrument(Instrument):

    def __init__(self, name, address=None, term_chars=None, **kwargs):
        super(VisaInstrument, self).__init__(name)

        self._ins = None
        self._address = None
        self._interrupted = False
        self._term_chars = term_chars
        self._timeout = DEFAULT_TIMEOUT

        self.add_parameter('address', type=types.StringType)
        self.add_parameter('timeout', type=types.IntType, value=DEFAULT_TIMEOUT,
                           units='ms',
                           help='Instrument read timeout')

        if not OLD_VISA:
            self._resource_manager = pyvisa.ResourceManager()

        if address:
            self.set_address(address)

        self.clear()
        self.set(kwargs)

    def interrupt(self):
        self._interrupted = True

    def do_set_address(self, val):
        if self._address == val and val is not None:
            return
        self._address = val
        self.reopen()

    def do_get_address(self):
        return self._address

    def do_set_timeout(self, val):
        self._timeout = val
        if self._ins:
            if OLD_VISA:
                self._ins.timeout = val / 1000
            else:
                self._ins.timeout = val


    def do_get_timeout(self):
        return self._timeout

    def set_term_chars(self, term_chars):
        self._term_chars = term_chars
        if self._ins:
            self._ins.read_termination = term_chars

    def open(self):
        logging.debug('Opening visa instrument at address %s, term_chars=%r', self._address, self._term_chars)
        try:
            if OLD_VISA:
                self._ins = pyvisa.visa.instrument(
                    self._address, term_chars=self._term_chars, timeout=self._timeout/1000.
                )
            else:
                self._ins = self._resource_manager.open_resource(self._address)
                self._ins.read_termination = self._term_chars
                self._ins.timeout = self._timeout

        except Exception, e:
            msg = 'Unable to open instrument %s' % (self._address,)
            logging.error(msg)

    def _check_ins(self):
        if self._ins is None:
            raise Exception("instrument not opened")

    def close(self):
        if self._ins:
            self._ins.close()
        self._ins = None

    def reopen(self):
        self.close()
        self.open()

    def read(self):
        self._check_ins()
        t0 = time.time()
        old_timeout = self._ins.timeout
        self._ins.timeout = 0
        try:
            while time.time() - t0 < self._timeout:
                try:
                    ret = self._ins.read()
                    break
                except VisaIOError as e:
                    if OLD_VISA:
                        print e
                        if e.error_code != VI_ERROR_TMO:
                            raise e
                    else:
                        if e.error_code != StatusCode.error_timeout:
                            raise e
                objsh.backend.main_loop(0)
            else:
                raise Exception("Instrument read timed out (timeout=%s)" % self._timeout)
            if self._interrupted:
                self._interrupted = False
                raise Exception("Interrupted")
        finally:
            self._ins.timeout = old_timeout

        return ret

    def read_raw(self):
        self._check_ins()
        return self._ins.read_raw()

    def write(self, cmd):
        self._check_ins()
        return self._ins.write(cmd)

    def write_raw(self, cmd):
        if OLD_VISA:
            self._ins.write(cmd)
        else:
            self._ins.write_raw(cmd)

    def ask(self, cmd, timeout=None):
        self._check_ins()
        if OLD_VISA:
            self._ins.write(cmd)
            return self._ins.read().strip()
        return self._ins.query(cmd).strip()

    def clear(self):
        self._check_ins()
        self._ins.clear()

    def get_visa_param(self, channel):
        p = self.get_parameter_options(channel)
        return self.ask(p['getfmt'])

    def set_visa_param(self, val, channel):
        p = self.get_parameter_options(channel)
        ret = self.write(p['setfmt']%val)
        for name in p.get('updates', []):
            print 'updating', name
            self.get(name)

    def add_visa_parameter(self, name, getfmt, setfmt, **kwargs):
        kwargs['getfmt'] = getfmt
        kwargs['setfmt'] = setfmt
        self.add_parameter(name,
            get_func=self.get_visa_param, set_func=self.set_visa_param,
            channel=name, **kwargs)

    # Close Visa handle if removed
    def remove(self):
        self.close()


class SCPI_Instrument(VisaInstrument):
    def add_scpi_parameter(self, name, scpi_cmd, scpi_fmt='%s', **kwargs):
        getfmt = scpi_cmd + "?"
        setfmt = scpi_cmd + " " + scpi_fmt
        self.add_visa_parameter(name, getfmt, setfmt, **kwargs)

    def check_last_command(self):
        esr = self.ask('*ESR?')
        assert int(esr) == 0, ('%s != 0' % esr)

    def test_commands(self):
        keys = []
        for k, v in self._parameters.iteritems():
            if v['flags'] & Instrument.FLAG_GET:
                self.get(k)
                self.check_last_command()

