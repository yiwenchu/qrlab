from visainstrument import VisaInstrument
import logging
import numpy


class Agilent_Generator(VisaInstrument):
    '''
    General Agilent generator class

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Agilent_E8257D', address='<GBIP address>, reset=<bool>')
    '''

    def __init__(self, name, address, reset=False, **kwargs):
        '''
        Initializes the Agilent_E8257D, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument Agilent_E8257D')
        VisaInstrument.__init__(self, name, address, tags=['physical'])

        options = self.ask("DIAG:INFO:OPT?").strip().split(",")

        self.add_visa_parameter('rf_on', 'OUTP?', 'OUTP %d', type=bool)
        self.add_visa_parameter('pulse_on', 'OUTP:MOD?', 'OUTP:MOD %d', type=bool)

        self.add_visa_parameter(
            'power', 'POW:AMPL?', 'POW:AMPL %s',
            units='dBm', minval=-135, maxval=30, type=float # SOME GENERATORS HAVE HIGHER POWER OPTIONS
        )

        self.add_visa_parameter(
            'phase', 'PHASE?', 'PHASE %s',
            units='rad', minval=-numpy.pi, maxval=numpy.pi, type=float
        )

        self.add_visa_parameter(
            'frequency', 'FREQ:CW?', 'FREQ:CW %s',
            units='Hz', minval=1e5, maxval=20e9, type=float
        )

        # Pulse Modulation Subsystem
        if "UNU" in options or "UNW" in options:
            self.add_modulation_options()

        self.add_function('reset')
        self.add_function('reset_sweep')

        if (reset):
            self.reset()
        else:
            self.get_all()
        self.set(kwargs)

    def add_modulation_options(self):
        self.add_visa_parameter(
            'pulse_modulation_enabled', 'PULM:STAT?', 'PULM:STAT %d', maxval=20e9, type=bool, gui_group='pulse'
        )

        self.add_visa_parameter(
            'pulse_width', 'PULM:INT:PWID?', 'PULM:INT:PWID %s',
            units='s', minval=1e-8, maxval=72.0, type=float, gui_group='pulse'
        )
        self.add_parameter(
            'pulse_trigger', gui_group='pulse', type=str,
            format_map={
                'FRUN': 'Internal Free-Run',
                'SQU': 'Internal Square',
                'TRIG': 'Internal Triggered',
                'DOUB': 'Internal Doublet',
                'GATE': 'Internal Gated',
                'EXT': 'External Pulse',
                'SCAL': 'Scalar Network Analyzer Input'
            }
        )

        # Sweep Subsystem
        self.add_visa_parameter(
            'sweep_frequency_mode', 'FREQ:MODE?', 'FREQ:MODE %s', gui_group='sweep', type=str,
            format_map = {
                'CW': 'Disabled',
                'SWE': 'Sweep',
                'LIST': 'List',
                }
        )

        self.add_visa_parameter(
            'sweep_amplitude_mode', 'POW:MODE?', 'POW:MODE %s', gui_group='sweep', type=str,
            format_map = {
                'FIX': 'Disabled',
                'SWE': 'Sweep',
                'LIST': 'List',
                }
        )

        self.add_visa_parameter(
            'sweep_trigger', 'TRIG:SOUR?', 'TRIG:SOUR %s', gui_group='sweep', type=str,
            format_map = {
                'BUS': 'Bus',
                'IMM': 'Immediate',
                'EXT': 'External',
                'KEY': 'Key'
            }
        )

        self.add_visa_parameter(
            'sweep_start_frequency', 'FREQ:STAR?', 'FREQ:STAR %.10E',
            type=float, gui_group='sweep', units='Hz'
        )

        self.add_visa_parameter(
            'sweep_stop_frequency', 'FREQ:STOP?', 'FREQ:STOP %.10E',
            type=float, gui_group='sweep', units='Hz'
        )

        self.add_visa_parameter(
            'sweep_start_amplitude', 'POW:STAR?', 'FREQ:STAR %.10fDBM',
            type=float, gui_group='sweep', units="dBm"
        )

        self.add_visa_parameter(
            'sweep_stop_amplitude', 'POW:STOP?', 'FREQ:STOP %.10fDBM',
            type=float, gui_group='sweep', units='dBm'
        )

        self.add_visa_parameter(
            'sweep_n_points', 'SWE:POIN?', "SWE:POIN %d",
            type=int, minval=2, maxval=65535, gui_group='sweep'
        )


    def reset(self):
        '''
        Resets the instrument to default values
        '''
        logging.info(__name__ + ' : resetting instrument')
        self.write('*RST')
        self.get_all()

    def do_set_pulse_trigger(self, val):
        if val in ('FRUN', 'SQU', 'TRIG', 'DOUB', 'GATE'):
            self.write('PULM:SOUR INT')
            self.write('PULM:SOUR:INT ' + val)
        else:
            self.write('PULM:SOUR ' + val)

    def do_get_pulse_trigger(self):
        val = self.ask('PULM:SOUR?')
        if val in ('EXT', 'SCAL'):
            return val
        return self.ask('PULM:SOUR:INT?')

    def reset_sweep(self):
        self.write('TSW')
