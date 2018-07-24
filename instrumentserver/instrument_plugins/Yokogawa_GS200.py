#==============================================================================
# Yokogawa GS200
# TODO: controller for 7651
#==============================================================================

import time
import visa
import numpy as np
from visainstrument import VisaInstrument
from instrument import Instrument
import types
import logging

class Yokogawa_GS200(VisaInstrument):

    def __init__(self, name, address, **kwargs):
        super(Yokogawa_GS200, self).__init__(name, address=address, term_chars='\n', **kwargs)

        self.add_visa_parameter('output_state', ':OUTP?', ':OUTP %d',
            type=types.BooleanType, flags=Instrument.FLAG_GETSET)
        
        self.add_visa_parameter('source_type', ':SOUR:FUNC?', ':SOUR:FUNC %s',
            type=types.StringType, flags=Instrument.FLAG_GETSET,
            option_list=('VOLT', 'CURR'))
        
        self.add_parameter('current_range', type=types.StringType,
                           flags=Instrument.FLAG_SET,
                           option_list=('MIN', 'MAX', 'UP', 'DOWN', '1E-3',
                                        '10E-3', '100E-3', '200E-3'),
                           )
        self.add_parameter('voltage_range', type=types.StringType,
                           flags=Instrument.FLAG_SET,
                           option_list=('MIN', 'MAX', 'UP', 'DOWN', '1E-3',
                                        '10E-3', '100E-3', '1E+0', '10E+0', '30E+0'),
                           )
        self.add_parameter('source_range', type=types.StringType,
                           flags=Instrument.FLAG_GETSET)

        self.add_parameter('max_current', type=types.FloatType,
                           flags=Instrument.FLAG_GETSET,
                           minval=0, maxval=200e-3)

        self.add_parameter('max_voltage', type=types.FloatType,
                           flags=Instrument.FLAG_GETSET,
                           minval=0, maxval=30)

        self.add_visa_parameter('slope', ':PROG:SLOP?', ':PROG:SLOP %s',
            type=types.FloatType, flags=Instrument.FLAG_GETSET,
            minval=0, maxval=3600)
        
        self.add_visa_parameter('repeat', ':PROG:REP?', ':PROG:REP %s',
            type=types.IntType, flags=Instrument.FLAG_GETSET)

        self.add_visa_parameter('voltage_limit', 'SOUR:PROT:VOLT?', 'SOUR:PROT:VOLT %s',
            type=types.FloatType, flags=Instrument.FLAG_GETSET,
            units='V')
        
        self.add_visa_parameter('current_limit', 'SOUR:PROT:CURR?', 'SOUR:PROT:CURR %s',
            type=types.FloatType, flags=Instrument.FLAG_GETSET,
            units='A')
        
        self.add_parameter('voltage', type=types.FloatType,
                           flags=Instrument.FLAG_GETSET,
                           units='V')
        self.add_parameter('current', type=types.FloatType,
                           flags=Instrument.FLAG_GETSET,
                           units='A')
        self.add_parameter('source_level', type=types.FloatType,
                           flags=Instrument.FLAG_GET)

        self._max_current = 10e-6
        self._max_voltage = 30.
        self.get_all()

    def do_set_voltage(self, level, range='AUTO'):
        if abs(self.get_max_voltage()) < abs(level):
            logging.warning('Level exceeds voltage limit currently set!')
            return False
        
        self.set_source_type('VOLT')
        self.set_source_level(level, range)

    def do_get_voltage(self):
        if self.get_source_type() == 'VOLT':
            return self.do_get_source_level()
        else:
            logging.warning('Yoko in current mode, cannot return voltage.')
            return None

    def do_set_max_voltage(self, val):
        self._max_voltage = val

    def do_get_max_voltage(self):
        return self._max_voltage

    def do_set_current(self, level, range='AUTO'):
        if abs(self.get_max_current()) < abs(level):
            logging.warning('Level exceeds current limit currently set!')
            return False
        
        self.set_source_type('CURR')
        self.set_source_level(level, range)

    def do_get_current(self):
        if self.get_source_type() == 'CURR':
            return self.do_get_source_level()
        else:
            logging.warning('Yoko in voltage mode, cannot return current.')
            return None

    def do_set_max_current(self, val):
        self._max_current = val

    def do_get_max_current(self):
        return self._max_current

    def set_interval(self, period):
        if period >= 0.0 and period <= 3600.0:
            self.write('PROG:INT %s' % period)

    def get_current_protection(self):
        return self.ask(':SOUR:PROT:CURR?')

    def do_set_source_range(self, range):
        self.write(':SOUR:RANG %s\n' % range)

    def do_get_source_range(self):
        return self.ask(':SOUR:RANG?')

    def do_get_source_level(self):
        return self.ask(':SOUR:LEV?')

    def do_set_voltage_range(self, range):
        # don't change state if at limits
        if self.get_source_range() == '1E-3' and range == 'DOWN':
            return
        if self.get_source_range() == '30E+0' and range == 'UP':
            return

        self.set_source_type('VOLT')
        self.set_source_range(range)

    def do_set_current_range(self, range):
        # don't change state if at limits
        if self.get_source_range() == '1E-3' and range == 'DOWN':
            return
        if self.get_source_range() == '200E-3' and range == 'UP':
            return

        self.set_source_type('CURR')
        self.set_source_range(range)

    # range auto updates at val * 1.2 (i.e. 1.21 V is 10 V scale)
    # here for simplicity just change scales if value is above the
    # current scale
    def select_voltage_range(self, value):
        ranges = ['10E-3', '100E-3', '1E+0', '10E+0', '30E+0']

        value = np.abs(value)
        if value <= float(ranges[0]):
            range = ranges[0]
        elif value <= float(ranges[1]):
            range = ranges[1]
        elif value <= float(ranges[2]):
            range = ranges[2]
        elif value <= float(ranges[3]):
            range = ranges[3]
        elif value <= float(ranges[4]):
            range = ranges[4]
        elif value <= float(ranges[5]):
            range = ranges[5]
        else:
            # voltage is out of range
            print 'voltage is out of range'
            range = -1
        return range

    def select_current_range(self, value):
        ranges = ['1E-3', '10E-3', '100E-3', '200E-3']

        value = np.abs(value)
        if value <= float(ranges[0]):
            range = ranges[0]
        elif value <= float(ranges[1]):
            range = ranges[1]
        elif value <= float(ranges[2]):
            range = ranges[2]
        elif value <= float(ranges[3]):
            range = ranges[3]
        else:
            print 'current is out of range'
            range = -1
        return range

    def set_source_level(self, level, range):
        if range=='AUTO' or range=='FIX':
            self.write('SOUR:LEV:%s %s\n' % (range, level))

    def find_ramp_time(self, initial, final, slew):
        '''
            calculates the appropriate slope and interval for a
            chosen slew rate
        '''

        return np.abs(final - initial) / float(slew)

    # yoko does not like changing ranges in ramp
    # slew is in V/s
    def set_voltage_ramp(self, level, slew=5.0):
        if abs(self.get_max_voltage()) < abs(level):
            logging.warning('Level exceeds voltage limit currently set!')
            return False

        if float(self.get_source_level()) == level:
            return
        self.set_output_state(1)
        self.set_source_type('VOLT')

        # set new range if required
        initial = float(self.get_source_level())
        if np.abs(level) - np.abs(initial) > 0:
            self.set_voltage_range(self.select_voltage_range(level))
        
        ### no clue why this is here...?
        self.set_voltage_range('30E+0')
        self.set_ramp(level, slew, initial)

    def set_current_ramp(self, level, slew=10e-6):
        if abs(self.get_max_current()) < abs(level):
            logging.warning('Level exceeds current limit currently set!')
            return False

        self.set_output_state(1)
        self.set_source_type('CURR')
        if float(self.get_source_level()) == level:
            return

        initial = float(self.get_source_level())
        if np.abs(level) - np.abs(initial) > 0:
            self.set_current_range(self.select_current_range(level))

        self.set_ramp(level, slew, initial)

    
    def set_ramp(self, level, slew, initial):
        self.set_repeat(0)

        # determine appropriate ramp time given slew rate
        ramp_time = self.find_ramp_time(initial, level, slew)
        self.set_slope(ramp_time)
        self.set_interval(ramp_time)
        self.write(':PROG:EDIT:STAR')
        self.set_source_level(level, range='FIX')
        self.write(':PROG:EDIT:END')
        self.write(':PROG:RUN')



    def get_voltage_protection(self):
        return self.ask(':SOUR:PROT:VOLT?')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    ins = Instrument.test(Yokogawa_GS200)


#yoko = Yokogawa_GS200('GPIB0::5')

#yoko.set_voltage(0.0)
#yoko.set_voltage_ramp(-5.0, slew=1)
#time.sleep(3)
#yoko.set_voltage_ramp(0, slew=1)

#yoko.close()

#yoko.set_output_state('ON')
#yoko.set_output_state('OFF')
#yoko.set_source_type('VOLT')
#print yoko.get_output_state()
#print yoko.get_source_type()
#print yoko.get_source_range()
#
#print '----'
#yoko.set_voltage_range('MIN')
#print yoko.get_source_range()
#yoko.set_voltage_range('UP')
#print yoko.get_source_range()
#yoko.set_voltage_range('UP')
#print yoko.get_source_range()
#yoko.set_voltage_range('UP')
#print yoko.get_source_range()
#print '----'
#
## check that set range does not return error if at limits
#yoko.set_voltage_range('MAX')
#yoko.set_voltage_range('UP')
#
#yoko.set_voltage(1.234, range='FasdfIX')
#print yoko.get_source_level()
#yoko.set_voltage(0.0)
#
#print '----'
#print yoko.get_voltage_protection()
#
#print '----'
#yoko.get_slope()

#print '----'
#time.sleep(2)
#yoko.set_slope(50.0)
#yoko.set_voltage(1.0)
#time.sleep(1)
#yoko.set_slope(10.0)
#yoko.set_voltage(0.0)

#print '---'
#yoko.set_output_state('ON')
#yoko.write('PROG:REP OFF')
#time.sleep(1)
#yoko.write('PROG:EDIT:STARt')
#yoko.set_source_type('VOLTAGE')
#yoko.set_slope(1000.0)
#yoko.write(':PROG:INT 1.0')
#yoko.set_voltage(5.0)
#yoko.write('PROG:EDIT:END')
#yoko.write('PROG:RUN')
#
#print yoko.ask('PROG:INT?')


#print '--- attempt 2'
#time.sleep(1)
#yoko.set_voltage(-1.0)
##time.sleep(1)
#yoko.set_voltage_ramp(10.0)
#yoko.set_voltage(0.0)
#yoko.write(':PROG:RUN')
#yoko.set_voltage_ramp(1.0)
#yoko.set_voltage_ramp(5.0)
#time.sleep(2)
#yoko.set_voltage_ramp(1.0)

#yoko.set_voltage_ramp(5, slew=5.0)
#yoko.write(':PROG:MEM "1.0, 10, V\n"')


#yoko.write(':PROG:LOAD "Prgm02.csv"')

#yoko.write(':PROG:RUN')

#yoko.write(':PROG:RUN')


#yoko.write(':PROG:INT 5.0')


#yoko.write(':PROG:RUN')


#yoko.write(':PROG:REP OFF')
#yoko.write(':PROG:INT 5.0')






#print '--- attempt 3'
#start = 0.0
#end   = 5.0
#dt = 100
#steps = np.linspace(start, end, dt)
#for val in steps:
#    yoko.set_voltage(val)
#    time.sleep(0.001)

#yoko.close()
