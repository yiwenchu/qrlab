#==============================================================================
# Yokogawa 7651

'''Still no access to ranges and limits (so instruments UI won't be able to get them)
Problem is realted to the way it reads output from the yoko - ask('OS;') needs to read multiple lines

Ranges fixed. Worked around reading, got info from output data using ask('OD;'). Bit of a kludge

TODO

'''
#==============================================================================

import time
import visa
import re
import numpy as np
from visainstrument import VisaInstrument
from instrument import Instrument
import types
import logging
from time import sleep

class Yokogawa_7651(VisaInstrument):

    def __init__(self, name, address, **kwargs):
        super(Yokogawa_7651, self).__init__(name, address=address, term_chars='\r\n', **kwargs)

        self.add_parameter('output_state', type=types.IntType,
                           flags=Instrument.FLAG_GETSET,
                           format_map={1: 'on', 0: 'off'})
        self.add_parameter('source_type', type=types.IntType,
                           flags=Instrument.FLAG_GETSET,
                           format_map={1: 'voltage', 5: 'current'})
        self.add_parameter('voltage_range', type=types.IntType,
                           flags=Instrument.FLAG_GETSET, units='V',
                           format_map={2: '10 mV',
                                       3: '100 mV',
                                       4: '1 V',
                                       5: '10 V',
                                       6: '30 V'})
        self.add_parameter('current_range', type=types.IntType,
                           flags=Instrument.FLAG_GETSET, units='mA',
                           format_map={4: '1 mA',
                                       5: '10 mA',
                                       6: '100 mA'})
        self.add_parameter('current_limit', type=types.IntType,
                           flags=Instrument.FLAG_GETSET, units='mA',
                           minvalue=5, maxvalue=120)
        self.add_parameter('voltage_limit', type=types.IntType,
                           flags=Instrument.FLAG_GETSET, units='V',
                           minvalue=1, maxvalue=30)
        self.add_parameter('voltage', type=types.FloatType,
                           flags=Instrument.FLAG_GETSET, units='V',
                           minvalue=-30.0, maxvalue=30.0)
        self.add_parameter('current', type=types.FloatType,
                           flags=Instrument.FLAG_GETSET, units='A',
                           minvalue=-0.1, maxvalue=0.1)
        self.add_parameter('polarity', type=types.IntType,
                           flags=Instrument.FLAG_GETSET,
                           format_map={0: 'positive',
                                       1: 'negative',
                                       2: 'invert'})
        self.add_parameter('output_data_value', type=types.StringType,
                           flags=Instrument.FLAG_GET)
        self.add_parameter('overload', type=types.BooleanType,
                           format_map={True: 'overload',
                                       False: 'normal'},
                            flags=Instrument.FLAG_GET)
        self.add_parameter('max_current', type=types.FloatType,
                           flags=Instrument.FLAG_GETSET,
                           minval=0, maxval=200e-3)

        self.add_parameter('max_voltage', type=types.FloatType,
                           flags=Instrument.FLAG_GETSET,
                           minval=0, maxval=30)


        self._max_current = 50e-3
        self._max_voltage = 30.
        self.max_current_step = 0.5e-3
        # self.get_all()
        
    def trigger(self):
        '''
        Triggering function for the Yokogawa 7651.
        
        After changing any parameters of the instrument (for example, output voltage), the device needs to be triggered before it will update.
        '''
        self.write('E;')
    
    def do_set_voltage_range(self,voltage):
        '''
        Function changes the voltage range of the power supply. 
        A float representing the desired voltage will adjust the range accordingly, or a string specifying the range will also work.
        
        Device can output a max of 30V.
        
        voltage: Desired voltage (float) or directly specified voltage range (string)
        voltage = {<0...+30.0>|10MV|100MV|1V|10V|30V},float/string
        '''
        
        if ( type(voltage) == type(float()) ):
            if voltage < 10e-3:
                yokoRange = 2
            elif ( voltage >= 10e-3 ) and ( voltage < 100e-3 ):
                yokoRange = 3
            elif ( voltage >= 100e-3 ) and ( voltage < 1 ):
                yokoRange = 4
            elif ( voltage >= 1 ) and ( voltage < 10 ):
                yokoRange = 5
            elif ( voltage >= 10 ) and ( voltage <= 30 ):
                yokoRange = 6
            else:
                yokoRange = 6
                print 'Highest voltage range is 30V.'
        else:
            yokoRange = voltage

        self.write( 'R%i;' % (yokoRange,) )
        self.trigger()
        #self.do_get_voltage_range()
    
    def do_set_current_range(self,current):
        '''
        Function changes the current range of the power supply.
        A float representing the desired current will adjust the range accordingly, or a string specifying the range will also work.
        
        Device has a output max of 100mA.
        
        current: Desired current (float) or directly specified current range (string)
        voltage = {<0...+0.1>|1MA|10MA|100MA},float/string
        '''
        if ( type(current) == type(float()) ):
            if current < 1e-3:
                yokoRange = 4
            elif ( current >= 1e-3 ) and ( current < 10e-3 ):
                yokoRange = 5
            elif ( current >= 10e-3 ) and ( current <= 100e-3 ):
                yokoRange = 6
            else:
                yokoRange = 6
                print 'Highest current range is 100mA.'
        else:
            yokoRange = current

        self.write( 'R%i;' % (yokoRange,) )
        self.trigger()
        #self.do_get_current_range()

    def do_set_voltage_limit(self,limit):

        self.write( 'LV%i;' % (limit,) )
        self.trigger()
        self.do_get_voltage_limit()

    def do_set_current_limit(self,limit):

        self.write( 'LA%i;' % (limit,) )
        self.trigger()
        self.do_get_current_limit()

    def do_set_source_type(self,func):
        '''
        Set the output of the Yokogawa 7651 to either constant voltage or constant current mode.
        
        func: Desired constant (voltage or current) mode.
        func = {VOLTAGE|CURRENT},string
        '''

        self.write( 'F%i;' % (func,) )
        self.trigger()
        self.do_get_source_type()
        self.do_get_output_state()
     
    def do_set_voltage(self,voltage):
        '''
        Set the output voltage of the power supply.
        
        voltage: Desired constant output voltage
        voltage = <0...+30.0>,float
        '''
#        if abs(self.do_get_max_voltage()) < abs(level):
#            logging.warning('Level exceeds voltage limit currently set!')
#            return False
            
        self.do_set_voltage_range(float(abs(voltage)))
        self.write( 'S%f;' % (abs(voltage),) )
        self.trigger()
        self.do_set_polarity(voltage < 0)
        self.do_get_voltage()

    def do_set_max_voltage(self, val):
        self._max_voltage = val

    def do_get_max_voltage(self):
        voltage_map = {2: '10 mV',
                       3: '100 mV',
                       4: '1 V',
                       5: '10 V',
                       6: '30 V'}
                       
        rg = self.do_get_voltage_range()
        max_voltage = voltage_map[rg]
        rg = max_voltage.split(' ') 
        if rg[1] == 'mV':
            multiplier = 1e-3;
        else:
            multiplier = 1
        
        return int(rg[0])*multiplier

        
    def do_set_current(self,current):
        '''
        Set the output current of the power supply.
        
        voltage: Desired constant output voltage
        voltage = <0...+0.1>,float
        '''
        absCurrent = float(np.abs(current))

        if abs(self.do_get_max_current()) < absCurrent:
            logging.warning('Level exceeds current limit currently set!')
            return False

        self.do_set_current_range(absCurrent)
        self.write( 'S%.8f;' % (absCurrent,) )
        self.trigger()
        self.do_set_polarity(current)
        self.do_get_current()

    # def do_set_current(self,current):
    #     '''
    #     Set the output current of the power supply.
        
    #     voltage: Desired constant output voltage
    #     voltage = <0...+0.1>,float
    #     '''
    #     absCurrent = float(np.abs(current))

    #     # curCurrent = self.do_get_current()
    #     # while np.abs(curCurrent-current)>self.max_current_step:
    #     #     curSet = curCurrent+np.sign(current-curCurrent)*max_current_step
    #     #     absCurSet = float(np.abs(curSet))
    #     #     if abs(self.do_get_max_current()) < absCurSet:
    #     #     logging.warning('Level exceeds current limit currently set!')
    #     #     return False

    #     #     self.do_set_current_range(absCurSet)       
    #     #     self.write( 'S%.8f;' % (absCurSet,) )
    #     #     self.trigger()
    #     #     self.do_set_polarity(curSet)
    #     #     curCurrent = self.do_get_current()
    #     #     print 'Setting current to'+ str(curCurrent)

    #     if abs(self.do_get_max_current()) < absCurrent:
    #         logging.warning('Level exceeds current limit currently set!')
    #         return False

    #     self.do_set_current_range(absCurrent)       
    #     self.write( 'S%.8f;' % (absCurrent,) )
    #     self.trigger()
    #     self.do_set_polarity(current)
    #     self.do_get_current()
    #     print 'Setting current to'+ str(absCurrent)


    def do_set_max_current(self, val):
        self._max_current = val

    def do_get_max_current(self):
        return self._max_current
    
    def do_set_output_state(self,setting):
        '''
        Enable or disable the output of the Yokogawa 7651.
        
        setting: Specify the state of the power supply output.
        setting = {0|1|OFF|ON},integer/string
        '''
        
        self.write('O%i;' % (int(setting),))
        self.trigger()
        self.do_get_output_state()
        
    def do_get_output_data_value(self):
        
        out_data = self.ask('OD;')
        sign = out_data[4]
        if sign == u'+':
            polarity = 1
        elif sign==u'-':
            polarity = -1
        else:
            raise Exception("do_get_output_data_value not reading instrument output correctly")
        
        val = float(out_data[5:-3])
        exp = int(out_data[-2:])
        value = polarity*val*(10.0**exp)
        return value
        
    def get_settings(self):
        
        self.write('OS;')
        model = self.read()
        function = self.read()
        program = self.read()
        limits = self.read()
        # print '\r\n'.join([model,function,program,limits])
        return (model,function,program,limits)
    
    def get_output_status(self):
        resp = self.ask('OC;')
        bitstr = resp[5:]
        os = bin(int(bitstr))[2:].zfill(8) # convert integer to binary
        return os
        # if bit is 1/0:
        # bit 8: cal switch on/off
        # bit 7: IC memory card in/out
        # bit 6: calibration/normal mode
        # bit 5: output on/off
        # bit 4: unstable/normal
        # bit 3: comm error/OK
        # bit 2: program executing/not
        # bit 1: program setting under execution/not
    
    def do_get_output_state(self):
        code = self.get_output_status()
        # print "Got output state: ",code[3]
        return int(code[3])
    
    def do_get_current_limit(self):
        
        limits = self.get_settings()[3]
        limit = re.search("LA(.+)",limits).group(1)
        # limit = int(limits[6:8])
        return limit

    def do_get_voltage_limit(self):
        
        limits = self.get_settings()[3]
        limit = re.search("LV(.+)LA",limits).group(1)
        # limit = int(limits[2:3])
        return limit
        
    def do_get_current_range(self):
     
        out = self.ask('OD;')
        
        if out[3] == 'A':
            units = 'mA'
            
            if out[8] == '.':
                crange = 6#100 mA
            elif out[7] == '.':
                crange = 5#10 mA
            elif out[6] == '.':
                crange = 4#1 mA
        else:
            print "Can only get current range when source_type is CURRENT."
            crange = None
            
        return crange        
        
#        function = self.get_settings()[1]
#        if int(function[1])==5:
#            rg = int(function[3])
#        else:
#            print "Can only get current range when source_type is CURRENT."
#            rg = None
#        return rg


    def do_get_voltage_range(self):
        
        out = self.ask('OD;')

            
        if out[3] == 'V':
            if out[-1] == '3':
                units = 'mV'
                if out[8] == '.':
                    vrange = 3#100 mV
                elif out[7] == '.':
                    vrange = 2#10 mV
               
            else:
                units = 'V'
                if out[8] == '.':
                    vrange = 6#30 V
                elif out[7] == '.':
                    vrange = 5#10 V
                elif out[6] == '.':
                    vrange = 4#1 V
        else:
            print "Can only get voltage range when source_type is VOLTAGE."
            vrange = None
            
        return vrange
            
        
#        function = self.get_settings()[1]
#        if int(function[1])==1:
#            rg = int(function[3])
#        else:
#            print "Can only get voltage range when source_type is VOLTAGE."
#            rg = None
#        return rg
        
    def do_get_source_type(self):
        
        out_data = self.ask('OD;')
        source = out_data[3]
        dic = {'V': 'voltage', 'A': 'current'}
        source = dic[source]
#        function = self.get_settings()[1]
#        source = int(function[1])
        return source

    def do_set_polarity(self, polarity):
        
        if ( type(polarity) == type(float()) ):
            if polarity > 0:
                yokoPolarity = 0
            else:
                yokoPolarity = 1
        else:
            yokoPolarity = polarity
        self.write('SG%i;' % (yokoPolarity,))
        self.trigger()
        
    def do_get_overload(self):
        
        status = self.do_get_output_data_value()
        if (status[0]=='E'):
            overload = True
        else:
            overload = False
        return overload

    def do_get_polarity(self):
        
        status = self.do_get_output_data_value()
        if (status[4]=='+'):
            pol = 0
        else:
            pol = 1
        return pol

    def do_get_voltage(self):
        if self.do_get_source_type()==u'voltage':
            return self.get_data_value()
        else:
            print "Can only get voltage when source_type is VOLTAGE."
            return None
        
    def do_get_current(self):
        if self.do_get_source_type()==u'current':
            return self.get_data_value()
        else:
            print "Can only get current when source_type is CURRENT."
            return None
        
    def get_data_value(self):
        
        status = self.do_get_output_data_value()
        return status#float(status[4:].rstrip())

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    ins = Instrument.test(Yokogawa_7651)
