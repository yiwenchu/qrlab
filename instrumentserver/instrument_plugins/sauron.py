# QTLab example instrument communicating by TCP/IP
# Reinier Heeres, 2009

from instrument import Instrument
import types
import socket
import re

# lazarus = instruments.create('fridge', 'sauron', host='sauron.central.yale.internal', port=33590, fridge='LZ')

class sauron(Instrument):

    def __init__(self, name, host, port, fridge, base_ruox_channel=6):
        Instrument.__init__(self, name, tags=['measure'])
        DICT_ON_OFF = {'ON': 'on', 'OFF': 'off'}
        self.TEMPCHANNEL = str(base_ruox_channel)
        self.HEATCHANNEL = '1'
        self.STILLHEATCHANNEL = '2'
        self.PTCCHANNEL = '1'
        self.TURBOCHANNEL = '1'

        self.add_parameter('temperature', units='K', type=types.FloatType,
                flags=Instrument.FLAG_GET, minval=0, maxval=400)
                
        self.add_parameter('temperature_setpoint', units='K', type=types.FloatType,
                flags=Instrument.FLAG_GETSET, minval=0, maxval=400)
                
        self.add_parameter('temperature_ramp_rate', units='K/min', type=types.FloatType,
                flags=Instrument.FLAG_GETSET, minval=0, maxval=10)

        self.add_parameter('still_heater_power', units='uW', type=types.FloatType,
                flags=Instrument.FLAG_GETSET, minval=0, maxval=99999)
                
        self.add_parameter('heater_power', units='uW', type=types.FloatType,
                flags=Instrument.FLAG_GETSET, minval=0, maxval=99999)
                
        self.add_parameter('heater_limit', units='mA', type=types.FloatType,
                flags=Instrument.FLAG_GETSET, minval=0, maxval=30)
                
        self.add_parameter('pulse_tube', type=types.StringType,
                flags=Instrument.FLAG_GETSET,
                format_map=DICT_ON_OFF,
                value='ON')
                
        self.add_parameter('turbo', type=types.StringType,
                flags=Instrument.FLAG_GETSET,
                format_map=DICT_ON_OFF,
                value='ON')
     
        self.add_parameter('temperature_ramp_enable', type=types.StringType,
                flags=Instrument.FLAG_GETSET,
                format_map=DICT_ON_OFF,
                value='ON')
           
        self.add_parameter('closed_loop_active', type=types.StringType,
                flags=Instrument.FLAG_GETSET,
                format_map=DICT_ON_OFF,
                value='ON')

        self.add_function('reset')

        self._host = host
        self._port = int(port)
        self._fridge = fridge
        self._connect()
        self.reset()

    def _connect(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._host, self._port))

    def send(self, data):
        if self._socket is None:
            self._connect()
        self._socket.send(data+'\r\n')

    def ask(self, data, bufsize=1024):
        # print "Beginning to query the fridge"
        self.send(data)
        return self._socket.recv(bufsize).rstrip()

    def do_get_temperature(self):
        channel = self.TEMPCHANNEL
        ret = self.ask('%s:READ:DEV:T%s:TEMP:SIG:TEMP'%(self._fridge,channel))
        # print "Got %s from fridge"%(ret,)
        try:
            match = re.search(":TEMP:SIG:TEMP:(.+)K",ret).group(1)
        except:
            match = -1
        return float(match)
        # how to return an error?
        
    def do_get_temperature_setpoint(self):
        channel = self.TEMPCHANNEL
        ret = self.ask('%s:READ:DEV:T%s:TEMP:LOOP:TSET'%(self._fridge,channel))  
        try:
            match = re.search(":TEMP:LOOP:TSET:(.+)K",ret).group(1)
        except:
            match = -1
        return float(match)

    def do_get_temperature_ramp_rate(self):
        channel = self.TEMPCHANNEL
        ret = self.ask('%s:READ:DEV:T%s:TEMP:LOOP:RAMP:RATE'%(self._fridge,channel))  
        try:
            match = re.search(":TEMP:LOOP:RAMP:RATE:(.+)K",ret).group(1)
        except:
            match = -1
        return float(match)

    def do_get_heater_limit(self):
        channel = self.TEMPCHANNEL
        ret = self.ask('%s:READ:DEV:T%s:TEMP:LOOP:RANGE'%(self._fridge,channel))  
        try:
            match = re.search(":TEMP:LOOP:RANGE:(.+)mA",ret).group(1)
        except:
            match = -1
        return float(match)

    def do_get_heater_power(self):
        channel = self.HEATCHANNEL
        ret = self.ask('%s:READ:DEV:H%s:HTR:SIG:POWR'%(self._fridge,channel))  
        try:
            match = re.search(":SIG:POWR:(.+)uW",ret).group(1)
        except:
            match = -1
        return float(match)
 
    def do_get_still_heater_power(self):
        channel = self.STILLHEATCHANNEL
        ret = self.ask('%s:READ:DEV:H%s:HTR:SIG:POWR'%(self._fridge,channel))  
        try:
            match = re.search(":SIG:POWR:(.+)uW",ret).group(1)
        except:
            match = -1
        return float(match)

    def do_get_pulse_tube(self):
        channel = self.PTCCHANNEL
        ret = self.ask('%s:READ:DEV:C%s:PTC:SIG:STATE'%(self._fridge,channel))
        # print "Got %s from fridge"%(ret,)
        try:
            match = re.search(":STATE:(.+)",ret).group(1)
        except Exception as e:
            match = e
        return match
        
    def do_get_turbo(self):
        channel = self.TURBOCHANNEL
        ret = self.ask('%s:READ:DEV:TURB%s:PUMP:SIG:STATE'%(self._fridge,channel))
        # print "Got %s from fridge"%(ret,)
        try:
            match = re.search(":STATE:(.+)",ret).group(1)
        except Exception as e:
            match = e
        return match

    def do_get_temperature_ramp_enable(self):
        channel = self.TEMPCHANNEL
        ret = self.ask('%s:READ:DEV:T%s:TEMP:LOOP:RAMP:ENAB'%(self._fridge,channel))
        # print "Got %s from fridge"%(ret,)
        try:
            match = re.search(":ENAB:(.+)",ret).group(1)
        except Exception as e:
            match = e
        return match

    def do_get_closed_loop_active(self):
        channel = self.TEMPCHANNEL
        ret = self.ask('%s:READ:DEV:T%s:TEMP:LOOP:MODE'%(self._fridge,channel))
        # print "Got %s from fridge"%(ret,)
        try:
            match = re.search(":MODE:(.+)",ret).group(1)
        except Exception as e:
            match = e
        return match

    def do_set_pulse_tube(self, state):
        channel = self.PTCCHANNEL
        reply = self.ask('%s:SET:DEV:C%s:PTC:SIG:STATE:%s'%(self._fridge,channel,state))
        print "Reply from Sauron:%s"%(reply,)
        return state
        
    def do_set_turbo(self, state):
        channel = self.TURBOCHANNEL
        reply = self.ask('%s:SET:DEV:TURB%s:PUMP:SIG:STATE:%s'%(self._fridge,channel,state))
        print "Reply from Sauron:%s"%(reply,)
        return state

    def do_set_closed_loop_active(self, state):
        channel = self.TEMPCHANNEL
        reply = self.ask('%s:SET:DEV:T%s:TEMP:LOOP:MODE:%s'%(self._fridge,channel,state))
        print "Reply from Sauron:%s"%(reply,)
        return state

    def do_set_temperature_ramp_enable(self, state):
        channel = self.TEMPCHANNEL
        reply = self.ask('%s:SET:DEV:T%s:TEMP:LOOP:RAMP:ENAB:%s'%(self._fridge,channel,state))
        print "Reply from Sauron:%s"%(reply,)
        return state
        
    def do_set_temperature_setpoint(self, temperature):
        channel = self.TEMPCHANNEL
        reply = self.ask('%s:SET:DEV:T%s:TEMP:LOOP:TSET:%f'%(self._fridge,channel,temperature))
        print "Reply from Sauron:%s"%(reply,)
        return temperature
        
    def do_set_temperature_ramp_rate(self, rate):
        channel = self.TEMPCHANNEL
        reply = self.ask('%s:SET:DEV:T%s:TEMP:LOOP:RAMP:RATE:%f'%(self._fridge,channel,rate))
        print "Reply from Sauron:%s"%(reply,)
        return rate

    def do_set_heater_limit(self, current):
        channel = self.TEMPCHANNEL
        reply = self.ask('%s:SET:DEV:T%s:TEMP:LOOP:RANGE:%f'%(self._fridge,channel,current))
        print "Reply from Sauron:%s"%(reply,)
        return current
        
    def do_set_heater_power(self, power):
        channel = self.HEATCHANNEL
        reply = self.ask('%s:SET:DEV:H%s:HTR:SIG:POWR:%f'%(self._fridge,channel,power))
        print "Reply from Sauron:%s"%(reply,)
        return power

    def do_set_still_heater_power(self, power):
        channel = self.STILLHEATCHANNEL
        reply = self.ask('%s:SET:DEV:H%s:HTR:SIG:POWR:%f'%(self._fridge,channel,power))
        print "Reply from Sauron:%s"%(reply,)
        return power

    def reset(self):
        pass
