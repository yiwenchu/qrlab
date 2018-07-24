# LeCroy_ArbStudio1104.py class, to perform the communication between the Wrapper and the device
# Jason T. Soo Hoo <jsoohoo@iqc.ca>, 2010
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
#
# Note: This program requires that the LeCroy .NET driver (AWG4000Control.dll) be in the .NET GAC (see Python variable sys.path for location --- c:\python26\DLLs).
# Further, it requires Python for .NET (http://pythonnet.sourceforge.net/) to be properly installed.
# This driver was developed and tested on Python 2.5.

from instrument import Instrument
import types
import logging
import numpy
# Hack to initalize the .NET <-> Python communication
# Hack. Initialization of .NET must be done here. Not sure why...
import clr
clr.AddReference("AWG4000Control")
clr.AddReference("Lecroy driver") # this "Lecroy driver.dll" file is written by Chunqing Deng <cdeng@iqc.ca>. It initialize the arbstudio device and set the sampling frequency. It locates at the same folder as "AWG4000Control.dll"
from ActiveTechnologies.Instruments.AWG4000.Control import DeviceSet, Device, Channel, ARBChannel, WaveformStruct, GenerationSequenceStruct, ARBBufferType
from ActiveTechnologies.Instruments.AWG4000.Control import TransferMode, TriggerMode, TriggerSource, TriggerAction, SensitivityEdge, FrequencyInterpolation
from ActiveTechnologies.Instruments.AWG4000.Control import Functionality, ClockSource, ChannelOutLogicOperation
from Lecroy_driver import Lecroy_1104
from System import Array, Double, Byte, Decimal
from System.Collections.Generic import List

class LeCroy_ArbStudio1104(Instrument):
    '''
    This is the python driver for the LeCroy ArbStudio 1104
    Arbitrary Waveform Generator

    Usage:
    Initialize with
    <name> = instruments.create('name', 'LeCroy_ArbStudio1104')
    '''

    def __init__(self, name, reset=True, leftClock = 250e6, rightClock = 250e6, clockSource = 'internal', exClock = 10e6):
        '''
        Initializes the AWG.

        Input:
            name (string)    : name of the instrument
            reset (bool)     : once the connection to the instrument is established, everything is reset automatically. So the instrument will be reset once this function is called.
            leftClock (float): sampling rate for ch1 and ch2
            rightClock (float): sampling rate for ch3 and ch4
            clockSource (string): 'internal' or 'external'
            exClock (float)  : samping rate of the external clock

        Output:
            None
        '''
        self._leftClock = leftClock
        self._rightClock = rightClock
        self._clockSource = clockSource
        self._exClock = exClock

        logging.debug(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])

        # Add parameters
        self.add_parameter('status', type=types.StringType,
            flags=Instrument.FLAG_GET, channels=(1, 4), channel_prefix='ch%d_')
        self.add_parameter('run_mode', type=types.StringType,
            flags=Instrument.FLAG_SET, channels=(1, 4), channel_prefix='ch%d_')
        self.add_parameter('trigger_in_delay', type=types.FloatType,
            flags=Instrument.FLAG_SET, channels=(1, 4), channel_prefix='ch%d_')
        self.add_parameter('trigger_out_delay', type=types.FloatType,
            flags=Instrument.FLAG_SET, channels=(1, 4), channel_prefix='ch%d_')

        # Add functions
        self.add_function('reset')
        self.add_function('get_all')
        self.add_function('clear_waveforms')
        self.add_function('init_device')
        self.add_function('force_trigger')
        self.add_function('get_error')
        self.add_function('send_waveform')
        self.add_function('force_stop')
        self.add_function('run')
        self.add_function('stop')
        self.add_function('setup_BNC_trigger_out') # this function set all the trigger out logic to OR.
        self.add_function('set_interpolation')
        self.add_function('set_trigger_out')
        self.add_function('set_trigger_in')

        if reset: # here, it does the same thing no matter you reset = True or False. It seems that the device has to be reset to work. That is it.
            self.reset()
        else:
            self.init_device()
            self.get_all()

    def get_dev(self):
        '''
        Only for testing the driver, do not call it.

        Input:
            None
        Output:
            None
        '''
        return self._device

    def init_device(self, set_clock = None):
        '''
        Initalizes device by setting the functionality of each channel (all set to ARB channel here).
        Not recommand to call it directly

        Input:
            None
        Output:
            None
        '''
        logging.debug(__name__  +' : Initializing device...')

        self._CDriver = Lecroy_1104()
        if set_clock == None:
            leftClockDec = Decimal(self._leftClock)
            rightClockDec = Decimal(self._rightClock)
        else:
            self._leftClock = set_clock
            self._rightClock = set_clock
            leftClockDec = Decimal(set_clock)
            rightClockDec = Decimal(set_clock)
        exClockDec = Decimal(self._exClock)

        source = ClockSource.Internal
        if (self._clockSource.upper() == "INTERNAL"):
            source = ClockSource.Internal
        elif (self._clockSource.upper() == "EXTERNAL"):
            source = ClockSource.External
        aterr = self._CDriver.SetSamplingFrequency(leftClockDec, rightClockDec, source, exClockDec)
        self._device = self._CDriver.GetDevice()

        if (self._device == None):
            logging.error(__name__  +' : Initialization failed.')
            print "Initialization failed."

    def reset(self):
        '''
        Resets the instrument to default values

        Input:
            None
        Output:
            None
        '''
        logging.info(__name__ + ' : Resetting instrument')
        self.init_device()
        self.get_all()

    def get_all(self):
        '''
        Reads all implemented parameters from the instrument,
        and updates the wrapper.

        Input:
            None
        Output:
            None
        '''
        logging.info(__name__ + ' : Reading all data from instrument')
        for i in range(1,5):
            self.get('ch%d_status' % i)

    def clear_waveforms(self):
        '''
        Clears the waveform memory on all channels

        Input:
            None
        Output:
            None
        '''
        logging.debug(__name__ + ' : Clear waveforms from channels')

        # loop over all 4 channels and reset samples and generation lists
        if (self._device != None):
            for i in range(0,4):
                self._device.GetChannel(i).ClearBuffer(ARBBufferType.WaveformsSamples)
                self._device.GetChannel(i).ClearBuffer(ARBBufferType.GenerationList)

    def do_set_run_mode(self, mode, channel):
        '''
        This function sets the running mode of each channel

        Input:
            mode (string): "single", 'continuous', 'stepped', or 'burst'
            channel (int): 1 - 4

        Output:
            None
        '''
        logging.debug(__name__  +' : Set trigger mode on channel %s to %s' % (channel, mode))
        if (mode.upper() == "SINGLE"):
            self._device.GetChannel(channel-1).SetTriggerMode(TriggerMode.Single)
        elif (mode.upper() == "CONTINUOUS"):
            self._device.GetChannel(channel-1).SetTriggerMode(TriggerMode.Continuous)
        elif (mode.upper() == "STEPPED"):
            self._device.GetChannel(channel-1).SetTriggerMode(TriggerMode.Stepped)
        elif (mode.upper() == "BURST"):
            self._device.GetChannel(channel-1).SetTriggerMode(TriggerMode.Burst)

    def do_set_trigger_in_delay(self, time, channel):
        '''
        This function trigger in delay.

        Input:
            time (float): in unit of seconds
            channel (int): 1 - 4
        Output:
            None
        '''
        if (channel <= 2):
            delaySample = int(time * self._leftClock)
            self._device.GetChannel(channel - 1).SetTriggerDelay(delaySample)
        else:
            delaySample = int(time * self._rightClock)
            self._device.GetChannel(channel - 1).SetTriggerDelay(delaySample)

    def do_set_trigger_out_delay(self, time, channel):
        '''
        This function does not work.
        do not use
        '''
        if (channel <= 2):
            delaySample = int(time * self._leftClock)
            self._device.GetChannel(channel - 1).SetTriggerOutDelay(delaySample)
        else:
            delaySample = int(time * self._rightClock)
            self._device.GetChannel(channel - 1).SetTriggerOutDelay(delaySample)

    def set_trigger_in(self, channel, source, action, edge):
        '''
        Sets the trigger level of the instrument

        Input:
            channel (int) : 1 to 4, the number of the designated channel
            source (string): 'BNC' or 'DC', type of trigger input
            action (string): 'start' or 'stop', trigger behavior
            edge (string): 'rising' or 'falling', what edge to trigger on

        Output:
            None
        '''
        logging.debug(__name__  + ' : Trigger set: Channel: %s Source: %s Edge: %s Action: %s ' %(channel, source, action, edge))

        # error checking
        if (source.upper() not in ['BNC', 'DC']):
            logging.debug(__name__ + ' : error - source %s' %source)
            return "error: source - %s" % source
        else:
            if (source.upper() == 'BNC'):
                source = TriggerSource.FPTriggerIN
            else:
                source = TriggerSource.DCTriggerIN

        if (action.upper() not in ['START', 'STOP']):
            logging.debug(__name__ + ' : error - action %s' %action)
            return "error: action - %s" % action
        else:
            if (action.upper() == 'START'):
                action = TriggerAction.TriggerStart
            else:
                action = TriggerAction.TriggerStop

        if (edge.upper() not in ['RISING', 'FALLING']):
            logging.debug(__name__ + ' : error - edge %s' %edge)
            return "error: edge - %s" % edge
        else:
            if (edge.upper() == 'RISING'):
                edge = SensitivityEdge.RisingEdge
            else:
                edge = SensitivityEdge.FallingEdge

        if (self._device != None):
            self._device.GetChannel(channel-1).SetExternalTrigger(source, edge, action)

    def set_trigger_out(self, channel, sources, edge):
        '''
        Set the trigger output for each channel. The trigger output setup is complicated, please refer to the AWG manual

        Input:
            channel (int): 1 - 4
            sources (list of string): ['NONE', 'STOP', 'START', 'EVENT_MARKER', 'DCTRIGGERIN', 'FPTRIGGERIN']
            edge (string): 'RISING' or 'FALLING'
        Output:
            None
        '''
        sourceClass = []
        edgeClass = []
        logging.debug(__name__  + ' : Trigger out: Channel: %s Source: %s Edge: %s ' %(channel, sources, edge))
        for oneSource in sources:
            if (oneSource.upper() not in ['NONE', 'STOP', 'START', 'EVENT_MARKER', 'DCTRIGGERIN', 'FPTRIGGERIN']):
                logging.debug(__name__ + ' : error - source %s' %oneSource)
                print "error: source - %s" % oneSource
            elif(oneSource.upper() == 'NONE'):
                sourceClass.append(TriggerSource.None)
            elif(oneSource.upper() == 'STOP'):
                sourceClass.append(TriggerSource.Stop)
            elif(oneSource.upper() == 'START'):
                sourceClass.append(TriggerSource.Start)
            elif(oneSource.upper() == 'EVENT_MARKER'):
                sourceClass.append(TriggerSource.Event_Marker)
            elif(oneSource.upper() == 'DCTRIGGERIN'):
                sourceClass.append(TriggerSource.DCTriggerIN)
            elif(oneSource.upper() == 'FPTRIGGERIN'):
                sourceClass.append(TriggerSource.FPTriggerIN)

        if (edge.upper() not in ['RISING', 'FALLING']):
            logging.debug(__name__ + ' : error - edge %s' %edge)
            print "error: edge - %s" % edge
        else:
            if (edge.upper() == 'RISINGEDGE'):
                edgeClass = SensitivityEdge.RisingEdge
            else:
                edgeClass = SensitivityEdge.FallingEdge

        return self._device.GetChannel(channel-1).SetTriggerOut(sourceClass, edgeClass)

    def do_get_status(self, channel):
        '''
        Gets the status of the designated channel.

        Input:
            channel (int) : 1 to 4 the number of the designated channel

        Output:
            'ILDE' or 'BUSY'
        '''
        logging.debug(__name__ + ' : Get status of channel %s' %channel)
        if (self._device != None):
            val = self._device.GetChannel(channel-1).IsChannelIdle()

            if val:
                return "ILDE"
            else:
                return "BUSY"
        else:
            return None

    def get_error(self):
        '''
        Reads the oerror register of the instrument
        Input:
            None
        Output:
            error (string) : current error message (if any)
        '''
        logging.debug(__name__ + ' : Get error message')

        if (self._device != None):
            return "Source: %s Description: %s" % (self._device.ErrorResult.ErrorSource, self._device.ErrorResult.ErrorDescription)
        else:
            return None

    # Set interpolation factors
    def set_interpolation(self, channel, factor):
        '''
        Sets the interpolation factor for the specified channel. Note that channels 1 and 2, and channels 3 and 4 are linked, so changing either
        channel in the pair changes the interpolation factor for both channels.

        Input:
            channel (int) - 1 to 4, the channel to set the interpolation factor
            factor (int) - 1, 2, or 4, the interpolation factor
        Output:
            error (string) : current error message (if any)
        '''
        logging.debug(__name__ + ' : Set interpolation to %s on channel pair %s' % (factor, channel))

        if (channel < 1 or channel > 4):
            logging.debug(__name__ + ' : Invalid channel %s' % channel)
            print __name__ + ' : Invalid channel %s' % channel

        if (factor not in [1,2,4]):
            logging.debug(__name__ + ' : Invalid factor %s' % factor)
            print __name__ + ' : Invalid factor %s' % factor

        if (self._device != None):
            if (channel == 1 or channel == 2):
                if (factor == 1):
                    return self._device.PairLeft.SetFrequencyInterpolation(FrequencyInterpolation.Frequency1X)
                elif (factor == 2):
                    return self._device.PairLeft.SetFrequencyInterpolation(FrequencyInterpolation.Frequency2X)
                else:
                    return self._device.PairLeft.SetFrequencyInterpolation(FrequencyInterpolation.Frequency4X)
            else:
                if (factor == 1):
                    return self._device.PairRight.SetFrequencyInterpolation(FrequencyInterpolation.Frequency1X)
                elif (factor == 2):
                    return self._device.PairRight.SetFrequencyInterpolation(FrequencyInterpolation.Frequency2X)
                else:
                    return self._device.PairRight.SetFrequencyInterpolation(FrequencyInterpolation.Frequency4X)
        else:
            return None


    # Send waveform to the device
    def send_waveform(self, waveform, channel):
        '''
        Sends a complete waveform. All parameters need to be specified.

        Input:
            waveform (float[numpoints]) : waveform
            channel (int) : 1 to 4, the number of the designated channel
        Output:
            None
        '''
        logging.debug(__name__ + ' : Sending waveform to instrument')

        wfs = Array.CreateInstance(WaveformStruct, 1)

        data = Array.CreateInstance(Double, len(waveform))

        ctr = 0

        for d in waveform:
            data.SetValue(float(d), ctr)
            ctr = ctr + 1

        # Done because Python assumes all variables are references
        x = wfs.GetValue(0)
        x.Sample = data
        markerarr = Array.CreateInstance(Double, 1)
        markerarr.SetValue(0, 0)
        x.Marker = markerarr
        wfs.SetValue(x, 0)

        if (self._device != None):
            oChannel = ARBChannel()
            oChannel = self._device.GetChannel(channel-1)
            res = oChannel.LoadWaveforms(wfs)

            if (res.ErrorSource != 0):
                return False

            seq = Array.CreateInstance(GenerationSequenceStruct, 1)

            # Done because Python assumes all variables are references
            x = seq.GetValue(0)
            x.WaveformIndex = 0
            x.Repetitions = 1
            seq.SetValue(x, 0)

            res = oChannel.LoadGenerationSequence(seq, TransferMode.NonReEntrant, True) # Hack to force true boolean type passing

            if (res.ErrorSource != 0):
                return False

            return True
        else:
            logging.debug(__name__ + ' : Could not get channel %s' % channel)
            print "Error getting channel!"
            return False

    def setup_BNC_trigger_out(self):
        '''
        see P45 ArbStudio-GSM_RevA.pdf for detail

        Input:
            None
        Output:
            None
        '''
        logging.debug(__name__ + ' : Setup the BNC trigger output.')
        result = self._device.SetupBNCTriggerOut(ChannelOutLogicOperation.OperationOR, ChannelOutLogicOperation.OperationOR, ChannelOutLogicOperation.OperationOR)
        # the input paras are ChannelOutLogicOperation, see P45 ArbStudio-GSM_RevA.pdf for detail
        print result.ErrorDescription

    def force_stop(self, channellist):
        '''
        Force stop running.

        Input:
            channellist (list of int): a list of channel No. e.g. [1,2,3,4]
        Output:
            None
        '''
        listByte = List[Byte]()
        for c in channellist:
            listByte.Add(c) # No -1 here because LeCroy changed conventions mid code.
        logging.debug(__name__ + ' : Force stop on channel %s' %(channellist))
        self._device.ForceStop(listByte.ToArray())

    def run(self, channellist):
        '''
        Run the channel on list.

        Input:
            channellist (list of int): a list of channel No. e.g. [1,2,3,4]
        Output:
            None
        '''
        listByte = List[Byte]()
        for c in channellist:
            listByte.Add(c) # No -1 here because LeCroy changed conventions mid code.
        logging.debug(__name__ + ' : run on channel %s' %(channellist))
        self._device.RUN(listByte.ToArray())

    def stop(self):
        '''
        Stop all running channels.

        Input:
            channellist (list of int): a list of channel No. e.g. [1,2,3,4]
        Output:
            None
        '''
        logging.debug(__name__ + ' : stop on all channels')
        self._device.STOP()

    def force_trigger(self, channellist):
        '''
        Force internal trigger the channels in the list.

        Input:
            channellist (list of int): a list of channel No. e.g. [1,2,3,4]
        Output:
            None
        '''
        listByte = List[Byte]()
        for c in channellist:
            listByte.Add(c) # No -1 here because LeCroy changed conventions mid code.
        logging.debug(__name__ + ' : Force trigger on channel %s' %(channellist))
        self._device.ForceTrigger(listByte.ToArray())