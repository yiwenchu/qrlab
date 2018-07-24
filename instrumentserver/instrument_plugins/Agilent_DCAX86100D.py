import visa
import time
import numpy as np
import matplotlib.pyplot as plt

'''
http://www.home.agilent.com/upload/cmc_upload/All/86100_Programming_Guide.pdf?&cc=US&lc=eng
'''

debug = False

NO_ERROR = '0,"No error"'




class FastScope:

    def __init__(self, address):
        self.ins = visa.instrument(address, timeout=5, term_chars='\n')
        self.reset()
        self.set_oscilloscope_mode()
        self.set_num_pts_manual()
        self.set_num_pts(10000)
        self.set_trig_frontpanel(level=0.5)

        self.x = np.array([])
        self.y = np.array([])

    def wr(self, command):
        self.ins.write(command+'\n')
        if debug:
            return self.error_check()

    def ask(self,command,bypass=False):
        ret = self.ins.ask(command+'?\n')
        if debug and not bypass:
            ''' The bypass command makes sure that error querying (an ask)
            doesn't lead to infinite recursion'''
            err = self.error_check()
            return ret
        else:
            return ret

    def reset(self):
        self.wr('*RST')

    def clear(self):
        '''Clears error queue, others'''
        self.wr('*CLS')

    def set_oscilloscope_mode(self):
        self.wr('SYST:MODE OSC')

    def set_trig_frontpanel(self,level=None):
        self.wr('TRIG:SOUR FPAN')
        self.wr('TRIG:BWL EDGE')
        if level:
            self.set_trig_level(level)

    def set_trig_freerun(self):
        self.wr('TRIG:SOUR FRUN')

    def set_trig_level(self,trig_lev):
        self.wr('TRIG:LEV %f' % trig_lev)

    def error_query(self):
        return self.ask('SYSTEM:ERROR',bypass=True)

    def report_error(self,error):
        if error != NO_ERROR:
            print '\tERROR: %s' % error

    def error_check(self):
        ret = self.error_query()
        if ret != NO_ERROR:
            self.report_error(ret)
            self.print_error_queue()
            return False #Return if there is an error.
        else:
            return True #No error return

    def print_error_queue(self):
        ret = self.error_query()
        while ret != NO_ERROR:
            self.report_error(ret)
            ret = self.error_query()

    def set_num_pts_manual(self):
        self.wr('ACQ:RLEN:AUT MAN')
        time.sleep(0.01)

    def set_num_pts_auto(self):
        self.wr('ACQ:RLEN:AUT AUT')

    def set_num_pts(self,num_pts):
        '''Requires being in manual num pts mode'''
        self.wr('ACQ:RLEN %d' % num_pts)

    def set_time_range(self,full_range_in_seconds):
        return self.wr(':TIM:RANG %f' %  full_range_in_seconds)

    def set_averaging(self, num_avg):
        if num_avg == 0:
            self.wr('ACQ:SMO NONE')
        else:
            self.wr('ACQ:SMO AVER')
            self.wr('ACQ:ECO %d' % num_avg)

    def set_time_offset(self,offset_in_seconds):
        #minimum 24ns
        self.wr(':TIM:POS %f' %  offset_in_seconds)

    def take_trace(self):
        x = self.ask('WAV:XYF:ASC:XDAT')
        x = np.array(x.split(','))
        self.x = x.astype(np.float)

        y = self.ask('WAV:XYF:ASC:YDAT')
        y = np.array(y.split(','))
        self.y = y.astype(np.float)

    def plot_trace(self):
        plt.plot(self.x,self.y)

    def set_voltage_range(self,voltage):
        '''set voltage in volts'''
        cmd = 'CHAN1:YSC %f' % (voltage/8.0,) #8 for 8 divs
        self.wr(cmd)

    def set_voltage_offset(self,offs):
        cmd = 'CHAN1:YOFF %f' % offs
        self.wr(cmd)


    def clear_and_wait_for_averages(self,num_avg):
        '''

        does not actually work.


        This command takes over until the FS has taken the specified number
        of averages'''
        self.wr('LTESt:ACQuire:STATe ON')
        self.wr('LTESt:ACQuire:CTYPe:WAVeforms %d' % num_avg)
        self.wr('ACQuire:SMOothing AVERage')
        self.wr('ACQuire:ECOunt %d' % num_avg)
        self.wr('ACQuire:CDISplay')
        self.wr(':ACQuire:RUN')

        done = False
        while not done:
            ret = self.ask('*OPC')
            print ret
            if ret == '1':
                done = True
            time.sleep(1)

#r = FastScope('TCPIP0::172.28.141.133::inst0::INSTR')
#time.sleep(2)
#r.take_trace()
#r = FastScope('fastscope')
##data = r.single_shot()
#data = r.take_single_channel(1)
#plt.plot(data, 'r-')