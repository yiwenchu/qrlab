# demo 2d scanner.
#
# author: wolfgang pfaff <w dot pfaff at tudelft dot nl>

from instrument import Instrument
# from cyclopean_instrument import CyclopeanInstrument
import types
import time
import numpy

import qt

class scan2d_demo(Instrument):
    def __init__(self, name):
        Instrument.__init__(self, name, tags=['measure'])

        #self._ins_linescan2d = ins_linescan2d
        #self._ins_pos2d = ins_pos2d
        #self._ins_z = ins_z
        #self._ins_opt3d = ins_opt3d

        # also get the counter, need to disable when linescanning
        # self._counters = qt.instruments['counters_demo']
        self._counter_was_running = False

        # add the relevant parameters for a 2D PL scanner
        self.add_parameter('pixel_time', type=types.FloatType,
                           flags=Instrument.FLAG_GETSET,
                           units='ms',
                           minval=1.0, maxval=99.0,
                           doc="""
                           Integration time per image pixel.
                           """)

        self.add_parameter('xstart',
                           type=types.FloatType,
                           flags=Instrument.FLAG_GETSET,
                           units='um')

        self.add_parameter('xstop',
                           type=types.FloatType,
                           flags=Instrument.FLAG_GETSET,
                           units='um')

        self.add_parameter('ystart',
                           type=types.FloatType,
                           flags=Instrument.FLAG_GETSET,
                           units='um')

        self.add_parameter('ystop',
                           type=types.FloatType,
                           flags=Instrument.FLAG_GETSET,
                           units='um')

        self.add_parameter('xsteps',
                           type=types.IntType,
                           flags=Instrument.FLAG_GETSET,
                           units='')

        self.add_parameter('ysteps',
                           type=types.IntType,
                           flags=Instrument.FLAG_GETSET,
                           units='')

        self.add_parameter('last_line_index',
                           type=types.ObjectType,
                           flags=Instrument.FLAG_GET,
                           units='',
                           doc="""
                           Returns the index of the last line of which data is
                           available.
                           """)

        self.add_parameter('last_line',
                           type=types.ObjectType,
                           flags=Instrument.FLAG_GET,
                           units='cps',
                           doc="""
                           Returns the last line of the measured data.
                           """)

        self.add_parameter('counter',
                           type=types.IntType,
                           flags=Instrument.FLAG_GETSET)

        self.add_parameter('counter2',
                           type=types.IntType,
                           flags=Instrument.FLAG_GETSET)

        self.add_parameter('counter3',
                           type=types.IntType,
                           flags=Instrument.FLAG_GETSET)

        self.add_parameter('counter4',
                           type=types.IntType,
                           flags=Instrument.FLAG_GETSET)


        # parameters to access the member instruments
        # self.add_parameter('position',
        #                    type = types.FloatType,
        #                    flags=Instrument.FLAG_GETSET,
        #                    units='um',
        #                    channels = ('x', 'y'),
        #                    channel_prefix='%s_',
        #                    )
        #
        # self.add_parameter('focus',
        #                    type = types.FloatType,
        #                    flags=Instrument.FLAG_GETSET,
        #                    units='um',
        #                    )
        #
        # # relevant functions to be visible outside
        self.add_function('get_line')
        self.add_function('get_lines')
        self.add_function('get_x')
        self.add_function('get_y')
        self.add_function('get_data')
        self.add_function('setup_data')
        self.add_function('move_abs_xy')
        #
        # # default params
        # self.set_pixel_time(1.0)
        # self.set_xstart(-2.0)
        # self.set_xstop(2.0)
        # self.set_ystart(-2.0)
        # self.set_ystop(2.0)
        # self.set_xsteps(11)
        # self.set_ysteps(11)
        # self.set_counter(1)
        #
        # # self._position = { 'x': self._ins_pos2d.get_x_position(),
        # #                    'y': self._ins_pos2d.get_y_position(), }
        # self._focus = 0 # self._ins_z.get_position()
        # self._position = { 'x': 0., 'y': 0. }
        #
        # # connect instruments
        # # def _ins_pos2d_changed(unused, changes, *arg, **kw):
        # #            for k in ['x_position', 'y_position']:
        # #                if k in changes: getattr(self, 'set_' + k)(changes[k])
        #
        # # self._ins_pos2d.connect('changed', _ins_pos2d_changed)
        # # self._ins_z.connect('changed', _ins_z_changed)
        #
        # # more set up
        # self.setup_data()

        self._current_line = 0
        self._last_line = 0
        self._busy = False
        self.deimudderseigsicht = 'ugly'

        self._supported = {
            'get_running': True,
            'get_recording': False,
            'set_running': False,
            'set_recording': False,
            'save': True,
            }


        ### debug, stuff from cyclopean instrument
        self.add_function('is_supported')
        self.add_function('supported')

        # save internally stored data (via a qtlab measurement)
        self.add_function('save')

    # get and set functions
    def do_set_pixel_time(self, val):
        self._pixel_time = val

    def do_get_pixel_time(self):
        return self._pixel_time

    def do_set_measuring(self, val):
        self._measuring = val

    def do_get_measuring(self):
        return self._measuring

    def do_set_xstart(self, val):
        self._xstart = val

    def do_get_xstart(self):
        return self._xstart

    def do_set_xstop(self, val):
        self._xstop = val

    def do_get_xstop(self):
        return self._xstop

    def do_set_ystart(self, val):
        self._ystart = val

    def do_get_ystart(self):
        return self._ystart

    def do_set_ystop(self, val):
        self._ystop = val

    def do_get_ystop(self):
        return self._ystop

    def do_set_xsteps(self, val):
        self._xsteps = val

    def do_get_xsteps(self):
        return self._xsteps

    def do_set_ysteps(self, val):
        self._ysteps = val

    def do_get_ysteps(self):
        return self._ysteps

    def do_get_last_line(self):
        return self._data[self._last_line,:].tolist()

    def do_get_last_line_index(self):
        return self._last_line

    def do_set_counter(self, val):
        self._counter = val

    def do_get_counter(self):
        return self._counter

    def do_set_counter2(self, val):
        self._counter = val

    def do_get_counter2(self):
        return self._counter

    def do_set_counter3(self, val):
        self._counter = val

    def do_get_counter3(self):
        return self._counter

    def do_set_counter4(self, val):
        self._counter = val

    def do_get_counter4(self):
        return self._counter

    # get and set for instruments
    def do_get_position(self, channel):
        return self._position[channel]

    def do_set_position(self, val, channel):
        self._position[channel] = val
        # f = getattr(self._ins_pos2d, 'set_' + channel + '_position')
        # f(val)

    def do_set_focus(self, val):
        self._focus = val
        # self._ins_z.set_position(val)

    def do_get_focus(self):
        return self._focus


    # the publicly visible functions declared above
    def get_line(self, line):
        return self._data[line,:].tolist()

    def get_lines(self, lines):
        return self._data[lines,:].tolist()

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_data(self):
        return self._data.tolist()

    def setup_data(self):
        self._data = numpy.zeros((self._ysteps, self._xsteps))
        self._x = numpy.r_[self._xstart:self._xstop:self._xsteps*1j]
        self._y = numpy.r_[self._ystart:self._ystop:self._ysteps*1j]
        #
        # # setup demo data
        # xx, yy = numpy.meshgrid(self._x, self._y)
        # self._demo_data = numpy.exp(-xx**2-yy**2)

### debugging: methods from cyclopean instrument
    def do_set_sampling_interval(self, val):
        self._sampling_interval = val

    def do_get_sampling_interval(self):
        return self._sampling_interval

    def do_set_is_running(self, val):
        self._is_running = val
        if val: self._start_running()
        else: self._stop_running()

    def do_get_is_running(self):
        return self._is_running

    def do_set_is_recording(self, val):
        self._is_recording = val
        if val: self._start_recording()
        else: self._stop_recording()

    def do_get_is_recording(self):
        return self._is_recording

    def is_supported(self, s):
        if self._supported.has_key(s):
            return self._supported[s]
        else:
            return False

    def supported(self):
        return self._supported

    def save(self, meta=""):
        pass # not implemented by default


    def _sampling_event(self):
        pass

    def _start_running(self):
        gobject.timeout_add(self._sampling_interval, self._sampling_event)

    def _stop_running(self):
        pass

    def _start_recording(self):
        pass

    def _stop_recording(self):
        pass

    # overloading save function
    # def save(self, meta=""):
    #     CyclopeanInstrument.save(self, meta)
    #     return

        # from wp_toolbox import qtlab_data
        # qtlab_data.save(self.get_name(), meta, x__x__um=self._x, y__y__um=self._y, z__counts__Hz=self._data)

    # internal functions
    # def _start_running(self):
        # CyclopeanInstrument._start_running(self)

        # make sure the counter is off.
        #if self._counters.get_is_running():
        #    self._counter_was_running = True
        #    self._counters.set_is_running(False)

        # self.setup_data()
        #         self._current_line = 0
        #         self._last_line = 0
        #         self._next_line()

    # def _stop_running(self):
    #     #if self._counter_was_running:
    #     #    self._counters.set_is_running(True)
    #     #    self._counter_was_running = False
    #     return
    #
    # def _sampling_event(self):
    #     return False
        # if not self._is_running:
        #             return False
        #
        #         if self._busy:
        #             if time.time() < self._line_start_time + self._x.size * self._pixel_time / 1000.:
        #                 return True
        #             else:
        #                 self._busy = False

        # self._data[self._current_line,:] = self._demo_data[self._current_line,:]
        # self._last_line = self._current_line
        # self._current_line += 1
        # self.get_last_line_index()

        # print some debug info
        # print 'got new line: '
        # print self._data[self._current_line-1,:]

        # if self._ins_linescan2d.get_is_running():
        #     return True
        # else:
        #     f = getattr(self._ins_linescan2d,
        #                 'get_counter' + str(self._counter) + '_values')
        #     self._data[self._current_line,:] = f()
        #     self._last_line = self._current_line
        #     self._current_line += 1
        #     self.get_last_line_index()
        #

        # if self._current_line <= self._y.size - 1:
        #      self._next_line()
        #      return True
        #  else:
        #      self._center_pos()
        #      self.save()
        #      self.set_is_running(False)
        #      if self._counter_was_running:
        #          self._counters.set_is_running(True)
        #          self._counter_was_running = False
        #      return False

    # instrument access
    def _center_pos(self):
        self.set_x_position((self._xstart+self._xstop)/2)
        self.set_y_position((self._ystart+self._ystop)/2)
        return

    def move_abs_xy(self, x, y):
        self.set_x_position(x)
        self.set_y_position(y)
        return

    # we use the linescanner instrument to get the actual data
    def _next_line(self):
        self._busy = True
        self._line_start_time = time.time()
        # y0 = self._y[self._current_line]
        # y1 = y0
        # x0 = self._x[0]
        # x1 = self._x[-1]
        # self._ins_linescan2d.set_start((x0, y0))
        # self._ins_linescan2d.set_end((x1, y1))
        # self._ins_linescan2d.set_nr_of_points(self._xsteps)
        # self._ins_linescan2d.set_pixel_time(self._pixel_time)
        # self._ins_linescan2d.set_is_running(True)
        return True


