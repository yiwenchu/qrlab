import types
from instrument import Instrument
import ctypes
import logging
logging.getLogger().setLevel(logging.WARNING)

class PositionerInfo(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int),
                ("locked", ctypes.c_bool)]

class AttocubeError(IOError):
    pass

def check_error(x):
    if x != 0:
        error_string = {
            -1: "NCB_Error",
            1: "NCB_Timeout",
            2: "NCB_NotConnected",
            3: "NCB_DriverError",
            4: "NCB_BootIgnored",
            5: "NCB_FileNotFound",
            6: "NCB_InvalidParam",
            7: "NCB_DeviceLocked",
            8: "NCB_NotSpecifiedParam"
        }[x]
        raise AttocubeError(error_string)


class Attocube_ANC350(Instrument):
    def __init__(self, name="ANC350", **kwargs):
        super(Attocube_ANC350, self).__init__(name)

        try:
            self.dll = ctypes.windll.LoadLibrary("hvpositionerv2")
        except Exception as e:
            logging.warn("Make sure hvpositionerv2.dll is in instrumentserver directory")
            raise

        positioner_info = PositionerInfo(0, False)
        # todo: Configure properly

        n_devices = self.dll.PositionerCheck(ctypes.byref(positioner_info))
        if n_devices == 0:
            raise AttocubeError("no ANC350 devices found")
        if positioner_info.locked:
            raise AttocubeError("ANC350 device is locked")

        self.add_parameter("position", units="nm", flags=Instrument.FLAG_GET)
        self.add_parameter("frequency", units="Hz")
        self.add_parameter("amplitude", units="V")
        self.add_parameter("speed", units="nm/s", flags=Instrument.FLAG_GET)
        self.add_parameter("step_width", units="nm", flags=Instrument.FLAG_GET)
        self.add_parameter("step_count", flags=Instrument.FLAG_SET)
        self.add_parameter("capacitance", units="pF", flags=Instrument.FLAG_GET)

        self.add_function("move_up_step")
        self.add_function("move_down_step")
        self.add_function("stop_moving")

        self.handle = ctypes.c_int(0)
        check_error(self.dll.PositionerConnect(0, ctypes.byref(self.handle)))
        self.enable_output()
        self.load_actor_profile(r"C:\Phil\Attocube\ANC350_GUI\general_APS_files\ANPz101res_70V.aps")
        self.set(kwargs)
        self.get_all()


    def add_parameter(self, name, **kwargs):
        if 'type' not in kwargs:
            kwargs['type'] = types.IntType
        return super(Attocube_ANC350, self).add_parameter(name, **kwargs)

    def get_int_from_func(self, func):
        v = ctypes.c_long(0)
        check_error(func(self.handle, 0, ctypes.byref(v)))
        return v.value

    def set_from_func(self, v, func):
        check_error(func(self.handle, 0, v))

    def do_get_position(self):
        return self.get_int_from_func(self.dll.PositionerGetPosition)

    def do_set_step_count(self, count):
        self.set_from_func(count, self.dll.PositionerStepCount)

    def do_get_speed(self):
        return self.get_int_from_func(self.dll.PositionerGetSpeed)

    def do_get_step_width(self):
        return self.get_int_from_func(self.dll.PositionerGetStepwidth)

    def do_get_amplitude(self):
        return self.get_int_from_func(self.dll.PositionerGetAmplitude)

    def do_set_amplitude(self, amp):
        self.set_from_func(amp, self.dll.PositionerAmplitude)

    def do_get_frequency(self):
        return self.get_int_from_func(self.dll.PositionerGetFrequency)

    def do_set_frequency(self, freq):
        self.set_from_func(freq, self.dll.PositionerFrequency)

    def do_get_capacitance(self):
        return self.get_int_from_func(self.dll.PositionerCapMeasure)

    def set_output(self, enabled):
        check_error(self.dll.PositionerSetOutput(self.handle, 0, enabled))

    def enable_output(self):
        self.set_output(True)

    def disable_output(self):
        self.set_output(False)

    def move_single_step(self, dir):
        check_error(self.dll.PositionerMoveSingleStep(self.handle, 0, dir))
        self.get_position()

    def move_up_step(self):
        self.move_single_step(0)

    def move_down_step(self):
        self.move_single_step(1)

    def stop_moving(self):
        check_error(self.dll.PositionerStopMoving(self.handle, 0))
        self.get_position()

    def load_actor_profile(self, filename):
        check_error(self.dll.PositionerLoad(self.handle, 0, filename))

if __name__ == '__main__':
    atc = Attocube_ANC350()
    atc.enable_output()
    atc.set_frequency(200)
    atc.set_step_count(100)
    atc.move_up_step()
