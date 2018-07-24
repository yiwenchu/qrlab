from instrument import Instrument
import types

ROTATIONS = (
    'Gaussian',
    'GaussianSquare',
    'Square',
    'Triangle',
    'Sinc',
    'Hanning',
    'Kaiser',
    'FlatTop',
)

class Qubit_Info(Instrument):

    def __init__(self, name, **kwargs):
        Instrument.__init__(self, name, tags=['virtual'])

#        self.add_parameter('rfsource', type=types.StringType,
#                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
#                set_func=lambda x: True)
        self.add_parameter('deltaf', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                units='Hz')
        self.add_parameter('sideband_period', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET)
        self.add_parameter('sideband_phase', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True)
        self.add_parameter('channels', type=types.StringType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True,
                doc='The physical channel this qubit should be in'
                )
        self.add_parameter('sideband_channels', type=types.StringType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True,
                doc='The sequencing channel for this qubit, sideband modulation will let it end up in the physical channels.'
                )
        self.add_parameter('rotation', type=types.StringType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                option_list=ROTATIONS,
                set_func=lambda x: True, value='Gaussian')

        self.add_parameter('rotation_selective', type=types.StringType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                option_list=ROTATIONS,
                set_func=lambda x: True, value='Gaussian')

        self.add_parameter('w', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True, value=10,
                doc='For Gaussian rotations sigma, for others the pulse width'
                )

        self.add_parameter('w_selective', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True, value=10,
                doc='For Gaussian rotations sigma, for others the pulse width'
                )

        self.add_parameter('pi_amp', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True, value=1.0,
                doc='''The amplitude for a pi pulse. If pi/2 amp is specified
                as well a quadratic interpolation will be performed''',
                )

        self.add_parameter('pi_amp_selective', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True, value=1.0,
                doc='''The amplitude for a pi pulse. If pi/2 amp is specified
                as well a quadratic interpolation will be performed''',
                )

        self.add_parameter('pi2_amp', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True, value=0,
                doc='The amplitude for a pi/2 pulse',
                )

        self.add_parameter('pi2_amp_selective', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True, value=0,
                doc='The amplitude for a pi/2 pulse',
                )

        self.add_parameter('drag', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True, value=0,
                doc='''The drag parameter, specifies which fraction of the
                derivative should be added to the other quadrature''',
                )

        self.add_parameter('drag_selective', type=types.FloatType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                set_func=lambda x: True, value=0,
                doc='''The drag parameter, specifies which fraction of the
                derivative should be added to the other quadrature''',
                )

        self.add_parameter('marker_channel', type=types.StringType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                value='',
                set_func=lambda x: True,
                doc='Marker channel for activity',
                )
        self.add_parameter('marker_ofs', type=types.IntType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                value=8,
                set_func=lambda x: True,
                doc='Marker offset with respect to activity, 0 = start simultaneously',
                )
        self.add_parameter('marker_bufwidth', type=types.IntType,
                flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                value=4,
                set_func=lambda x: True,
                doc='Marker buffer width, total buffer = activity + bufwidth',
                )
        self.set(kwargs)

    def do_set_deltaf(self, val, calc=True):
        if not calc:
            return

        if val == 0:
            period = 1e50

        else:
            period = 1e9 / val
        self.set_sideband_period(period, calc=False)

    def do_set_sideband_period(self, val, calc=True):
        if not calc:
            return
        if val == 0:    # This shouldn't occur for period...
            deltaf = 1e50
        else:
            deltaf = 1e9 / val
        self.set_deltaf(deltaf, calc=False)
