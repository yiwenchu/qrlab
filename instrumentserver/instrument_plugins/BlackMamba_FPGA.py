# -*- coding: utf-8 -*-
from instrument import Instrument
from instrumentserver.instrument_plugins.Yngwie_FPGA import Yngwie_FPGA
from YngwieInterface import BlackMambaInterface, YngwieBase
import time

class BlackMamba_FPGA(Instrument):
    def __init__(self, name='BM'):
        super(BlackMamba_FPGA, self).__init__(name)
        self.bm = BlackMambaInterface()
        soft_set = Instrument.FLAG_SET | Instrument.FLAG_SOFTGET
        self.add_parameter('n_cards', type=int, flags=soft_set)
        self.add_parameter('card_name', type=str, flags=soft_set, channels=range(4))

    def get_card(self, n=0):
        instruments = self.get_instruments()
        card_name = getattr(self, 'get_card_name%d' % n)()
        card = instruments[card_name]
        if card is None:
            raise EnvironmentError('Could not find card ' + card_name)
        return card

    def get_cards(self):
        return [self.get_card(n) for n in range(self.get_n_cards())]

    def set_dump_path(self, path):
        for card in self.get_cards():
            card.set_dump_path(path)

    def load_tables(self, tables_prefix):
        for i, card in enumerate(self.get_cards()):
            card.load_tables(tables_prefix)

    def accept_stream(self, card_n, *args, **kwargs):
        self.get_card(card_n).accept_stream(card_n, *args, **kwargs)

    def check_trigger(self):
        print 'check trigger'
        return all(c.check_trigger() for c in self.get_cards())

    def update_modes(self):
        for c in self.get_cards():
            c.update_modes()

    def start(self):
        self.bm.start()
        time.sleep(0.2)
        print 'bm start'
        for c in self.get_cards():
            c.start()
            print 'card start'
        self.bm.trigger()
        print 'bm trigger'
        
    def stop(self, close_bm=False):
        for c in self.get_cards():
            c.stop()
        if close_bm:
            self.bm.stop()

if __name__ == '__main__':
    bm = BlackMamba_FPGA()
