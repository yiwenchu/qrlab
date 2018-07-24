from math import ceil

from PyQt4.QtCore import QSettings
from PyQt4.QtGui import QApplication, QProgressBar
from pyqtgraph.widgets.SpinBox import SpinBox
from pyqt_utils import settings_window

from scripts.vna.vna_widget import VNAWidget


class WideSpanVNAWidget(VNAWidget):
    def __init__(self):
        super(WideSpanVNAWidget, self).__init__()
        self.start_spin = SpinBox(value=3e9)
        self.form_layout.addRow("Start Frequency", self.start_spin)
        self.stop_spin = SpinBox(value=9e9)
        self.form_layout.addRow("Stop Frequency", self.stop_spin)
        self.step_spin = SpinBox(value=1e6)
        self.form_layout.addRow("Delta Frequency", self.step_spin)
        self.progress_bar = QProgressBar()
        self.form_layout.addRow(self.progress_bar)

    def grab_trace(self):
        vna = self.get_vna()
        n_points_per_trace = vna.get_points()
        start = self.start_spin.value()
        stop = self.stop_spin.value()
        step = self.step_spin.value()
        span = stop - start
        n_traces = int(ceil(span / (step * n_points_per_trace)))
        trace_size = span / n_traces
        trace_starts = [start + i*trace_size for i in range(n_traces)]
        trace_ends = [start + (i+1)*trace_size for i in range(n_traces)]
        self.freqs = []
        self.mags = []
        for i, (start_f, stop_f) in enumerate(zip(trace_starts, trace_ends)):
            self.progress_bar.setValue(int((100.0*(1+i))/n_traces))
            vna.set_start_freq(start_f)
            vna.set_stop_freq(stop_f)
            self.freqs.extend(vna.do_get_xaxis())
            m, _ = vna.do_get_data(opc=True)
            self.mags.extend(m)
            self.replot()
            QApplication.instance().processEvents()
        self.progress_bar.setValue(0)

if __name__ == '__main__':
    settings = QSettings('philreinhold', 'wide_span_vna_widget')
    settings_window.run(WideSpanVNAWidget, settings)
