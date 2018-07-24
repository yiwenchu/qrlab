from PyQt4.QtCore import QSettings
from PyQt4.QtGui import QPushButton, QApplication
import numpy
from numpy.fft import rfft, fftfreq
from pyqtgraph import PlotWidget
from dataserver import dataserver_helpers

from pyqt_utils import settings_window
from scripts.vna.vna_widget import VNAWidget


__ui_version__ = 1

class VNAVibrationWidget(VNAWidget):
    def __init__(self):
        super(VNAVibrationWidget, self).__init__()
        self.setWindowTitle("VNA Vibration Analyzer")
        self.fft_plot = PlotWidget()
        self.mean_fft_plot = PlotWidget()
        self.mean_fft_plot.addLegend()
        self.plot_layout.addWidget(self.fft_plot)
        self.plot_layout.addWidget(self.mean_fft_plot)
        continuous_button = QPushButton("Continuous Acquire")
        break_button = QPushButton("Break")
        reset_button = QPushButton("Reset Average")
        self.button_layout.addWidget(continuous_button)
        self.button_layout.addWidget(break_button)
        self.button_layout.addWidget(reset_button)
        continuous_button.clicked.connect(self.continuous_acquire)
        break_button.clicked.connect(self.set_break_acquire)
        reset_button.clicked.connect(self.reset_averaging)
        self.reset_averaging()

    def reset_averaging(self):
        self.acquisition_number = 0
        self.fft_points = {}
        self.mean_fft_points = {}
        self.mean_fft_plot.clear()

    def grab_trace(self):
        super(VNAVibrationWidget, self).grab_trace()
        self.acquisition_number += 1

        # Frequency Axis
        sweep_time = self.get_vna().get_sweep_time()
        n_points = self.get_vna().get_points()

        self.fft_plot.clear()
        self.mean_fft_plot.clear()
        self.mean_fft_plot.plotItem.legend.setParent(None)
        self.mean_fft_plot.addLegend()

        for name, dataset, pen in [('mag', self.mags, 'r'), ('phase', self.phases, 'g')]:
            self.fft_points[name] = numpy.abs(rfft(dataset - dataset.mean()))
            if name not in self.mean_fft_points:
                self.mean_fft_points[name] = self.fft_points[name]
            else:
                self.mean_fft_points[name] += self.fft_points[name]
                self.mean_fft_points[name] /= float(self.acquisition_number-1) / self.acquisition_number

            self.fft_freqs = fftfreq(n_points, sweep_time / n_points)[:len(self.fft_points[name])]


            self.fft_plot.plot(self.fft_freqs, self.fft_points[name], pen=pen)
            self.mean_fft_plot.plot(self.fft_freqs, self.mean_fft_points[name], pen=pen, name=name)
        self.mean_fft_plot.autoRange()


    def save_trace(self):
        super(VNAVibrationWidget, self).save_trace()
        group = self.get_h5group()
        group['fft_freqs'] = self.fft_freqs
        for name in ('mag', 'phase'):
            group[name+'_fft'] = self.fft_points[name]
            group[name+'_mean_fft'] = self.mean_fft_points[name]
            dataserver_helpers.set_scale(group, 'fft_freqs', name+'_fft')
            dataserver_helpers.set_scale(group, 'fft_freqs', name+'_mean_fft')
        group.attrs['n_averages'] = self.acquisition_number
        group.file.close()

    def set_break_acquire(self):
        self.break_acquire = True

    def continuous_acquire(self):
        self.break_acquire = False
        while not self.break_acquire:
            self.grab_trace()
            self.message('Acquisition Number %d' % self.acquisition_number)
            QApplication.instance().processEvents()


if __name__ == "__main__":
    settings = QSettings('philreinhold', 'vna_vibration_widget')
    settings_window.run(VNAVibrationWidget, settings, __ui_version__)
