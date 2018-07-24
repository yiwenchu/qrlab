from PyQt4.QtCore import QSettings
from PyQt4.QtGui import QDoubleSpinBox, QPushButton
from numpy import arange, array
from pyqtgraph import ImageView, PlotWidget
from dataserver import dataserver_helpers
from pyqt_utils import settings_window

from scripts.vna.vna_grab_trace_and_fit import VNAFitWidget


class VNAPowerSweepWidget(VNAFitWidget):
    def __init__(self):
        super(VNAPowerSweepWidget, self).__init__()
        self.power_start = QDoubleSpinBox()
        self.power_stop = QDoubleSpinBox()
        self.power_step = QDoubleSpinBox()
        self.form_layout.addRow("Power Start", self.power_start)
        self.form_layout.addRow("Power Stop", self.power_stop)
        self.form_layout.addRow("Power Stop", self.power_step)

        self.start_sweep_button = QPushButton("Start Sweep")
        self.start_sweep_button.clicked.connect(self.start_sweep)
        self.button_layout.addWidget(self.start_sweep_button)

        self.traces_plot = ImageView()
        self.plot_layout.addWidget(self.traces_plot)
        self.qints_plot = PlotWidget()
        self.plot_layout.addWidget(self.qints_plot)

        self.current_power = None
        self.traces = []
        self.qints = []

    def start_sweep(self):
        powers = arange(self.power_start.value(), self.power_stop.value(), self.power_step.value())
        vna = self.get_vna()
        for i, power in enumerate(powers):
            self.current_power = power
            vna.set_power(power)
            self.grab_trace()
            self.fit_trace()
            self.save_trace()

            self.traces.append(self.mags)
            self.traces_plot.setImage(array(self.traces))

            self.qints.append(self.fit_params['qint'].value)
            self.qints_plot.clear()
            self.qints_plot.plot(self.powers[:i], self.qints)
        self.current_power = None
        g = self.get_h5group()
        g['freqs'] = vna.get_xaxis()
        g['powers'] = powers
        g['qints'] = self.qints
        g['traces'] = self.traces
        dataserver_helpers.set_scale(g, 'powers', 'qints')
        dataserver_helpers.set_scale(g, 'powers', 'traces')
        dataserver_helpers.set_scale(g, 'powers', 'freqs', dim=1)

    def get_h5group(self):
        if self.current_power is not None:
            dataset = str(self.dataset_edit.text())
            self.dataset_edit.setText(dataset + "/%.1f" % self.current_power)
            g = super(VNAPowerSweepWidget, self).get_h5group()
            self.dataset_edit.setText(dataset)
        else:
            return super(VNAPowerSweepWidget, self).get_h5group()

    def grab_trace(self):
        super(VNAPowerSweepWidget, self).grab_trace()
        power_array = self.get_h5group(use_power_child=False)



if __name__ == '__main__':
    settings = QSettings('philreinhold', 'vna_power_sweep_widget')
    settings_window.run(VNAPowerSweepWidget, settings, 1)
