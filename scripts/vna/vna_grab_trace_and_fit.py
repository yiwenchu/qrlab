from PyQt4.QtCore import QSettings
from PyQt4.QtGui import QComboBox, QPushButton

from scripts.vna.cavity_analysis import fit_asymmetric_db_hanger, fit_db_lorentzian
from dataserver import dataserver_helpers
from pyqt_utils import settings_window
from scripts.vna.vna_widget import VNAWidget



# TODO: Fit-Range Selectors
# TODO: Circle-Plot

__ui_version__ = 1


class VNAFitWidget(VNAWidget):
    def __init__(self):
        super(VNAFitWidget, self).__init__()
        self.fit_combo_box = QComboBox()
        self.fit_combo_box.addItems(["Lorentzian", "Asymmetric Hanger"])
        self.form_layout.addRow("Fit Type", self.fit_combo_box)
        fit_button = QPushButton("Fit")
        self.button_layout.addWidget(fit_button)
        self.fitted_mags = None
        self.vna_params = None
        self.fit_params = {}

        fit_button.clicked.connect(self.fit_trace)

    def grab_trace(self):
        self.fitted_mags = None
        self.fit_params = {}
        super(VNAFitWidget, self).grab_trace()

    def replot(self):
        super(VNAFitWidget, self).replot()
        if self.fitted_mags is not None:
            self.mag_plot.plot(self.freqs, self.fitted_mags, pen='r', name='Fit')

    def fit_trace(self):
        fit_type = str(self.fit_combo_box.currentText())
        if fit_type == "Asymmetric Hanger":
            self.fit_params, self.fitted_mags = fit_asymmetric_db_hanger(self.freqs, self.mags)
        elif fit_type == "Lorentzian":
            self.fit_params, self.fitted_mags = fit_db_lorentzian(self.freqs, self.mags)
        self.fit_params = {k: v.value for k, v in self.fit_params.items()}
        self.message("Parameters")
        self.message("----------")
        for pn, pv in self.fit_params.items():
            self.message(pn, ":", pv)
        self.message("")
        self.replot()

    def save_trace(self):
        super(VNAFitWidget, self).save_trace()
        group = self.get_h5group()
        if 'fitted_mags' in group:
            del group['fitted_mags']
        if self.fitted_mags is not None:
            group['fitted_mags'] = self.fitted_mags
            dataserver_helpers.set_scale(group, 'freqs', 'fitted_mags')


if __name__ == '__main__':
    settings = QSettings('philreinhold', 'vna_fit_widget')
    settings_window.run(VNAFitWidget, settings, __ui_version__)
