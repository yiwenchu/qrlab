import os
from PyQt4.QtCore import QSettings
import h5py
import numpy
from PyQt4.QtGui import QWidget, QFormLayout, QLineEdit, QPushButton, QHBoxLayout, \
    QPlainTextEdit, QFileDialog, QMessageBox, QVBoxLayout, QDialog, QDialogButtonBox, QAbstractItemView
from pyqtgraph import PlotWidget, time
from mclient import instruments, datasrv
from pyqt_utils.h5_widgets import H5File, H5View

import qt_helpers
import h5helpers

# TODO: Circle-Plot

__ui_version__ = 3

class VNAWidget(QWidget):
    settings = None
    def __init__(self):
        super(VNAWidget, self).__init__()
        self.setWindowTitle("VNA Window")
        layout = QVBoxLayout(self)

        self.plot_layout = QHBoxLayout()
        layout.addLayout(self.plot_layout)
        self.mag_plot = PlotWidget()
        self.mag_plot.addLegend()
        self.plot_layout.addWidget(self.mag_plot)

        self.message_box = QPlainTextEdit()
        layout.addWidget(self.message_box)

        self.form_layout = QFormLayout()
        layout.addLayout(self.form_layout)

        self.dataset_edit = QLineEdit("TestSample")
        save_file_button = QPushButton("HDF5 File")
        self.save_file_edit = QLineEdit("C:\\_Data\\test.h5")
        # self.form_layout.addRow("VNA Address", self.address_combo_box)
        self.form_layout.addRow(save_file_button, self.save_file_edit)
        dataset_button = QPushButton("Dataset")
        self.form_layout.addRow(dataset_button, self.dataset_edit)

        self.button_layout = QHBoxLayout()
        layout.addLayout(self.button_layout)
        grab_trace_button = QPushButton("Grab Trace")
        save_button = QPushButton("Save")
        self.button_layout.addWidget(grab_trace_button)
        self.button_layout.addWidget(save_button)

        self.freqs = None
        self.mags = None
        self.phases = None
        self.vna = None
        self.current_vna_addr = None
        self.vna_params = None

        save_file_button.clicked.connect(self.change_save_file)
        dataset_button.clicked.connect(self.change_dataset)
        grab_trace_button.clicked.connect(self.grab_trace)
        save_button.clicked.connect(self.save_trace)

    def get_vna(self):
        return instruments['VNA']

    def grab_trace(self):
        vna = self.get_vna()
        self.freqs = vna.do_get_xaxis()
        self.mags, self.phases = vna.do_get_data()
        self.replot()
        self.vna_params =  vna.get_parameter_values(query=True)
        self.message("VNA Params")
        self.message("----------")
        for k, v in self.vna_params.items():
            self.message(k, ":", v)
        self.message("")

    def replot(self):
        self.mag_plot.clear()
        self.mag_plot.plotItem.legend.setParent(None)
        self.mag_plot.addLegend()
        if self.freqs is None:
            return
        self.mag_plot.plot(self.freqs, self.mags, pen='g', name='Data')

    def message(self, *objs):
        self.message_box.appendPlainText(" ".join([str(o) for o in objs]))

    def change_save_file(self):
        filename = QFileDialog.getSaveFileName(
            self, "Save File", self.save_file_edit.text(),
            "HDF5 files (*.h5 *.hdf5)", options=QFileDialog.DontConfirmOverwrite
        )
        self.save_file_edit.setText(filename)

    def change_dataset(self):
        dialog = QDialog()
        layout = QVBoxLayout(dialog)
        model = H5File(h5py.File(str(self.save_file_edit.text())))
        tree_view = H5View()
        tree_view.setModel(model)
        tree_view.setSelectionMode(QAbstractItemView.SingleSelection)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(tree_view)
        layout.addWidget(button_box)
        if dialog.exec_():
            #self.dataset_edit.setText(tree_view.selected_path()[1:])
            dsname = model.itemFromIndex(tree_view.selectedIndexes()[0]).fullname[1:]
            self.dataset_edit.setText(dsname)


    def get_h5group(self):
        filename = str(self.save_file_edit.text())
        h5file = datasrv.get_file(filename)
        path = str(self.dataset_edit.text())
        return h5helpers.resolve_path(h5file, path)

    def save_trace_csv(self):
        default_name = time.strftime("trace_%Y%m%d_%H%M%S.dat")
        save_path = str(self.save_path_edit.text())
        default_filename = os.path.join(save_path, default_name)
        filename = str(QFileDialog.getSaveFileName(self, "Save Trace", default_filename))
        data = numpy.array([self.freqs, self.mags, self.phases]).transpose()
        numpy.savetxt(filename, data)

    def save_trace(self):
        group = self.get_h5group()
        if 'freqs' in group:
            reply = QMessageBox.question(self, "", "Dataset exists in file, Overwrite?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        group['freqs'] = self.freqs
        group['mags'] = self.mags
        group['phases'] = self.phases
        h5helpers.set_scale(group, 'freqs', 'mags')
        h5helpers.set_scale(group, 'freqs', 'phases')
        h5helpers.update_attrs(group, self.vna_params)
        self.message("Saved in file %s" % group.get_fullname())


if __name__ == '__main__':
    settings = QSettings('philreinhold', 'vna_widget')
    qt_helpers.run(VNAWidget, settings, __ui_version__)
