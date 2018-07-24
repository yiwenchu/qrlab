import argparse
from PyQt4 import QtCore, QtGui
import objectsharer as objsh
import logging
logging.getLogger().setLevel(logging.INFO)
import sys
import types
import instrument
import math
from collections import defaultdict
import json

import localconfig

GETSET_BUTSIZE = QtCore.QSize(30, 20)
EPSILON = 1e-12

GREEN = QtGui.QColor(90,255,70)
RED = QtGui.QColor(255,0,0)
GRAY = QtGui.QColor(245,245,245)

def dict_to_ordered_tuples(dic):
    '''Convert a dictionary to a list of tuples, sorted by key.'''
    if dic is None:
        return []
    keys = dic.keys()
    keys.sort()
    ret = [(key, dic[key]) for key in keys]
    return ret

#########################################
# Control widgets for different types.
#########################################

def eval_input(txt):
    if txt is None:
        return None
    txt = str(txt)
    if txt == '':
        return None
    txt = txt.replace(' ', '')
    try:
        ret = eval(txt)
    except:
        ret = None
    return ret

class IntWidget(QtGui.QSpinBox):
    def __init__(self, ins, param, opts):
        super(IntWidget, self).__init__()
        self._ins = ins
        self._param = param
        self._opts = opts
        self.setMinimum(opts.get('minval', -1000000000))
        self.setMaximum(opts.get('maxval', +1000000000))

    def update(self, val, cb=None):
        if val is not None:
            self.setValue(int(val))
        if cb:
            cb(val is not None)

    def do_get(self, query=True, cb=None):
        self._ins.get(self._param, query=query, callback=lambda x: self.update(x, cb=cb))

    def do_set(self, cb=None):
        val = eval_input(self.value())
        if val is not None:
            self._ins.set(self._param, int(val), callback=cb)

class FloatWidget(QtGui.QDoubleSpinBox):
    def __init__(self, ins, param, opts):
        super(FloatWidget, self).__init__()
        self._ins = ins
        self._param = param
        self._opts = opts
        self.setMinimum(opts.get('minval', -1e12))
        self.setMaximum(opts.get('maxval', 1e12))
        self.setDecimals(opts.get('decimals', 2))

    def update(self, val, cb=None):
        if val is not None:
            self.setValue(float(val))
        if cb:
            cb(val is not None)

    def do_get(self, query=True, cb=None):
        self._ins.get(self._param, query=query, callback=lambda x: self.update(x, cb=cb))

    def do_set(self, cb=None):
        val = eval_input(self.value())
        if val is not None:
            self._ins.set(self._param, float(val), callback=cb)

class SciWidget(QtGui.QLineEdit):
    '''
    A widget for displaying scientific numbers.

    opts is an option dictionary, and can contain the following items:
    - 'format': an explicit format string
    - 'display_scale': a number representing the requested order to use when
    displaying the value, i.e. display_order=3 will format as 123.456e3
    - 'multdigitis': number of digits used will be a multiple of this (default 3)
    - 'maxdigits': maximum number of digits to use (default 6)
    '''

    def __init__(self, ins, param, opts):
        super(SciWidget, self).__init__()
        self._ins = ins
        self._param = param
        self._opts = opts
        self._maxdigits = opts.get('maxdigits', 6)
        self._multdigits = opts.get('multdigits', 3)

    def update(self, val, cb=None):
        if val is not None:
            self.setText(self.format(val))
        if cb:
            cb(val is not None)

    def _count_digits(self, val):
        '''
        Return number of digits behind the dot.
        '''
        digits = -self._multdigits
        order = 1.0
        while abs(val) > EPSILON and digits < self._maxdigits:
            val -= round(val / order) * order
            order /= 10**self._multdigits
            digits += self._multdigits
        return max(0, digits)

    def _add_spaces(self, valstr):
        idx = valstr.find('.')
        while idx != -1 and (idx + 4) < len(valstr) and valstr[idx+4] not in ('e', 'E'):
            valstr = valstr[:idx+4] + ' ' + valstr[idx+4:]
            idx += 4
        return valstr

    def format(self, val):
        if 'format' in self._opts:
            return self._opts['format'] % (val,)

        if 'display_scale' in self._opts:
            dispscale = self._opts['display_scale']
        else:
            if val == 0 or math.floor(math.log10(abs(val))) in (-1.0, -2.0):
                dispscale = 0
            else:
                dispscale = int(3 * round(math.floor(math.log10(abs(val)) / 3.0)))

        val /= 10.0**dispscale
        if dispscale == 0:
            fmt = '%%.%02df' % (self._count_digits(val),)
            return self._add_spaces(fmt % val)
        else:
            fmt = '%%.%02dfe%%d' % (self._count_digits(val),)
            return self._add_spaces(fmt % (val, dispscale))

    def do_get(self, query=True, cb=None):
        self._ins.get(self._param, query=query, callback=lambda x: self.update(x, cb=cb))

    def do_set(self, cb=None):
        val = eval_input(self.text())
        if val is not None:
            self._ins.set(self._param, val, callback=cb)

class StringWidget(QtGui.QLineEdit):
    def __init__(self, ins, param, opts):
        super(StringWidget, self).__init__('')
        self._ins = ins
        self._param = param
        self._opts = opts

    def update(self, val, cb=None):
        if val is not None:
            self.setText(str(val))
        if cb:
            cb(val is not None)

    def do_get(self, query=True, cb=None):
        self._ins.get(self._param, query=query, callback=lambda x: self.update(x, cb=cb))

    def do_set(self, cb=None):
        val = self.text()
        if val is not None:
            self._ins.set(self._param, str(val), callback=cb)

class ComplexWidget(QtGui.QLineEdit):
    def __init__(self, ins, param, opts):
        super(ComplexWidget, self).__init__('')
        self._ins = ins
        self._param = param
        self._opts = opts

    def update(self, val, cb=None):
        if val is not None:
            self.setText(str(val))
        if cb:
            cb(val is not None)

    def do_get(self, query=True, cb=None):
        self._ins.get(self._param, query=query, callback=lambda x: self.update(x, cb=cb))

    def do_set(self, cb=None):
        val = self.text()
        if val is not None:
            self._ins.set(self._param, complex(str(val)), callback=cb)

class StringLabelWidget(QtGui.QLabel):
    def __init__(self, ins, param, opts):
        super(StringLabelWidget, self).__init__()
        self._ins = ins
        self._param = param
        self._opts = opts

    def update(self, val, cb=None):
        if val is not None:
            self.setText(str(val))
        if cb:
            cb(val is not None)

    def do_get(self, query=True, cb=None):
        self._ins.get(self._param, query=query, callback=lambda x: self.update(x, cb=cb))

    def do_set(self, cb=None):
        val = self.value()
        if val is not None:
            self._ins.set(self._param, val, callback=cb)

class DropdownWidget(QtGui.QComboBox):
    def __init__(self, ins, param, opts):
        super(DropdownWidget, self).__init__()
        self._ins = ins
        self._param = param
        self._opts = opts
        self.set_options()

    def set_options(self):
        self._val_to_idx = {}
        self._idx_to_val = {}

        if 'format_map' in self._opts:
            for i, (k, v) in enumerate(dict_to_ordered_tuples(self._opts['format_map'])):
                self.addItem(str(v))
                self._idx_to_val[i] = k
                self._val_to_idx[k] = i

        elif 'option_list' in self._opts:
            self._map = self._opts['option_list']
            for i, k in enumerate(self._map):
                self.addItem(str(k))
                if type(k) is types.StringType:
                    k = k.upper()
                self._idx_to_val[i] = k
                self._val_to_idx[k] = i

    def update(self, val, cb=None):
        if type(val) is types.StringType:
            val = val.upper()
        if val in self._val_to_idx:
            self.setCurrentIndex(self._val_to_idx[val])
        if cb:
            cb(val is not None)

    def do_get(self, query=True, cb=None):
        self._ins.get(self._param, query=query, callback=lambda x: self.update(x, cb=cb))

    def do_set(self, cb=None):
        val = self.currentIndex()
        if val in self._idx_to_val:
            self._ins.set(self._param, self._idx_to_val[val], callback=cb)

class InstrumentDropdownWidget(QtGui.QComboBox):

    def __init__(self, ins, param, opts):
        super(InstrumentDropdownWidget, self).__init__()
        self._ins = ins
        self._param = param
        self._opts = opts
        self._idx_to_val = {}
        self._val_to_idx = {}

        instruments.connect('instrument-added', self.update_instruments)
        instruments.connect('instrument-removed', self.update_instruments)
        self.update_instruments()

    def update_instruments(self, name=None):
        curval = self._idx_to_val.get(self.currentIndex(), None)
        self._idx_to_val = {}
        self._val_to_idx = {}

        selfname = self._ins.get_name()
        i = 0
        for name in instruments.list_instruments():
            if name == selfname:
                continue
            self.addItem(name)
            self._idx_to_val[i] = name
            i += 1

        self.update(curval)

    def update(self, val, cb=None):
        if val in self._val_to_idx:
            self.setCurrentIndex(self._val_to_idx[val])
        if cb:
            cb(val is not None)

    def do_get(self, query=True, cb=None):
        self._ins.get(self._param, query=query, callback=lambda x: self.update(x, cb=cb))

    def do_set(self, cb=None):
        val = self.currentIndex()
        if val in self._idx_to_val:
            self._ins.set(self._param, self._idx_to_val[val], callback=cb)

class JSONModel(QtGui.QStandardItemModel):
    def __init__(self, fn):
        super(JSONModel, self).__init__()
        self.setColumnCount(2)
        self.add_data(json.load(open(fn, 'r')), self)

    def add_data(self, data, parent):
        for name, val in data.items():
            name_item = QtGui.QStandardItem(name)
            if isinstance(val, dict):
                name_item.setCheckable(True)
                name_item.setCheckState(QtCore.Qt.Checked)
                parent.appendRow(name_item)
                self.add_data(val, name_item)
            else:
                val_item = QtGui.QStandardItem(str(val))
                parent.appendRow([name_item, val_item])

    def get_selected(self):
        for i in range(self.rowCount()):
            item = self.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                yield str(item.text())

    def check_all(self, check_state=QtCore.Qt.Checked):
        for i in range(self.rowCount()):
            self.item(i).setCheckState(check_state)

    def check_none(self):
        self.check_all(QtCore.Qt.Unchecked)

class InstrumentSelectorDialog(QtGui.QDialog):
    def __init__(self, fn):
        super(InstrumentSelectorDialog, self).__init__()
        self.model = JSONModel(fn)
        tree_view = QtGui.QTreeView()
        tree_view.setModel(self.model)
        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok |
            QtGui.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        all_button = buttons.addButton('All', QtGui.QDialogButtonBox.ActionRole)
        all_button.clicked.connect(lambda: self.model.check_all())
        none_button = buttons.addButton('None', QtGui.QDialogButtonBox.ActionRole)
        none_button.clicked.connect(lambda: self.model.check_none())

        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(tree_view)
        layout.addWidget(buttons)

    def exec_(self):
        if super(InstrumentSelectorDialog, self).exec_():
            return list(self.model.get_selected())
        return None


class InstrumentsContainer(QtGui.QTabWidget):

    def __init__(self, instruments, parent=None):
        self.instruments = instruments
        super(InstrumentsContainer, self).__init__(parent)
        self.setTabsClosable(False)
        self.setMovable(True)

        self._ins_widgets = {}
        self.show_tab_actions = {}

        self.instruments.connect('instrument-added', self._ins_added_cb)
        self.instruments.connect('instrument-removed', self._ins_removed_cb)

        names = self.instruments.list_instruments()
        names.sort()
        for name in names:
            self._ins_added_cb(name)

        create_instruments_action = QtGui.QAction('Create from File', self)
        create_instruments_action.triggered.connect(self.create_from_file)
        self.addAction(create_instruments_action)
        save_settings_action = QtGui.QAction('Save Settings', self)
        save_settings_action.triggered.connect(self.save_settings)
        self.addAction(save_settings_action)
        load_settings_action = QtGui.QAction('Load Settings', self)
        load_settings_action.triggered.connect(self.load_settings)
        self.addAction(load_settings_action)
        reload_current_action = QtGui.QAction('Reload current', self)
        reload_current_action.triggered.connect(self.reload_current)
        self.addAction(reload_current_action)
        self.settings_dir = None

    def reload_current(self):
        current = str(self.tabText(self.currentIndex()))
        self.instruments.reload(current)

    def create_from_file(self):
        self.load_settings(create=True)

    def save_settings(self, _checked=False):
        fn = QtGui.QFileDialog.getSaveFileName(directory=self.settings_dir)
        if fn:
            self.instruments.save_instruments(fn)

    def load_settings(self, _checked=False, create=False):
        fn = QtGui.QFileDialog.getOpenFileName(directory=self.settings_dir)
        if fn:
            inslist = InstrumentSelectorDialog(fn).exec_()
            if inslist:
                self.instruments.load_settings_from_file(fn, inslist, create=create, async=True)

    def _ins_added_cb(self, name):
        logging.info('Instrument %s added', name)
        ins = self.instruments.get(name)
        if hasattr(ins, 'on_disconnect'):
            ins.on_disconnect(lambda: self._ins_removed_cb(name))
        if ins is None:
            logging.error('Unable to get instrument %s', name)
            return
        tab = InstrumentTab(name, ins)
        tab_scroll_area = QtGui.QScrollArea()
        tab_scroll_area.setWidget(tab)
        tab_scroll_area.setWidgetResizable(True)
        self._ins_widgets[name] = tab_scroll_area
        self.show_tab_actions[name] = action = QtGui.QAction(name, self)
        action.setCheckable(True)
        action.setChecked(True)
        action.toggled.connect(lambda enabled: self.show_tab(name, enabled))
        self.addTab(tab_scroll_area, name)

    def _ins_removed_cb(self, name):
        logging.info('Instrument %s removed', name)
        if name in self._ins_widgets:
            tab = self._ins_widgets[name]
            idx = self.indexOf(tab)
            self.removeTab(idx)
            del self._ins_widgets[name]

    def show_tab(self, name, enabled):
        widget = self._ins_widgets[name]
        currently_enabled = self.indexOf(widget) >= 0
        if enabled == currently_enabled:
            return
        if enabled:
            self.addTab(widget, name)
        else:
            self.removeTab(self.indexOf(widget))

    def sizeHint(self):
        return QtCore.QSize(350, 700)

class ColorPushButton(QtGui.QPushButton):
    def __init__(self, text, parent=None):
        super(ColorPushButton, self).__init__(text, parent)
        self.set_bg(GRAY)

    def set_bg(self, color):
        self.setStyleSheet(
            "QPushButton { background-color: %s }"
            "QPushButton:pressed { background-color: %s }" % (
            color.name(), color.light(125).name()
            )
        )

    def _highlight(self, color=GREEN):
        self.set_bg(color)
        QtCore.QTimer.singleShot(1000, self._unhighlight)

    def _unhighlight(self):
        self.set_bg(QtGui.QColor(GRAY))

class GetButton(ColorPushButton):

    def __init__(self, entry, parent=None):
        icon = QtGui.QIcon.fromTheme('zoom-in')
        super(GetButton, self).__init__('Get', parent)
        self.setFixedSize(GETSET_BUTSIZE)
        self.entry = entry
        self.connect(self, QtCore.SIGNAL("clicked()"), self._clicked_cb)

    def _clicked_cb(self):
        self.entry.do_get(cb=lambda x: self._highlight(GREEN if x else RED))

class SetButton(ColorPushButton):

    def __init__(self, entry, parent=None):
        icon = QtGui.QIcon.fromTheme('zoom-out')
        super(SetButton, self).__init__('Set', parent)
        self.setFixedSize(GETSET_BUTSIZE)
        self.entry = entry
        self.connect(self, QtCore.SIGNAL("clicked()"), self._clicked_cb)
        entry.connect(entry, QtCore.SIGNAL("returnPressed()"), self._clicked_cb)

    def _clicked_cb(self):
        self.set_bg(GRAY)
        self.entry.do_set(cb=lambda x: self._highlight(GREEN if x else RED))

class FunctionButton(QtGui.QPushButton):
    def __init__(self, ins, funcname):
        super(FunctionButton, self).__init__(funcname)
        self._ins = ins
        self._funcname = funcname
        self.connect(self, QtCore.SIGNAL("clicked()"), self._clicked_cb)

    def _clicked_cb(self):
        f = getattr(self._ins, self._funcname)
        f(callback=self._func_callback)

    def _func_callback(self, val):
        print 'Function executed: %s' % (val, )

class InstrumentTab(QtGui.QWidget):

    def __init__(self, name, ins, parent=None):
        self._name = name
        self._ins = ins
        self._ins.connect('changed', self._ins_changed_cb)
        super(InstrumentTab, self).__init__(parent)

        self.setWindowTitle(self._name)
        self.setGeometry(300, 300, 300, 150)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.connect(self, QtCore.SIGNAL("customContextMenuRequested(const QPoint&)"), self._context_menu_cb)

        self.form = QtGui.QFormLayout(self)
        self.setLayout(self.form)

        self.setup_widgets()
        self.show()

    def _create_entry(self, param, opts):
        if not opts['flags'] & instrument.Instrument.FLAG_SET:
            entry = StringLabelWidget(self._ins, param, opts)

        elif 'format_map' in opts or 'option_list' in opts:
            entry = DropdownWidget(self._ins, param, opts)
        elif opts['type'] in (types.IntType, ):
            entry = SciWidget(self._ins, param, opts)
        elif opts['type'] in (types.FloatType,):
#            if abs(opts.get('minval', 0)) > 1e4 or abs(opts.get('maxval', 0)) > 1e4:
            entry = SciWidget(self._ins, param, opts)
#            else:
#                entry = FloatWidget(self._ins, param, opts)
        elif opts['type'] in (types.ComplexType,):
            entry = ComplexWidget(self._ins, param, opts)
        elif opts['type'] is types.BooleanType:
            opts['format_map'] = {False: 'False', True: 'True'}
            entry = DropdownWidget(self._ins, param, opts)
        elif opts['type'] == instrument.TYPE_INSTRUMENT:
            entry = InstrumentDropdownWidget(self._ins, param, opts)
        else:
            entry = StringWidget(self._ins, param, opts)

        entry.do_get(query=False)

        return entry

    def setup_widgets(self):
        self._entry_widgets = {}
        params = self._ins.get_shared_parameters()
        names = params.keys()
        names.sort()
        groups = defaultdict(lambda: [])

        for name in names:
            groups[params[name].get('gui_group', 'default')].append(name)

        group_widgets = {}
        for gui_group_name, child_names in groups.items():
            if gui_group_name == 'default':
                group_form = self.form
            else:
                group_widgets[gui_group_name] = group_widget = QtGui.QWidget()
                group_form = QtGui.QFormLayout(group_widget)

            for name in child_names:
                opts = params[name]

                label = name
                if 'units' in opts:
                    label += ' [%s]' % (opts['units'],)
                if 'doc' in opts or 'help' in opts:
                    label += ' [?]'
                lbl = QtGui.QLabel(label)
                if 'doc' in opts:
                    lbl.setToolTip(opts['doc'])
                if 'help' in opts:
                    lbl.setToolTip(opts['help'])

                entry = self._create_entry(name, opts)
                self._entry_widgets[name] = entry
                hbox = QtGui.QHBoxLayout()
                hbox.addWidget(entry)
                hbox.addWidget(GetButton(entry))
                hbox.addWidget(SetButton(entry))
                group_form.addRow(lbl, hbox)

        if group_widgets:
            groups_tab_widget = QtGui.QTabWidget()
            self.form.addRow(groups_tab_widget)
            for group_name, group_widget in group_widgets.items():
                groups_tab_widget.addTab(group_widget, group_name)

        hbox = QtGui.QHBoxLayout()
        for funcname in ['get_all'] + self._ins.get_function_names():
            if self._ins.get_function_parameters(funcname) is None:
                hbox.addWidget(FunctionButton(self._ins, funcname))
        self.form.addRow(hbox)

    def _ins_changed_cb(self, changes):
        for key, val in changes.iteritems():
            entry = self._entry_widgets.get(key, None)
            if entry:
                entry.update(val)

    def _reopen(self):
        win = InstrumentTab(self._name, self._ins)

    def _context_menu_cb(self, point):
        m = QtGui.QMenu(self)
        act = QtGui.QAction('Reopen in new window', self)
        act.triggered.connect(self._reopen)
        m.addAction(act)

        m.exec_(self.mapToGlobal(point))

class InstrumentsMainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(InstrumentsMainWindow, self).__init__()
        self.setWindowTitle('Instruments')
        self.tab_widget = InstrumentsContainer(instruments)
        self.setCentralWidget(self.tab_widget)

        file_menu = self.menuBar().addMenu("File")
        for action in self.tab_widget.actions():
            file_menu.addAction(action)

if __name__ == '__main__':
    isrvname = localconfig.instrument_server_alias
    isrvaddr = localconfig.instrument_server_addr
    isrvport = localconfig.instrument_server_port
    isrv = 'tcp://{}:{}'.format(isrvaddr, isrvport)    
    
    parser = argparse.ArgumentParser(description='Connect to instrument server')
    parser.add_argument('--isrv', type=str, default=isrv,
        help='Instruments server location')
    parser.add_argument('--isrvname', type=str, default=isrvname,
        help='Instruments server alias as registered to objectsharer')
    '''
    parser.add_argument('--isrv', type=str, default='tcp://127.0.0.1:55555',
        help='Instruments server location')
    parser.add_argument('--isrvname', type=str, default='instruments',
        help='Instruments server alias as registered to objectsharer')
    '''
    args = parser.parse_args()

    if hasattr(objsh, 'ZMQBackend'):
        backend = objsh.ZMQBackend()
    else:
        backend = objsh.backend
    
    backend.start_server(isrvaddr)
    '''
    backend.start_server('127.0.0.1')
    '''
    backend.connect_to(args.isrv)     # Instruments server

    instruments = objsh.helper.find_object(args.isrvname)
    app = QtGui.QApplication(sys.argv)

    mainwin = InstrumentsMainWindow()
    mainwin.show()

    backend.add_qt_timer()
    sys.exit(app.exec_())

