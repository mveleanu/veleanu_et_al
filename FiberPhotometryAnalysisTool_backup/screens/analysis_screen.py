import sys
import numpy as np
import json
from time import perf_counter
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt import QtCore
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from screens.plotting_screen import Plotter
from functions.analysis_data_loader import AnalysisDataLoader
from screens.event_editing_screen import EventEditor


class AnalysisScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_parameter_screen()
        self.group_loader = None
        self.processed_data = None
        self.database = {}
        self.data_loader = AnalysisDataLoader()
        self.plotter = None
        self.events_updated = False
        self.metric_list = {}

        self.setCentralWidget(self.area)


    def init_ui(self):
        self.area = DockArea()
        self.resize(1200, 700)

        self.parameters_area = Dock("Parameters", size=(0.4, 1), hideTitle=True)
        self.psth_area = Dock("PSTH", size=(0.6, 0.5), hideTitle=True)
        self.peak_amplitude_area = Dock("Peak Amplitude", size=(0.4, 0.5), hideTitle=True)
        self.heatmap_area = Dock("Heatmap", size=(0.6, 0.5), hideTitle=True)
        self.overall_peak_amplitdes_area = Dock("Overall Peak Amplitudes", size=(0.4, 0.5), hideTitle=True)


        self.area.addDock(self.parameters_area, 'left')
        self.area.addDock(self.psth_area, 'right', self.parameters_area)
        self.area.addDock(self.peak_amplitude_area, 'right', self.psth_area)
        self.area.addDock(self.heatmap_area, 'bottom', self.psth_area)
        self.area.addDock(self.overall_peak_amplitdes_area, 'bottom', self.peak_amplitude_area)


    def init_parameter_screen(self):
        self.parameter_screen = QWidget()
        self.parameter_layout = QVBoxLayout()
        self.parameter_screen.setLayout(self.parameter_layout)

        self.parameter_screen.show()
        
        # Sampling Frquency Box
        samp_freq_box = QHBoxLayout()
        samp_freq_label = QLabel('Sampling Frequency of DFF (Hz):')
        samp_freq_box.addWidget(samp_freq_label)
        self.samp_freq_input = QtWidgets.QSpinBox()
        self.samp_freq_input.setValue(20)
        self.samp_freq_input.setRange(1, 1000)
        self.samp_freq_input.setSingleStep(1)
        samp_freq_box.addWidget(self.samp_freq_input)
        self.parameter_layout.addLayout(samp_freq_box)

        # Baseline Correction Option
        baseline_corr_box = QHBoxLayout()
        baseline_corr_label = QLabel('Baseline Correction:')
        baseline_corr_box.addWidget(baseline_corr_label)
        self.baseline_correct_input = QtWidgets.QCheckBox()
        self.baseline_correct_input.setChecked(True)
        baseline_corr_box.addWidget(self.baseline_correct_input)
        self.parameter_layout.addLayout(baseline_corr_box)

        # Set PSTH Start and End Times
        psth_start_time_box = QHBoxLayout()
        psth_start_label = QLabel('PSTH Start Time (s):')   
        psth_start_time_box.addWidget(psth_start_label)
        self.psth_start_input = QtWidgets.QSpinBox()
        self.psth_start_input.setMinimum(-10000)
        self.psth_start_input.setValue(-5)
        self.psth_start_input.setRange(-10000, 0)
        self.psth_start_input.setSingleStep(1)
        psth_start_time_box.addWidget(self.psth_start_input)
        self.parameter_layout.addLayout(psth_start_time_box)

        psth_end_time_box = QHBoxLayout()
        psth_end_label = QLabel('PSTH End Time (s):')
        psth_end_time_box.addWidget(psth_end_label)
        self.psth_end_input = QtWidgets.QSpinBox()
        self.psth_end_input.setValue(5)
        self.psth_end_input.setRange(0, 10000)
        self.psth_end_input.setSingleStep(1)
        psth_end_time_box.addWidget(self.psth_end_input)

        self.parameter_layout.addLayout(psth_end_time_box)

        # Adding Spinbox for Setting Peak Detection Width
        peak_detection_width_box = QHBoxLayout()
        peak_detection_width_label = QLabel('Peak Detection Width (s):')
        peak_detection_width_box.addWidget(peak_detection_width_label)
        self.peak_detection_width_input = QtWidgets.QSpinBox()
        self.peak_detection_width_input.setValue(1)
        self.peak_detection_width_input.setRange(0, 10000)
        self.peak_detection_width_input.setSingleStep(1)
        peak_detection_width_box.addWidget(self.peak_detection_width_input)

        self.parameter_layout.addLayout(peak_detection_width_box)

        # Adding Radion Button for Baseline zscore
        baseline_zscore_box = QHBoxLayout()
        baseline_zscore_label = QLabel('Units:')
        baseline_zscore_box.addWidget(baseline_zscore_label)
        self.baseline_zscore_input = QRadioButton('Baseline z-score')
        baseline_zscore_box.addWidget(self.baseline_zscore_input)
        self.dff_input = QRadioButton('DFF', checked=True)
        baseline_zscore_box.addWidget(self.dff_input)
        self.parameter_layout.addLayout(baseline_zscore_box)
        
        # Adding Load Presaved Tree Box
        load_tree_box = QHBoxLayout()
        load_tree_button = QPushButton('Load Previous Analysis')
        load_tree_button.clicked.connect(self.load_previous_analysis)
        load_tree_box.addWidget(load_tree_button)

        self.parameter_layout.addLayout(load_tree_box)
        
        # Adding Edit Events Button
        edit_events_button = QPushButton('Edit Events')
        edit_events_button.clicked.connect(self.edit_events)
        self.parameter_layout.addWidget(edit_events_button)

        # Adding Groups Box
        groups_box = QVBoxLayout()
        add_group_button = QPushButton('Add Group')
        groups_box.addWidget(add_group_button)
        add_group_button.clicked.connect(self.add_group)
        groups_list_box = QtWidgets.QTreeView()
        groups_list_box.doubleClicked.connect(self.edit_group_info)
        groups_list_box.setContextMenuPolicy(Qt.CustomContextMenu)
        groups_list_box.customContextMenuRequested.connect(self.show_context_menu)

        self.tree_model = QStandardItemModel()
        self.tree_model.setHorizontalHeaderLabels(['Group Name'])
        groups_list_box.setModel(self.tree_model)

        groups_box.addWidget(groups_list_box)
        scrollbarr = QtWidgets.QScrollBar()
        groups_list_box.setVerticalScrollBar(scrollbarr)
        self.parameter_layout.addLayout(groups_box)

        self.analyze_button = QPushButton('Analyze')
        self.analyze_button.clicked.connect(self.analyze)
        self.parameter_layout.addWidget(self.analyze_button)



        self.parameters_area.addWidget(self.parameter_screen)

        # Save Analysis Button
        save_analysis_button = QPushButton('Save Analysis')
        save_analysis_button.clicked.connect(self.save_analysis)
        self.parameter_layout.addWidget(save_analysis_button)

        # Export Data Buttom
        export_data_button = QPushButton('Export Data')
        export_data_button.clicked.connect(self.export_data)
        self.parameter_layout.addWidget(export_data_button)

    def show_context_menu(self, pos):
        index = self.sender().indexAt(pos)
        if index.isValid():
            menu = QMenu()
            delete_action = menu.addAction('Delete Group')
            delete_action.triggered.connect(lambda: self.delete_item(index))
            menu.exec_(self.sender().viewport().mapToGlobal(pos))

    def delete_item(self, index):
        item = self.tree_model.itemFromIndex(index)
        self.database.pop(item.text())
        self.tree_model.removeRow(index.row())


    def save_analysis(self):
        print('Saving Analysis')
        file_path = QFileDialog.getSaveFileName(self, 'Save File', '', 'Analysis Files (*.json)')
        if file_path[0] == '':
            return

        # Saving the databse as json
        if ".json" not in file_path[0]:
            file_path[0] = file_path[0] + '.json'

        save_data = {"database": self.database, "metric_list": self.metric_list}
        with open(file_path[0], 'w') as f:
            json.dump(save_data, f)

    def export_data(self):
        print('Exporting Data')
        file_path = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if file_path == '':
            return

        if self.plotter is None:
            return
        
        self.plotter.export_data(file_path)

        # Showing a popup message when done
        msg = QMessageBox()
        msg.setWindowTitle('Export Data')
        msg.setText('Data Exported Successfully')
        msg.exec_()


    def load_previous_analysis(self):
        print('Loading Previous Analysis')
        file_path = QFileDialog.getOpenFileName(self, 'Open File', '', 'Analysis Files (*.json)')
        if file_path[0] == '':
            return
        
        with open(file_path[0], 'r') as f:
            data = json.load(f)
        
        self.database = data['database']
        self.metric_list = data['metric_list']

        # Clearing the tree model
        self.tree_model.clear()

        for group in self.database.keys():
            group_item = QStandardItem(group)
            self.tree_model.setHorizontalHeaderLabels(['Group Name'])
            group_item.setEditable(False)
            group_item.setCheckable(True)
            group_item.setCheckState(Qt.Checked)
            self.tree_model.appendRow(group_item)

    def edit_events(self):
        database = {}
        
        # Seeing if tree has any event
        if self.tree_model.rowCount() == 0:
            QMessageBox.warning(self, 'No Groups Added', 'Please add a group to continue')
            return


        # Checking if any group is selected
        for i in range(self.tree_model.rowCount()):
            if self.tree_model.item(i).checkState() == Qt.Checked:
                database[self.tree_model.item(i).text()] = self.database[self.tree_model.item(i).text()]

        self.processed_data = self.data_loader.load_data(database, 
                                                         self.samp_freq_input.value(), 
                                                         self.psth_start_input.value(), 
                                                         self.psth_end_input.value(), 
                                                         self.baseline_correct_input.isChecked(), 
                                                         peak_detection_width=self.peak_detection_width_input.value(), 
                                                         use_zscore=self.baseline_zscore_input.isChecked())
        if self.processed_data is None:
            return
        
        self.event_editor = EventEditor(database=self.processed_data)
        self.event_editor.editComplete.connect(self.update_events)

    def update_events(self):
        self.processed_data = self.event_editor.database
        self.events_updated = self.event_editor.events_updated

    def analyze(self):  
        if self.tree_model.rowCount() == 0:
            QMessageBox.warning(self, 'No Groups Added', 'Please add a group to continue')
            return

        for widget in self.psth_area.widgets:
            widget.setParent(None)
        for widget in self.heatmap_area.widgets:
            widget.setParent(None)
        for widget in self.peak_amplitude_area.widgets:
            widget.setParent(None)
        for widget in self.overall_peak_amplitdes_area.widgets:
            widget.setParent(None)
        
        database = {}
        # Checking if any group is selected
        for i in range(self.tree_model.rowCount()):
            if self.tree_model.item(i).checkState() == Qt.Checked:
                database[self.tree_model.item(i).text()] = self.database[self.tree_model.item(i).text()]

        group_order = []
        for i in range(self.tree_model.rowCount()):
            if self.tree_model.item(i).checkState() == Qt.Checked:
                group_order.append(self.tree_model.item(i).text())

        if self.processed_data is not None and self.events_updated:
            self.plotter = Plotter(self.processed_data, self.psth_area, self.heatmap_area, self.peak_amplitude_area, self.overall_peak_amplitdes_area, group_order=group_order, use_zscore=self.baseline_zscore_input.isChecked(), metric_list=self.metric_list)
        else:
            self.processed_data = self.data_loader.load_data(database, self.samp_freq_input.value(), self.psth_start_input.value(), self.psth_end_input.value(), self.baseline_correct_input.isChecked(), peak_detection_width=self.peak_detection_width_input.value(), use_zscore=self.baseline_zscore_input.isChecked())
            self.plotter = Plotter(self.processed_data, self.psth_area, self.heatmap_area, self.peak_amplitude_area, self.overall_peak_amplitdes_area, group_order=group_order, use_zscore=self.baseline_zscore_input.isChecked(), metric_list=self.metric_list)

        
    def add_group(self):
        print('Adding Group...')
        if self.group_loader is None:
            self.group_loader = GroupLoader()
            self.group_loader.submitClicked.connect(self.add_group_to_tree)
            self.group_loader.show()
        else:
            self.group_loader.close()
            self.group_loader = None

    def add_group_to_tree(self, group_dict):
        self.group_loader = None
        if group_dict['group_name'] in self.database:
            print('Group already exists')
            QMessageBox.warning(self, 'Group Already Exists', 'Group with the same name already exists')
            return
        
        group_name = group_dict['group_name']
        group_item = QStandardItem(group_name)
        group_item.setEditable(False)
        group_item.setCheckable(True)
        group_item.setCheckState(Qt.Checked)
        self.tree_model.appendRow(group_item)
        # Showing row children


        # Saving to the main database file 
        self.database[group_name] = group_dict['group_files']
        self.metric_list[group_name] = group_dict['peak_amplitude_metric']

    def edit_group_info(self, index):
        item = self.tree_model.itemFromIndex(index)
        self.group_loader = GroupLoader()
        self.group_loader.database = self.database[self.tree_model.itemFromIndex(index).text()]
        self.group_loader.group_name_input.setText(item.text())
        self.group_loader.group_files_table.setRowCount(0)
        for key, value in self.database[item.text()].items():
            self.group_loader.group_files_table.setRowCount(self.group_loader.group_files_table.rowCount() + 1)
            self.group_loader.group_files_table.setItem(self.group_loader.group_files_table.rowCount() - 1, 0, QTableWidgetItem(key))
            self.group_loader.group_files_table.setItem(self.group_loader.group_files_table.rowCount() - 1, 1, QTableWidgetItem(value['dff_file_path']))
            self.group_loader.group_files_table.setItem(self.group_loader.group_files_table.rowCount() - 1, 2, QTableWidgetItem(value['timestamps_file_path']))

        self.group_loader.submitClicked.connect(self.update_group)
        self.update_group_old_name = item.text()
        self.group_loader.show()

    def update_group(self, group_dict):
        self.database.pop(self.update_group_old_name)
        self.tree_model.removeRow(self.tree_model.indexFromItem(self.tree_model.findItems(self.update_group_old_name)[0]).row())

        group_name = group_dict['group_name']
        group_item = QStandardItem(group_name)
        group_item.setEditable(False)
        group_item.setCheckable(True)
        group_item.setCheckState(Qt.Checked)
        self.tree_model.appendRow(group_item)

        # Saving to the main database file 
        self.database[group_name] = group_dict['group_files']
        self.metric_list[group_name] = group_dict['peak_amplitude_metric']


class GroupLoader(QMainWindow):
    submitClicked = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setCentralWidget(self.window)
        self.add_item_screen = None
        self.item_from_add_group_screen = {}
        self.database = {}



    def init_ui(self):
        self.window = QWidget()
        self.setWindowTitle('Add Group')
        self.resize(400, 400)

        self.layout = QVBoxLayout()
        self.window.setLayout(self.layout)

        # Group Name Box
        group_name_box = QHBoxLayout()
        group_name_label = QLabel('Group Name:')
        group_name_box.addWidget(group_name_label)
        self.group_name_input = QLineEdit()
        group_name_box.addWidget(self.group_name_input)
        self.layout.addLayout(group_name_box)

        # Show Group Files Table
        self.group_files_table = QTableWidget()
        self.group_files_table.setColumnCount(3)
        self.group_files_table.setHorizontalHeaderLabels(['Name', 'DFF File Path', 'Timestamps File Path'])

        # Adding Right Click Menu
        self.group_files_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.group_files_table.customContextMenuRequested.connect(self.show_context_menu)
        self.layout.addWidget(self.group_files_table)

        # Adding Option to select Peak amplitude metric
        peak_amplitude_box = QHBoxLayout()
        peak_amplitude_label = QLabel('Peak Amplitude Metric:')
        peak_amplitude_box.addWidget(peak_amplitude_label)
        self.peak_amplitude_input = QComboBox()
        self.peak_amplitude_input.addItems(['max', 'min'])
        peak_amplitude_box.addWidget(self.peak_amplitude_input)

        self.layout.addLayout(peak_amplitude_box)

        # Add Group Button
        add_group_button = QPushButton('Add Animal')
        add_group_button.clicked.connect(self.add_group)
        self.layout.addWidget(add_group_button)

        # Save Button
        save_button = QPushButton('Save')
        save_button.clicked.connect(self.save)
        self.layout.addWidget(save_button)

    def show_context_menu(self, pos):
        index = self.group_files_table.indexAt(pos)
        if index.isValid():
            menu = QMenu()
            delete_action = menu.addAction('Delete Item')
            delete_action.triggered.connect(lambda: self.delete_item(index))
            menu.exec_(self.group_files_table.viewport().mapToGlobal(pos))

    def delete_item(self, index):
        self.group_files_table.removeRow(index.row())


    def add_group(self):
        # Getting all names
        names = []
        for i in range(self.group_files_table.rowCount()):
            names.append(self.group_files_table.item(i, 0).text())

        if self.add_item_screen is None:
            self.add_item_screen = AddItemScreen(names=names)
            self.add_item_screen.submitClicked.connect(self.add_item)
            self.add_item_screen.show()
        else:
            self.add_item_screen.close()
            self.add_item_screen = None

    def add_item(self, item_dict):
        self.item_from_add_group_screen[self.add_item_screen.name_input.text()] = item_dict
        self.add_item_screen = None
        self.group_files_table.setRowCount(self.group_files_table.rowCount() + 1)
        self.group_files_table.setItem(self.group_files_table.rowCount() - 1, 0, QTableWidgetItem(item_dict['name']))
        self.group_files_table.setItem(self.group_files_table.rowCount() - 1, 1, QTableWidgetItem(item_dict['dff_file_path']))
        self.group_files_table.setItem(self.group_files_table.rowCount() - 1, 2, QTableWidgetItem(item_dict['timestamps_file_path']))
        

    def save(self):
        save_name = self.group_name_input.text()
        if save_name == '':
            QMessageBox.warning(self, 'No Group Name', 'Please enter a group name')
            return

        table_files = {}
        for i in range(self.group_files_table.rowCount()):
            name = self.group_files_table.item(i, 0).text()

            if name in self.database.keys() and name not in self.item_from_add_group_screen.keys():
                table_files[name] = self.database[name]
            else:
                table_files[name] = self.item_from_add_group_screen[name]
        metric = self.peak_amplitude_input.currentText()

        group_dict = {
            'group_name': save_name,
            'group_files': table_files,
            'peak_amplitude_metric': metric
        }

        self.submitClicked.emit(group_dict)
        self.close()




class AddItemScreen(QMainWindow):
    submitClicked = pyqtSignal(dict)

    def __init__(self, names=[]):
        super().__init__()
        self.init_ui()
        self.setCentralWidget(self.window)
        self.dff_file_path = None
        self.timestamp_file_path = None
        self.names = names

    def init_ui(self):
        self.window = QWidget()
        self.setWindowTitle('Add Animal')
        self.resize(400, 200)

        self.layout = QVBoxLayout()
        self.window.setLayout(self.layout)

        # Item Name Box
        name_box = QHBoxLayout()
        name_label = QLabel('Name:')
        name_box.addWidget(name_label)
        self.name_input = QLineEdit()
        name_box.addWidget(self.name_input)
        self.layout.addLayout(name_box)

        # DFF File Path Box
        dff_file_path_box = QHBoxLayout()
        dff_file_path_label = QLabel('DFF File Path:')
        dff_file_path_box.addWidget(dff_file_path_label)
        self.dff_file_path_input = QPushButton('Select File')
        self.dff_file_path_input.clicked.connect(self.select_dff_file)
        dff_file_path_box.addWidget(self.dff_file_path_input)
        self.layout.addLayout(dff_file_path_box)

        # Select Column Name
        column_name_box = QHBoxLayout()
        column_name_label = QLabel('DFF Column Name:')
        column_name_box.addWidget(column_name_label)
        self.dff_column_name_input = QComboBox()
        column_name_box.addWidget(self.dff_column_name_input)
        
        self.layout.addLayout(column_name_box)

        # Timestamp File Path Box
        timestamp_file_path_box = QHBoxLayout()
        timestamp_file_path_label = QLabel('Timestamp File Path:')
        timestamp_file_path_box.addWidget(timestamp_file_path_label)
        self.timestamp_file_path_input = QPushButton('Select File')
        self.timestamp_file_path_input.clicked.connect(self.select_timestamp_file)
        timestamp_file_path_box.addWidget(self.timestamp_file_path_input)
        self.layout.addLayout(timestamp_file_path_box)

        # Select Column Name
        column_name_box = QHBoxLayout()
        column_name_label = QLabel('Column Name:')
        column_name_box.addWidget(column_name_label)
        self.timestamp_column_name_input = QComboBox()
        column_name_box.addWidget(self.timestamp_column_name_input)

        self.layout.addLayout(column_name_box)

        # Binned Peaks Box
        binned_peaks_box = QGroupBox('Binned Peaks')
        binned_peaks_layout = QVBoxLayout()
        binned_peaks_box.setLayout(binned_peaks_layout)

        # Enable Binned Peaks
        enable_binned_peaks_box = QHBoxLayout()
        enable_binned_peaks_label = QLabel('Enable Binned Peaks:')
        enable_binned_peaks_box.addWidget(enable_binned_peaks_label)
        self.enable_binned_peaks_input = QCheckBox()
        self.enable_binned_peaks_input.setChecked(False)
        self.enable_binned_peaks_input.stateChanged.connect(self.enable_binned_peaks)
        enable_binned_peaks_box.addWidget(self.enable_binned_peaks_input)
        binned_peaks_layout.addLayout(enable_binned_peaks_box)

        # Binned Peaks Start Time
        binned_peaks_start_time_box = QHBoxLayout()
        binned_peaks_start_time_label = QLabel('Start Times:')
        binned_peaks_start_time_box.addWidget(binned_peaks_start_time_label)
        self.binned_peaks_start_time_input = QComboBox()
        self.binned_peaks_start_time_input.setEnabled(False)
        binned_peaks_start_time_box.addWidget(self.binned_peaks_start_time_input)

        binned_peaks_layout.addLayout(binned_peaks_start_time_box)

        # Binned Peaks End Time
        binned_peaks_end_time_box = QHBoxLayout()
        binned_peaks_end_time_label = QLabel('End Times:')
        binned_peaks_end_time_box.addWidget(binned_peaks_end_time_label)
        self.binned_peaks_end_time_input = QComboBox()
        self.binned_peaks_end_time_input.setEnabled(False)
        binned_peaks_end_time_box.addWidget(self.binned_peaks_end_time_input)
        binned_peaks_layout.addLayout(binned_peaks_end_time_box)

        self.layout.addWidget(binned_peaks_box)

        # Save Button
        save_button = QPushButton('Save')
        save_button.clicked.connect(self.save)
        self.layout.addWidget(save_button)

    def enable_binned_peaks(self):
        # CHekc if timestamps file is selected
        if self.timestamp_file_path is None:
            self.enable_binned_peaks_input.setChecked(False)
            QMessageBox.warning(self, 'No Timestamps File', 'Please select a timestamp file first')
            return

        self.binned_peaks_start_time_input.setEnabled(self.enable_binned_peaks_input.isChecked())
        self.binned_peaks_end_time_input.setEnabled(self.enable_binned_peaks_input.isChecked())

        # Getting all items from the column name input
        columns = pd.read_csv(self.timestamp_file_path).columns

        if len(columns) < 2:
            QMessageBox.warning(self, 'Not Enough Columns', 'Please select a timestamp file with atleast 2 columns')
            self.enable_binned_peaks_input.setChecked(False)
            self.binned_peaks_start_time_input.setEnabled(False)
            self.binned_peaks_end_time_input.setEnabled(False)
            return

        self.binned_peaks_start_time_input.clear()
        self.binned_peaks_end_time_input.clear()
        self.binned_peaks_start_time_input.addItems(columns)
        self.binned_peaks_end_time_input.addItems(columns)

        self.binned_peaks_start_time_input.setCurrentIndex(0)
        self.binned_peaks_end_time_input.setCurrentIndex(1)


    def select_dff_file(self):
        # Select DFF File this file is either a csv or a h5 file
        folder_path = QFileDialog.getOpenFileName(self, 'Select DFF File', '', 'DFF Files (*.csv *.h5)')
        self.dff_file_path = folder_path[0]

        if self.dff_file_path.endswith('.csv'):
            d = pd.read_csv(self.dff_file_path)
            columns = d.columns 

            if d.empty or len(columns) == 0:
                QMessageBox.warning(self, 'Empty File', 'File is empty')
                return
            self.dff_column_name_input.clear()
            self.dff_column_name_input.addItems(columns)
            self.dff_column_name_input.setCurrentIndex(0)
        else:
            self.dff_column_name_input.clear()
            self.dff_column_name_input.addItem('dff')
        

    def select_timestamp_file(self):
        # Select Timestamp File
        folder_path = QFileDialog.getOpenFileName(self, 'Select Timestamp File', '', 'Timestamp Files (*.csv)')
        self.timestamp_file_path = folder_path[0]
        d = pd.read_csv(self.timestamp_file_path)
        columns = d.columns
        if d.empty or len(columns) == 0:
            QMessageBox.warning(self, 'Empty File', 'File is empty')
            return

        self.timestamp_column_name_input.clear()
        self.timestamp_column_name_input.addItems(columns)
        self.timestamp_column_name_input.setCurrentIndex(0)

    def save(self):
        filename = self.name_input.text()

        if filename == '':
            QMessageBox.warning(self, 'No Name', 'Please enter a name')
            return
        
        if self.dff_file_path is None:
            QMessageBox.warning(self, 'No DFF File', 'Please select a DFF file')
            return
    
        if self.timestamp_file_path is None:
            QMessageBox.warning(self, 'No Timestamps File', 'Please select a Timestamps file')
            return
        
        if filename in self.names:
            QMessageBox.warning(self, 'Name Already Exists', 'Name already exists')
            return

        dff_file_path = self.dff_file_path
        timestamp_file_path = self.timestamp_file_path
        timestamp_column_name = self.timestamp_column_name_input.currentText()
        dff_column_name = self.dff_column_name_input.currentText()

        if self.enable_binned_peaks_input.isChecked():
            start_time = self.binned_peaks_start_time_input.currentText()
            end_time = self.binned_peaks_end_time_input.currentText()

            if start_time == end_time:
                QMessageBox.warning(self, 'Start and End Times are same', 'Please select different start and end times')
                return
        else:
            start_time = None
            end_time = None

        event_dict = {
            'name': filename,
            'dff_file_path': dff_file_path,
            'dff_column_name': dff_column_name,
            'timestamps_file_path': timestamp_file_path,
            'timestamp_column_name': timestamp_column_name,
            'binned_peaks': self.enable_binned_peaks_input.isChecked(),
            'start_time': start_time,
            'end_time': end_time

        }

        self.submitClicked.emit(event_dict)
        self.close()