import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt import QtCore
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pandas as pd


class EventEditor(QMainWindow):
    editComplete = pyqtSignal()

    def __init__(self, database):
        super().__init__()
        self.initUI()
        self.database = database
        self.groups = []
        self.animal_ids = {}
        self.events_updated = False

        self.init_data()
        self.init_event_editor_screen()
        self.init_plot_screen()
        self.setCentralWidget(self.area)
        self.data_fetcher()
        self.plot_data()

    def initUI(self):
        self.setWindowTitle('Event Editor')
        self.setGeometry(100, 100, 800, 600)
        
        self.area = DockArea()

        # Adding the the plot screen to the dock area
        self.plot_screen_dock = Dock("Plot Screen", size=(0.7, 1), hideTitle=True)
        self.event_screen_dock = Dock("Event Editor", size=(0.3, 1), hideTitle=True)

        self.area.addDock(self.event_screen_dock, 'left')
        self.area.addDock(self.plot_screen_dock, 'right')
        
        
        self.show()

    def init_event_editor_screen(self):
        # Variables In this screen

        self.event_screen = QWidget()
        self.event_layout = QVBoxLayout()
        
        self.event_screen.setLayout(self.event_layout)
        self.event_screen.show()

        # Box to select the Group
        groups_box = QHBoxLayout()
        group_selector_label = QLabel("Select Group:")
        groups_box.addWidget(group_selector_label)

        self.group_selector = QComboBox()
        self.group_selector.addItems(self.groups)
        self.group_selector.currentIndexChanged.connect(self.group_selector_changed)
        self.group_selector.setCurrentIndex(0)
        groups_box.addWidget(self.group_selector)

        self.event_layout.addLayout(groups_box)

        # Box to select the animal ids
        animal_id_box = QHBoxLayout()
        event_selector_label = QLabel("Select Animal ID:")
        animal_id_box.addWidget(event_selector_label)

        self.animal_id_selector = QComboBox()
        self.animal_id_selector.addItems(self.animal_ids[self.groups[0]])
        self.animal_id_selector.currentIndexChanged.connect(self.data_fetcher)
        self.animal_id_selector.setCurrentIndex(0)
        animal_id_box.addWidget(self.animal_id_selector)

        self.event_layout.addLayout(animal_id_box)


        # List of Events
        self.event_list = QListView()
        self.event_list_model = QStandardItemModel()
        self.event_list.clicked.connect(self.plot_data)
        self.event_list.setModel(self.event_list_model)
        self.event_layout.addWidget(self.event_list)

        # Buttons for event table controls
        event_table_controls = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self.select_all)
        event_table_controls.addWidget(select_all_button)

        deselect_all_button = QPushButton("Deselect All")  
        deselect_all_button.clicked.connect(self.deselect_all)
        event_table_controls.addWidget(deselect_all_button)

        mean_button = QPushButton("Mean")  
        mean_button.clicked.connect(self.mean)
        event_table_controls.addWidget(mean_button)

        self.event_layout.addLayout(event_table_controls)

        # Export Timestamps Button
        export_timestamps_button = QPushButton("Export Timestamps")
        export_timestamps_button.clicked.connect(self.export_timestamps)
        self.event_layout.addWidget(export_timestamps_button)

        # Export PSTH and Peak Amplitudes Button
        export_psth_button = QPushButton("Export PSTH and Peak Amplitudes")
        export_psth_button.clicked.connect(self.export_psth)
        self.event_layout.addWidget(export_psth_button)

        # Adding space to the layout
        self.event_layout.addStretch()

        # Save Events Button
        save_events_button = QPushButton("Save Events")
        save_events_button.clicked.connect(self.save_events)
        self.event_layout.addWidget(save_events_button)


        self.event_screen_dock.addWidget(self.event_screen)



    def group_selector_changed(self):
        self.animal_id_selector.clear()
        self.animal_id_selector.addItems(self.animal_ids[self.groups[self.group_selector.currentIndex()]])
        self.animal_id_selector.setCurrentIndex(0)
        self.plot_screen.clear()
        self.data_fetcher()


    def init_data(self):
        # Loading the groups
        self.groups = list(self.database.keys())
        # Loading the animal ids
        for group in self.groups:
            self.animal_ids[group] = [id for id in self.database[group]["event_related_activity"].keys() if "group_mean" not in id]

        # Loading the events
        self.events = {}
        for group in self.groups:
            self.events[group] = {}
            for animal_id in self.animal_ids[group]:
                self.events[group][animal_id] = {"event_id":{}, "event_name":[]}
                for event in self.database[group]["event_related_activity"][animal_id].columns:
                    event_name = str(str(round(float(event.split(", ")[0]))) + ":" + str(round(float(event.split(", ")[1]))))
                    self.events[group][animal_id]["event_name"].append(event_name)
                    self.events[group][animal_id]["event_id"][event_name] = event
                    

    def data_fetcher(self):
        self.event_list_model.clear()
        group = self.groups[self.group_selector.currentIndex()]
        animal_id = self.animal_ids[group][self.animal_id_selector.currentIndex()]

        for event in self.events[group][animal_id]["event_name"]:
            item = QStandardItem(event)
            item.setCheckable(True)
            item.setCheckState(Qt.Checked)
            item.setEditable(False)
            item.emitDataChanged()
            self.event_list_model.appendRow(item)

        self.event_list.setModel(self.event_list_model)

        self.data = self.database[group]["event_related_activity"][animal_id]
        self.plot_column_names = self.events[group][animal_id]["event_id"]

        self.plot_data()

    def select_all(self):
        for i in range(self.event_list_model.rowCount()):
            item = self.event_list_model.item(i)
            item.setCheckState(Qt.Checked)
        self.plot_data()
        

    def deselect_all(self):
        for i in range(self.event_list_model.rowCount()):
            item = self.event_list_model.item(i)
            item.setCheckState(Qt.Unchecked)
        self.plot_data()
        self.plot_screen.clear()

    def mean(self):
        mean_data = pd.DataFrame()

        for i in range(self.event_list_model.rowCount()):
            item = self.event_list_model.item(i)
            if item.checkState() == Qt.Checked:
                mean_data[item.text()] = self.data[self.plot_column_names[item.text()]]
                
        self.plot_screen.clear()
        self.plot_screen.plot(mean_data.index, mean_data.mean(axis=1).values, pen=(255, 0, 0))

        std_data = mean_data.std(axis=1)
        brush = pg.mkBrush(255, 0, 0, int(0.3 * 255))
        pen = pg.mkPen(255, 0, 0, width=0.5)
        upper = self.plot_screen.plot(mean_data.index, mean_data.mean(axis=1).values + std_data, pen=pen)
        lower = self.plot_screen.plot(mean_data.index, mean_data.mean(axis=1).values - std_data, pen=pen)
        fill = pg.FillBetweenItem(curve1=lower, curve2=upper, brush=brush)
        self.plot_screen.addItem(fill)



    def export_timestamps(self):
        # Gettitng Selected Events
        checked_events = []
        for i in range(self.event_list_model.rowCount()):
            item = self.event_list_model.item(i)
            if item.checkState() == Qt.Checked:
                name = self.plot_column_names[item.text()]
                name = float(name.split(", ")[0]) + float(name.split(", ")[1])
                checked_events.append(name)

        if len(checked_events) == 0:
            QMessageBox.about(self, "Error", "No events selected")
            return
        else:
            filename = QFileDialog.getSaveFileName(self, "Save File", "", "CSV (*.csv)")
            if filename[0] == "":
                return
            else:
                filename = filename[0]
            
            pd.DataFrame({"timestamps": checked_events}).to_csv(filename, index=False)
    
    def export_psth(self):
        # Gettitng Selected Events
        checked_events = []
        for i in range(self.event_list_model.rowCount()):
            item = self.event_list_model.item(i)
            if item.checkState() == Qt.Checked:
                checked_events.append(item.text())

        if len(checked_events) == 0:
            QMessageBox.about(self, "Error", "No events selected")
            return
        else:
            filename = QFileDialog.getSaveFileName(self, "Save File", "", "CSV (*.csv)")
            if filename[0] == "":
                return
            else:
                filename = filename[0]

            # Exporting the PSTH
            data = pd.DataFrame()

            for event in checked_events:
                data[self.plot_column_names[event]] = self.data[self.plot_column_names[event]]

            data.to_csv(filename.replace(".csv", "_psth.csv"))

            # Exporting the Peak Amplitudes
            amplitudes = pd.DataFrame()
            for event in checked_events:
                d = self.data[self.plot_column_names[event]]
                amp = d[d.index > 0].max()  
                baseline = d[d.index < 0].mean()
                amplitudes[event] = [amp - baseline]

            amplitudes = amplitudes.T.reset_index()
            amplitudes.columns = ["timestamp", "peak_amplitude"]
            amplitudes.to_csv(filename.replace(".csv", "_peak_amplitudes.csv"), index=False)

            # Dialog Box
            QMessageBox.about(self, "Export Complete", "Export Complete")
            
        


    def save_events(self):
        data = pd.DataFrame()

        for i in range(self.event_list_model.rowCount()):
            item = self.event_list_model.item(i)
            if item.checkState() == Qt.Checked:
                data[self.plot_column_names[item.text()]] = self.data[self.plot_column_names[item.text()]]

        if len(data.columns) == 0:
            QMessageBox.about(self, "Error", "No events selected")
            return
        else:
            answer = QMessageBox.question(self, "Save Events", "Are you sure you want to overwrite the events?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer == QMessageBox.No:
                return

        group = self.groups[self.group_selector.currentIndex()]
        animal_id = self.animal_ids[group][self.animal_id_selector.currentIndex()]

        self.database[group]["event_related_activity"][animal_id] = data
        self.init_data()
        self.data_fetcher()
        self.events_updated = True
        self.editComplete.emit()



    def init_plot_screen(self):
        self.plot_screen = pg.PlotWidget()
        self.plot_screen_dock.addWidget(self.plot_screen)


    def plot_data(self):
        # All checked events
        checked_events = []
        for i in range(self.event_list_model.rowCount()):
            item = self.event_list_model.item(i)
            if item.checkState() == Qt.Checked:
                checked_events.append(item.text())

        if len(checked_events) == 0:
            print("No events selected")
            self.plot_screen.clear()
            return

        self.plot_screen.clear()

        for event in checked_events:
            self.plot_screen.plot(self.data[self.plot_column_names[event]].index, self.data[self.plot_column_names[event]].values, pen=(255, 0, 0))



