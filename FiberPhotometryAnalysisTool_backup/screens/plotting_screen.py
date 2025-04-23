import os
import numpy as np
from time import perf_counter
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt import QtCore
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
import pandas as pd

def load_data(self, database, signal_sampling_rate, start_time, end_time, baseline_correction, peak_detection_width=1, use_zscore=False):
    processed_data = {}

    for group in database.keys():
        processed_data[group] = {"event_related_activity":{}, "global_activity":{}, "binned_peaks":pd.DataFrame([])}
        processed_data[group]["event_related_activity"]["group_mean"] = pd.DataFrame([])
        processed_data[group]["global_activity"]["group_data"] = {"group":[], "peaks_amplitude":[], "peaks_number":[], "animal_id":[]}
        for animal_id in database[group].keys():
            dff_column_name = database[group][animal_id]["dff_column_name"]

            if database[group][animal_id]["dff_file_path"].endswith(".csv"):
                dff = pd.read_csv(database[group][animal_id]["dff_file_path"])[dff_column_name].values
            elif database[group][animal_id]["dff_file_path"].endswith(".h5"):
                with h5py.File(database[group][animal_id]["dff_file_path"], "r") as f:
                    guppy_dff = f["data"][()]
                dff = pd.DataFrame({"dff": guppy_dff})
                dff = dff["dff"].values
            else:
                return

            column_name = database[group][animal_id]['timestamp_column_name']

            timestamps = pd.read_csv(database[group][animal_id]["timestamps_file_path"])
            timestamps = (timestamps[column_name] * signal_sampling_rate).astype(int)

            psth = self.get_psth(dff, timestamps, baseline_correction, start_time=start_time, end_time=end_time)

            if use_zscore:
                psth = self.get_zscore(psth, start_idx=-2, baseline_correct=baseline_correction)

            processed_data[group]['event_related_activity'][animal_id] = psth
            processed_data[group]['event_related_activity']['group_mean'][animal_id] = psth.mean(axis=1)

            peaks, info = find_peaks(dff, width=peak_detection_width*signal_sampling_rate)
            processed_data[group]['global_activity'][animal_id] = {"peaks":peaks, "info":info}

            processed_data[group]['global_activity']["group_data"]["group"].append(group)
            processed_data[group]['global_activity']["group_data"]["peaks_amplitude"].append(info["prominences"].mean())
            processed_data[group]['global_activity']["group_data"]["peaks_number"].append(len(peaks))
            processed_data[group]['global_activity']["group_data"]["animal_id"].append(animal_id)

            # Getting the binned peaks information
            if database[group][animal_id]["binned_peaks"]:
                start_time_col = database[group][animal_id]["start_time"]
                end_time_col = database[group][animal_id]["end_time"]

                binned_data_timestamps = pd.read_csv(database[group][animal_id]["timestamps_file_path"])
                binned_data_timestamps = binned_data_timestamps.dropna(subset=[start_time_col, end_time_col])

                start_times = (binned_data_timestamps[start_time_col] * signal_sampling_rate).astype(int)
                end_times = (binned_data_timestamps[end_time_col] * signal_sampling_rate).astype(int)

                binned_peaks_number = []
                binned_peaks_amplitude = []
                peak_rates = []

                for i in range(len(start_times)):
                    p = peaks[(peaks > start_times[i]) & (peaks < end_times[i])]
                    if len(p) > 0:
                        binned_peaks_number.append(len(p))
                        # Calculate peak rate (peaks per second)
                        episode_duration = (end_times[i] - start_times[i]) / signal_sampling_rate
                        peak_rate = len(p) / episode_duration
                        peak_rates.append(peak_rate)

                    a = info["prominences"][(peaks > start_times[i]) & (peaks < end_times[i])]
                    if len(a) > 0:
                        binned_peaks_amplitude.append(a.mean())

                average_peaks_number = np.mean(binned_peaks_number)
                average_peaks_amplitude = np.mean(binned_peaks_amplitude)
                average_peak_rate = np.mean(peak_rates)  # Average peak rate (peaks per second)
                total_peaks = np.sum(binned_peaks_number)

                binned_peaks = pd.DataFrame({
                    "group": [group],
                    "animal_id": [animal_id],
                    "average_peaks_number": [average_peaks_number],
                    "average_peaks_amplitude": [average_peaks_amplitude],
                    "average_peak_rate": [average_peak_rate],  # Add average peak rate to DataFrame
                    "total_peaks": [total_peaks]
                })
                processed_data[group]["binned_peaks"] = pd.concat([processed_data[group]["binned_peaks"], binned_peaks], axis=0)

        processed_data[group]['event_related_activity']['group_mean'] = pd.DataFrame(processed_data[group]['event_related_activity']['group_mean'])

    # Debugging: Print the binned peaks data
    binned_peaks_data = pd.concat([processed_data[group]["binned_peaks"] for group in processed_data.keys()])
    print("Binned peaks data prepared for export:", binned_peaks_data.head())

    return processed_data

class Plotter:
    def __init__(self, data, psth_area, heatmap_area, peak_amplitude_area, overall_peak_amplitude_area, group_order, use_zscore, metric_list):
        self.data = data
        self.psth_area = psth_area
        self.heatmap_area = heatmap_area
        self.peak_amplitude_area = peak_amplitude_area
        self.overall_peak_amplitude_area = overall_peak_amplitude_area
        self.group_order = group_order
        self.use_zscore = use_zscore
        self.metric_list = metric_list
        self.color_map = self.generate_colormap()
        self.init_ui()
        self.plot_psth()
        self.plot_heatmap()
        self.plot_peak_amplitude()

        # Assign binned peaks data for export
        self.binned_peaks_data = pd.concat([self.data[group]["binned_peaks"] for group in self.data.keys()])
        print("Binned peaks data assigned for export:", self.binned_peaks_data.head())

    def init_ui(self):
        # PSTH Screen
        self.psth_screen = pg.PlotWidget(name="PSTH")
        self.psth_screen.setTitle("PSTH")
        if self.use_zscore:
            self.psth_screen.setLabel("left", "z-score")
        else:
            self.psth_screen.setLabel("left", "dF/F")
        self.psth_screen.setLabel("bottom", "Time (s)")
        self.psth_screen.addLegend()

        self.psth_area.addWidget(self.psth_screen)

        # Heatmap Screen
        self.heatmap_screen = None

        self.group_selector = QComboBox()
        self.group_selector.addItems(list(self.data.keys()))
        # self.group_selector.currentIndexChanged.connect(self.plot_heatmap)
        self.heatmap_area.addWidget(self.group_selector)

        self.aid_selector = QComboBox()
        list_items = list(self.data[self.group_selector.currentText()]["event_related_activity"].keys())
        list_items.remove("group_mean")
        self.aid_selector.addItems(list_items)
        self.heatmap_area.addWidget(self.aid_selector)
        self.color_bar = None
    
        # Linking Group and Animal ID Selector
        self.group_selector.currentIndexChanged.connect(lambda: self.aid_selector.clear())
        self.group_selector.currentIndexChanged.connect(lambda: self.aid_selector.addItems(self.get_groups_for_heatmap()))
        # self.group_selector.currentIndexChanged.connect(lambda: self.plot_heatmap())

        self.aid_selector.currentIndexChanged.connect(self.plot_heatmap)

        # Peak Amplitude Screen
        self.peak_amplitude_screen = pg.PlotWidget(name="Peak Amplitude")
        self.peak_amplitude_screen.setTitle("Peak Amplitude")
        self.peak_amplitude_screen.setLabel("left", "dF/F")
        self.peak_amplitude_screen.setLabel("bottom", "Time (s)")
        self.peak_amplitude_area.addWidget(self.peak_amplitude_screen)

        # Overall Peak Amplitude Screen
        self.select_peak_type = QComboBox()
        self.select_peak_type.addItems(["Amplitude", "Number", "Binned Peak Amplitude", "Binned Peak Numbers", "Binned Peak Total"])
        self.overall_peak_amplitude_area.addWidget(self.select_peak_type)
        self.select_peak_type.setCurrentIndex(0)
        self.select_peak_type.currentIndexChanged.connect(self.plot_overall_peak_amplitude)

        self.overall_peak_amplitude_screen = pg.PlotWidget(name="Overall Peak Amplitude")
#        self.overall_peak_amplitude_screen.setTitle("Overall Peak Amplitude")
        self.overall_peak_amplitude_screen.setLabel("left", "dF/F")
        self.overall_peak_amplitude_screen.setLabel("bottom", "Time (s)")
        self.overall_peak_amplitude_area.addWidget(self.overall_peak_amplitude_screen)

        self.plot_overall_peak_amplitude()




    def get_groups_for_heatmap(self):
        list_items = list(self.data[self.group_selector.currentText()]["event_related_activity"].keys())
        list_items.remove("group_mean")
        return list_items
    
    def plot_psth(self):
        self.psth_screen.clear()
        self.psth_data = pd.DataFrame()

        for group in self.data.keys():
            mean = self.data[group]["event_related_activity"]["group_mean"].mean(axis=1)
            sem = self.data[group]["event_related_activity"]["group_mean"].sem(axis=1)
            
            
            color = self.color_map[group]
            brush = pg.mkBrush(color[0], color[1], color[2], int(0.3 * 255))
            pen = pg.mkPen(color=color, width=2)

            # Plot mean line
            self.psth_screen.plot(mean.index, mean.values.flatten(), pen=pen, name=group)
            
            # Plot error bars
            pen = pg.mkPen(color=color, width=0.5)
            upper = self.psth_screen.plot(mean.index, mean.values.flatten() + sem.values.flatten(), pen=pen)
            lower = self.psth_screen.plot(mean.index, mean.values.flatten() - sem.values.flatten(), pen=pen)
            fill = pg.FillBetweenItem(curve1=lower, curve2=upper, brush=brush)
            self.psth_screen.addItem(fill)

            self.psth_data[group+"_mean"] = mean.values.flatten()
            self.psth_data[group+"_sem"] = sem.values.flatten()
            self.psth_data.index = mean.index
        


        self.region_selector = pg.LinearRegionItem([-1,1])
        self.region_selector.setZValue(-10)
        self.psth_screen.addItem(self.region_selector)

        self.region_selector.sigRegionChanged.connect(self.plot_peak_amplitude)


    def plot_heatmap(self):
        group = self.group_selector.currentText()
        aid = self.aid_selector.currentText()

        try:
            data = self.data[group]["event_related_activity"][aid]
        except:
            return

        if self.heatmap_screen is not None:
            self.heatmap_screen.setParent(None)

        self.heatmap_screen = pg.PlotWidget(name="Heatmap")
        self.heatmap_area.addWidget(self.heatmap_screen)
        

        self.heatmap_screen.setTitle(f"Heatmap: {group} - {aid}")
        self.heatmap_screen.setLabel("left", "Events")
        self.heatmap_screen.setLabel("bottom", "Time (s)")
        


        self.img = pg.ImageItem(image=data.to_numpy())

        self.heatmap_screen.addItem(self.img)
        self.heatmap_screen.autoRange()

        if self.color_bar is not None:
            self.heatmap_screen.removeItem(self.color_bar)

        self.color_bar = self.heatmap_screen.addColorBar(self.img, colorMap='inferno', interactive=False, values=(data.min().min(), data.max().max()))

        input_range = (0, data.shape[0])
        output_range = (data.index.min(), data.index.max())
        axis = self.heatmap_screen.getAxis('bottom')
        
        axis.tickStrings = lambda values, scale, spacing: [str(np.round(np.interp(v, input_range, output_range), 1)) for v in values]



    def generate_colormap(self):
        cmap = plt.get_cmap('hsv') 
        colors = [cmap(i / len(self.data)) for i in range(len(self.data))]
        colors = [(int(r*255), int(g*255), int(b*255), int(a*255)) for r, g, b, a in colors]
        keys = list(self.data.keys())
        return {keys[i]: colors[i] for i in range(len(keys))}
    
    def plot_peak_amplitude(self):
        self.peak_amplitude_screen.clear()
        start_point = self.region_selector.getRegion()[0]
        end_point = self.region_selector.getRegion()[1]

        # Getting Peak Data
        self.peak_amplitude_data = pd.DataFrame()

        peak_data = {"group":[], "peak_amplitude":[], "animal_id":[]}
        
        try:
            for group in self.group_order:
                d = []
                for animal_id in self.data[group]["event_related_activity"].keys():
                    if animal_id == "group_mean":
                        continue

                    data = self.data[group]["event_related_activity"][animal_id]
                    data = data.loc[start_point:end_point]


                    peak_data["group"].append(group)
                    peak_data["animal_id"].append(animal_id)

                    if self.metric_list[group] == "min":
                        peak_data["peak_amplitude"].append(data.min().mean())
                    else:
                        peak_data["peak_amplitude"].append(data.max().mean())
        except:
            pass

        peak_data = pd.DataFrame(peak_data)
        peak_data["group"] = pd.Categorical(peak_data["group"], categories=self.group_order, ordered=True)
        self.peak_amplitude_data = peak_data

        # Plotting Peak Data as Bar Graph
        bargraph_data = peak_data.groupby("group", observed=True)["peak_amplitude"].mean()
        bg = pg.BarGraphItem(x=np.arange(len(bargraph_data)), height=bargraph_data.values, width=0.6, brushes=[self.color_map[group] for group in bargraph_data.index])
        self.peak_amplitude_screen.addItem(bg)

        # Scatter Plot
        sp = pg.ScatterPlotItem(hoverable=True, hoverPen='g', pen=None, symbol='o', symbolPen=None, symbolBrush="white")

        for i, group in enumerate(self.group_order):
            d = peak_data[peak_data["group"] == group].reset_index(drop=True)
            for j in range(len(d)):
                sp.addPoints(x=[i], y=[d["peak_amplitude"][j]], size=10, brush="white", data=f"{d['animal_id'][j]}: {d['peak_amplitude'][j]:.2f}")

        self.peak_amplitude_screen.addItem(sp)


        # for i, group in enumerate(self.group_order):
        #     d = peak_data[peak_data["group"] == group].reset_index(drop=True)
        #     self.peak_amplitude_screen.plot([i]*len(d), d["peak_amplitude"], pen=None, symbol='o', symbolPen=None, symbolBrush="white", hoverable=True)

        self.peak_amplitude_screen.setLabel("left", "Peak Amplitude")
        self.peak_amplitude_screen.setLabel("bottom", "Group")
        self.peak_amplitude_screen.getAxis("bottom").setTicks([[(i, group) for i, group in enumerate(self.group_order)]])


    def plot_overall_peak_amplitude(self):
        self.overall_peak_amplitude_screen.clear()
        peak_data = pd.DataFrame()
        for group in self.group_order:
            peak_data = pd.concat([peak_data, pd.DataFrame(self.data[group]["global_activity"]["group_data"])])

        peak_data["group"] = pd.Categorical(peak_data["group"], categories=self.group_order, ordered=True)
        self.overall_peak_amplitude_data = peak_data

        if self.select_peak_type.currentText() == "Amplitude":
            bar_graph_data = peak_data.groupby("group", observed=True)["peaks_amplitude"].mean()

            bg = pg.BarGraphItem(x=np.arange(len(bar_graph_data)), height=bar_graph_data.values, width=0.6, brushes=[self.color_map[group] for group in bar_graph_data.index])
            self.overall_peak_amplitude_screen.addItem(bg)

            self.overall_peak_amplitude_screen.setLabel("left", "Peak Amplitude")
            self.overall_peak_amplitude_screen.setLabel("bottom", "Group")

            self.overall_peak_amplitude_screen.getAxis("bottom").setTicks([[(i, group) for i, group in enumerate(bar_graph_data.index)]])

            # Scatter Plot
            sp = pg.ScatterPlotItem(hoverable=True, hoverPen='g', pen=None, symbol='o', symbolPen=None, symbolBrush="white")

            for i, group in enumerate(self.group_order):
                data = peak_data[peak_data["group"] == group]
                for j in range(len(data)):
                    sp.addPoints(x=[i], y=[data["peaks_amplitude"].iloc[j]], size=10, brush="white", data=f"{data['animal_id'].iloc[j]}: {data['peaks_amplitude'].iloc[j]:.2f}")

            self.overall_peak_amplitude_screen.addItem(sp)


            # for i, group in enumerate(self.group_order):
            #     data = peak_data[peak_data["group"] == group]
            #     self.overall_peak_amplitude_screen.plot([i]*len(data), data["peaks_amplitude"], pen=None, symbol='o', symbolPen=None, symbolBrush="white")
        elif self.select_peak_type.currentText() == "Number":
            bar_graph_data = peak_data.groupby("group", observed=True)["peaks_number"].mean()

            bg = pg.BarGraphItem(x=np.arange(len(bar_graph_data)), height=bar_graph_data.values, width=0.6, brushes=[self.color_map[group] for group in bar_graph_data.index])
            self.overall_peak_amplitude_screen.addItem(bg)

            self.overall_peak_amplitude_screen.setLabel("left", "Peak Number")
            self.overall_peak_amplitude_screen.setLabel("bottom", "Group")

            self.overall_peak_amplitude_screen.getAxis("bottom").setTicks([[(i, group) for i, group in enumerate(bar_graph_data.index)]])

            # Scatter Plot
            # for i, group in enumerate(self.group_order):
            #     data = peak_data[peak_data["group"] == group]
            #     self.overall_peak_amplitude_screen.plot([i]*len(data), data["peaks_number"], pen=None, symbol='o', symbolPen=None, symbolBrush="white")
            sp = pg.ScatterPlotItem(hoverable=True, hoverPen='g', pen=None, symbol='o', symbolPen=None, symbolBrush="white")

            for i, group in enumerate(self.group_order):
                data = peak_data[peak_data["group"] == group]
                for j in range(len(data)):
                    sp.addPoints(x=[i], y=[data["peaks_number"].iloc[j]], size=10, brush="white", data=f"{data['animal_id'].iloc[j]}: {data['peaks_number'].iloc[j]}")

            self.overall_peak_amplitude_screen.addItem(sp)

        elif self.select_peak_type.currentText() == "Binned Peak Amplitude":
            d = pd.DataFrame()

            for group in self.group_order:
                d = pd.concat([d, self.data[group]["binned_peaks"]])

            if len(d) == 0:
                return

            d["group"] = pd.Categorical(d["group"], categories=self.group_order, ordered=True)
            bar_graph_data = d.groupby("group", observed=True)["average_peaks_amplitude"].mean()

            bg = pg.BarGraphItem(x=np.arange(len(bar_graph_data)), height=bar_graph_data.values, width=0.6, brushes=[self.color_map[group] for group in bar_graph_data.index])
            self.overall_peak_amplitude_screen.addItem(bg)

            self.overall_peak_amplitude_screen.setLabel("left", "Peak Amplitude")
            self.overall_peak_amplitude_screen.setLabel("bottom", "Group")

            self.overall_peak_amplitude_screen.getAxis("bottom").setTicks([[(i, group) for i, group in enumerate(bar_graph_data.index)]])
            
            # Scatter Plot
            sp = pg.ScatterPlotItem(hoverable=True, hoverPen='g', pen=None, symbol='o', symbolPen=None, symbolBrush="white")

            for i, group in enumerate(self.group_order):
                data = d[d["group"] == group]
                for j in range(len(data)):
                    sp.addPoints(x=[i], y=[data["average_peaks_amplitude"].iloc[j]], size=10, brush="white", data=f"{data['animal_id'].iloc[j]}: {data['average_peaks_amplitude'].iloc[j]:.2f}")

            self.overall_peak_amplitude_screen.addItem(sp)

        elif self.select_peak_type.currentText() == "Binned Peak Numbers":
            d = pd.DataFrame()

            for group in self.group_order:
                d = pd.concat([d, self.data[group]["binned_peaks"]])

            if len(d) == 0:
                return

            d["group"] = pd.Categorical(d["group"], categories=self.group_order, ordered=True)
            bar_graph_data = d.groupby("group", observed=True)["average_peaks_number"].mean()

            bg = pg.BarGraphItem(x=np.arange(len(bar_graph_data)), height=bar_graph_data.values, width=0.6, brushes=[self.color_map[group] for group in bar_graph_data.index])
            self.overall_peak_amplitude_screen.addItem(bg)

            self.overall_peak_amplitude_screen.setLabel("left", "Peak Number")
            self.overall_peak_amplitude_screen.setLabel("bottom", "Group")

            self.overall_peak_amplitude_screen.getAxis("bottom").setTicks([[(i, group) for i, group in enumerate(bar_graph_data.index)]])

            # Scatter Plot
            sp = pg.ScatterPlotItem(hoverable=True, hoverPen='g', pen=None, symbol='o', symbolPen=None, symbolBrush="white")

            for i, group in enumerate(self.group_order):
                data = d[d["group"] == group]
                for j in range(len(data)):
                    sp.addPoints(x=[i], y=[data["average_peaks_number"].iloc[j]], size=10, brush="white", data=f"{data['animal_id'].iloc[j]}: {data['average_peaks_number'].iloc[j]}")

            self.overall_peak_amplitude_screen.addItem(sp)
        elif self.select_peak_type.currentText() == "Binned Peak Total":
            d = pd.DataFrame()

            for group in self.group_order:
                d = pd.concat([d, self.data[group]["binned_peaks"]])
            
            if len(d) == 0:
                return

            d["group"] = pd.Categorical(d["group"], categories=self.group_order, ordered=True)
            bar_graph_data = d.groupby("group", observed=True)["total_peaks"].mean()

            bg = pg.BarGraphItem(x=np.arange(len(bar_graph_data)), height=bar_graph_data.values, width=0.6, brushes=[self.color_map[group] for group in bar_graph_data.index])
            self.overall_peak_amplitude_screen.addItem(bg)

            self.overall_peak_amplitude_screen.setLabel("left", "Total Peaks")
            self.overall_peak_amplitude_screen.setLabel("bottom", "Group")

            self.overall_peak_amplitude_screen.getAxis("bottom").setTicks([[(i, group) for i, group in enumerate(bar_graph_data.index) ]])

            # Scatter Plot
            sp = pg.ScatterPlotItem(hoverable=True, hoverPen='g', pen=None, symbol='o', symbolPen=None, symbolBrush="white")

            for i, group in enumerate(self.group_order):
                data = d[d["group"] == group]
                for j in range(len(data)):
                    sp.addPoints(x=[i], y=[data["total_peaks"].iloc[j]], size=10, brush="white", data=f"{data['animal_id'].iloc[j]}: {data['total_peaks'].iloc[j]}")

            self.overall_peak_amplitude_screen.addItem(sp)


    def export_data(self, file_path):
        """
        Save all results in appropriate subfolders within the chosen directory
        """
        # Ensure the export directory exists
        os.makedirs(file_path, exist_ok=True)

        # Exporting PSTH Data
        psth_output_path = os.path.join(file_path, "psth.csv")
        self.psth_data.to_csv(psth_output_path, index=True)
        print(f"PSTH data exported to {psth_output_path}")

        # Export ALL individual animal PSTHs with proper naming
        for group in self.data.keys():
            for animal_id in self.data[group]["event_related_activity"].keys():
                if animal_id == "group_mean":
                    continue
                    
                # Get PSTH data for this animal
                animal_psth = self.data[group]["event_related_activity"][animal_id]
                
                # Create filename with format: "animalIDgroup_psth.csv"
                animal_filename = f"{animal_id}{group}_psth.csv"
                animal_psth_path = os.path.join(file_path, animal_filename)
                
                # Export to CSV
                animal_psth.to_csv(animal_psth_path, index=True)
                print(f"Individual PSTH exported to {animal_psth_path}")


        # Exporting Binned Peaks Data
        if hasattr(self, 'binned_peaks_data'):
            print("Binned peaks data available for export.")
            binned_peaks_output_path = os.path.join(file_path, "binned_peaks.csv")
            self.binned_peaks_data.to_csv(binned_peaks_output_path, index=False)
            print(f"Binned peaks data exported to {binned_peaks_output_path}")
        else:
            print("Binned peaks data not available for export.")

        # Exporting Average Peak Rate Data
        if hasattr(self, 'average_peak_rate_data'):
            average_peak_rate_output_path = os.path.join(file_path, "average_peak_rate.csv")
            self.average_peak_rate_data.to_csv(average_peak_rate_output_path, index=False)
            print(f"Average peak rate data exported to {average_peak_rate_output_path}")

        # ===== ADDED CODE FROM VISUALIZE_RESULTS.PY =====
        # Calculate and export AUC and peak response CSV files using the same calculations as in visualize_results.py
        
        # Prepare data structures for statistics
        stats_data = {
            'mouse_id': [],
            'day4_peak': [],
            'day6_peak': [],
            'day4_mean': [],
            'day6_mean': [],
            'day4_auc': [],
            'day6_auc': []
        }
        
        # Process each group and animal
        for group in self.data.keys():
            for animal_id in self.data[group]["event_related_activity"].keys():
                if animal_id == "group_mean":
                    continue
                
                # Group name follows the pattern: 'D4' or 'D6'
                day = group[:2]  # Extract 'D4' or 'D6'
                
                # Calculate metrics from PSTH
                psth = self.data[group]["event_related_activity"][animal_id]
                time_index = psth.index
                
                # Define windows exactly as in visualize_results.py
                peak_window = (time_index >= -1) & (time_index <= 2)
                response_window = (time_index >= -1) & (time_index <= 2)
                
                # Calculate mean across trials (columns)
                mean_response = psth.mean(axis=1)
                
                # If this is the first time seeing this animal, add to mouse_id
                if animal_id not in stats_data['mouse_id']:
                    stats_data['mouse_id'].append(animal_id)
                
                # Find the index for this animal
                animal_idx = stats_data['mouse_id'].index(animal_id)
                
                # Extend the lists if needed
                while len(stats_data['day4_peak']) <= animal_idx:
                    stats_data['day4_peak'].append(None)
                    stats_data['day6_peak'].append(None)
                    stats_data['day4_mean'].append(None)
                    stats_data['day6_mean'].append(None)
                    stats_data['day4_auc'].append(None)
                    stats_data['day6_auc'].append(None)
                
                # Calculate metrics based on day
                if day == 'D4':
                    stats_data['day4_peak'][animal_idx] = np.max(mean_response[peak_window])
                    stats_data['day4_mean'][animal_idx] = np.mean(mean_response[response_window])
                    stats_data['day4_auc'][animal_idx] = np.trapz(mean_response[response_window], time_index[response_window])
                elif day == 'D6':
                    stats_data['day6_peak'][animal_idx] = np.max(mean_response[peak_window])
                    stats_data['day6_mean'][animal_idx] = np.mean(mean_response[response_window])
                    stats_data['day6_auc'][animal_idx] = np.trapz(mean_response[response_window], time_index[response_window])
        
        # Create and save the max_peak_amplitude.csv file
        peak_df = pd.DataFrame({
            'Mouse_ID': stats_data['mouse_id'],
            'Day_4': stats_data['day4_peak'],
            'Day_6': stats_data['day6_peak']
        })
        peak_df.to_csv(os.path.join(file_path, "max_peak_amplitude.csv"), index=False)
        print(f"Peak response data exported to {os.path.join(file_path, 'max_peak_amplitude.csv')}")
        
        # Create and save the auc_response.csv file
        auc_df = pd.DataFrame({
            'Mouse_ID': stats_data['mouse_id'],
            'Day_4': stats_data['day4_auc'],
            'Day_6': stats_data['day6_auc']
        })
        auc_df.to_csv(os.path.join(file_path, "auc_response.csv"), index=False)
        print(f"AUC response data exported to {os.path.join(file_path, 'auc_response.csv')}")
        
        # Also export mean_response.csv for completeness
        mean_df = pd.DataFrame({
            'Mouse_ID': stats_data['mouse_id'],
            'Day_4': stats_data['day4_mean'],
            'Day_6': stats_data['day6_mean']
        })
        mean_df.to_csv(os.path.join(file_path, "mean_response.csv"), index=False)
        print(f"Mean response data exported to {os.path.join(file_path, 'mean_response.csv')}")
        # ===== END ADDED CODE =====

        # Exporting Heatmap Data
        for group in self.data.keys():
            for aid in self.data[group]["event_related_activity"].keys():
                if aid == "group_mean":
                    continue
                data = self.data[group]["event_related_activity"][aid]
                data.to_csv(os.path.join(file_path, f"heatmap_{group}_{aid}.csv"))