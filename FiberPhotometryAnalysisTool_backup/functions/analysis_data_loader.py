import pandas as pd
import numpy as np
import os
import h5py
from scipy.signal import find_peaks

class AnalysisDataLoader:
    def __init__(self):
        pass

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


    def get_psth(self, dff, event_indices, baseline_correct=True, start_time=-2, end_time=5):
        start_idx = start_time*20
        end_idx = end_time*20

        vals = []
        events_corrected = []
        for e in event_indices:
            if e > np.abs(start_idx) and e < (len(dff) - end_idx):
                vals.append(dff[e+start_idx:e+end_idx])
                events_corrected.append(int(e))

        time = np.around(np.arange(start_time, end_time, 1/20), decimals=3)
        psth = pd.DataFrame(vals).T.fillna(0)
        psth.index = time
        psth.columns = [str(np.around(divmod(vals/20, 60), decimals=2)[0]) + ", " + str(np.around(divmod(vals/20, 60), decimals=2)[1])  for vals in events_corrected]
        psth = psth.loc[:,~psth.columns.duplicated()].copy()

        if baseline_correct:
            mean_vals = psth[psth.index < -2].median()
            for id in mean_vals.index:
                psth[id] = psth[id].values - mean_vals.loc[id]

        return psth

    def get_zscore(self, data, start_idx=-1, baseline_correct=True):
        for col in data.columns:
            mean = data[data.index < start_idx][col].mean()
            std = data[data.index < start_idx][col].std()
            data[col] = (data[col] - mean)/std

        if baseline_correct:
            for col in data.columns:
                mean = data[data.index < start_idx][col].mean()
                data[col] = data[col] - mean

        return data
