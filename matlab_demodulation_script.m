% MATLAB Demodulation Script for Fiber Photometry Data
% This script processes .lvm files from fiber photometry recordings to extract 
% control and signal channels and calculate dF/F
% 
% Author: Veleanu Maxime - Adapted from Yaroslav Sych
% Last Updated: 05/01/2025
% 
% Usage:
%   1. Set the dirPath variable to point to your .lvm files directory
%   2. Run the script to process all .lvm files in the directory
%
% Outputs:
%   - Periodogram plots (.png)
%   - Demodulated signal plots (.png)
%   - dF/F plots (.png)
%   - Control and signal CSV files for further analysis
%   - dF/F CSV files (integrated functionality from guppy_dff.py)
%
% Depends on:
%   - lvm_import2.m function for importing LabVIEW Measurement files

clear all

% Define the directory where the .lvm files are located
dirPath = 'C:\Users\veleanu\Documents\GitHub\veleanu_et_al\fiberphotometry_example_dataset\processed';

% Get a list of all .lvm files in the directory
fileList = dir(fullfile(dirPath, '*.lvm'));

% Loop over all files
for fileIdx = 1:length(fileList)
    % Get the current file name
    fileName = fileList(fileIdx).name;
    
    % Import the data
    data = lvm_import2(fullfile(dirPath, fileName));
    
    % Create a new directory for each file
    newDir = fullfile(dirPath, fileName(1:end-4)); % Remove the .lvm extension
    mkdir(newDir);
    
    % Create subdirectories
    mkdir(fullfile(newDir, 'plot'));
    mkdir(fullfile(newDir, 'guppy'));
    mkdir(fullfile(newDir, 'guppy', 'output'));
    
    % Check for nans in ch1
    y1 = data.Segment1.data(:,2);
    y1(isnan(y1)) = 0;
    timestamps = data.Segment1.data(:,1);
    
    % Digital lock-in for the reference channel
    Fs = 2e3;  % Sampling frequency in Hz
    L = length(y1); 
    NFFT = 2^nextpow2(L);
    Y = fft(y1,NFFT)/L;
    f = Fs/2*linspace(0,1,NFFT/2+1);
    
    % Power spectral density
    figure;
    hold on
    [psdestx,Fxx] = periodogram(y1,rectwin(length(y1)),length(y1),Fs);
    plot(Fxx,10*log10(psdestx)); grid on;
    xlabel('Hz'); ylabel('Power/Frequency (dB/Hz)');
    title('Periodogram Power Spectral Density Estimate');
    
    % Save the plot as a .png file in the 'plot' subdirectory
    saveas(gcf, fullfile(newDir, 'plot', [fileName '_Periodogram.png']));
    
    % Take a power in frequency band
    k = 1;
    demodSig = struct();
    N_samples = 0.05*Fs; % Window size for demodulation (50 ms)
    
    for l = 1:(length(y1)/N_samples-1)
        % Sub vector corresponding to 50 us
        ySig_sub = y1(k:k + N_samples);
        
        % Extract power of harmonic from subvector
        demodSig.control(l) = bandpower(ySig_sub,Fs,[473 493]);
        demodSig.green(l) = bandpower(ySig_sub,Fs,[190 210]);
        
        k = k + N_samples;
    end
    
    % Plot and save the demodSig data
    figure;
    subplot(2,1,1);
    plot(demodSig.control);
    title('control');
    subplot(2,1,2);
    plot(demodSig.green);
    title('green');
    saveas(gcf, fullfile(newDir, 'plot', [fileName '_demodulated.png']));
    
    % Save the demodSig data as .csv files in the 'guppy' subdirectory
    event_nb = length(demodSig.green)/2;
    timestamps = 0.5:0.5:event_nb;
    timestamps2 = timestamps/10;
    guppymatrix_control = [timestamps2' demodSig.control'];
    guppymatrix_signal = [timestamps2' demodSig.green'];
    
    % Create a cell array for the header
    header = {'timestamps', 'data', 'sampling_rate'};
    
    % Convert the data matrices to cell arrays
    guppycontrol_cell = num2cell(guppymatrix_control);
    guppysignal_cell = num2cell(guppymatrix_signal);
    
    % Add the "sampling rate" to the second row of the third column
    guppycontrol_cell{1,3} = 20;
    guppysignal_cell{1,3} = 20;
    
    % Combine the header and the data
    guppycontrol_cell = [header; guppycontrol_cell];
    guppysignal_cell = [header; guppysignal_cell];
    
    % Write the cell arrays to .csv files with suffixes instead of prefixes
    [~, fileNameWithoutExt, ~] = fileparts(fileName);
    writecell(guppycontrol_cell, fullfile(newDir, 'guppy', [fileNameWithoutExt '_C.csv']));
    writecell(guppysignal_cell, fullfile(newDir, 'guppy', [fileNameWithoutExt '_S.csv']));
    
    % Calculate dff and plot
    test_signal = demodSig.green(1:end);
    Fb = prctile(test_signal,10);
    dff = (test_signal - Fb)/Fb;
    figure;
    plot(1:length(dff),dff);
    
    % Save the dff plot in the 'plot' subdirectory
    saveas(gcf, fullfile(newDir, 'plot', [fileName '_DFF.png']));
    
    %% NEW CODE: Integrate Python guppy_dff.py functionality
    % Apply moving average filter to demodulated data
    filter_window = 15; % Same as FILTER_WINDOW in Python script
    
    % Create filter coefficient
    b = ones(filter_window, 1) / filter_window;
    
    % Apply filter to control and signal data
    filtered_control = filtfilt(b, 1, demodSig.control);
    filtered_signal = filtfilt(b, 1, demodSig.green);
    
    % Fit control to signal (linear regression)
    [p, ~] = polyfit(filtered_control, filtered_signal, 1);
    fitted_control = polyval(p, filtered_control);
    
    % Apply adjustment factor (equivalent to Python implementation)
    signal_mean = mean(filtered_signal);
    fitted_mean = mean(fitted_control);
    adjustment_factor = 0.95;
    adjusted_fitted_control = (fitted_control - fitted_mean) * adjustment_factor + signal_mean;
    
    % Calculate dF/F in percent
    dff_percent = ((filtered_signal - adjusted_fitted_control) ./ adjusted_fitted_control) * 100;
    
    % Create dF/F CSV file
    dff_matrix = [timestamps2' dff_percent'];
    dff_cell = num2cell(dff_matrix);
    dff_cell = [{'timestamps', 'data'}; dff_cell]; % Header
    
    % Save dF/F CSV
    dff_filename = fullfile(newDir, 'guppy', 'output', [fileNameWithoutExt '_dff.csv']);
    writecell(dff_cell, dff_filename);
    
    % Additional plot for the new dF/F calculation method
    figure;
    plot(timestamps2, dff_percent);
    title('dF/F (%) - Moving Average Method');
    xlabel('Time (s)');
    ylabel('dF/F (%)');
    saveas(gcf, fullfile(newDir, 'plot', [fileName '_DFF_MA.png']));
    
    % Close all figures to avoid memory issues
    close all;
    
    disp(['Successfully processed: ' fileName]);
end

disp('Batch processing complete!');