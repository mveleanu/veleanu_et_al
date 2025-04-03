clear all

% Define the directory where the .lvm files are located
dirPath = 'C:\Users\veleanu\Documents\Fiber';

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
    
    % Check for nans in ch1
    y1 = data.Segment1.data(:,2);
    y1(isnan(y1)) = 0;
    timestamps = data.Segment1.data(:,1);
    
    % Digital lock-in for the reference channel
    Fs = 2e3;
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
    N_samples = 0.05*Fs;
    
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
end
