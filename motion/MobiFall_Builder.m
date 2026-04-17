function MobiFall_Builder()
    % ============================================================
    % MobiFall Builder (Fixed-Rate Version)
    %
    % Main changes:
    % 1) Resamples ALL sensors to a fixed uniform timeline.
    % 2) Keeps 256-sample windows.
    % 3) Stores XCNN as [256 x 1 x 6 x N] for the CNN.
    % 4) Stores XRaw as [8 x 1 x 256 x N] so that:
    %       raw_x = squeeze(XVal_Raw(:,1,:,i));
    %    gives an 8x256 matrix and raw_x(7,:), raw_x(8,:) work.
    % 5) Uses multiple ADL windows per trial by default to create
    %    harder negative examples than a single center slice.
    % ============================================================

    clearvars -except ans;
    clc;

    % ---------------- CONFIGURATION ----------------
    dataRoot = '.';
    windowSize = 256;

    % Fixed sampling rate for training
    % Recommendation: 100 Hz first. If you want easier Android runtime,
    % set this to 50 and retrain from scratch with that choice.
    targetFs = 100;
    dt_ns = 1e9 / targetFs;

    % How many ADL windows to extract from each ADL trial
    % 1 = only one window per ADL trial
    % 2 = center + one hard negative (recommended starting point)
    % 3 = even more ADL diversity
    numADLWindowsPerTrial = 2;

    % Minimum spacing between ADL window centers (in seconds)
    minADLSeparationSec = 1.0;
    minADLSeparationSamples = round(minADLSeparationSec * targetFs);

    % Reproducibility
    rng(42);

    fallCodes = {'FOL', 'FKL', 'BSC', 'SDL'};
    adlCodes  = {'STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'CSI', 'CSO'};

    fprintf('Scanning dataset directory: %s\n', dataRoot);
    files = dir(fullfile(dataRoot, '**', '*.txt'));

    if isempty(files)
        error('No .txt files found. Make sure the dataset is inside: %s', pwd);
    end

    % ---------------- INDEX ALL TRIAL FILES ----------------
    trials = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for i = 1:length(files)
        fileName = files(i).name;
        folder   = files(i).folder;

        % Expected pattern: WAL_acc_5_1.txt
        parsed = regexp(fileName, '^([A-Za-z]{3})_(acc|gyro|ori)_(\d+)_(\d+)\.txt$', 'tokens');
        if isempty(parsed)
            continue;
        end

        code    = upper(parsed{1}{1});
        sensor  = lower(parsed{1}{2});
        subject = str2double(parsed{1}{3});
        trialNum = str2double(parsed{1}{4});

        if ~ismember(code, fallCodes) && ~ismember(code, adlCodes)
            continue;
        end

        key = sprintf('%s_%d_%d', code, subject, trialNum);

        if ~isKey(trials, key)
            trials(key) = struct( ...
                'acc', '', ...
                'gyro', '', ...
                'ori', '', ...
                'code', code, ...
                'subject', subject, ...
                'trialNum', trialNum, ...
                'isFall', ismember(code, fallCodes));
        end

        tmp = trials(key);
        tmp.(sensor) = fullfile(folder, fileName);
        trials(key) = tmp;
    end

    fprintf('Found %d trial groups. Processing...\n', trials.Count);

    % ---------------- BUILD DATASET ----------------
    trialKeys = keys(trials);

    X_raw_list = {};
    Y_list = [];

    keptTrials = 0;
    skippedTrials = 0;
    fallSamples = 0;
    adlSamples = 0;

    for k = 1:length(trialKeys)
        t_info = trials(trialKeys{k});

        % Skip incomplete trials
        if isempty(t_info.acc) || isempty(t_info.gyro) || isempty(t_info.ori)
            skippedTrials = skippedTrials + 1;
            continue;
        end

        % 1) Read files
        [acc_t,  acc_data]  = readSensorData(t_info.acc);
        [gyro_t, gyro_data] = readSensorData(t_info.gyro);
        [ori_t,  ori_data]  = readSensorData(t_info.ori);

        if isempty(acc_t) || isempty(gyro_t) || isempty(ori_t)
            skippedTrials = skippedTrials + 1;
            continue;
        end

        % 2) Remove duplicate timestamps
        [acc_t, uniqueIdx] = unique(acc_t, 'stable');
        acc_data = acc_data(uniqueIdx, :);

        [gyro_t, uniqueIdx] = unique(gyro_t, 'stable');
        gyro_data = gyro_data(uniqueIdx, :);

        [ori_t, uniqueIdx] = unique(ori_t, 'stable');
        ori_data = ori_data(uniqueIdx, :);

        % Need at least 2 samples for interpolation
        if numel(acc_t) < 2 || numel(gyro_t) < 2 || numel(ori_t) < 2
            skippedTrials = skippedTrials + 1;
            continue;
        end

        % 3) Find overlapping region
        start_ns = max([acc_t(1), gyro_t(1), ori_t(1)]);
        end_ns   = min([acc_t(end), gyro_t(end), ori_t(end)]);

        if end_ns <= start_ns
            skippedTrials = skippedTrials + 1;
            continue;
        end

        % 4) Create a FIXED uniform timeline
        ref_t = (double(start_ns):dt_ns:double(end_ns))';

        if numel(ref_t) < 10
            skippedTrials = skippedTrials + 1;
            continue;
        end

        % 5) Interpolate ALL sensors to this fixed timeline
        % Orientation file in MobiFall is typically [azimuth, pitch, roll],
        % so we keep columns 2:3 = pitch, roll.
        try
            acc_synced  = interp1(double(acc_t),  acc_data,        ref_t, 'linear');
            gyro_synced = interp1(double(gyro_t), gyro_data,       ref_t, 'linear');
            ori_synced  = interp1(double(ori_t),  ori_data(:,2:3), ref_t, 'linear');
        catch
            skippedTrials = skippedTrials + 1;
            continue;
        end

        % Drop any rows that became NaN after interpolation
        validRows = ...
            ~any(isnan(acc_synced), 2) & ...
            ~any(isnan(gyro_synced), 2) & ...
            ~any(isnan(ori_synced), 2);

        acc_synced  = acc_synced(validRows, :);
        gyro_synced = gyro_synced(validRows, :);
        ori_synced  = ori_synced(validRows, :);

        if isempty(acc_synced)
            skippedTrials = skippedTrials + 1;
            continue;
        end

        % Combined matrix = [N x 8]
        % [AccX AccY AccZ GyroX GyroY GyroZ Pitch Roll]
        combined_data = [acc_synced, gyro_synced, ori_synced];

        % Acc magnitude for event localization
        acc_mag = vecnorm(acc_synced(:,1:3), 2, 2);

        % 6) Window extraction
        if t_info.isFall
            % One fall-centered window around highest accel magnitude
            [~, peakIdx] = max(acc_mag);
            window = extractWindow(combined_data, peakIdx, windowSize);

            X_raw_list{end+1} = window; %#ok<AGROW>
            Y_list(end+1) = 1; %#ok<AGROW>
            fallSamples = fallSamples + 1;
        else
            % Multiple ADL windows to create harder negatives
            centers = chooseADLWindowCenters( ...
                acc_mag, ...
                numADLWindowsPerTrial, ...
                minADLSeparationSamples, ...
                windowSize);

            for c = 1:numel(centers)
                window = extractWindow(combined_data, centers(c), windowSize);
                X_raw_list{end+1} = window; %#ok<AGROW>
                Y_list(end+1) = 0; %#ok<AGROW>
                adlSamples = adlSamples + 1;
            end
        end

        keptTrials = keptTrials + 1;
    end

    fprintf('Kept trials:   %d\n', keptTrials);
    fprintf('Skipped trials:%d\n', skippedTrials);
    fprintf('Fall samples:  %d\n', fallSamples);
    fprintf('ADL samples:   %d\n', adlSamples);

    if isempty(X_raw_list)
        error('No usable windows were generated.');
    end

    % ---------------- FORMAT OUTPUT TENSORS ----------------
    fprintf('Formatting tensors...\n');
    numSamples = numel(X_raw_list);

    % CNN input: [Time x 1 x Channels x Samples] = [256 x 1 x 6 x N]
    XCNN = zeros(windowSize, 1, 6, numSamples, 'single');

    % Raw heuristic input: [Channels x 1 x Time x Samples] = [8 x 1 x 256 x N]
    % This matches your training script:
    % raw_x = squeeze(XVal_Raw(:,1,:,i));  % -> 8 x 256
    XRaw = zeros(8, 1, windowSize, numSamples, 'single');

    for i = 1:numSamples
        mat = single(X_raw_list{i});   % [256 x 8]

        % First 6 channels for CNN
        XCNN(:,1,:,i) = permute(mat(:,1:6), [1 3 2]);   % [256 x 1 x 6]

        % All 8 channels for posture / raw heuristic
        XRaw(:,1,:,i) = permute(mat.', [1 3 2]);        % [8 x 1 x 256]
    end

    Y_Categorical = categorical(Y_list');

    % Save metadata too
    save('MobiFall_Ready.mat', ...
        'XCNN', 'XRaw', 'Y_Categorical', ...
        'targetFs', 'windowSize', ...
        'numADLWindowsPerTrial', 'minADLSeparationSec', ...
        '-v7.3');

    fprintf('\nSuccess! MobiFall_Ready.mat generated.\n');
    fprintf('Fixed sampling rate: %d Hz\n', targetFs);
    fprintf('Window size: %d samples (%.2f seconds)\n', windowSize, windowSize / targetFs);
end

% ============================================================
% HELPER FUNCTIONS
% ============================================================

function [t, data] = readSensorData(filePath)
    fid = fopen(filePath, 'r');
    if fid == -1
        t = [];
        data = [];
        return;
    end

    % Skip header until @DATA
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if strcmpi(line, '@DATA')
            break;
        end
    end

    C = textscan(fid, '%f %f %f %f', ...
        'Delimiter', ',', ...
        'CommentStyle', '#');

    fclose(fid);

    if numel(C) < 4 || isempty(C{1})
        t = [];
        data = [];
        return;
    end

    t = C{1};
    data = [C{2}, C{3}, C{4}];

    validRows = ~isnan(t) & ~any(isnan(data), 2);
    t = t(validRows);
    data = data(validRows, :);
end

function window = extractWindow(data, centerIdx, windowSize)
    N = size(data, 1);
    half = floor(windowSize / 2);

    startIdx = centerIdx - half;
    endIdx   = startIdx + windowSize - 1;

    if startIdx < 1
        startIdx = 1;
        endIdx = min(windowSize, N);
    end

    if endIdx > N
        endIdx = N;
        startIdx = max(1, N - windowSize + 1);
    end

    if N < windowSize
        window = zeros(windowSize, size(data, 2), 'like', data);
        window(1:N, :) = data;
    else
        window = data(startIdx:endIdx, :);
    end
end

function centers = chooseADLWindowCenters(accMag, numWindows, minSeparationSamples, windowSize)
    N = numel(accMag);

    if N == 0
        centers = [];
        return;
    end

    if N < windowSize
        centers = round(N/2);
        return;
    end

    centers = [];

    % 1) Always include the center window
    centerIdx = round(N / 2);
    centers(end+1) = centerIdx; %#ok<AGROW>

    if numWindows == 1
        return;
    end

    % 2) Add hard negatives around the strongest accel peaks,
    %    but keep them separated from existing centers.
    [~, sortIdx] = sort(accMag, 'descend');

    for i = 1:numel(sortIdx)
        idx = sortIdx(i);

        if all(abs(idx - centers) >= minSeparationSamples)
            centers(end+1) = idx; %#ok<AGROW>
            if numel(centers) >= numWindows
                centers = unique(centers, 'stable');
                return;
            end
        end
    end

    % 3) If still short, fill with evenly spaced centers
    fillCandidates = round(linspace(1, N, max(numWindows*4, 8)));

    for i = 1:numel(fillCandidates)
        idx = fillCandidates(i);

        if all(abs(idx - centers) >= max(1, floor(minSeparationSamples/2)))
            centers(end+1) = idx; %#ok<AGROW>
            if numel(centers) >= numWindows
                break;
            end
        end
    end

    centers = unique(centers, 'stable');
end