clear all; clc; close all; 

% ==========================================
% FALLGUARD DEMO DAY TRAINING SCRIPT
% ==========================================

Network;

disp('Loading Master Dataset...');
load('MobiFall_Ready.mat');

% 1. Create an 80/20 Train/Validation Split
disp('Splitting data into 80% Train and 20% Validation...');
numSamples = size(XCNN, 4); 
rng(42); 
% cv = cvpartition(numSamples, 'HoldOut', 0.20);
cv = cvpartition(Y_Categorical, 'HoldOut', 0.20);

idxTrain = training(cv); % 80% of indices
idxVal = test(cv);       % 20% of indices

% 2. Route the data into the correct variables
XTrain = XCNN(:, :, :, idxTrain);
YTrain = Y_Categorical(idxTrain);

XVal_CNN = XCNN(:, :, :, idxVal);
XVal_Raw = XRaw(:, :, :, idxVal); % Keep the 8-channel raw data for the heuristic!
YVal = Y_Categorical(idxVal);

disp(['Training samples: ', num2str(sum(idxTrain))]);
disp(['Validation samples: ', num2str(sum(idxVal))]);

% 3. Setup the Optimizer
options = trainingOptions('adam', ...
    'MaxEpochs', 36, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'ValidationData', {XVal_CNN, YVal}, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ... % The crucial Demo Day plot!
    'Verbose', false);

% 4. TRAIN THE NETWORK
disp('Starting Neural Network Training...');
FallNet = trainnet(XTrain, YTrain, layers, "crossentropy", options);
disp('Training Complete! Network saved as FallNet.');

% ==========================================
% POSTURE VERIFICATION (THE LEAKY FILTER FIX)
% ==========================================
disp('Running Post-Processing Heuristic on Validation Set...');

% 1. Get raw probabilities from the trained network using the CNN slice
probs = predict(FallNet, XVal_CNN); 

POSTURE_SHIFT_THRESHOLD = 30.0;

% 2. Apply the Auxiliary Orientation Check
for i = 1:size(XVal_Raw, 4)
    prob_fall = probs(i, 2); % Column 2 is Fall Probability
    
    % VET EVERYTHING: If there's a >10% chance of a fall, check posture!
    if prob_fall >= 0.10
        % Extract the raw 8x256 matrix for this specific trial
        raw_x = squeeze(XVal_Raw(:, 1, :, i)); 
        
        % In MATLAB, arrays are 1-indexed. Row 7 is Pitch, Row 8 is Roll.
        pitch_window = raw_x(7, :); 
        roll_window  = raw_x(8, :);
        
        % Calculate Peak-to-Peak angular excursion
        delta_pitch = max(pitch_window) - min(pitch_window);
        delta_roll  = max(roll_window) - min(roll_window);
        
        % If neither angle changed significantly, cancel the alarm!
        if delta_pitch < POSTURE_SHIFT_THRESHOLD && delta_roll < POSTURE_SHIFT_THRESHOLD
            probs(i, 2) = 0.0; % Absolute Zero Fall
            probs(i, 1) = 1.0; % Reassign to ADL
        end
    end
end

% 3. Generate the Final Confusion Matrix
% Convert logicals (true/false) into numbers (1/0)
predicted_numeric = double(probs(:,2) >= 0.5); 

% REBUILD CATEGORICALS WITH PRESENTATION LABELS
% Use the categorical constructor: categorical(data, numeric_values, category_names)
% This maps 0 -> ADL, and 1 -> Fall for the predictions
final_predictions = categorical(predicted_numeric, [0, 1], {'ADL (Negative)', 'Fall (Positive)'});

% Do the same for Ground Truth. double() converts existing categories to indices (1 and 2)
yval_indices = double(YVal); 
YVal_Chart = categorical(yval_indices, [1, 2], {'ADL (Negative)', 'Fall (Positive)'});

% Create the figure
figure('Name', 'Demo Day Final Confusion Matrix', 'Position', [100, 100, 800, 550]);
cm = confusionchart(YVal_Chart, final_predictions);

% ==========================================
% DEMO DAY VISUAL CUSTOMIZATIONS
% ==========================================

% 1. Plain English Titles and Axes
cm.Title = 'Hybrid AI + Kinematics Fall Detection';
cm.XLabel = 'What the System Predicted';
cm.YLabel = 'What Actually Happened (Ground Truth)';

% 2. Add a subtitle to act as a "legend" for the audience explaining the quadrants
% cm.Subtitle = ['Top-Left: True Negative (Correctly ignored)  |  ', ...
%                'Bottom-Right: True Positive (Caught the Fall)'];

% 3. Keep the professional summaries
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% 4. Bump up the font size for the presentation screen
cm.FontSize = 14;

% ==========================================
% EXTRACT Z-SCORE STATS FOR ANDROID JAVA
% ==========================================
disp('Calculating Normalization Statistics for Android...');

% Initialize empty arrays for the 6 channels
mat_mean = zeros(1, 6);
mat_std  = zeros(1, 6);

% Loop through the 6 channels in XTrain
% XTrain shape is [256 (Time), 1, 6 (Channels), N (Samples)]
for c = 1:6
    % Extract every single data point for this specific channel
    channelData = XTrain(:, 1, c, :);
    
    % Flatten it into a 1D column
    channelData = channelData(:); 
    
    % Calculate Mean and Standard Deviation
    mat_mean(c) = mean(channelData);
    mat_std(c)  = std(channelData);
end

% Print the results formatted perfectly for Java!
fprintf('\n\n========== COPY THIS INTO ANDROID STUDIO ==========\n');
fprintf('final float[] MAT_MEAN = {%.4ff, %.4ff, %.4ff, %.4ff, %.4ff, %.4ff};\n', mat_mean);
fprintf('final float[] MAT_STD  = {%.4ff, %.4ff, %.4ff, %.4ff, %.4ff, %.4ff};\n', mat_std);
fprintf('===================================================\n\n');