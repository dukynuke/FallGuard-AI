clear; clc; close all;
rng(0);   % Reproducibility

%% =========================================================================
% 1) Point to SAFE dataset folder
% =========================================================================
dataFolder = 'FallGuardAudio';

if ~isfolder(dataFolder)
    error('Dataset folder "%s" was not found.', dataFolder);
end

ads = audioDatastore(dataFolder, ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.wav');

numFiles = numel(ads.Files);
fprintf('Total audio files found: %d\n', numFiles);

%% =========================================================================
% 2) Parse SAFE filenames and assign labels
%    SAFE format: AA-BBB-CC-DDD-FF.wav
%    AA = fold, FF = class (01 Fall, 02 NonFall)
% =========================================================================
[~, namesOnly, ~] = cellfun(@fileparts, ads.Files, 'UniformOutput', false);
namesOnly = string(namesOnly);

foldIDs  = strings(numFiles,1);
classIDs = strings(numFiles,1);

for i = 1:numFiles
    parts = split(namesOnly(i), '-');
    
    if numel(parts) ~= 5
        error('Unexpected filename format: %s', namesOnly(i));
    end
    
    foldIDs(i)  = parts(1);   % AA
    classIDs(i) = parts(5);   % FF
end

% Map class IDs to categorical labels
ads.Labels = categorical(classIDs, ["01","02"], ["Fall","NonFall"]);

if any(isundefined(ads.Labels))
    badIdx = find(isundefined(ads.Labels), 1, 'first');
    error('Undefined label detected in file: %s', ads.Files{badIdx});
end

fprintf('\nOverall dataset label counts:\n');
disp(countEachLabel(ads));

%% =========================================================================
% 3) Strict split by fold to avoid leakage
%    Example:
%    Train = folds 01-07
%    Val   = fold 08
%    Test  = folds 09-10
% =========================================================================
trainFolds = compose("%02d", 1:7);
valFolds   = "08";
testFolds  = ["09","10"];

isTrain = ismember(foldIDs, trainFolds);
isVal   = ismember(foldIDs, valFolds);
isTest  = ismember(foldIDs, testFolds);

if any((isTrain & isVal) | (isTrain & isTest) | (isVal & isTest))
    error('Fold split overlap detected.');
end

if ~all(isTrain | isVal | isTest)
    warning('Some files are not assigned to train/val/test.');
end

adsTrain = subset(ads, find(isTrain));
adsVal   = subset(ads, find(isVal));
adsTest  = subset(ads, find(isTest));

fprintf('\nSplit summary:\n');
fprintf('Training samples   : %d\n', numel(adsTrain.Files));
fprintf('Validation samples : %d\n', numel(adsVal.Files));
fprintf('Testing samples    : %d\n', numel(adsTest.Files));

fprintf('\nTraining label counts:\n');
disp(countEachLabel(adsTrain));

fprintf('Validation label counts:\n');
disp(countEachLabel(adsVal));

fprintf('Testing label counts:\n');
disp(countEachLabel(adsTest));

%% =========================================================================
% 4) Mel-spectrogram feature extractor
% =========================================================================
fs = 48000;   % SAFE dataset sampling rate

afe = audioFeatureExtractor( ...
    'SampleRate', fs, ...
    'Window', hann(round(0.03*fs), 'periodic'), ... % 30 ms
    'OverlapLength', round(0.015*fs), ...           % 15 ms
    'melSpectrum', true);

% Optional: reduce/standardize mel bands if you want
% setExtractorParams(afe, 'melSpectrum', 'NumBands', 64);

%% =========================================================================
% 5) Transform audio into CNN-ready mel-spectrogram tensors
% =========================================================================
tdsTrain = transform(adsTrain, @(data, info) formatForCNN(data, info, afe), 'IncludeInfo', true);
tdsVal   = transform(adsVal,   @(data, info) formatForCNN(data, info, afe), 'IncludeInfo', true);
tdsTest  = transform(adsTest,  @(data, info) formatForCNN(data, info, afe), 'IncludeInfo', true);

% Read one example to get CNN input size
sample = read(tdsTrain);
sampleX = sample{1};

imageSize = [size(sampleX,1), size(sampleX,2), 1];

reset(tdsTrain);

fprintf('\nCNN input size: [%d %d %d]\n', imageSize(1), imageSize(2), imageSize(3));
%% =========================================================================
% 6) Define CNN architecture
% =========================================================================
layers = [
    imageInputLayer(imageSize, 'Normalization', 'zerocenter', 'Name', 'input')

    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    fullyConnectedLayer(64, 'Name', 'fc1')
    dropoutLayer(0.5, 'Name', 'drop1')

    fullyConnectedLayer(2, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classOutput')
];

%% =========================================================================
% 7) Training options
% =========================================================================
miniBatchSize = 64;
validationFrequency = max(1, floor(numel(adsTrain.Files) / miniBatchSize));

options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 1e-3, ...
    'L2Regularization', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', tdsVal, ...
    'ValidationFrequency', validationFrequency, ...
    'ValidationPatience', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% =========================================================================
% 8) Train network
% =========================================================================
disp('Training Audio CNN...');
[net, trainInfo] = trainNetwork(tdsTrain, layers, options);

%% =========================================================================
% 9) Evaluate on held-out test set
% =========================================================================
reset(tdsTest);
YPred = classify(net, tdsTest);
YTest = adsTest.Labels;

accuracy = mean(YPred == YTest);
fprintf('\nHeld-out Test Accuracy: %.2f%%\n', accuracy * 100);

% =========================================================================
% STANDALONE DEMO DAY CONFUSION MATRIX
% =========================================================================

% =========================================================================
% STANDALONE DEMO DAY CONFUSION MATRIX (CORE 2x2 ONLY)
% =========================================================================

% 1. Define the plain-English names
presentationLabels = {'Non-Fall (Negative)', 'Fall (Positive)'};
originalLabels = {'NonFall', 'Fall'};

% 2. Rename the categories for the chart
YTest_Chart = renamecats(YTest, originalLabels, presentationLabels);
YPred_Chart = renamecats(YPred, originalLabels, presentationLabels);

% 3. Force the order so 'Negative' is top-left and 'Positive' is bottom-right
YTest_Chart = reordercats(YTest_Chart, presentationLabels);
YPred_Chart = reordercats(YPred_Chart, presentationLabels);

% 4. Create the Figure and Chart (Slightly smaller width since summaries are gone)
figure('Name', 'Demo Day Audio Confusion Matrix', 'Position', [150, 150, 650, 500]);
cm = confusionchart(YTest_Chart, YPred_Chart);

% 5. Apply Plain English Titles (Using the cell array trick for the subtitle)
cm.Title = {
    'Audio CNN Fall Detection', ... % Main Title
    'Top-Left: True Negative (Ignored background noise)  |  Bottom-Right: True Positive (Heard the Fall)' % Subtitle
};

% 6. Clean up the axes and apply Demo Day formatting
cm.XLabel = 'What the AI Heard (Prediction)';
cm.YLabel = 'What Actually Happened (Ground Truth)';

% --- REMOVE THE PERCENTAGE SUMMARIES ---
cm.RowSummary = 'off';
cm.ColumnSummary = 'off';

cm.FontSize = 14;
%% =========================================================================
% 11) Optional: save trained model
% =========================================================================
save('SAFE_AudioCNN_Model.mat', 'net', 'trainInfo', ...
    'trainFolds', 'valFolds', 'testFolds', 'accuracy');

disp('Done. Model saved to SAFE_AudioCNN_Model.mat');

%% =========================================================================
% Helper function
% =========================================================================
function [dataOut, info] = formatForCNN(data, info, afe)
    % Ensure mono
    if size(data,2) > 1
        data = mean(data, 2);
    end
    
    % Ensure column vector
    data = data(:);

    % Enforce exact 3-second length at 48 kHz
    targetSamples = 3 * 48000;   % 144000
    
    currentSamples = size(data,1);

    if currentSamples < targetSamples
        data = [data; zeros(targetSamples - currentSamples, 1)];
    elseif currentSamples > targetSamples
        data = data(1:targetSamples);
    end

    % Extract mel-spectrogram
    features = extract(afe, data);

    % Log compression
    features = log10(features + eps);

    % CNN expects H x W x C
    features = reshape(features, [size(features,1), size(features,2), 1]);

    % Output predictor/label pair
    dataOut = {features, info.Label};
end