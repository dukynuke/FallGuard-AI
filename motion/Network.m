layers = [
    % Treat the 256-length window as a 256x1 "image" with 6 channels
    imageInputLayer([256 1 6], 'Normalization', 'none', 'Name', 'input')
    
    % Block 1 (Filter size 7x1, padding 3 on top/bottom, 0 on left/right)
    convolution2dLayer([7 1], 32, 'Padding', [3 3 0 0], 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool1')
    
    % Block 2
    convolution2dLayer([5 1], 64, 'Padding', [2 2 0 0], 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool2')
    
    % Block 3
    convolution2dLayer([5 1], 128, 'Padding', [2 2 0 0], 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    
    % Block 4
    convolution2dLayer([3 1], 128, 'Padding', [1 1 0 0], 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    
    % Global Average Pooling & Dense Layers
    globalAveragePooling2dLayer('Name', 'gap')
    dropoutLayer(0.25, 'Name', 'drop1')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc')
    dropoutLayer(0.25, 'Name', 'drop2')
    
    % MATLAB requires 2 output nodes (ADL vs Fall)
    fullyConnectedLayer(2, 'Name', 'fc2')
    
    % Keep Softmax to convert outputs to probabilities
    softmaxLayer('Name', 'softmax')
];