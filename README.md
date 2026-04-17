<p align="center">
  <img src="assets/banner.png" alt="FallGuard banner" width="100%">
</p>

<h1 align="center">FallGuard</h1>

<p align="center">
  Hybrid smartphone fall detection for AI competition demos using motion sensors, posture verification, and audio-based CNN classification.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/MATLAB-R2023a%2B-orange?style=for-the-badge" alt="MATLAB badge">
  <img src="https://img.shields.io/badge/Domain-Fall%20Detection-blue?style=for-the-badge" alt="Fall detection badge">
  <img src="https://img.shields.io/badge/Modalities-IMU%20%2B%20Audio-green?style=for-the-badge" alt="Multimodal badge">
  <img src="https://img.shields.io/badge/Competition-Demo%20Ready-purple?style=for-the-badge" alt="Competition badge">
</p>

## Overview

**FallGuard** is our AI competition project for automatic fall detection using a smartphone-centered sensing pipeline.

The repository contains two main learning paths:

- a **motion-based fall detection pipeline** built from motion signals such as acceleration, gyroscope, and orientation
- an **audio-based fall detection pipeline** that converts audio clips into mel-spectrograms and classifies **Fall** vs **NonFall**

The project is designed to be **demo-friendly**: it combines a neural network with a simple posture-based verification stage to reduce false alarms in practical smartphone scenarios.

---

## Why this project matters

Falls can become dangerous very quickly, especially when the person cannot call for help immediately. FallGuard is built around three practical ideas:

1. **Use the smartphone people already have**  
   No extra wearable hardware is required for the core sensing pipeline.

2. **Combine AI with physical reasoning**  
   The system does not rely only on a neural network. The motion pipeline also checks posture change to suppress suspicious false positives.

3. **Support multiple sensing channels**  
   Motion and audio can be explored as complementary paths for robust fall detection.

---

## Repository structure

```text
FallGuard/
├── audio/
│   └── FallGuardAudioCode.m
├── motion/
│   ├── FallGuardCode.m
│   └── MobiFall_Builder.m
├── assets/
│   ├── banner.png
│   └── screenshots/
│       ├── hero-placeholder.png
│       ├── motion-confusion-placeholder.png
│       ├── audio-confusion-placeholder.png
│       └── training-placeholder.png
├── docs/
│   └── UPLOAD_GUIDE.md
├── README.md
└── .gitignore
```

---

## Pipeline summary

```mermaid
flowchart LR
    A[MobiFall raw motion files] --> B[MobiFall_Builder.m]
    B --> C[MobiFall_Ready.mat]
    C --> D[FallGuardCode.m]
    D --> E[Motion CNN]
    D --> F[Posture verification using pitch and roll]
    E --> G[Final motion decision]

    H[SAFE audio dataset] --> I[FallGuardAudioCode.m]
    I --> J[Mel-spectrogram extraction]
    J --> K[Audio CNN]
    K --> L[Fall / NonFall prediction]
```

---

## What each file does

### `motion/MobiFall_Builder.m`
Builds the motion dataset for training.

Main responsibilities:
- scans the MobiFall-style dataset directory
- groups accelerometer, gyroscope, and orientation files by trial
- resamples all sensors to a **fixed uniform timeline**
- extracts **256-sample windows**
- stores:
  - `XCNN` for CNN training
  - `XRaw` for raw posture-based heuristics
  - `Y_Categorical` labels
- saves everything into `MobiFall_Ready.mat`

### `motion/FallGuardCode.m`
Main motion AI training and validation script.

Main responsibilities:
- loads `MobiFall_Ready.mat`
- creates an **80/20 train-validation split**
- trains the motion network
- applies a **posture verification heuristic** based on pitch and roll
- plots the final confusion matrix
- computes normalization statistics for Android deployment

### `audio/FallGuardAudioCode.m`
Main audio training and evaluation script.

Main responsibilities:
- loads a SAFE-style audio dataset folder
- parses filenames into labels and folds
- performs a **strict fold split** for train / validation / test
- extracts **mel-spectrogram** features
- trains a CNN for **Fall vs NonFall**
- evaluates on a held-out test set
- saves the trained model to `SAFE_AudioCNN_Model.mat`

---

## Demo visuals

### Motion pipeline result
<p align="center">
  <img src="assets/screenshots/motion-confusion-matrix.png" alt="Motion AI confusion matrix" width="92%">
</p>

### Audio pipeline result
<p align="center">
  <img src="assets/screenshots/audio-confusion-matrix.png" alt="Audio AI confusion matrix" width="92%">
</p>

### Training screenshots
<p align="center">
  <img src="assets/screenshots/motionAI_trainingCurve.png" alt="Motion AI Training curve with MobiFall v2.0 Dataset" width="92%">
</p>

<p align="center">
  <img src="assets/screenshots/audioAI_trainingCurve.png" alt="Motion AI Training curve with MobiFall v2.0 Dataset" width="92%">
</p>

---

## Quick start

### 1) Motion pipeline

Prepare your MobiFall-style dataset and place the files where the builder can access them.

Run:

```matlab
cd motion
MobiFall_Builder
FallGuardCode
```

Expected output:
- `MobiFall_Ready.mat`
- trained motion model in memory
- confusion matrix for the validation split
- printed normalization constants for Android integration

### 2) Audio pipeline

Prepare the SAFE audio dataset in a folder named:

```text
FallGuardAudio/
```

Then run:

```matlab
cd audio
FallGuardAudioCode
```

Expected output:
- audio CNN training progress
- held-out test accuracy
- confusion matrix
- `SAFE_AudioCNN_Model.mat`

---

## Neural network setup

### Motion AI 
The motion branch is trained on fixed-length sensor windows generated from the MobiFall pipeline.  
Its training configuration is:

#### Training configuration
- **Optimizer:** Adam
- **Loss function:** Cross-entropy
- **Epochs:** 36
- **Mini-batch size:** 32
- **Initial learning rate:** 1e-3
- **Validation split:** 80/20 holdout
- **Extra logic:** post-processing with **pitch/roll posture verification** to suppress suspicious false positives

### Audio AI 
The audio branch converts 3-second audio clips into **mel-spectrograms** and feeds them into a compact CNN.

#### Audio feature front end
- **Sample rate:** 48 kHz
- **Window size:** 30 ms Hann window
- **Overlap:** 15 ms
- **Representation:** mel-spectrogram
- **Input normalization:** zerocenter

#### Audio CNN architecture
- Image input layer
- Conv2D(16) + BatchNorm + ReLU + MaxPool
- Conv2D(32) + BatchNorm + ReLU + MaxPool
- Conv2D(64) + BatchNorm + ReLU
- Fully connected layer (64)
- Dropout (0.5)
- Fully connected layer (2 classes)
- Softmax + classification layer

#### Training configuration
- **Optimizer:** Adam
- **Loss function:** classification loss
- **Epochs:** 15
- **Mini-batch size:** 64
- **Initial learning rate:** 1e-3
- **L2 regularization:** 1e-4
- **Validation patience:** 5
- **Execution environment:** auto

---

## Datasets

This repository is structured around two data sources:

### Motion dataset
The motion pipeline expects a **MobiFall-style** directory of text files containing:
- accelerometer
- gyroscope
- orientation data

### Audio dataset
The audio pipeline expects a SAFE-style audio folder with filenames following the pattern:

```text
AA-BBB-CC-DDD-FF.wav
```

where:
- `AA` = fold ID
- `FF` = class ID
- `01` = Fall
- `02` = NonFall

---

## Method highlights

### Motion branch
- fixed-rate sensor synchronization
- window extraction for fall and ADL samples
- CNN-based classification
- auxiliary pitch/roll posture verification
- Android-friendly normalization export

### Audio branch
- mel-spectrogram front-end
- fold-based split to reduce leakage
- compact CNN for binary audio classification
- test-time confusion matrix for clear demo presentation

---

## Android / deployment note

The motion training script prints `MAT_MEAN` and `MAT_STD` arrays formatted for direct use in Android Studio.  
That makes it easier to keep preprocessing consistent between MATLAB training and mobile inference.

---



## Acknowledgment

Built as part of an AI competition project focused on practical, accessible fall detection using everyday devices for 2026 Applied AI Challenge in Iowa State University.
