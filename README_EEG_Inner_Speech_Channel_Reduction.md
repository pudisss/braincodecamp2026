# EEG Inner Speech Channel Reduction Research

## Purpose of This README

This file is intended to give Codex enough context to understand the current research project when used together with the latest `.ipynb` notebook.

The notebook contains the latest implementation. This README explains:

- the research question
- the dataset and where it comes from
- how the EEG data are structured
- which parts of the dataset are currently being used
- preprocessing assumptions
- label/event mappings
- the baseline ML and DL approaches discussed
- the channel-reduction experiment design
- reproducibility rules
- known implementation issues already discovered
- what should and should not be changed without a good reason
- possible future experiments after the main channel-reduction experiment is complete

The project is still **in progress**. The main experiment has not been finalized yet.

---

# 1. Main Research Question

The project studies **EEG channel reduction for inner-speech classification**.

The main question is:

> How much classification performance can be retained when the number of EEG channels is reduced from a high-density 128-channel setup?

The planned channel counts are:

```text
128 -> 64 -> 32 -> 16
```

The goal is not initially to find the "best possible" electrodes.

The primary experiment is intended to measure the effect of reducing EEG spatial density while keeping the selected electrodes reasonably distributed across the scalp.

The central idea is:

```text
Same dataset
Same labels
Same preprocessing
Same train/validation/test protocol
Same model settings
Same selected channel subsets for every subject
Same selected channel subsets for every model

Only the number of EEG channels changes:
128 -> 64 -> 32 -> 16
```

This is important because the project wants **channel count** to be the main controlled experimental variable.

---

# 2. Broader Experiment Plan

The planned comparison is across approximately three model families:

```text
1. Classical Machine Learning
2. Deep Learning trained from scratch
3. Pretrained EEG / neuroscience model
```

Each model family should eventually be evaluated under the same channel configurations:

| Model family | 128 | 64 | 32 | 16 |
|---|---:|---:|---:|---:|
| Classical ML | yes | yes | yes | yes |
| Scratch DL | yes | yes | yes | yes |
| Pretrained EEG model | planned | planned | planned | planned |

The current focus is:

1. understand and load the dataset correctly
2. establish a model baseline
3. define a reproducible channel-reduction rule
4. run the channel-reduction experiment
5. only after the main experiment is stable, consider further signal-processing or feature-engineering experiments

Future preprocessing or representation experiments such as FFT/DFT, STFT, wavelets, alternative PSD approaches, etc. are intentionally secondary for now.

---

# 3. Dataset

## Dataset name

The data come from the **Thinking Out Loud / Inner Speech Recognition EEG dataset** by Nicolás Nieto and collaborators.

Canonical/original dataset:

```text
OpenNeuro ds003626
```

Associated publication:

```text
Nieto, N., Peterson, V., Rufiner, H. L., et al.
"Thinking out loud, an open-access EEG-based BCI dataset for inner speech recognition."
Scientific Data, 2022.
```

## Kaggle copy

The project is currently being developed using the Kaggle dataset:

```text
https://www.kaggle.com/datasets/truthisneverlinear/inner-speech-recognition
```

Important:

- the Kaggle version is **not the official/original publication source**
- it appears to be a **third-party mirror/repackaging**
- the underlying dataset itself is legitimate
- for research citations, cite the original paper/OpenNeuro dataset, not the Kaggle uploader
- Kaggle is being used because it is more convenient for storage and compute

Suggested wording for a research methodology section:

> Data were obtained from the Thinking Out Loud Inner Speech Dataset (OpenNeuro ds003626); a Kaggle-hosted mirror was used for computational convenience.

---

# 4. Participants and Recording Setup

The original dataset contains:

```text
10 subjects
3 sessions per subject
128 EEG channels
8 additional external EOG/EMG channels
original recording sampling rate: 1024 Hz
24-bit recording
```

The experimental classes are directional words:

```text
0 = Up
1 = Down
2 = Right
3 = Left
```

The actual spoken/internal words in the experiment were Spanish directional words:

```text
Arriba
Abajo
Derecha
Izquierda
```

The project treats them as the four semantic classes:

```text
Up / Down / Right / Left
```

There are three task conditions:

```text
0 = Pronounced Speech
1 = Inner Speech
2 = Visualized condition
```

The project currently focuses on:

```text
condition == 1
```

which is **Inner Speech**.

---

# 5. Important Trial Counts

The dataset contains approximately:

```text
5640 total trials
```

but this does NOT mean there are 5640 inner-speech trials.

The approximate condition counts discussed are:

```text
Pronounced Speech: 1128
Inner Speech:       2236
Visualized:         2276
```

The main project currently uses:

```text
2236 inner-speech trials
```

The four inner-speech classes are balanced at approximately:

```text
559 trials per class
```

Chance accuracy for the 4-class task is:

```text
25%
```

---

# 6. Trial vs Epoch

Important terminology:

## Trial

A **trial** is one repetition of the experimental procedure.

Example:

```text
participant sees cue
-> internally says/thinks "Left"
-> rests
```

That entire repeated experimental event is one trial.

## Epoch

An **epoch** is a selected time segment cut from a continuous EEG signal.

Conceptually:

```text
continuous EEG
-------------------------------> time

             event occurs
                  |
                  v

        |----------------|
        selected EEG window
             = epoch
```

Therefore:

```text
Trial = experimental concept
Epoch = signal/data-processing concept
```

In this dataset, the derivative data are already epoched, and approximately one epoch corresponds to one trial, so the words can feel interchangeable in practice.

---

# 7. Epoching / Trigger Context

In raw EEG processing, epochs are normally created from a long continuous signal using events/triggers.

A STIM or trigger channel may contain markers such as:

```text
stimulus onset
button press
task start
task end
```

Typical MNE flow:

```text
Raw EEG
-> trigger/event detection
-> select time interval around event
-> create MNE Epochs
```

Relevant MNE concepts include:

```python
mne.find_events(...)
mne.Epochs(...)
```

`Raw.pick(...)` only selects channels; it does not itself perform epoch creation.

This project currently uses the authors' already-preprocessed derivative epochs, so raw epoch creation is not required for the main experiment.

---

# 8. Raw vs Derivative Data

The dataset contains raw recordings and derivative/preprocessed data.

## Raw

Raw EEG is approximately:

```text
continuous .bdf recordings
1024 Hz
128 EEG + 8 EOG/EMG
```

## Derivatives

The `derivatives/` folder contains data derived from the raw recordings.

For this dataset, the relevant EEG derivative files are the authors' processed EEG epochs.

Typical flow:

```text
Raw EEG
-> re-referencing / preprocessing
-> filtering
-> notch filtering
-> downsampling
-> ICA/artifact handling
-> event correction / epoching
-> derivative .fif files
```

The project currently uses the derivative `.fif` data.

This is intentional.

There is no need to repeat ICA/filtering/artifact removal unless a later research question specifically requires a different preprocessing pipeline.

---

# 9. Important Sampling-Rate Note

The original recordings were collected at:

```text
1024 Hz
```

The processed derivatives are around:

```text
~256 Hz
```

There was discussion of a small inconsistency in dataset documentation around 254 vs 256 Hz.

Therefore:

**Never hard-code the sampling rate if it can be read from the actual MNE file.**

Always use:

```python
fs = epochs.info["sfreq"]
```

---

# 10. Derivative EEG File Structure

The relevant EEG files are MNE Epochs `.fif` files.

Typical filename:

```text
sub-01_ses-01_eeg-epo.fif
```

Load with:

```python
epochs = mne.read_epochs(
    eeg_path,
    preload=True,
    verbose=False
)
```

Then:

```python
X = epochs.get_data()
```

The basic tensor format is:

```text
X[trial, channel, time]
```

Typical session-level shape is approximately:

```text
(number_of_trials, 128, 1154)
```

Example:

```python
X[5]
```

means:

```text
trial 5
all 128 EEG channels
all time points
```

and:

```python
X[5, 20]
```

means:

```text
trial 5
channel 20
all time points
```

---

# 11. Event File Format — VERY IMPORTANT

A major issue was already discovered here.

The event file is named something like:

```text
sub-01_ses-01_events.dat
```

Do **NOT** use:

```python
np.loadtxt(...)
```

on this file.

That produced incorrect huge integer values such as:

```text
8750614427111457408
8461470288145507118
...
```

The `.dat` file is actually stored as a **Pandas pickle** in this dataset.

Correct loading:

```python
import pandas as pd
import numpy as np

events = pd.read_pickle(event_path)
events = np.asarray(events)
```

This is a critical implementation detail that should be preserved.

---

# 12. Event Matrix Structure

The event matrix has four columns.

The working interpretation is:

```text
events[:, 0] = event sample / timestamp information
events[:, 1] = class
events[:, 2] = condition
events[:, 3] = session information
```

Class mapping:

```python
CLASS_NAMES = {
    0: "Up",
    1: "Down",
    2: "Right",
    3: "Left"
}
```

Condition mapping:

```python
CONDITION_NAMES = {
    0: "Pronounced",
    1: "Inner Speech",
    2: "Visualized"
}
```

Always verify alignment:

```python
assert len(X) == len(events)
```

One EEG epoch/trial should correspond to one row of the event table.

---

# 13. Inner-Speech Filtering

The main experiment uses only:

```python
inner_mask = events[:, 2] == 1
```

Then:

```python
X_inner = X[inner_mask]
y_inner = events[inner_mask, 1].astype(np.int64)
```

After combining all subjects/sessions, the expected total number of inner-speech trials is approximately:

```text
2236
```

---

# 14. Trial Timeline and Current Time Window

The approximate experimental trial timeline discussed is:

```text
0.0 - 0.5 s : concentration
0.5 - 1.0 s : visual directional cue
1.0 - 3.5 s : action / inner speech
3.5 - 4.5 s : relaxation
```

There is an important confound:

If the full epoch is used, the model could learn activity associated with seeing the visual directional cue instead of only the internal word.

Therefore the current recommended main interval is:

```text
1.0 s <= time < 3.5 s
```

Code:

```python
times = epochs.times

action_mask = (
    (times >= 1.0) &
    (times < 3.5)
)

X = X[:, :, action_mask]
```

At approximately 256 Hz, the action period should contain about:

```text
~640 time samples
```

Therefore the combined model-ready tensor is expected to look approximately like:

```text
X.shape = (2236, 128, ~640)
y.shape = (2236,)
subjects.shape = (2236,)
```

Do not hard-code `640`; derive the time mask from `epochs.times`.

---

# 15. Combined Dataset Structure

The notebook is intended to create:

```python
X
y
subjects
```

with:

```text
X:
[trial, channel, time]

y:
[class ID for each trial]

subjects:
[subject ID for each trial]
```

Expected:

```text
X        -> (~2236, 128, ~640)
y        -> (~2236,)
subjects -> (~2236,)
```

---

# 16. Loading All Subjects/Sessions

The intended structure is roughly:

```python
for eeg_path in eeg_files:
    epochs = mne.read_epochs(...)
    X = epochs.get_data()

    events = pd.read_pickle(event_path)
    events = np.asarray(events)

    assert len(X) == len(events)

    # select inner speech
    inner_mask = events[:, 2] == 1
    X = X[inner_mask]
    y = events[inner_mask, 1]

    # select action interval
    action_mask = (
        (epochs.times >= 1.0) &
        (epochs.times < 3.5)
    )
    X = X[:, :, action_mask]

    # save X, y, subject ID
```

Then:

```python
X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
subjects = np.concatenate(all_subjects, axis=0)
```

---

# 17. Dataset Sanity Checks

Before model training, always check:

```python
print(X.shape)
print(y.shape)
print(subjects.shape)

print(np.isnan(X).sum())
print(np.isinf(X).sum())
```

Expected:

```text
NaN: 0
Inf: 0
```

Class counts:

```python
for label in range(4):
    print(label, np.sum(y == label))
```

Subject counts:

```python
for subject in np.unique(subjects):
    print(subject, np.sum(subjects == subject))
```

Do not trust a model result until these basic checks pass.

---

# 18. Interactive EEG Visualization

Plotly was preferred for EEG exploration because the user wants interactive plots.

Desired features:

```text
zoom
pan
hover
toggle channels
inspect exact time/amplitude
```

Recommended exploration hierarchy:

```text
subject
-> session
-> trial
-> channel(s)
```

Do not initially draw all subjects and all 128 channels in one Plotly figure because it becomes cluttered and slow.

Useful conversion:

```python
eeg_microvolts = eeg_volts * 1e6
```

MNE provides:

```python
epochs.times
epochs.ch_names
epochs.info["sfreq"]
```

---

# 19. Machine Learning Baseline

The intended classical ML baseline is:

```text
Preprocessed EEG
-> frequency-domain feature extraction
-> bandpower features
-> standardization
-> SVM
```

Do not initially flatten the raw:

```text
128 x ~640
```

signal directly for classical ML, because that creates roughly 80k raw features per trial.

Instead use EEG bandpower features.

Bands discussed:

```text
Delta: 1-4 Hz
Theta: 4-8 Hz
Alpha: 8-13 Hz
Beta: 13-30 Hz
Gamma: 30-45 Hz
```

Recommended feature extraction:

```python
from scipy.signal import welch
```

Conceptual feature shape with all 128 channels:

```text
128 channels x 5 bands = 640 features per trial
```

A representative feature extractor:

```python
def extract_bandpower_features(X, fs):
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 45)
    }

    all_features = []

    for trial in X:
        freqs, psd = welch(
            trial,
            fs=fs,
            nperseg=min(256, trial.shape[-1]),
            axis=-1
        )

        trial_features = []

        for low, high in bands.values():
            mask = (freqs >= low) & (freqs < high)

            band_power = psd[:, mask].mean(axis=1)

            band_power = np.log(
                band_power + 1e-12
            )

            trial_features.append(band_power)

        trial_features = np.array(
            trial_features
        ).flatten()

        all_features.append(trial_features)

    return np.array(all_features)
```

Then:

```text
StandardScaler
-> SVC / SVM
```

Standardization must be fit on training data only:

```python
scaler.fit(X_train_features)
scaler.transform(X_val_features)
scaler.transform(X_test_features)
```

Never fit the scaler using validation/test data.

---

# 20. Deep Learning Baseline

For DL, the current idea is to use the raw preprocessed time-domain EEG directly:

```text
[batch, channels, time]
```

instead of feeding manually extracted bandpower features.

The discussed baseline is an EEGNet-style compact CNN.

General idea:

```text
raw preprocessed EEG
-> temporal convolution
-> spatial convolution across electrodes
-> separable temporal convolution
-> pooling
-> classifier
-> 4 classes
```

Representative input:

```text
[B, 128, ~640]
```

internally reshaped to:

```text
[B, 1, 128, ~640]
```

The baseline class discussed was approximately:

```python
class EEGNet(nn.Module):

    def __init__(
        self,
        n_channels,
        n_classes=4,
        dropout=0.5
    ):
        super().__init__()

        F1 = 8
        D = 2
        F2 = F1 * D

        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                F1,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False
            ),
            nn.BatchNorm2d(F1)
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                F1,
                F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        self.separable = nn.Sequential(
            nn.Conv2d(
                F2,
                F2,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F2,
                bias=False
            ),
            nn.Conv2d(
                F2,
                F2,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(
            F2,
            n_classes
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)
```

The notebook may contain a newer version. The notebook should take priority over this README if implementation details differ.

---

# 21. DL Normalization

The discussed DL normalization is per-channel normalization based only on the training set.

Example:

```python
mean = X_train.mean(
    axis=(0, 2),
    keepdims=True
)

std = X_train.std(
    axis=(0, 2),
    keepdims=True
)

std = np.maximum(std, 1e-8)

X_train_dl = (X_train - mean) / std
X_val_dl   = (X_val - mean) / std
X_test_dl  = (X_test - mean) / std
```

Again:

**Never compute normalization statistics using validation/test data.**

---

# 22. Train/Validation/Test Splitting

EEG is strongly subject-dependent.

Avoid a naive random trial-level split across the entire dataset because this can place EEG from the same participant in both training and test sets.

For development, a simple subject split was discussed:

```python
train_subjects = [1, 2, 3, 4, 5, 6, 7]
val_subjects   = [8]
test_subjects  = [9, 10]
```

with:

```python
train_mask = np.isin(subjects, train_subjects)
val_mask   = np.isin(subjects, val_subjects)
test_mask  = np.isin(subjects, test_subjects)
```

For the final scientific evaluation, **Leave-One-Subject-Out (LOSO)** is preferred.

LOSO concept:

```text
Fold 1:
train = subjects 2-10
test  = subject 1

Fold 2:
train = subjects 1,3-10
test  = subject 2

...

Fold 10:
train = subjects 1-9
test  = subject 10
```

The project is interested in how well the model generalizes to an unseen participant.

---

# 23. Evaluation Metrics

Metrics discussed/planned:

```text
Accuracy
Macro F1
Balanced Accuracy
Confusion Matrix
```

Potential channel-retention metric:

```text
Retention(C) =
    performance using C channels
    ----------------------------
    performance using 128 channels
    x 100
```

Example:

```text
128 channels = 60% accuracy
32 channels  = 55% accuracy

Retention =
55 / 60 * 100
= 91.7%
```

This gives an intuitive measure of how much full-channel performance remains after channel reduction.

---

# 24. Training Loop / tqdm Note

A Kaggle/Jupyter issue occurred with:

```python
from tqdm.auto import tqdm
```

which produced:

```text
Error displaying widget: model not found
```

This is related to notebook widget support, not model training.

Using:

```python
from tqdm import tqdm
```

avoids the widget dependency.

However, nested tqdm bars created too many lines in Kaggle.

The preferred training display is therefore:

```text
ONE tqdm progress bar for epochs only
```

Do not wrap every training and validation DataLoader in an additional text tqdm unless specifically desired.

Conceptual structure:

```python
pbar = tqdm(range(EPOCHS), desc="Training EEGNet")

for epoch in pbar:
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc, ... = evaluate(...)

    pbar.set_postfix(
        train_acc=...,
        val_acc=...,
        best=...,
        loss=...
    )
```

---

# 25. Checkpointing

The discussed DL baseline saves the best model based on validation accuracy:

```python
best_val = 0

if val_acc > best_val:
    best_val = val_acc

    torch.save(
        model.state_dict(),
        "best_eegnet.pt"
    )
```

Then reload:

```python
model.load_state_dict(
    torch.load(
        "best_eegnet.pt",
        weights_only=True
    )
)
```

---

# 26. Channel Reduction — Main Research Design

This is currently the most important experimental design section.

The project wants:

```text
128 -> 64 -> 32 -> 16 channels
```

with the subsets chosen **before model training** and kept fixed.

The main reduction strategy should be:

> Nested, spatially distributed channel subsets that preserve broad scalp coverage and approximate left/right balance.

The goal is to study channel-count reduction, not to optimize electrode locations using classification accuracy.

---

# 27. Channel-Reduction Rules

## Rule 1 — Nested subsets

Required:

```text
16 subset of 32
32 subset of 64
64 subset of 128
```

Symbolically:

```text
16 ⊂ 32 ⊂ 64 ⊂ 128
```

This makes the interpretation cleaner.

---

## Rule 2 — Preserve spatial coverage

Each subset should try to retain coverage of:

```text
Frontal
Central
Temporal
Parietal
Occipital
```

and both:

```text
Left hemisphere
Right hemisphere
```

The subsets should not accidentally become concentrated in one part of the scalp.

---

## Rule 3 — Approximate bilateral balance

If possible, the reduced configurations should remain approximately balanced across:

```text
left
right
midline
```

This does not need to be perfectly symmetrical if the BioSemi geometry does not permit exact pairs.

---

## Rule 4 — Do not select channels using classification performance

For the **main experiment**, do not:

```text
train model
-> find highest-importance channels
-> keep those
```

That would turn the experiment into:

```text
channel optimization + channel reduction
```

rather than a clean channel-density reduction study.

Data-driven channel selection can be studied later as a separate experiment.

---

## Rule 5 — Same subsets for every subject and model

This is extremely important.

If the 32-channel configuration is defined once, exactly those same 32 electrode names must be used for:

```text
Subject 1
Subject 2
...
Subject 10
```

and:

```text
SVM
EEGNet
pretrained model
```

Otherwise the experiment changes both:

```text
channel count
+
channel location
```

and becomes harder to interpret fairly.

---

# 28. BioSemi 128 Montage and Channel Coordinates

The data use a BioSemi 128-channel layout.

The montage can be attached in MNE using:

```python
montage = mne.channels.make_standard_montage(
    "biosemi128"
)

epochs.set_montage(
    montage,
    match_case=False,
    on_missing="warn"
)
```

Then retrieve coordinates:

```python
ch_pos = (
    epochs
    .get_montage()
    .get_positions()["ch_pos"]
)
```

Create ordered arrays:

```python
channel_names = np.array(
    epochs.ch_names
)

positions = np.array([
    ch_pos[ch]
    for ch in channel_names
])
```

Expected:

```text
channel_names.shape = (128,)
positions.shape     = (128, 3)
```

---

# 29. Rough Scalp Regions

A rough coordinate-based region assignment was discussed.

The coordinate interpretation used:

```text
x = left/right
y = posterior/anterior
z = lower/upper
```

A rough region classifier used thresholds like:

```python
def get_region(x, y, z):

    if y > 0.045:
        region = "Frontal"

    elif y > 0.015:
        region = "Front-Central"

    elif y > -0.020:
        region = "Central"

    elif y > -0.050:
        region = "Parietal"

    else:
        region = "Occipital"

    if abs(x) > 0.060 and -0.045 < y < 0.045:
        region = "Temporal"

    return region
```

Hemisphere:

```python
def get_hemisphere(x, threshold=0.005):

    if x < -threshold:
        return "Left"

    elif x > threshold:
        return "Right"

    else:
        return "Midline"
```

These are **heuristic geographic regions**, not clinical anatomical labels.

If a future publication requires rigorous anatomical region definitions, verify and document a more formal mapping.

---

# 30. Farthest-Point Sampling for Channel Reduction

The current proposed channel selection method is **Farthest Point Sampling (FPS)** based on the 3D electrode positions.

The intuition:

```text
start with one electrode
-> choose the electrode furthest from selected set
-> repeat
```

This tends to spread retained electrodes across the scalp.

Representative implementation:

```python
def farthest_point_sampling(positions):

    n_channels = len(positions)

    center_target = np.array([
        0,
        0,
        positions[:, 2].max()
    ])

    start = np.argmin(
        np.linalg.norm(
            positions - center_target,
            axis=1
        )
    )

    selected = [start]

    remaining = set(
        range(n_channels)
    )

    remaining.remove(start)

    while remaining:

        best_channel = None
        best_distance = -1

        for candidate in remaining:

            distances = np.linalg.norm(
                positions[candidate]
                - positions[selected],
                axis=1
            )

            min_distance = distances.min()

            if min_distance > best_distance:
                best_distance = min_distance
                best_channel = candidate

        selected.append(best_channel)
        remaining.remove(best_channel)

    return np.array(selected)
```

Then:

```python
channel_order = farthest_point_sampling(
    positions
)
```

---

# 31. Channel Configurations

The nested configurations are then:

```python
channel_configs_idx = {
    128: channel_order[:128],
    64:  channel_order[:64],
    32:  channel_order[:32],
    16:  channel_order[:16]
}
```

Channel-name version:

```python
channel_configs = {
    n: channel_names[idx].tolist()
    for n, idx in channel_configs_idx.items()
}
```

Nestedness checks:

```python
assert set(channel_configs[16]).issubset(
    channel_configs[32]
)

assert set(channel_configs[32]).issubset(
    channel_configs[64]
)

assert set(channel_configs[64]).issubset(
    channel_configs[128]
)
```

---

# 32. Applying the Channel Configurations

Current simple approach:

```python
X_configs = {
    n: X[:, idx, :]
    for n, idx in channel_configs_idx.items()
}
```

Expected:

```text
X_configs[128] -> (trials, 128, time)
X_configs[64]  -> (trials, 64, time)
X_configs[32]  -> (trials, 32, time)
X_configs[16]  -> (trials, 16, time)
```

This lets the same exact configurations be reused across ML and DL.

---

# 33. Reproducibility

The project explicitly wants the channel configurations to be reproducible.

The configuration should be saved, not regenerated casually for every experiment.

Recommended:

```python
np.save(
    "channel_configs_idx.npy",
    channel_configs_idx,
    allow_pickle=True
)
```

and:

```python
np.save(
    "channel_configs_names.npy",
    channel_configs,
    allow_pickle=True
)
```

Load:

```python
channel_configs_idx = np.load(
    "channel_configs_idx.npy",
    allow_pickle=True
).item()
```

Saving both indices and names is preferred.

Why?

Indices such as:

```text
[0, 17, 43, 81, ...]
```

only remain meaningful if channel ordering never changes.

Channel names such as:

```text
["A1", "B12", "C7", ...]
```

provide an explicit record of the selected electrodes.

---

# 34. Very Important Reproducibility Rule

Once the final:

```text
128
64
32
16
```

channel subsets are accepted for the main experiment:

**do not regenerate or change them between model runs.**

Every experiment should reuse the exact saved channel configurations.

If Codex rewrites code, it should preferentially load the saved configuration files rather than silently recomputing a new selection.

---

# 35. Potential Improvement to FPS

The current FPS implementation provides broad spatial distribution but does not explicitly enforce perfect left/right symmetry or a fixed number of electrodes per named region.

Possible future refinement:

```text
spatial FPS
+
regional quotas
+
approximate bilateral symmetry
```

However:

**Do not change the main channel-selection rule casually after results have already been generated.**

Any major change to the reduction rule should be treated as a new experimental condition.

---

# 36. Future Secondary Experiment — Data-Driven Channel Selection

After the primary spatial-density experiment is complete, a possible secondary experiment is:

```text
Experiment A:
Spatially distributed fixed reduction

vs

Experiment B:
Data-driven / learned channel selection
```

Possible data-driven approaches:

```text
channel ablation importance
mutual information
feature importance
sparse approaches
model-based electrode ranking
```

If data-driven selection is used, it must be performed using **training data only** inside each train/test fold.

Never:

```text
all subjects
-> select best channels
-> perform LOSO
```

because the held-out subject would influence channel selection.

Correct:

```text
LOSO fold
-> training subjects only
-> select channels
-> train model
-> evaluate on unseen subject
```

---

# 37. Further Signal Processing — Deferred for Now

It is valid to perform further signal processing on the already-preprocessed EEG.

Examples:

```text
DFT / FFT
PSD
Welch PSD
STFT
Wavelets
additional feature extraction
```

Preprocessed data do not mean that no further transformation is possible.

The distinction is:

```text
preprocessing
!=
feature extraction
```

Example:

```text
Raw EEG
-> preprocessing
-> clean EEG
-> FFT / PSD / bandpower
-> ML model
```

However, the current project priority is:

```text
finish the channel-reduction experiment first
```

before introducing many additional preprocessing/feature-engineering variables.

---

# 38. Current Recommended Experiment Order

Recommended sequence:

```text
1. Verify derivative data loading
2. Verify event labels
3. Select inner-speech trials
4. Select action interval
5. Combine subjects
6. Run data sanity checks
7. Create/reuse fixed channel configurations
8. Establish 128-channel ML baseline
9. Establish 128-channel DL baseline
10. Run 64-channel experiments
11. Run 32-channel experiments
12. Run 16-channel experiments
13. Compare retention
14. Move to LOSO / stronger validation
15. Add pretrained EEG model
16. Run statistical testing
17. Only then consider extra signal-processing experiments
```

---

# 39. Statistical Analysis Direction

The project has discussed eventually doing statistical comparison across subjects.

Because the same subjects are evaluated under multiple channel conditions and possibly multiple model families, the data naturally have a repeated-measures structure.

The intended conceptual unit for statistics is:

```text
subject-level performance
```

not individual EEG trials treated as independent statistical samples.

Potential future tests should compare subject-level metrics across:

```text
128
64
32
16
```

and across model families.

Exact statistical methodology has not yet been finalized.

Do not implement a final significance-testing pipeline without re-checking the experiment design first.

---

# 40. Current Research Interpretation Goal

The key result should ideally answer something like:

> How much performance is retained when reducing EEG channel density while preserving broad spatial scalp coverage?

Example interpretation:

```text
128 channels: 60%
64 channels:  58%
32 channels:  55%
16 channels:  47%
```

Then:

```text
32-channel retention =
55 / 60 * 100
= 91.7%
```

A result like this would support discussion about whether much lower-density EEG configurations can preserve a useful fraction of full-cap performance.

---

# 41. Research Motivation

The motivation is practical:

High-density EEG systems can require:

```text
many electrodes
longer preparation time
more complex setup
more hardware
greater practical burden
```

If similar classification performance can be achieved with fewer electrodes, this may help make EEG-based systems simpler and more practical.

The project is not claiming this has already been achieved.

The experiment is intended to test it.

---

# 42. Portfolio-Level Description of the Project

The project is currently still in progress.

A concise description is:

> This research investigates inner-speech classification from EEG and studies how reducing the number of EEG channels affects model performance. Using a 128-channel inner-speech EEG dataset, the experiment compares configurations of 128, 64, 32, and 16 spatially distributed electrodes while keeping the same channel subsets across subjects and models. The project combines signal processing, neuroscience, machine learning, deep learning, and controlled experimental design to study how much decoding performance can be retained with fewer electrodes.

---

# 43. Researcher Learning Goals

The project is also being used to learn and apply:

```text
EEG signal structure
epochs and trials
event/trigger systems
signal preprocessing
frequency-domain analysis
feature extraction
classical machine learning
deep learning
pretrained neural models
channel selection/reduction
cross-subject evaluation
reproducibility
statistical testing
experimental design
```

---

# 44. Implementation Philosophy

When modifying the notebook, prioritize:

```text
correctness
reproducibility
clear experiment boundaries
no leakage
same conditions across models
simple baseline first
one change at a time
```

Do not optimize everything simultaneously.

The project intentionally follows a controlled progression:

```text
baseline
-> channel reduction
-> stronger validation
-> pretrained model
-> statistics
-> optional advanced preprocessing
```

---

# 45. Things Codex Should NOT Do Automatically

Unless explicitly requested, do not:

1. switch to random trial-level train/test splits across subjects
2. use test data to fit scalers
3. use test data to choose channels
4. regenerate channel configurations independently for each model
5. use different channel subsets for different subjects
6. use different channel subsets for ML vs DL
7. silently change class mappings
8. treat the `.dat` event file as plain text
9. hard-code the derivative sampling rate
10. re-run heavy preprocessing such as ICA on the derivative files without a research reason
11. replace the main channel-reduction rule with accuracy-based channel selection
12. compare model results produced under different data splits as if they were directly comparable
13. train with the entire 4.5-second epoch without recognizing the visual-cue confound
14. assume the Kaggle uploader is the official dataset author
15. treat the project as complete; it is still ongoing

---

# 46. Things Codex SHOULD Preserve

Codex should preserve:

```text
X shape convention:
[trial, channel, time]

class mapping:
0 Up
1 Down
2 Right
3 Left

condition mapping:
0 Pronounced
1 Inner Speech
2 Visualized

main condition:
Inner Speech only

main action interval:
1.0 <= t < 3.5 seconds

channel counts:
128 / 64 / 32 / 16

channel subsets:
nested
fixed
spatially distributed
same across subjects
same across models

evaluation:
subject-aware
eventually LOSO

ML representation:
bandpower / Welch PSD baseline

DL representation:
raw preprocessed EEG

primary experimental variable:
number of EEG channels
```

---

# 47. Recommended Project File Organization

A clean structure would be:

```text
project/
|
|-- README.md
|
|-- notebooks/
|   |-- latest_experiment.ipynb
|
|-- configs/
|   |-- channel_configs_idx.npy
|   |-- channel_configs_names.npy
|
|-- checkpoints/
|   |-- best_eegnet.pt
|
|-- results/
|   |-- baseline_128.csv
|   |-- channel_ablation.csv
|   |-- subject_metrics.csv
|
|-- src/
|   |-- data.py
|   |-- features.py
|   |-- models.py
|   |-- train.py
|   |-- evaluate.py
|
|-- figures/
    |-- channel_layouts/
    |-- confusion_matrices/
    |-- learning_curves/
```

This is optional; the current notebook may still be monolithic.

---

# 48. Suggested Refactoring Boundaries

If Codex later refactors the notebook into Python files, sensible modules are:

## `data.py`

Responsibilities:

```text
discover EEG files
load MNE epochs
load pickled event files
extract subject IDs
filter inner-speech trials
crop time windows
combine datasets
perform sanity checks
```

## `channels.py`

Responsibilities:

```text
load/save channel configs
attach BioSemi montage
get XYZ coordinates
farthest-point sampling
region/hemisphere metadata
apply 128/64/32/16 subsets
```

## `features.py`

Responsibilities:

```text
Welch PSD
bandpower
future FFT/STFT/wavelet features
```

## `models.py`

Responsibilities:

```text
EEGNet
other scratch DL models
future pretrained adapters
```

## `train.py`

Responsibilities:

```text
PyTorch Dataset/DataLoader
training loop
checkpointing
seed control
```

## `evaluate.py`

Responsibilities:

```text
accuracy
macro F1
balanced accuracy
confusion matrix
subject-level metrics
retention calculation
```

---

# 49. Reproducibility Seeds

The exact final random seed has not been finalized in this project discussion.

When experiments become official, define and log seeds for at least:

```python
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

Also consider deterministic settings if needed.

The important rule is:

```text
same seed policy across comparable experiments
```

---

# 50. Experiment Logging

For every official run, ideally log:

```text
model name
channel count
channel names/config hash
train subjects
validation subjects
test subjects
sampling rate
time window
feature method
normalization method
random seed
learning rate
batch size
epochs
best validation epoch
accuracy
macro F1
balanced accuracy
per-subject metrics
```

This will make the later research write-up much easier.

---

# 51. Baseline Before Optimization

The baseline is meant to prove:

```text
the dataset loads correctly
labels align correctly
the model can learn above chance
the pipeline can be evaluated consistently
```

The first baseline does **not** need to be the best published architecture.

Avoid spending too much time optimizing the baseline before the channel-reduction experiment works end-to-end.

---

# 52. Main Scientific Fairness Rule

When comparing channel counts:

```text
128 vs 64 vs 32 vs 16
```

keep all other experimental choices fixed as much as possible.

For example:

```text
same split
same model family
same hyperparameters where structurally possible
same preprocessing
same temporal window
same class definitions
same metrics
same random-seed protocol
```

For architectures whose input spatial dimension depends on channel count, reinitialize and retrain the model for each configuration.

Example:

```python
model = EEGNet(
    n_channels=32,
    n_classes=4
)
```

for the 32-channel experiment.

Do not train a 128-channel EEGNet and simply delete channels at test time unless that is a separate robustness experiment.

---

# 53. Important ML vs DL Representation Difference

Do not force ML and DL to use the same input representation merely for superficial consistency.

The intended baseline comparison is:

```text
Classical ML:
EEG
-> engineered bandpower features
-> SVM

Deep Learning:
EEG
-> raw preprocessed time-domain signal
-> EEGNet
```

This is intentional.

Classical ML typically benefits from explicit feature extraction.

The DL model is intended to learn temporal/spatial representations directly.

---

# 54. Potential Future Pretrained Model Track

The project intends to eventually compare:

```text
ML
vs
scratch DL
vs
pretrained EEG model
```

No final pretrained model has been selected yet.

Earlier discussion mentioned the general idea of pretrained neuroscience/EEG models and autoencoder-style pretraining, but the final architecture should be selected later based on:

```text
input channel compatibility
sampling-rate compatibility
availability of weights
license
cross-subject suitability
ability to adapt to 128/64/32/16 channels
```

Do not lock in a pretrained model without checking these constraints.

---

# 55. Storage / Compute Constraints

The user prefers Kaggle Notebook because local storage and compute are limited.

Therefore:

```text
prefer working directly from /kaggle/input
avoid unnecessary duplication of the full dataset
avoid loading multiple redundant copies into memory
avoid downloading raw data locally unless needed
```

GPU availability in Kaggle can vary.

The dataset itself is not extremely large after derivative preprocessing, but efficient loading is still preferred.

---

# 56. User Preferences for Code Assistance

For this project, when the user asks specifically for code:

- provide code directly
- keep explanations short unless the user asks for explanation
- do not waste tokens restating obvious concepts
- preserve the existing notebook conventions when possible
- avoid unnecessary abstraction unless it improves reproducibility

The user prefers learning by building and inspecting code.

---

# 57. Current State of the Project

As of the latest discussion, the project has reached approximately:

```text
Dataset provenance understood
✓

Derivative/preprocessed EEG identified
✓

Epoch/trial concepts understood
✓

MNE loading established
✓

Event .dat loading bug discovered and fixed
✓

Inner-speech filtering defined
✓

Action interval defined
✓

Interactive Plotly visualization discussed
✓

ML baseline design discussed
✓

EEGNet-style baseline discussed
✓

tqdm/Kaggle display issue handled
✓

Main channel-reduction philosophy defined
✓

BioSemi montage coordinate extraction discussed
✓

Nested FPS-based channel configurations discussed
✓

Saving .npy channel configs for reproducibility discussed
✓

Full official channel-reduction experiment
NOT YET COMPLETED

LOSO final evaluation
NOT YET COMPLETED

Pretrained EEG baseline
NOT YET COMPLETED

Final statistical analysis
NOT YET COMPLETED
```

---

# 58. Immediate Next Steps

The most logical next steps are:

```text
1. Inspect the latest notebook and identify exactly which parts are already implemented.

2. Verify the saved channel configuration files:
   - channel_configs_idx.npy
   - channel_configs_names.npy

3. Confirm the selected 16/32/64 subsets look spatially reasonable.

4. Establish stable 128-channel baselines:
   - SVM + bandpower
   - EEGNet

5. Refactor training/evaluation into reusable functions.

6. Run the same baseline under:
   128
   64
   32
   16

7. Store results in a single structured table.

8. Move from development subject split to LOSO evaluation.

9. Add subject-level metrics and retention.

10. Only after the main experiment is stable:
    investigate pretrained EEG models and additional signal-processing variants.
```

---

# 59. Suggested Results Table

Eventually store results in a dataframe like:

```text
model
channel_count
fold
test_subject
accuracy
macro_f1
balanced_accuracy
seed
```

Example schema:

```python
results = pd.DataFrame(columns=[
    "model",
    "channel_count",
    "fold",
    "test_subject",
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "seed"
])
```

This structure will make statistical analysis easier later.

---

# 60. Final Mental Model for Codex

The entire project can be summarized as:

```text
Thinking Out Loud EEG dataset
        |
        v
Authors' preprocessed derivative epochs
        |
        v
Load .fif with MNE
Load events.dat with pd.read_pickle
        |
        v
Select condition == Inner Speech
        |
        v
Crop to 1.0-3.5 s action interval
        |
        v
X = [trial, 128 channels, time]
        |
        v
Fixed spatial channel configurations
128 -> 64 -> 32 -> 16
        |
        +--------------------------+
        |                          |
        v                          v
Bandpower features             Raw EEG
        |                          |
        v                          v
       SVM                      EEGNet
        |                          |
        +-------------+------------+
                      |
                      v
          subject-aware evaluation
                      |
                      v
           compare performance
                      |
                      v
          calculate retention
                      |
                      v
 eventually:
 LOSO + pretrained model + statistics
```

---

# 61. Priority Rule When README and Notebook Differ

The latest notebook represents the newest implementation.

Therefore:

> If this README describes an older code detail but the notebook clearly contains a newer intentional implementation, treat the notebook as the implementation source of truth.

However, do **not** silently override the research-design constraints in this README.

Especially preserve:

```text
subject-aware evaluation
no leakage
fixed saved channel subsets
same subsets across subjects/models
inner-speech-only condition
action-window awareness
reproducibility
```

If Codex wants to change one of those core experimental assumptions, it should surface the change explicitly instead of silently modifying it.

---

# 62. Short Research Statement

> This ongoing research investigates how reducing EEG electrode density affects inner-speech classification. Using the Thinking Out Loud 128-channel EEG dataset, the project compares fixed, spatially distributed configurations of 128, 64, 32, and 16 channels across classical machine learning, deep learning, and eventually pretrained EEG models. The experiment is designed so that channel count is the main changing variable, while subject splits, preprocessing, channel locations, labels, temporal windows, and evaluation procedures remain controlled.

