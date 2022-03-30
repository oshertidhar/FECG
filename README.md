# FECG
M.Sc Final Project @ TAU. Created By Osher Tidhar

## General
- My model is implemented in `model.py`.
- Synthetic dataset can be found here: PhysioNet - Fetal ECG Synthetic Database. 127k samples which are sampled at 250Hz.
https://physionet.org/content/fecgsyndb/1.0.0/

## How to run
# Train Synthectic Dataset
1. Download dataset from https://physionet.org/content/fecgsyndb/1.0.0/
2. If needed, change PHYSIONET_PATH in `main.py, FromDatToMat.py` with the corresponding path where the dataset that was downloaded.
3. Run in the following order:
  a. FromDatToMat.py
  b. SimulatedMergingAndWindowing.py
  c. main.py
