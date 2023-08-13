# FECG
M.Sc Final Project @ TAU. Created By Osher Tidhar

## General
- My model is implemented in `model.py`.
- Synthetic dataset can be found here: PhysioNet - Fetal ECG Synthetic Database. 127k samples which are sampled at 250Hz.
https://physionet.org/content/fecgsyndb/1.0.0/

# Branches
1. OldModel - without injection of TECG
2. TrainSim:
   Model Pre-Training on Simulated Data
4. TrainReal:
   Model Fine-Tune Training on Real-World Data 

## How to run
1. FromDat2Mat.py
2. SimulatedMergingAndWindowing.py
3. Main.py
   
# Train Synthectic Dataset 
* BRANCH: TrainSim
1. Download dataset from https://physionet.org/content/fecgsyndb/1.0.0/
2. If needed, change PHYSIONET_PATH in `main.py, FromDatToMat.py` with the corresponding path where the dataset that was downloaded.
3. Run in the following order:

  a. FromDatToMat.py
  b. SimulatedMergingAndWindowing.py
  c. main.py

# Train Real Dataset 
* BRANCH: TrainReal
1. Download dataset from https://physionet.org/content/fecgsyndb/1.0.0/
2. If needed, change PHYSIONET_PATH in `main.py, FromDatToMat.py` with the corresponding path where the dataset that was downloaded.
3. Run in the following order:

  a. FromDatToMat.py
  b. SimulatedMergingAndWindowing.py
  c. main.py
