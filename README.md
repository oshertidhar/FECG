# FECG
M.Sc Final Project @ TAU. Created By Osher Tidhar

## General
- My model is implemented in `model.py`.
- Synthetic dataset can be found here: PhysioNet - Fetal ECG Synthetic Database. 127k samples which are sampled at 250Hz.
https://physionet.org/content/fecgsyndb/1.0.0/

# Branches
1. **OldModel** - without injection of TECG (See OldInjectionScheme.png)
2. **TrainSim**: (See NewInjectionScheme.png)
   Model Pre-Training on Simulated Data
3. **TrainReal**: (See NewInjectionScheme.png)
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
1. Download dataset from https://physionet.org/content/nifeadb/1.0.0/
2. If needed, change PHYSIONET_PATH in `main.py, FromDatToMat.py` with the corresponding path where the dataset that was downloaded.
3. In order to get a standardized input size, in SimulatedMergingAndWindowing.py, make sure the real-world data is divided into windows of size 1024.
4. Run in the following order:

  a. FromDatToMat.py
  b. SimulatedMergingAndWindowing.py
  c. main.py


## Real-World Data
The Non-Invasive Fetal ECG Arrhythmia (NIFEA) (https://www.physionet.org/content/nifeadb/1.0.0/) database was chosen to be the real-world data for out model.
This database provides 12 fetal arrhythmias recordings and 14 normal rhythm recordings. 
For each recording, a set of four or five abdominal channels and one chest maternal channel were recorded. The sampling frequency was 500 Hz or 1 kHz. 


