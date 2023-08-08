from ResnetNetwork import *
import torch.optim as optim
import torch.utils.data as data
from CenterLoss import *
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import math
from model import *
import dataloader
from scipy.io import loadmat

#SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../AECG/simulated_signals_windows_l1to2_baseline_and_c0to2_11_Oct_22_Ch19_21_23")
REAL_DATASET_TECG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../AECG/real_signals_nifeadb_1024windows_NR_10_20_Dec_22/NR_10_ch0")
REAL_DATASET_AECG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../AECG/real_signals_nifeadb_1024windows_NR_10_20_Dec_22/NR_10_ch2")

#LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../MasterAECG+MECG/Losses/Ch_19_21_23/mecg2IsShiftedBy1NotNoised")
LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses/Overfit/NR_10/TrainAllSet")

BATCH_SIZE = 32
#epochs = 50
epochs = 100
learning_rate_real = 1e-5
learning_rate = 1e-3

LOAD_CHCKPONTS = True
BEST_MODEL_ACC = 0.3660

def main():
    pl.seed_everything(1234)

    # (TODO) OVERFIT_REAL: uncomment ALL LINES BELOW after we replace to real data instead of simulated data
    real_dataset = dataloader.RealOverfitDataset(REAL_DATASET_AECG, REAL_DATASET_TECG)
    print("real_dataset",len(real_dataset))
    train_size_real = int(len(real_dataset))
    train_dataset_real = torch.utils.data.Subset(real_dataset, range(train_size_real))

    # A good rule of thumb is: num_workers = 4 * num_GPU
    train_data_loader_real = data.DataLoader(train_dataset_real, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)  #num_workers=12)

    #  use ALL the available GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TWO input channels - 1. AECG = MECG + FECG    2. TECG (thoracic)
    resnet_model = ResNet(2)
    resnet_model = nn.DataParallel(resnet_model)
    resnet_model.to(device)
    print(device)

    print('Params before Freezing')
    NumOfFrozenParams(resnet_model.parameters())

    # Freezing F_Decoder Weights
    resnet_model.module.Fdecoder.requires_grad_(False)
    #resnet_model.module.encoder.requires_grad_(False)

    print('Params after Freezing')
    NumOfFrozenParams(resnet_model.parameters())

    if LOAD_CHCKPONTS:
        resnet_model.load_state_dict(torch.load(os.path.join(network_load_folder_sim, network_file_name_best_sim)))

    print('Params after LOADING')
    NumOfFrozenParams(resnet_model.parameters())

    #best_model_accuracy_real = - math.inf
    best_model_accuracy_real = math.inf

    criterion = nn.L1Loss().cuda()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=512*64, use_gpu=device)
    params = list(resnet_model.parameters()) + list(criterion_cent.parameters())
    optimizer_model_real = optim.SGD(params, lr=learning_rate_real, momentum=0.9, weight_decay=1e-5)
    scheduler_real = torch.optim.lr_scheduler.OneCycleLR(optimizer_model_real, max_lr=1e-2, steps_per_epoch=int(
        np.ceil(len(train_data_loader_real.dataset) / BATCH_SIZE)), epochs=epochs + 1)

    train_loss_ecg_list = []

    for epoch in range(epochs):
        #Train
        resnet_model.train()
        best_model_accuracy_real, train_loss_real =train_overfit_real(resnet_model,
              train_data_loader_real,
              optimizer_model_real,
              epoch,
              epochs,
              criterion,
              criterion_cent,
              train_loss_ecg_list,
              scheduler_real,
            best_model_accuracy_real)

    #Saving graphs training
    path_losses = os.path.join(LOSSES, "TL1ECG")
    np.save(path_losses, np.array(train_loss_ecg_list))

    del resnet_model
    del real_dataset
    del train_data_loader_real
    torch.cuda.empty_cache()


def NumOfFrozenParams(parameters):
    #params = resnet_model.state_dict()
    #print(params.keys())
    #print(len(params.keys()))
    cntr_false = 0
    cntr_true = 0
    for param in parameters:
        if not param.requires_grad:
            #print(param)
            cntr_false += 1
        if param.requires_grad:
            #print(param)
            cntr_true += 1
    print('cntr_false: ', cntr_false)
    print('cntr_true: ', cntr_true)


if __name__=="__main__":
    main()
    path_losses = os.path.join(LOSSES, "TL1ECG.npy")
    train_loss_m_list = np.load(path_losses)	
    # plotting validation and training losses and saving them
    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(train_loss_m_list, label="training")	
    ax1.set_ylabel("L1 M - ECG")
    ax1.set_xlabel("Epoch")	
    plt.show()
    plt.close()

    for filename in os.listdir(ECG_OUTPUTS_REAL):  # present the fecg outputs
        if "ecg_all" in filename:
            print(filename)
            number_file = filename.index("g") + 1
            end_path = filename[number_file+4:]

            ecg_all = os.path.join(ECG_OUTPUTS_REAL, "ecg_all" + end_path)
            mecg_label = os.path.join(ECG_OUTPUTS_REAL, "label_m" + end_path)
            mecg = os.path.join(ECG_OUTPUTS_REAL, "mecg" + end_path)
            fecg = os.path.join(ECG_OUTPUTS_REAL, "fecg" + end_path)
            fecg_shifted = os.path.join(ECG_OUTPUTS_REAL, "fecg_shifted" + end_path)

            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
            ax1.plot(np.load(mecg_label))
            ax1.set_ylabel("ECG")
            ax2.plot(np.load(mecg))
            ax2.set_ylabel("MECG")
            ax3.plot(np.load(ecg_all))
            ax3.set_ylabel("Abdomen_2")
            ax4.plot(np.load(fecg_shifted))
            ax4.set_ylabel("FECG_shifted")
            ax5.plot(np.load(fecg))
            ax5.set_ylabel("FECG")
            plt.show()
            plt.close()
