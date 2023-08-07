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

REAL_DATASET_TECG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "real_signals_nifeadb_1024windows_NR_07_15_Dec_22/NR_07_ch0")
REAL_DATASET_AECG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "real_signals_nifeadb_1024windows_NR_07_15_Dec_22/NR_07_ch3")

LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses/Overfit/NR_07/Jan02")

BATCH_SIZE = 32
#epochs = 50
epochs = 1
learning_rate_real = 1e-5
learning_rate = 1e-3

LOAD_CHCKPONTS = True
BEST_MODEL_ACC = 0.3660

def main():
    pl.seed_everything(1234)

    real_dataset = dataloader.RealOverfitDataset(REAL_DATASET_AECG, REAL_DATASET_TECG)
    print("real_dataset",len(real_dataset))
    train_size_real = int(0.6 * len(real_dataset))
    val_size_real = int(0.2 * len(real_dataset))
    test_size_real = int(0.2 * len(real_dataset))
    train_dataset_real = torch.utils.data.Subset(real_dataset, range(train_size_real))
    val_dataset_real = torch.utils.data.Subset(real_dataset, range(train_size_real, train_size_real + val_size_real))
    test_dataset_real = torch.utils.data.Subset(real_dataset, range(train_size_real + val_size_real, train_size_real + val_size_real + test_size_real))

    # A good rule of thumb is: num_workers = 4 * num_GPU
    train_data_loader_real = data.DataLoader(train_dataset_real, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)  #num_workers=12)
    val_data_loader_real = data.DataLoader(val_dataset_real, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader_real = data.DataLoader(test_dataset_real, batch_size=BATCH_SIZE, shuffle=False)

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

    val_loss_real = 0
    criterion = nn.L1Loss().cuda()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=512*64, use_gpu=device)
    params = list(resnet_model.parameters()) + list(criterion_cent.parameters())
    optimizer_model_real = optim.SGD(params, lr=learning_rate_real, momentum=0.9, weight_decay=1e-5)
    scheduler_real = torch.optim.lr_scheduler.OneCycleLR(optimizer_model_real, max_lr=1e-2, steps_per_epoch=int(
        np.ceil(len(train_data_loader_real.dataset) / BATCH_SIZE)), epochs=epochs + 1)

    train_loss_ecg_list = []
    validation_loss_ecg_list = []
    validation_corr_ecg_list = []

    for epoch in range(epochs):
        #Train
        resnet_model.train()
        train_overfit_real(resnet_model,
              train_data_loader_real,
              optimizer_model_real,
              epoch,
              epochs,
              criterion,
              criterion_cent,
              train_loss_ecg_list,
              scheduler_real)

        # Validation Real
        resnet_model.eval()
        best_model_accuracy_real, val_loss_real = val_overfit_real(
               val_data_loader_real,
               resnet_model,
               criterion,
               criterion_cent,
               epoch,
               epochs,
               validation_loss_ecg_list,
               validation_corr_ecg_list,
               best_model_accuracy_real)

    #Saving graphs training
    path_losses = os.path.join(LOSSES, "TL1ECG")
    np.save(path_losses, np.array(train_loss_ecg_list))
    #Saving graphs validation
    path_losses = os.path.join(LOSSES, "VL1ECG")
    np.save(path_losses, np.array(validation_loss_ecg_list))
    path_losses = os.path.join(LOSSES, "CorrECG")
    np.save(path_losses, np.array(validation_corr_ecg_list))

    # Test
    test_loss_ecg, test_corr_ecg = test_overfit_real(str(network_save_folder_real + network_file_name_best_real),
                                        test_data_loader_real)
    with open("test_loss.txt", 'w') as f:
        f.write(",test_loss_ecg = {:.4f},test_corr_ecg = {:.4f}".format(test_loss_ecg, test_corr_ecg))

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
    path_losses = os.path.join(LOSSES, "VL1ECG.npy")	
    validation_loss_m_list = np.load(path_losses)	
    path_losses = os.path.join(LOSSES, "CorrECG.npy")		
    correlation_f_list = np.load(path_losses)	
    # plotting validation and training losses and saving them	
    fig, (ax1,ax2) = plt.subplots(2, 1)	
    ax1.plot(train_loss_m_list, label="training")	
    ax1.plot(validation_loss_m_list, label="validation")	
    ax1.set_ylabel("L1 M - ECG")
    ax1.set_xlabel("Epoch")	
    ax2.plot(correlation_f_list)
    ax2.set_ylabel("CorrM")
    ax2.set_xlabel("Epoch")
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
