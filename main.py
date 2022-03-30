from ResnetNetwork import *
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
from CenterLoss import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import math
from model import *
import dataloader
from scipy.io import loadmat
import wfdb
from EarlyStopping import *
import scipy.fftpack as function
from SignalPreprocessing.data_preprocess_function import *

# SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "PartialDataset")
SIMULATED_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "simulated_windows_noise_without_c3")
#ECG_OUTPUTS_TEST_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RealTest")
#ECG_OUTPUTS_TEST_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Osher_Synt_SanityCheck_06_Mar")
#ECG_OUTPUTS_TEST_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Osher_TestReal/Using_HPF/ALL_100")
ECG_OUTPUTS_TEST_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Osher_TestReal/real_signals_challenge2013_Plot_08_Mar")


#REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RealSignals")
#REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "NewReal")

#REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Real_Preprocessing/real_windows/")
#REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Real_Preprocessing/best_windows/Synt_SanityCheck")
#REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Real_Preprocessing/best_windows/Using_HPF/ALL_100")
REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/home/otidhar/FECG/real_signals_challenge2013_windows_08_Mar_b10Tob19")
#REAL_DATASET = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Real_Preprocessing/best_windows/real_signals_ninfea_windows_08_Mar_10")

#InferenceOnReal = False
InferenceOnReal = True

if not os.path.exists(ECG_OUTPUTS_TEST_REAL):
    os.mkdir(ECG_OUTPUTS_TEST_REAL)

LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Losses")
#LOSSES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/home/otidhar/FECG/Losses50000")

if not os.path.exists(LOSSES):
    os.mkdir(LOSSES)

BAR_LIST_TEST = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BarListTest")
if not os.path.exists(BAR_LIST_TEST):
    os.mkdir(BAR_LIST_TEST)

BATCH_SIZE = 32
epochs = 30
learning_rate = 1e-3


def inference(filename, test_data_loader_real):
    resnet_model = ResNet(1)
    #resnet_model.load_state_dict(torch.load('Models/best_model'))
    resnet_model.load_state_dict(torch.load('/home/otidhar/FECG/Models_Sub1_7_10_Ch1_3_5_19_21_23/last_model'))
    #resnet_model.load_state_dict(torch.load('/home/otidhar/FECG/Models/best_model'))
    resnet_model.eval()

    resnet_model.cuda()

    criterion = nn.L1Loss().cuda()

    test_loss_ecg = 0
    test_corr_ecg = 0
    with torch.no_grad():
        for i, batch_features in enumerate(test_data_loader_real):
            batch_for_model_test = Variable(batch_features.float().cuda())
            outputs_m_test_real, _, outputs_f_test_real, _ = resnet_model(batch_for_model_test)
            test_loss_ecg += criterion(outputs_m_test_real + outputs_f_test_real, batch_for_model_test)

            for j, elem in enumerate(outputs_f_test_real):
                test_corr_ecg += np.corrcoef((outputs_m_test_real[j] + outputs_f_test_real[j]).cpu().detach().numpy(), batch_for_model_test.cpu().detach().numpy()[j])[0][1]
                path = os.path.join(ECG_OUTPUTS_TEST_REAL, "label_ecg" + str(j) + str(i))
                np.save(path, batch_features[j].cpu().detach().numpy())

                path = os.path.join(ECG_OUTPUTS_TEST_REAL, "ecg" + str(j) + str(i))
                np.save(path, (outputs_m_test_real[j] + outputs_f_test_real[j]).cpu().detach().numpy())
                #sio.savemat(path + '_mat.mat',{'data': np.float64((outputs_m_test_real[j] + outputs_f_test_real[j]))}, oned_as='column')

                path = os.path.join(ECG_OUTPUTS_TEST_REAL, "mecg" + str(j) + str(i))
                np.save(path, (outputs_m_test_real[j]).cpu().detach().numpy())
                #sio.savemat(path + '_mat.mat',{'data': np.float64((outputs_m_test_real[j]))}, oned_as='column')

                path = os.path.join(ECG_OUTPUTS_TEST_REAL, "fecg" + str(j) + str(i))
                np.save(path, (outputs_f_test_real[j]).cpu().detach().numpy())
                #sio.savemat(path + '_mat.mat',{'data': np.float64((outputs_f_test_real[j]))}, oned_as='column')

        test_loss_ecg /= len(test_data_loader_real.dataset)
        test_corr_ecg /= len(test_data_loader_real.dataset)
        print('L1: ' + str(test_loss_ecg.item()))
        print('Corr: ' + str(test_corr_ecg))


def main():

    pl.seed_everything(1234)
    list_simulated = simulated_database_list(SIMULATED_DATASET)#[:127740]

    list_simulated = remove_nan_signals(list_simulated)

    simulated_dataset = dataloader.SimulatedDataset(SIMULATED_DATASET,list_simulated)
    print("simulated_dataset",len(simulated_dataset))
    train_size_sim = int(0.6 * len(simulated_dataset))
    val_size_sim = int(0.2 * len(simulated_dataset))
    test_size_sim = int(0.2 * len(simulated_dataset))

    train_dataset_sim, val_dataset_sim, test_dataset_sim = torch.utils.data.random_split(simulated_dataset, [train_size_sim, val_size_sim,test_size_sim])

    train_data_loader_sim = data.DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    #train_data_loader_sim = data.DataLoader(train_dataset_sim, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader_sim = data.DataLoader(val_dataset_sim, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader_sim = data.DataLoader(test_dataset_sim, batch_size=BATCH_SIZE, shuffle=False)

    #  use gpu if available
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    resnet_model = ResNet(1).cuda()
    best_model_accuracy = - math.inf
    val_loss = 0
    early_stopping = EarlyStopping(delta_min=0.001, patience=6, verbose=True)
    criterion = nn.L1Loss().cuda()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=512*64, use_gpu=device)
    params = list(resnet_model.parameters()) + list(criterion_cent.parameters())
    optimizer_model = optim.SGD(params, lr=learning_rate, momentum=0.9,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_model, milestones=[20], gamma=0.1)

    train_loss_f_list = []
    train_loss_m_list = []
    train_loss_average_list = []
    validation_loss_f_list = []
    validation_loss_m_list = []
    validation_loss_average_list = []
    validation_corr_m_list = []
    validation_corr_f_list = []

    for epoch in range(epochs):
        #Train
        resnet_model.train()
        train(resnet_model,
              train_data_loader_sim,
              optimizer_model,
              criterion,
              criterion_cent,
              epoch,
              epochs,
              train_loss_f_list,
              train_loss_m_list,
              train_loss_average_list)
        #Evaluation
        resnet_model.eval()
        best_model_accuracy, val_loss = val(val_data_loader_sim,
                                    resnet_model,
                                    criterion,
                                    epoch,
                                    epochs,
                                    validation_loss_m_list,
                                    validation_loss_f_list,
                                    validation_loss_average_list,
                                    validation_corr_m_list,
                                    validation_corr_f_list,
                                    best_model_accuracy,
                                    criterion_cent)
        scheduler.step()
        early_stopping(val_loss, resnet_model)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    #Saving graphs training
    path_losses = os.path.join(LOSSES, "TL1M")
    np.save(path_losses, np.array(train_loss_m_list))
    path_losses = os.path.join(LOSSES, "TL1F")
    np.save(path_losses, np.array(train_loss_f_list))
    path_losses = os.path.join(LOSSES, "TL1Avg")
    np.save(path_losses, np.array(train_loss_average_list))

    #Saving graphs validation
    path_losses = os.path.join(LOSSES, "VL1M")
    np.save(path_losses, np.array(validation_loss_m_list))
    path_losses = os.path.join(LOSSES, "VL1F")
    np.save(path_losses, np.array(validation_loss_f_list))
    path_losses = os.path.join(LOSSES, "VL1Avg")
    np.save(path_losses, np.array(validation_loss_average_list))

    path_losses = os.path.join(LOSSES, "CorrM")
    np.save(path_losses, np.array(validation_corr_m_list))
    path_losses = os.path.join(LOSSES, "CorrF")
    np.save(path_losses, np.array(validation_corr_f_list))

    #Test
    test_loss_m, test_loss_f, test_loss_avg, test_corr_m, test_corr_f, test_corr_average,\
        list_bar_good_example_noisetype, list_bar_bad_example_noisetype,\
        list_bar_good_example_snr,list_bar_bad_example_snr, \
        list_bar_good_example_snrcase, list_bar_bad_example_snrcase = test(str(network_save_folder_orig + network_file_name_best),test_data_loader_sim)

    path_bar = os.path.join(BAR_LIST_TEST, "list_bar_good_example_noisetype")
    np.save(path_bar, np.array(list_bar_good_example_noisetype))
    path_bar = os.path.join(BAR_LIST_TEST, "list_bar_bad_example_noisetype")
    np.save(path_bar, np.array(list_bar_bad_example_noisetype))
    path_bar = os.path.join(BAR_LIST_TEST, "list_bar_good_example_snr")
    np.save(path_bar, np.array(list_bar_good_example_snr))
    path_bar = os.path.join(BAR_LIST_TEST, "list_bar_bad_example_snr")
    np.save(path_bar, np.array(list_bar_bad_example_snr))
    path_bar = os.path.join(BAR_LIST_TEST, "list_bar_good_example_snrcase")
    np.save(path_bar, np.array(list_bar_good_example_snrcase))
    path_bar = os.path.join(BAR_LIST_TEST, "list_bar_bad_example_snrcase")
    np.save(path_bar, np.array(list_bar_bad_example_snrcase))

    with open("test_loss.txt", 'w') as f:
        f.write("test_loss_m = {:.4f},test_loss_f = {:.4f},test_loss_avg = {:.4f},"
                "test_corr_m = {:.4f},test_corr_f = {:.4f},test_corr_avg = {:.4f}\n".format(test_loss_m, test_loss_f, test_loss_avg,
                                                                                            test_corr_m,test_corr_f,test_corr_average))
    del resnet_model
    del simulated_dataset
    del train_data_loader_sim
    torch.cuda.empty_cache()


if __name__=="__main__":

    if not InferenceOnReal:
        main()
    if InferenceOnReal:
        real_dataset = dataloader.RealDataset(REAL_DATASET)
        test_data_loader_real = data.DataLoader(real_dataset, batch_size=BATCH_SIZE, shuffle=False)
        inference(network_save_folder_orig, test_data_loader_real)
    if not InferenceOnReal:
        for filename in os.listdir(ECG_OUTPUTS_TEST):  # present the fecg outputs
            if "fecg" in filename:
                # num_of_f += 1
                path = os.path.join(ECG_OUTPUTS_TEST, filename)
                number_file = filename.index("g") + 1
                end_path = filename[number_file:]
                fecg = os.path.join(ECG_OUTPUTS_TEST, "fecg" + end_path)
                fecg_label = os.path.join(ECG_OUTPUTS_TEST, "label_f" + end_path)
                mecg = os.path.join(ECG_OUTPUTS_TEST, "mecg" + end_path)
                mecg_label = os.path.join(ECG_OUTPUTS_TEST, "label_m" + end_path)
                ecg= os.path.join(ECG_OUTPUTS_TEST, "ecg_all" + end_path)
                real = np.load(path)
                fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
                ax1.plot(np.load(mecg))
                ax1.set_ylabel("MECG")
                ax2.plot(np.load(mecg_label))
                ax2.set_ylabel("MECG_LABEL")
                ax3.plot(np.load(fecg))
                ax3.set_ylabel("FECG")
                ax4.plot(np.load(fecg_label))
                ax4.set_ylabel("FECG_LABEL")
                ax5.plot(np.load(ecg))
                ax5.set_ylabel("ECG")
                plt.show()
                plt.close()

    #     for filename in os.listdir(ECG_OUTPUTS_TEST):  # present the fecg outputs
    #         if "fecg" in filename:
    #             #num_of_f += 1
    #             path = os.path.join(ECG_OUTPUTS_TEST, filename)
    #             number_file = filename.index("g") + 1
    #             end_path = filename[number_file:]
    #             path_label = os.path.join(ECG_OUTPUTS_TEST, "label_f" + end_path)
    #             real = np.load(path)
    #             label = np.load(path_label)
    #             correlation = check_correlation(real, label)
    # #            if (correlation < 0.70):
    #               #  correlation_f += 1
    #             fig, (ax1, ax2) = plt.subplots(2, 1)
    #             ax1.plot(real)
    #             ax1.set_ylabel("FECG")
    #             ax2.plot(label)
    #             ax2.set_ylabel("LABEL")
    #             plt.show()
    #             plt.close()
    #
    #         if "mecg" in filename:
    #             path = os.path.join(ECG_OUTPUTS_TEST, filename)
    #             number_file = filename.index("g") + 1
    #             end_path = filename[number_file:]
    #             path_label = os.path.join(ECG_OUTPUTS_TEST, "label_m" + end_path)
    #             real = np.load(path)
    #             label = np.load(path_label)
    #             fig, (ax1, ax2) = plt.subplots(2, 1)
    #             ax1.plot(np.load(path))
    #             ax1.set_ylabel("MECG")
    #             ax2.plot(np.load(path_label))
    #             ax2.set_ylabel("LABEL")
    #             plt.show()
    #             plt.close()
    if InferenceOnReal:
        for filename in os.listdir(ECG_OUTPUTS_TEST_REAL):  # present the fecg outputs
            if "label_ecg" in filename and not "mat" in filename:
                path_label = os.path.join(ECG_OUTPUTS_TEST_REAL, filename)
                number_file = filename.index("g") + 1
                end_path = filename[number_file:]
                path = os.path.join(ECG_OUTPUTS_TEST_REAL, "ecg" + end_path)
                mecg_label = os.path.join(ECG_OUTPUTS_TEST_REAL, "mecg" + end_path)
                fecg_label = os.path.join(ECG_OUTPUTS_TEST_REAL, "fecg" + end_path)
                fig, (ax1, ax2,ax3,ax4) = plt.subplots(4, 1)
                ax1.plot(np.load(path)[0])
                ax1.set_ylabel("ECG")
                ax2.plot(np.load(path_label)[0])
                ax2.set_ylabel("LABEL ECG")
                ax3.plot(np.load(mecg_label)[0])
                ax3.set_ylabel("MECG")
                ax4.plot(np.load(fecg_label)[0])
                ax4.set_ylabel("FECG")
                plt.show()
                plt.close()

    BAR_LIST = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BarListTest")

    if not InferenceOnReal:
        # BAR REPRESENTATION
        ind = np.arange(4)
        x_labels = ['NONE', 'MA', 'MA+EM', 'MA+EM+BW']
        results = np.load(os.path.join(BAR_LIST, "list_bar_bad_example_noisetype.npy"))
        sum = np.sum(np.matrix(results))
        plt.bar(ind, results)
        plt.title('Failing signals according to noise type. Total: {}'.format(sum))
        plt.xticks(ind, ('NONE', 'MA', 'MA+EM', 'MA+EM+BW'))
        plt.show()
        plt.close()

        ind = np.arange(4)
        x_labels = ['NONE', 'MA', 'MA+EM', 'MA+EM+BW']
        results = np.load(os.path.join(BAR_LIST, "list_bar_good_example_noisetype.npy"))
        sum = np.sum(np.matrix(results))
        plt.bar(ind, results)
        plt.title('Successful signals according to noise type. Total: {}'.format(sum))
        plt.xticks(ind, ('NONE', 'MA', 'MA+EM', 'MA+EM+BW'))
        plt.show()
        plt.close()

        ind = np.arange(5)
        x_labels = ['00', '03', '06', '09', '12']
        results = np.load(os.path.join(BAR_LIST, "list_bar_bad_example_snr.npy"))
        sum = np.sum(np.matrix(results))
        plt.bar(ind, results)
        plt.title('Failing signals according to SNR [dB]. Total: {}'.format(sum))
        plt.xticks(ind, ('00', '03', '06', '09', '12'))
        plt.show()
        plt.close()

        ind = np.arange(5)
        x_labels = ['00', '03', '06', '09', '12']
        results = np.load(os.path.join(BAR_LIST, "list_bar_good_example_snr.npy"))
        sum = np.sum(np.matrix(results))
        plt.bar(ind, results)
        plt.title('Successful signals according to SNR [dB]. Total: {}'.format(sum))
        plt.xticks(ind, ('00', '03', '06', '09', '12'))
        plt.show()
        plt.close()

        X = np.arange(7)
        data = np.load(os.path.join(BAR_LIST, "list_bar_bad_example_snrcase.npy"))
        print(np.sum(data, axis=0))
        sum = np.sum(np.matrix(data))
        a = plt.bar(X, data[0], color='b', width=0.1)
        b = plt.bar(X + 0.1, data[1], color='g', width=0.1)
        c = plt.bar(X + 0.2, data[2], color='r', width=0.1)
        d = plt.bar(X + 0.3, data[3], color='c', width=0.1)
        e = plt.bar(X + 0.4, data[4], color='y', width=0.1)
        plt.legend((a, b, c, d, e), ('00', '03', '06', '09', '12'))
        plt.xticks(X, ('CO', 'C1', 'C2', 'C3', 'C4', 'C5', 'BASELINE'))
        plt.title('Failing signals according to physiological case and SNR [dB]. Total: {}'.format(sum))
        plt.show()
        plt.close()

        X = np.arange(7)
        data = np.load(os.path.join(BAR_LIST, "list_bar_good_example_snrcase.npy"))
        print(np.sum(data, axis=0))
        sum = np.sum(np.matrix(data))
        a = plt.bar(X, data[0], color='b', width=0.1)
        b = plt.bar(X + 0.1, data[1], color='g', width=0.1)
        c = plt.bar(X + 0.2, data[2], color='r', width=0.1)
        d = plt.bar(X + 0.3, data[3], color='c', width=0.1)
        e = plt.bar(X + 0.4, data[4], color='y', width=0.1)
        plt.legend((a, b, c, d, e), ('00', '03', '06', '09', '12'))
        plt.xticks(X, ('CO', 'C1', 'C2', 'C3', 'C4', 'C5', 'BASELINE'))
        plt.title('Successful signals according to physiological case and SNR [dB]. Total: {}'.format(sum))
        plt.show()
        plt.close()

        path_losses = os.path.join(LOSSES, "TL1M.npy")
        train_loss_m_list = np.load(path_losses)
        path_losses = os.path.join(LOSSES, "TL1F.npy")
        train_loss_f_list = np.load(path_losses)
        path_losses = os.path.join(LOSSES, "TL1Avg.npy")
        train_loss_average_list = np.load(path_losses)
        path_losses = os.path.join(LOSSES, "VL1M.npy")
        validation_loss_m_list = np.load(path_losses)
        path_losses = os.path.join(LOSSES, "VL1F.npy")
        validation_loss_f_list = np.load(path_losses)
        path_losses = os.path.join(LOSSES, "VL1Avg.npy")
        validation_loss_average_list = np.load(path_losses)

        path_losses = os.path.join(LOSSES, "CorrF.npy")
        correlation_f_list = np.load(path_losses)
        path_losses = os.path.join(LOSSES, "CorrM.npy")
        correlation_m_list = np.load(path_losses)

        # plotting validation and training losses and saving them
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(train_loss_m_list, label="training")
        ax1.plot(validation_loss_m_list, label="validation")
        ax1.set_ylabel("L1 M")
        ax1.set_xlabel("Epoch")
        ax2.plot(train_loss_f_list, label="training")
        ax2.plot(validation_loss_f_list, label="validation")
        ax2.set_ylabel("L1 F")
        ax2.set_xlabel("Epoch")
        ax3.plot(train_loss_average_list, label="training")
        ax3.plot(validation_loss_average_list, label="validation")
        ax3.set_ylabel("L1 Avg")
        ax3.set_xlabel("Epoch")
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.show()
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(correlation_f_list)
        ax1.set_ylabel("CorrF")
        ax1.set_xlabel("Epoch")
        ax2.plot(correlation_m_list)
        ax2.set_ylabel("CorrM")
        ax2.set_xlabel("Epoch")
        plt.show()
        plt.close()

    if not InferenceOnReal:
        for filename in os.listdir(SIMULATED_DATASET): #present the fecg outputs
            if "fecg" in filename:
                print(filename)
                path = os.path.join(SIMULATED_DATASET, filename)
                number_file = filename.index("g") + 2
                start_path = filename[:number_file -5]
                end_path = filename[number_file:]
                print(end_path)
                path_label = os.path.join(SIMULATED_DATASET,start_path + "mecg" + end_path)
                print(path_label)
                real = loadmat(path)['data']
                label = loadmat(path_label)['data']
                fig, (ax1, ax2) = plt.subplots(2, 1)
                ax1.plot(real)
                ax1.set_ylabel("FECG")
                ax2.plot(label)
                ax2.set_ylabel("LABEL")
                plt.show()
                plt.close()

