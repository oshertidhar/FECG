import torch
import os
from ResnetNetwork import *
from torch.autograd import Variable
import math
from HelpFunctions import *

network_save_folder_orig = "./Models"
network_file_name_last = "/last_model"
network_file_name_best = "/best_model"

BAR_LIST_TRAIN = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BarListTrain")
if not os.path.exists(BAR_LIST_TRAIN):
    os.mkdir(BAR_LIST_TRAIN)

if not os.path.exists(network_save_folder_orig):
    os.mkdir(network_save_folder_orig)

delta = 3

fecg_lamda = 100
cent_lamda = 0.001
hinge_lamda = 0.5

mecg_weight = 1.
fecg_weight = 1.
cent_weight = 1.
hinge_weight = 1.

include_mecg_loss = True
include_fecg_loss = True
include_center_loss = True
include_hinge_loss = True

ECG_OUTPUTS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputsTrain")
if not os.path.exists(ECG_OUTPUTS):
    os.mkdir(ECG_OUTPUTS)

ECG_OUTPUTS_VAL = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "ECGOutputsVal")
if not os.path.exists(ECG_OUTPUTS_VAL):
    os.mkdir(ECG_OUTPUTS_VAL)

ECG_OUTPUTS_TEST = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "ECGOutputsTest")
if not os.path.exists(ECG_OUTPUTS_TEST):
    os.mkdir(ECG_OUTPUTS_TEST)

def train(resnet_model,
              train_data_loader_sim,
              optimizer_model,
              criterion,
              criterion_cent,
              epoch,
              epochs,
              train_loss_f_list,
              train_loss_m_list,
              train_loss_average_list):

    total_loss_epoch = 0.
    total_loss_m = 0.
    total_loss_f = 0.
    total_loss_cent = 0.
    total_loss_hinge = 0.

    list_bar_bad_example_noisetype = [0, 0, 0, 0]
    list_bar_good_example_noisetype = [0, 0, 0, 0]
    list_bar_bad_example_snr = [0, 0, 0, 0, 0]
    list_bar_good_example_snr = [0, 0, 0, 0, 0]
    list_bar_bad_example_snrcase = [[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0]]
    list_bar_good_example_snrcase = [[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0]]

    for i, batch_features in enumerate(train_data_loader_sim):
        optimizer_model.zero_grad()
        batch_for_model = Variable(batch_features[0].transpose(1, 2).float().cuda())
        batch_for_m = Variable(batch_features[1].transpose(1, 2).float().cuda())
        batch_for_f = Variable(batch_features[2].transpose(1, 2).float().cuda())
        batch_for_noise_test = batch_features[6].cpu().detach().numpy()
        batch_for_snr_test = batch_features[7].cpu().detach().numpy()
        batch_for_case_test = batch_features[8].cpu().detach().numpy()

        outputs_m, one_before_last_m, outputs_f, one_before_last_f = resnet_model(batch_for_model)

        if epoch + 1 == epochs:
            for j, elem in enumerate(outputs_f):
                corr_f = \
                np.corrcoef(outputs_f.cpu().detach().numpy()[j], batch_for_f.cpu().detach().numpy()[j])[0][1]
                if (corr_f < 0.4):
                    list_bar_bad_example_noisetype[batch_for_noise_test[j]] += 1
                    list_bar_bad_example_snr[batch_for_snr_test[j]] += 1
                    list_bar_bad_example_snrcase[batch_for_snr_test[j]][batch_for_case_test[j]] += 1
                else:
                    list_bar_good_example_noisetype[batch_for_noise_test[j]] += 1
                    list_bar_good_example_snr[batch_for_snr_test[j]] += 1
                    list_bar_good_example_snrcase[batch_for_snr_test[j]][batch_for_case_test[j]] += 1

            path_bar = os.path.join(BAR_LIST_TRAIN, "list_bar_good_example_noisetype")
            np.save(path_bar, np.array(list_bar_good_example_noisetype))
            path_bar = os.path.join(BAR_LIST_TRAIN, "list_bar_bad_example_noisetype")
            np.save(path_bar, np.array(list_bar_bad_example_noisetype))
            path_bar = os.path.join(BAR_LIST_TRAIN, "list_bar_good_example_snr")
            np.save(path_bar, np.array(list_bar_good_example_snr))
            path_bar = os.path.join(BAR_LIST_TRAIN, "list_bar_bad_example_snr")
            np.save(path_bar, np.array(list_bar_bad_example_snr))
            path_bar = os.path.join(BAR_LIST_TRAIN, "list_bar_good_example_snrcase")
            np.save(path_bar, np.array(list_bar_good_example_snrcase))
            path_bar = os.path.join(BAR_LIST_TRAIN, "list_bar_bad_example_snrcase")
            np.save(path_bar, np.array(list_bar_bad_example_snrcase))

            if not os.path.exists(ECG_OUTPUTS):
                os.mkdir(ECG_OUTPUTS)
            path = os.path.join(ECG_OUTPUTS, "ecg_all" + str(i))
            np.save(path, batch_features[0][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS, "label_m" + str(i))
            np.save(path, batch_features[1][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS, "label_f" + str(i))
            np.save(path, batch_features[2][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS, "fecg" + str(i))
            np.save(path, outputs_f[0][0].cpu().detach().numpy())
            path = os.path.join(ECG_OUTPUTS, "mecg" + str(i))
            np.save(path, outputs_m[0][0].cpu().detach().numpy())

        # COST(M,M^)
        train_loss_mecg = criterion(outputs_m, batch_for_m)

        # COST(F,F^)
        train_loss_fecg = criterion(outputs_f, batch_for_f)
        flatten_m, flatten_f = torch.flatten(one_before_last_m, start_dim=1), torch.flatten(one_before_last_f,
                                                                                            start_dim=1)
        hinge_loss = criterion_hinge_loss(one_before_last_m, one_before_last_f, delta)
        batch_size = one_before_last_m.size()[0]
        labels_center_loss = Variable(torch.cat((torch.zeros(batch_size), torch.ones(batch_size))).cuda())
        loss_cent = criterion_cent(torch.cat((flatten_f, flatten_m), 0), labels_center_loss)

        total_loss = mecg_weight * train_loss_mecg + fecg_weight * fecg_lamda * train_loss_fecg
        if include_center_loss:
            total_loss += cent_weight * cent_lamda * loss_cent
        if include_hinge_loss:
            total_loss += hinge_weight * hinge_lamda * hinge_loss

        total_loss.backward()
        optimizer_model.step()

        total_loss_m += mecg_weight * train_loss_mecg.item()
        total_loss_f += fecg_weight * fecg_lamda * train_loss_fecg.item()

        total_loss_cent += cent_weight * cent_lamda * loss_cent.item()
        total_loss_hinge += hinge_weight * hinge_lamda * hinge_loss.item()
        total_loss_epoch += total_loss.item()

    total_loss_m = total_loss_m / (len(train_data_loader_sim.dataset))
    total_loss_f = total_loss_f / (len(train_data_loader_sim.dataset))
    train_loss_f_list.append(total_loss_f)
    train_loss_m_list.append(total_loss_m)
    train_loss_average_list.append((total_loss_m+total_loss_f)/2)

    total_loss_cent = total_loss_cent / (len(train_data_loader_sim.dataset))
    total_loss_hinge = total_loss_hinge / (len(train_data_loader_sim.dataset))
    total_loss_epoch = total_loss_epoch / (len(train_data_loader_sim.dataset))
    # display the epoch training loss
    print("epoch S : {}/{}  total_loss = {:.8f}".format(epoch + 1, epochs, total_loss_epoch))
    if include_mecg_loss:
        print("loss_mecg = {:.8f} ".format(total_loss_m))
    if include_fecg_loss:
        print("loss_fecg = {:.8f} ".format(total_loss_f))
    if include_center_loss:
        print("loss_cent = {:.8f} ".format(total_loss_cent))
    if include_hinge_loss:
        print("loss_hinge = {:.8f} ".format(total_loss_hinge))
    print("\n")
    if epoch + 1 == epochs:
        with open("train_loss_last_epoch.txt", 'w') as f:
            f.write("L1 M = {:.4f},L1 F= {:.4f},LCent = {:.4f},"
                    "LHinge = {:.4f},LTot = {:.4f}\n".format(total_loss_m,
                                                            total_loss_f,
                                                            total_loss_cent,
                                                            total_loss_hinge,
                                                            total_loss_epoch))

def val(val_data_loader_sim,
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
        criterion_cent):
    total_val_loss_m = 0
    total_val_loss_f = 0
    val_loss_m = 0
    val_loss_f = 0
    val_corr_m = 0
    val_corr_f = 0
    with torch.no_grad():
        for i, batch_features in enumerate(val_data_loader_sim):
            batch_for_model_val = Variable(batch_features[0].transpose(1, 2).float().cuda())
            batch_for_m_val = Variable(batch_features[1].transpose(1, 2).float().cuda())
            batch_for_f_val = Variable(batch_features[2].transpose(1, 2).float().cuda())
            outputs_m_val, _, outputs_f_val, _ = resnet_model(batch_for_model_val)
            val_loss_m = criterion(outputs_m_val, batch_for_m_val)
            val_loss_f = criterion(outputs_f_val, batch_for_f_val)

            for j, elem in enumerate(outputs_m_val):
                val_corr_m += \
                np.corrcoef(outputs_m_val.cpu().detach().numpy()[j], batch_for_m_val.cpu().detach().numpy()[j])[0][1]
                val_corr_f += \
                np.corrcoef(outputs_f_val.cpu().detach().numpy()[j], batch_for_f_val.cpu().detach().numpy()[j])[0][1]

            if epoch + 1 == epochs:
                if not os.path.exists(ECG_OUTPUTS_VAL):
                    os.mkdir(ECG_OUTPUTS_VAL)
                path = os.path.join(ECG_OUTPUTS_VAL, "ecg_all" + str(i))
                np.save(path, batch_features[0][0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS_VAL, "label_m" + str(i))
                np.save(path, batch_features[1][0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS_VAL, "label_f" + str(i))
                np.save(path, batch_features[2][0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS_VAL, "fecg" + str(i))
                np.save(path, outputs_f_val[0][0].cpu().detach().numpy() )
                path = os.path.join(ECG_OUTPUTS_VAL, "mecg" + str(i))
                np.save(path, outputs_m_val[0][0].cpu().detach().numpy())
            total_val_loss_m += val_loss_m.item() * mecg_weight
            total_val_loss_f += val_loss_f.item() * fecg_weight * fecg_lamda


    total_val_loss_m /= len(val_data_loader_sim.dataset)
    total_val_loss_f /= len(val_data_loader_sim.dataset)
    val_corr_m /= len(val_data_loader_sim.dataset)
    val_corr_f /= len(val_data_loader_sim.dataset)
    val_corr_average = (val_corr_m + val_corr_f) / 2
    val_loss_average = (total_val_loss_m + total_val_loss_f) / 2

    # saving validation losses
    validation_loss_m_list.append(total_val_loss_m)
    validation_loss_f_list.append(total_val_loss_f)
    validation_loss_average_list.append(val_loss_average)
    validation_corr_m_list.append(val_corr_m)
    validation_corr_f_list.append(val_corr_f)
    if epoch + 1 == epochs:
        with open("val_loss_last_epoch.txt", 'w') as f:
            f.write("L1 M = {:.4f},L1 F= {:.4f},LAvg = {:.4f},"
                    "CorrM = {:.4f},CorrF = {:.4f}, CorrAvg = {:.4f}\n".format(total_val_loss_m,
                                                            total_val_loss_f,
                                                            val_loss_average,
                                                            val_corr_m,
                                                            val_corr_f,
                                                            val_corr_average))
    # saving last model

    torch.save(resnet_model.state_dict(), str(network_save_folder_orig + network_file_name_last))
    # saving best model
    if (val_corr_average > best_model_accuracy):
        best_model_accuracy = val_corr_average
        torch.save(resnet_model.state_dict(), str(network_save_folder_orig + network_file_name_best))
        torch.save(criterion_cent.state_dict(), str(network_save_folder_orig + '/criterion_cent'))
        print("saving best model")
        with open("best_model_epoch.txt", 'w') as f:
            f.write(str(epoch))

    print(
        'Validation: Average loss M: {:.4f}, Average Loss F: {:.4f}, Average Loss M+F: {:.4f}, Correlation M: {:.4f},Correlation F: {:.4f},Correlation Average: {:.4f})\n'.format(
            total_val_loss_m, total_val_loss_f, val_loss_average, val_corr_m, val_corr_f, val_corr_average))
    return best_model_accuracy,val_corr_average


def test(filename,test_data_loader_sim):
    resnet_model = ResNet(1)
    resnet_model.load_state_dict(torch.load(filename))
    resnet_model.eval()

    resnet_model.cuda()

    criterion = nn.L1Loss().cuda()

    test_loss_m = 0
    test_loss_f = 0
    test_corr_m = 0
    test_corr_f = 0

    list_bar_bad_example_noisetype = [0, 0, 0, 0]
    list_bar_good_example_noisetype = [0, 0, 0, 0]
    list_bar_bad_example_snr = [0, 0, 0, 0, 0]
    list_bar_good_example_snr = [0, 0, 0, 0, 0]
    list_bar_bad_example_snrcase = [[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0]]
    list_bar_good_example_snrcase = [[0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0]]

    with torch.no_grad():
        for i, batch_features in enumerate(test_data_loader_sim):
            batch_for_model_test = Variable(batch_features[0].transpose(1, 2).float().cuda())
            batch_for_m_test = Variable(batch_features[1].transpose(1, 2).float().cuda())
            batch_for_f_test = Variable(batch_features[2].transpose(1, 2).float().cuda())
            batch_for_noise_test = batch_features[6].cpu().detach().numpy()
            batch_for_snr_test = batch_features[7].cpu().detach().numpy()
            batch_for_case_test = batch_features[8].cpu().detach().numpy()
            outputs_m_test, _, outputs_f_test, _ = resnet_model(batch_for_model_test)
            test_loss_m += criterion(outputs_m_test, batch_for_m_test)
            test_loss_f += criterion(outputs_f_test, batch_for_f_test)
            for j, elem in enumerate(outputs_m_test):
                corr_m = np.corrcoef(outputs_m_test.cpu().detach().numpy()[j], batch_for_m_test.cpu().detach().numpy()[j])[0][1]
                test_corr_m += corr_m
                corr_f = np.corrcoef(outputs_f_test.cpu().detach().numpy()[j], batch_for_f_test.cpu().detach().numpy()[j])[0][1]
                test_corr_f += corr_f
                if(corr_f < 0.4):
                    list_bar_bad_example_noisetype[batch_for_noise_test[j]] += 1
                    list_bar_bad_example_snr[batch_for_snr_test[j]] += 1
                    list_bar_bad_example_snrcase[batch_for_snr_test[j]][batch_for_case_test[j]] += 1
                else:
                    list_bar_good_example_noisetype[batch_for_noise_test[j]] += 1
                    list_bar_good_example_snr[batch_for_snr_test[j]] += 1
                    list_bar_good_example_snrcase[batch_for_snr_test[j]][batch_for_case_test[j]] += 1

            path = os.path.join(ECG_OUTPUTS_TEST, "ecg_all" + str(i))
            np.save(path, batch_features[0][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS_TEST, "label_m" + str(i))
            np.save(path, batch_features[1][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS_TEST, "label_f" + str(i))
            np.save(path, batch_features[2][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS_TEST, "fecg" + str(i))
            np.save(path, outputs_f_test[0][0].cpu().detach().numpy())
            path = os.path.join(ECG_OUTPUTS_TEST, "mecg" + str(i))
            np.save(path, outputs_m_test[0][0].cpu().detach().numpy())

    test_loss_m /= len(test_data_loader_sim.dataset)
    test_loss_f /= len(test_data_loader_sim.dataset)
    test_loss_average = (test_loss_m + test_loss_f) / 2
    test_corr_m /= len(test_data_loader_sim.dataset)
    test_corr_f /= len(test_data_loader_sim.dataset)
    test_corr_average = (test_corr_m + test_corr_f) / 2

    return test_loss_m, test_loss_f, test_loss_average, test_corr_m, test_corr_f, test_corr_average,\
           list_bar_good_example_noisetype,list_bar_bad_example_noisetype, \
            list_bar_good_example_snr,list_bar_bad_example_snr, \
            list_bar_good_example_snrcase,list_bar_bad_example_snrcase