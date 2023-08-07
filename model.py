import torch
import os
import time
from ResnetNetwork import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

network_file_name_last = "/last_model"
network_file_name_best_sim = "best_model_real"
network_load_folder_sim = "./Models/Overfit/NR_07"
network_save_folder_real = "./Models/Overfit/NR_07/Jan02"
network_file_name_best_real = "/best_model_real"

fecg_lamda = 10.
save_mecg_orig = False
SAVE_M_EVERY = 3
shift_mecg2 = True
noise_mecg2 = False

# real parameters
ecg_lamda = 1.
ecg_weight = 1.

ECG_OUTPUTS_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputsTrainReal/NR_07/Jan02")
if not os.path.exists(ECG_OUTPUTS_REAL):
    os.mkdir(ECG_OUTPUTS_REAL)

ECG_OUTPUTS_VAL_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "ECGOutputsValReal/NR_07/Jan02")
if not os.path.exists(ECG_OUTPUTS_VAL_REAL):
    os.mkdir(ECG_OUTPUTS_VAL_REAL)
ECG_OUTPUTS_TEST_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "ECGOutputsTestReal/NR_07/Jan02")


def train_overfit_real(resnet_model,
              train_data_loader_real,
              optimizer_model,
              epoch,
              epochs,
              criterion,
              criterion_cent,
              train_loss_ecg_list,
              scheduler_real):

    total_loss_epoch = 0.
    total_loss_ecg = 0.
    t0 = time.time()

    for i, batch_features in enumerate(train_data_loader_real):
        optimizer_model.zero_grad()
        #print('batch_features[0] size:', batch_features[0].size())

        batch_for_model = Variable(1000. * batch_features[0].float().cuda())
        batch_for_m = Variable(1000. * batch_features[1].float().cuda())

        # OVERFIT_REAL: now we should have a REAL input of 2xn
        # Concatenation:
        batch_for_model_2d = torch.cat((batch_for_model, batch_for_m), 1)

        outputs_m_orig, one_before_last_m, outputs_f, _ = resnet_model(batch_for_model_2d)
        #print('outputs_f size: ',outputs_f.size())

        # Design Change: 2xn input - ADDING TECG source to the output of M_Decoder
        outputs_m = outputs_m_orig + batch_for_m # Delta + TECG

        if epoch + 1 == epochs:
            save_signals(ECG_OUTPUTS_REAL, i, batch_features, outputs_m, outputs_m_orig, outputs_f)

        # COST(M,M^)
        train_loss_mecg = criterion(outputs_m, batch_for_m.cuda())
        # OVERFIT_REAL: now we want to train ONLY the self-supervised part
        train_loss_ecg = train_loss_mecg

        total_loss = ecg_weight * ecg_lamda * train_loss_ecg
        total_loss.backward()
        optimizer_model.step()
        scheduler_real.step()
        total_loss_ecg += train_loss_ecg.item() * ecg_lamda * ecg_weight
        total_loss_epoch += total_loss.item()

    total_loss_ecg = total_loss_ecg / (len(train_data_loader_real.dataset))
    train_loss_ecg_list.append(total_loss_ecg)

    total_loss_epoch = total_loss_epoch / (len(train_data_loader_real.dataset))

    print("epoch S : {}/{}  total_loss = {:.8f}".format(epoch + 1, epochs, total_loss_epoch))
    print("loss_ecg = {:.8f} ".format(total_loss_ecg))
    print("\n")

    if epoch + 1 == epochs:
        with open("train_loss_last_epoch.txt", 'w') as f:
            f.write("L1ECG = {:.4f},LTot = {:.4f}\n".format(total_loss_ecg,
                                                             total_loss_epoch))
    print('{} seconds'.format(time.time() - t0))
    print("\n")


def val_overfit_real(val_data_loader_real,
        resnet_model,
        criterion,
        criterion_cent,
        epoch,
        epochs,
        validation_loss_ecg_list,
        validation_corr_ecg_list,
        best_model_accuracy):

    val_loss_ecg = 0
    val_corr_average = 0

    for i, batch_features in enumerate(val_data_loader_real):
        batch_for_model_val = Variable(1000. * batch_features[0].float().cuda())
        batch_for_m_val = Variable(1000. * batch_features[1].float().cuda())

        # (TODO) OVERFIT_REAL: delete from here to concat. - now we should have a REAL input of 2xn
        # Concatenation:
        batch_for_model_val_2d = torch.cat((batch_for_model_val, batch_for_m_val), 1)

        outputs_m_val_orig, one_before_last_m, outputs_f_val, _ = resnet_model(batch_for_model_val_2d)
        outputs_m_val = outputs_m_val_orig + batch_for_m_val

        # COST(M,M^)
        # (TODO) OVERFIT_REAL: now we want to train only the self-supervised part
        val_loss_mecg = criterion(outputs_m_val, batch_for_m_val.cuda())
        val_loss_ecg = val_loss_mecg

        val_loss_ecg += ecg_weight * ecg_lamda * val_loss_ecg


        for j, elem in enumerate(outputs_m_val):
            val_corr_average += np.corrcoef(outputs_m_val.cpu().detach().numpy()[j], batch_for_m_val.cpu().detach().numpy()[j])[0][1]

        if epoch + 1 == epochs:
            if not os.path.exists(ECG_OUTPUTS_VAL_REAL):
                os.mkdir(ECG_OUTPUTS_VAL_REAL)
            path = os.path.join(ECG_OUTPUTS_VAL_REAL, "ecg_all" + str(i))
            np.save(path, batch_features[0][0].cpu().detach().numpy()[0, :])
            path = os.path.join(ECG_OUTPUTS_VAL_REAL, "label_m" + str(i))
            np.save(path, batch_features[1][0].cpu().detach().numpy()[0, :])
            path = os.path.join(ECG_OUTPUTS_VAL_REAL, "mecg" + str(i))
            np.save(path, outputs_m_val[0][0].cpu().detach().numpy() / 1000.)
            path = os.path.join(ECG_OUTPUTS_VAL_REAL, "fecg" + str(i))
            np.save(path, outputs_f_val[0][0].cpu().detach().numpy() / 1000.)

    val_loss_ecg /= len(val_data_loader_real.dataset)
    val_corr_average /= len(val_data_loader_real.dataset)

    validation_loss_ecg_list.append(val_loss_ecg.cpu().detach())
    validation_corr_ecg_list.append(val_corr_average)

    if epoch + 1 == epochs:
        with open("val_loss_last_epoch.txt", 'w') as f:
            f.write("LECG = {:.4f},CorrECG = {:.4f}".format(val_loss_ecg,val_corr_average))
            torch.save(resnet_model.state_dict(), str(network_save_folder_real+ 'last_model'))
    if (val_loss_ecg < best_model_accuracy):
    #if (val_corr_average > best_model_accuracy):
        best_model_accuracy = val_loss_ecg
        torch.save(resnet_model.state_dict(), str(network_save_folder_real + network_file_name_best_real))
        print("saving best model")
        with open("best_model_epoch_real.txt", 'w') as f:
            f.write(str(epoch))
    print(
    'Validation: Average loss ECG: {:.4f},Correlation Average ECG: {:.4f})\n'.format(
        val_loss_ecg,val_corr_average))
    return best_model_accuracy,val_loss_ecg


def test_overfit_real(filename_real, test_data_loader_real):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model_real = ResNet(2)
    resnet_model_real = nn.DataParallel(resnet_model_real)
    resnet_model_real.to(device)
    resnet_model_real.load_state_dict(torch.load(filename_real))
    resnet_model_real.eval()

    resnet_model_real.cuda()

    criterion = nn.L1Loss().cuda()

    test_loss_ecg = 0
    test_corr_ecg = 0
    with torch.no_grad():
        for i, batch_features in enumerate(test_data_loader_real):
            batch_for_model_test = Variable(batch_features[0].float().cuda())
            batch_for_m_test = Variable(batch_features[1].float().cuda())


            # (TODO) OVERFIT_REAL: delete from here to concat. - now we should have a REAL input of 2xn
            batch_for_m_test = batch_for_m_test.cuda()
            batch_for_model_test_2d = torch.cat((batch_for_model_test, batch_for_m_test), 1)

            outputs_m_test_orig, _, outputs_f_test, _ = resnet_model_real(batch_for_model_test_2d)
            outputs_m_test = outputs_m_test_orig + batch_for_m_test

            test_loss_m = criterion(outputs_m_test, batch_for_m_test.cuda())
            test_loss_ecg = test_loss_m

            test_loss_ecg += ecg_weight * ecg_lamda * test_loss_ecg

            for j, elem in enumerate(outputs_m_test):
                test_corr_ecg += np.corrcoef((outputs_m_test[j]).cpu().detach().numpy(), batch_for_m_test.cpu().detach().numpy()[j])[0][1]
            if not os.path.exists(ECG_OUTPUTS_TEST_REAL):
                os.mkdir(ECG_OUTPUTS_TEST_REAL)
            path = os.path.join(ECG_OUTPUTS_TEST_REAL, "ecg_all" + str(i))
            np.save(path, batch_features[0][0].cpu().detach().numpy()[0, :])
            path = os.path.join(ECG_OUTPUTS_TEST_REAL, "label_m" + str(i))
            np.save(path, batch_features[1][0].cpu().detach().numpy()[0, :])
            path = os.path.join(ECG_OUTPUTS_TEST_REAL, "mecg" + str(i))
            np.save(path, outputs_m_test[0][0].cpu().detach().numpy() / 1000.)
            path = os.path.join(ECG_OUTPUTS_TEST_REAL, "fecg" + str(i))
            np.save(path, outputs_f_test[0][0].cpu().detach().numpy() / 1000.)

    test_loss_ecg /= len(test_data_loader_real.dataset)
    test_corr_ecg /= len(test_data_loader_real.dataset)

    return test_loss_ecg,test_corr_ecg


def save_signals(folder, i, batch_features, outputs_m, outputs_m_orig, outputs_f, SaveOrigMecg = False):
    if not os.path.exists(ECG_OUTPUTS_REAL):
        os.mkdir(ECG_OUTPUTS_REAL)
    path = os.path.join(ECG_OUTPUTS_REAL, "ecg_all_win0_batch" + str(i))
    np.save(path, batch_features[0][0].cpu().detach().numpy()[0, :])
    path = os.path.join(ECG_OUTPUTS_REAL, "ecg_all_win1_batch" + str(i))
    np.save(path, batch_features[0][1].cpu().detach().numpy()[0, :])
    path = os.path.join(ECG_OUTPUTS_REAL, "ecg_all_win2_batch" + str(i))
    np.save(path, batch_features[0][2].cpu().detach().numpy()[0, :])
    path = os.path.join(ECG_OUTPUTS_REAL, "label_m_win0_batch" + str(i))
    np.save(path, batch_features[1][0].cpu().detach().numpy()[0, :])
    path = os.path.join(ECG_OUTPUTS_REAL, "label_m_win1_batch" + str(i))
    np.save(path, batch_features[1][1].cpu().detach().numpy()[0, :])
    path = os.path.join(ECG_OUTPUTS_REAL, "label_m_win2_batch" + str(i))
    np.save(path, batch_features[1][2].cpu().detach().numpy()[0, :])
    path = os.path.join(ECG_OUTPUTS_REAL, "mecg_win0_batch" + str(i))
    np.save(path, outputs_m[0][0].cpu().detach().numpy() / 1000.)
    path = os.path.join(ECG_OUTPUTS_REAL, "mecg_win1_batch" + str(i))
    np.save(path, outputs_m[1][0].cpu().detach().numpy() / 1000.)
    path = os.path.join(ECG_OUTPUTS_REAL, "mecg_win2_batch" + str(i))
    np.save(path, outputs_m[2][0].cpu().detach().numpy() / 1000.)
    if SaveOrigMecg:
        path = os.path.join(ECG_OUTPUTS_REAL, "mecg_orig_win0_batch" + str(i))
        np.save(path, outputs_m_orig[0][0].cpu().detach().numpy() / 1000.)
        path = os.path.join(ECG_OUTPUTS_REAL, "mecg_orig_win1_batch" + str(i))
        np.save(path, outputs_m_orig[1][0].cpu().detach().numpy() / 1000.)
        path = os.path.join(ECG_OUTPUTS_REAL, "mecg_orig_win2_batch" + str(i))
        np.save(path, outputs_m_orig[2][0].cpu().detach().numpy() / 1000.)
    path = os.path.join(ECG_OUTPUTS_REAL, "fecg_win0_batch" + str(i))
    np.save(path, outputs_f[0][0].cpu().detach().numpy() / 1000.)
    path = os.path.join(ECG_OUTPUTS_REAL, "fecg_win1_batch" + str(i))
    np.save(path, outputs_f[1][0].cpu().detach().numpy() / 1000.)
    path = os.path.join(ECG_OUTPUTS_REAL, "fecg_win2_batch" + str(i))
    np.save(path, outputs_f[2][0].cpu().detach().numpy() / 1000.)

    for j in range(3):
        fecg_shifted = shift((outputs_f[j][0].cpu().detach().numpy() / 1000.), -248, cval=0) + shift((outputs_f[j+1][0].cpu().detach().numpy() / 1000.), 776, cval=0)
        path = os.path.join(ECG_OUTPUTS_REAL, "fecg_shifted_win"+ str(j) + "_batch" + str(i))
        np.save(path, fecg_shifted)


def add_noise(tensor, percentage):
    i = tensor.size()
    #print("size is:" + str(i))
    # create uniform noise:
    noise = torch.normal(0, 1, i) * percentage
    # add it:
    output = tensor + noise
    return output
