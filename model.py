import torch
import os
import time
from ResnetNetwork import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

network_file_name_last = "/last_model"

network_file_name_best_sim = "best_model"
network_load_folder_sim = "../MasterAECG+MECG/Models/Ch_19_21_23/mecg2IsShiftedBy1NotNoised"

network_save_folder_real = "./Models/Overfit/NR_10/TrainAllSet"
network_file_name_best_real = "/best_model_real"

fecg_lamda = 10.
save_mecg_orig = False
SAVE_M_EVERY = 3
shift_mecg2 = True
noise_mecg2 = False

# real parameters
ecg_lamda = 1.
ecg_weight = 1.

ECG_OUTPUTS_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputsTrainReal/NR_10/TrainAllSet")
if not os.path.exists(ECG_OUTPUTS_REAL):
    os.mkdir(ECG_OUTPUTS_REAL)

def train_overfit_real(resnet_model,
              train_data_loader_real,
              optimizer_model,
              epoch,
              epochs,
              criterion,
              criterion_cent,
              train_loss_ecg_list,
              scheduler_real,
              best_model_accuracy):

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
    if (total_loss_ecg < best_model_accuracy):
        best_model_accuracy = total_loss_ecg
        torch.save(resnet_model.state_dict(), str(network_save_folder_real + network_file_name_best_real))
        print("saving best model")
        with open("best_model_epoch_real.txt", 'w') as f:
            f.write(str(epoch))
    return best_model_accuracy,total_loss_ecg

    print('{} seconds'.format(time.time() - t0))
    print("\n")



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
