import torch
import os
import time
from ResnetNetwork import *
from torch.autograd import Variable

#ECG_OUTPUTS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../MasterAECG+MECG/ECGOutputs/Ch_19_21_23/mecg2IsShiftedBy1NotNoised")
ECG_OUTPUTS_CHECK = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../MasterAECG+MECG/ECGOutputsCheck/Ch_19_21_23/mecg2IsShiftedBy1NotNoised")
ECG_OUTPUTS_VAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../MasterAECG+MECG/ECGOutputsVal/Ch_19_21_23/mecg2IsShiftedBy1NotNoised")
ECG_OUTPUTS_TEST = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../MasterAECG+MECG/ECGOutputsTest/Ch_19_21_23/mecg2IsShiftedBy1NotNoised")

#network_save_folder = "./Models"
#network_save_folder = "../MasterAECG+MECG/Models/Ch_19_21_23/mecg2IsShiftedBy1NotNoised"
network_file_name_last = "/last_model"
network_file_name_best = "/best_model"

network_save_folder_orig = "./Models/Overfit"
#network_file_name_best_sim = "/best_model_sim"
#network_file_name_best_cent = "/criterion_cent"
network_file_name_best_real = "/best_model_real"

delta = 3

fecg_lamda = 10.
cent_lamda = 0.01
hinge_lamda = 0.5

mecg_weight = fecg_weight = 1.
cent_weight = 1.
hinge_weight = 1.

include_mecg_loss = True
include_fecg_loss = True
include_center_loss = False
include_hinge_loss = False

save_mecg_orig = False
SAVE_M_EVERY = 3
shift_mecg2 = True
noise_mecg2 = False

# real parameters
ecg_lamda = 1.
ecg_weight = 1.

ECG_OUTPUTS_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ECGOutputsTrainReal")
if not os.path.exists(ECG_OUTPUTS_REAL):
    os.mkdir(ECG_OUTPUTS_REAL)

ECG_OUTPUTS_VAL_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "ECGOutputsValReal")
if not os.path.exists(ECG_OUTPUTS_VAL_REAL):
    os.mkdir(ECG_OUTPUTS_VAL_REAL)
ECG_OUTPUTS_TEST_REAL = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "ECGOutputsTestReal")


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
        batch_for_model = Variable(1000. * batch_features[0].float().cuda())
        batch_for_m = Variable(1000. * batch_features[1].float())

        # (TODO) OVERFIT_REAL: delete from here to concat. - now we should have a REAL input of 2xn
        if shift_mecg2:
            mecg2 = torch.roll(batch_for_m, 1, 2).cuda()  # size is: torch.Size([32, 1, 1024])

        if noise_mecg2:
            mecg2 = add_noise(batch_for_m, 0.05).cuda()
        # Concatenation:
        batch_for_model_2d = torch.cat((batch_for_model, mecg2), 1)

        outputs_m_orig, one_before_last_m, _, _ = resnet_model(batch_for_model_2d)
        # (TODO) OVERFIT_REAL: now we want to train only the self-supervised part
        outputs_m = outputs_m_orig
        for j, elem in enumerate(outputs_m):
            path = os.path.join(ECG_OUTPUTS_REAL, "ecg_all" + str(i))
            np.save(path, batch_features[0][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS_REAL, "label_m" + str(i))
            np.save(path, batch_features[1][0].cpu().detach().numpy()[:, 0])
            #path = os.path.join(ECG_OUTPUTS_REAL, "label_ecg" + str(j) + str(i) + str(epoch))
            #np.save(path, batch_features[j].cpu().detach().numpy())
            path = os.path.join(ECG_OUTPUTS_REAL, "mecg" + str(i))
            np.save(path, outputs_m[0][0].cpu().detach().numpy() / 1000.)

        # COST(M,M^)
        # (TODO) OVERFIT_REAL: now we want to train only the self-supervised part
        train_loss_mecg = criterion(outputs_m, batch_for_m.cuda())
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

def train(resnet_model,
              train_data_loader_sim,
              optimizer_model,
              optimizer_centloss,
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
    #total_loss_ecg = 0. #TODO add when real data
    total_loss_cent = 0.
    total_loss_hinge = 0.
    t0 = time.time()

    # real_epoch = False #TODO add when real data
    for i, batch_features in enumerate(train_data_loader_sim):
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()

        batch_for_model = Variable(1000. * batch_features[0].transpose(1, 2).float().cuda())
        batch_for_m = Variable(1000. * batch_features[1].transpose(1, 2).float())
        batch_for_f = Variable(1000. * batch_features[2].transpose(1, 2).float().cuda())

        # SHIFT batch_for_m - shouldn't be identical
        #print(batch_for_m[1][:,0:10])
        if shift_mecg2:
            mecg2 = torch.roll(batch_for_m, 1, 2).cuda()  # size is: torch.Size([32, 1, 1024])

        # noise batch_for_m - shouldn't be identical
        if noise_mecg2:
            mecg2 = add_noise(batch_for_m, 0.05).cuda()
        # Concatenation:
        batch_for_model_2d = torch.cat((batch_for_model, mecg2), 1)

        outputs_m_orig, one_before_last_m, outputs_f, one_before_last_f = resnet_model(batch_for_model_2d)
        # Design Change: 2xn input - ADDING TECG source to the output of M_Decoder
        outputs_m = outputs_m_orig + mecg2

        if save_mecg_orig:
            if (epoch + 1) % SAVE_M_EVERY == 0:
                if not os.path.exists(ECG_OUTPUTS_CHECK):
                    os.mkdir(ECG_OUTPUTS_CHECK)
                path = os.path.join(ECG_OUTPUTS_CHECK, "ecg_all" + str(i))
                np.save(path, batch_features[0][0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS_CHECK, "label_m" + str(i))
                np.save(path, batch_features[1][0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS_CHECK, "label_f" + str(i))
                np.save(path, batch_features[2][0].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS_CHECK, "fecg" + str(i))
                np.save(path, outputs_f[0][0].cpu().detach().numpy() / 1000.)
                path = os.path.join(ECG_OUTPUTS_CHECK, "mecg_orig" + str(i))
                np.save(path, outputs_m_orig[0][0].cpu().detach().numpy() / 1000.)
                path = os.path.join(ECG_OUTPUTS_CHECK, "mecg" + str(i))
                np.save(path, outputs_m[0][0].cpu().detach().numpy() / 1000.)
                # TODO: ADD one_before_last_m TO PLOT!!
                #path = os.path.join(ECG_OUTPUTS_CHECK, "one_before_last_f" + str(i))
                #np.save(path, one_before_last_f[0][0].cpu().detach().numpy() / 1000.)

        if epoch + 1 == epochs:
            if not os.path.exists(ECG_OUTPUTS):
                os.mkdir(ECG_OUTPUTS)
            path = os.path.join(ECG_OUTPUTS, "ecg_all" + str(i))
            np.save(path, batch_features[0][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS, "mecg_2" + str(i))
            np.save(path, mecg2[0][0].cpu().detach().numpy())
            path = os.path.join(ECG_OUTPUTS, "label_m" + str(i))
            np.save(path, batch_features[1][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS, "label_f" + str(i))
            np.save(path, batch_features[2][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS, "fecg" + str(i))
            np.save(path, outputs_f[0][0].cpu().detach().numpy() / 1000.)
            path = os.path.join(ECG_OUTPUTS, "mecg" + str(i))
            np.save(path, outputs_m[0][0].cpu().detach().numpy() / 1000.)

        # if not real_epoch: #TODO add when real data
        # COST(M,M^)
        train_loss_mecg = criterion(outputs_m, batch_for_m.cuda())
        #train_loss_mecg = criterion(outputs_m, torch.zeros(batch_for_m.size())).cuda()

        # COST(F,F^)
        train_loss_fecg = criterion(outputs_f, batch_for_f)

        # else: #TODO add when real data
        #   outputs_m += outputs_f
        #   train_loss_ecg = criterion(outputs_m, batch_for_model)

        flatten_m, flatten_f = torch.flatten(one_before_last_m, start_dim=1), torch.flatten(one_before_last_f,
                                                                                            start_dim=1)
        hinge_loss = criterion_hinge_loss(one_before_last_m, one_before_last_f, delta)
        batch_size = one_before_last_m.size()[0]
        labels_center_loss = Variable(torch.cat((torch.zeros(batch_size), torch.ones(batch_size))).cuda())
        loss_cent = criterion_cent(torch.cat((flatten_f, flatten_m), 0).cuda(), labels_center_loss)

        # if not real_epoch: #TODO add when real data
        total_loss = mecg_weight * train_loss_mecg + fecg_weight * fecg_lamda * train_loss_fecg
        if include_center_loss:
            total_loss += cent_weight * cent_lamda * loss_cent
        if include_hinge_loss:
            total_loss += hinge_weight * hinge_lamda * hinge_loss
        # else: #TODO add when real data
        #     total_loss = train_loss_ecg + cent_weight*cent_lamda*loss_cent + hinge_weight*hinge_lamda*hinge_loss #TODO: check lamda for ecg and change loss ecg

        total_loss.backward()
        optimizer_model.step()
        optimizer_centloss.step()

        # if not real_epoch: #TODO add when real data
        total_loss_m += mecg_weight * train_loss_mecg.item()
        total_loss_f += fecg_weight * fecg_lamda * train_loss_fecg.item()

        # else: #TODO add when real data
        #   total_loss_ecg += train_loss_ecg.item()

        total_loss_cent += cent_weight * cent_lamda * loss_cent.item()
        total_loss_hinge += hinge_weight * hinge_lamda * hinge_loss.item()
        total_loss_epoch += total_loss.item()
        batch_features, batch_for_model, batch_for_m, batch_for_f, total_loss, outputs_m, one_before_last_m, \
        outputs_f, one_before_last_f, train_loss_mecg, train_loss_fecg = None, None, None, None, None, None, None, \
                                                                         None, None, None, None

    # compute the epoch training loss
    # if not real_epoch: #TODO add when real data
    total_loss_m = total_loss_m / (len(train_data_loader_sim.dataset))
    total_loss_f = total_loss_f / (len(train_data_loader_sim.dataset))
    train_loss_f_list.append(total_loss_f)
    train_loss_m_list.append(total_loss_m)
    train_loss_average_list.append((total_loss_m+total_loss_f)/2)

    # else: #TODO add when real data
    #    total_loss_ecg = total_loss_ecg / (len(train_data_loader_sim.dataset))

    total_loss_cent = total_loss_cent / (len(train_data_loader_sim.dataset))
    total_loss_hinge = total_loss_hinge / (len(train_data_loader_sim.dataset))
    total_loss_epoch = total_loss_epoch / (len(train_data_loader_sim.dataset))

    # display the epoch training loss
    # if not real_epoch: #TODO add when real data
    print("epoch S : {}/{}  total_loss = {:.8f}".format(epoch + 1, epochs, total_loss_epoch))
    if include_mecg_loss:
        print("loss_mecg = {:.8f} ".format(total_loss_m))
    if include_fecg_loss:
        print("loss_fecg = {:.8f} ".format(total_loss_f))
    if include_center_loss:
        print("loss_cent = {:.8f} ".format(total_loss_cent))
    if include_hinge_loss:
        print("loss_hinge = {:.8f} ".format(total_loss_hinge))

    print('{} seconds'.format(time.time() - t0))
    print("\n")

    # else: #TODO add when real data
    #    print("epoch R : {}/{}, total_loss = {:.8f}, loss_ecg = {:.8f}, loss_cent = {:.8f}, loss_hinge = {:.8f}".format(
    #            epoch + 1, epochs, total_loss_epoch, total_loss_ecg, total_loss_cent, total_loss_hinge))


def add_noise(tensor, percentage):
    i = tensor.size()
    #print("size is:" + str(i))
    # create uniform noise:
    noise = torch.normal(0, 1, i) * percentage
    # add it:
    output = tensor + noise
    return output


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
        best_model_accuracy):
    val_loss_m = 0
    val_loss_f = 0
    val_corr_m = 0
    val_corr_f = 0
    with torch.no_grad():
        for i, batch_features in enumerate(val_data_loader_sim):
            batch_for_model_val = Variable(1000. * batch_features[0].transpose(1, 2).float().cuda())
            batch_for_m_val = Variable(1000. * batch_features[1].transpose(1, 2).float())
            batch_for_f_val = Variable(1000. * batch_features[2].transpose(1, 2).float().cuda())

            # SHIFT batch_for_m - shouldn't be identical
            if shift_mecg2:
                mecg2 = torch.roll(batch_for_m_val, 1, 2).cuda()  # size is: torch.Size([32, 1, 1024])

            # noise batch_for_m - shouldn't be identical
            if noise_mecg2:
                mecg2 = add_noise(batch_for_m_val, 0.05).cuda()

            batch_for_model_val_2d = torch.cat((batch_for_model_val, mecg2), 1)

            outputs_m_orig_test, _, outputs_f_test, _ = resnet_model(batch_for_model_val_2d)
            outputs_m_test = outputs_m_orig_test + mecg2

            val_loss_m += criterion(outputs_m_test, batch_for_m_val.cuda())
            #val_loss_m += criterion(outputs_m_test, torch.zeros(batch_for_m_val.size())).cuda()
            val_loss_f += criterion(outputs_f_test, batch_for_f_val).cuda()
            for j, elem in enumerate(outputs_m_test):
                val_corr_m += \
                np.corrcoef(outputs_m_test.cpu().detach().numpy()[j], batch_for_m_val.cpu().detach().numpy()[j])[0][1]
                val_corr_f += \
                np.corrcoef(outputs_f_test.cpu().detach().numpy()[j], batch_for_f_val.cpu().detach().numpy()[j])[0][1]
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
                np.save(path, outputs_f_test[0][0].cpu().detach().numpy() / 1000.)
                path = os.path.join(ECG_OUTPUTS_VAL, "mecg" + str(i))
                np.save(path, outputs_m_test[0][0].cpu().detach().numpy() / 1000.)
    val_loss_m /= len(val_data_loader_sim.dataset)
    val_loss_f /= len(val_data_loader_sim.dataset)
    val_corr_m /= len(val_data_loader_sim.dataset)
    val_corr_f /= len(val_data_loader_sim.dataset)
    val_corr_average = (val_corr_m + val_corr_f) / 2
    val_loss_average = (val_loss_m + val_loss_f) / 2

    # saving validation losses
    validation_loss_m_list.append(val_loss_m.cpu().detach())
    validation_loss_f_list.append(val_loss_f.cpu().detach())
    validation_loss_average_list.append(val_loss_average.cpu().detach())
    validation_corr_m_list.append(val_corr_m)
    validation_corr_f_list.append(val_corr_f)

    # saving last model
    torch.save(resnet_model.state_dict(), str(network_save_folder + network_file_name_last))
    # saving best model
    if (val_loss_average < best_model_accuracy):
        best_model_accuracy = val_loss_average
        torch.save(resnet_model.state_dict(), str(network_save_folder + network_file_name_best))
        print("saving best model")

    print(
        'Validation: Average loss M: {:.4f}, Average Loss F: {:.4f}, Average Loss M+F: {:.4f}, Correlation M: {:.4f},Correlation F: {:.4f},Correlation Average: {:.4f})\n'.format(
            val_loss_m, val_loss_f, val_loss_average, val_corr_m, val_corr_f, val_corr_average))
    return best_model_accuracy

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
        batch_for_model_val = Variable(1000. * batch_features[0].transpose(1, 2).float().cuda())
        batch_for_m_val = Variable(1000. * batch_features[1].transpose(1, 2).float())

        # (TODO) OVERFIT_REAL: delete from here to concat. - now we should have a REAL input of 2xn
        if shift_mecg2:
            mecg2 = torch.roll(batch_for_model_val, 1, 2).cuda()  # size is: torch.Size([32, 1, 1024])

        if noise_mecg2:
            mecg2 = add_noise(batch_for_model_val, 0.05).cuda()
        # Concatenation:
        batch_for_model_val_2d = torch.cat((batch_for_model_val, mecg2), 1)

        outputs_m_val, one_before_last_m, _, _ = resnet_model(batch_for_model_val_2d)
        # COST(M,M^)
        # (TODO) OVERFIT_REAL: now we want to train only the self-supervised part
        val_loss_mecg = criterion(outputs_m_val, batch_for_m_val.cuda())
        val_loss_ecg = val_loss_mecg

        val_loss_ecg += ecg_weight * ecg_lamda * val_loss_ecg

        for j, elem in enumerate(outputs_m_val):
            val_corr_average += np.corrcoef((outputs_m_val[j]).cpu().detach().numpy(), batch_for_model_val.cpu().detach().numpy()[j])[0][1]
            path = os.path.join(ECG_OUTPUTS_VAL_REAL, "label_ecg" + str(j) + str(i) + str(epoch))
            np.save(path, batch_features[j].cpu().detach().numpy())
            path = os.path.join(ECG_OUTPUTS_VAL_REAL, "mecg" + str(j) + str(i) + str(epoch))
            np.save(path, (outputs_m_val[j]).cpu().detach().numpy())

    val_loss_ecg /= len(val_data_loader_real.dataset)
    val_corr_average /= len(val_data_loader_real.dataset)

    validation_loss_ecg_list.append(val_loss_ecg.cpu().detach())
    validation_corr_ecg_list.append(val_corr_average)

    if epoch + 1 == epochs:
        with open("val_loss_last_epoch.txt", 'w') as f:
            f.write("LECG = {:.4f},CorrECG = {:.4f}".format(val_loss_ecg,val_corr_average))
            torch.save(resnet_model.state_dict(), str(network_save_folder_orig + 'last_model'))
    if (val_corr_average > best_model_accuracy):
        best_model_accuracy = val_corr_average
        torch.save(resnet_model.state_dict(), str(network_save_folder_orig + network_file_name_best_real))
        print("saving best model")
        with open("best_model_epoch_real.txt", 'w') as f:
            f.write(str(epoch))
    print(
    'Validation: Average loss ECG: {:.4f},Correlation Average ECG: {:.4f})\n'.format(
        val_loss_ecg,val_corr_average))
    return best_model_accuracy,val_loss_ecg


def test_overfit_real(filename_real, test_data_loader_real):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model_real = ResNet(2)  # 2 input channels - 1. AECG = MECG_1 + FECG 2. MECG_2
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
            if shift_mecg2:
                mecg2 = torch.roll(batch_for_m_test, 1, 2).cuda()  # size is: torch.Size([32, 1, 1024])

            if noise_mecg2:
                mecg2 = add_noise(batch_for_m_test, 0.05).cuda()
            batch_for_m_test = batch_for_m_test.cuda()

            batch_for_model_test_2d = torch.cat((batch_for_model_test, mecg2), 1)
            outputs_m_test, _, _, _ = resnet_model_real(batch_for_model_test_2d)
            test_loss_m = criterion(outputs_m_test, batch_for_m_test.cuda())
            test_loss_ecg = test_loss_m

            test_loss_ecg += ecg_weight * ecg_lamda * test_loss_ecg

            for j, elem in enumerate(outputs_m_test):
                test_corr_ecg += np.corrcoef((outputs_m_test[j]).cpu().detach().numpy(), batch_for_m_test.cpu().detach().numpy()[j])[0][1]
                if not os.path.exists(ECG_OUTPUTS_TEST_REAL):
                    os.mkdir(ECG_OUTPUTS_TEST_REAL)
                path = os.path.join(ECG_OUTPUTS_TEST_REAL, "label_ecg" + str(i))
                np.save(path, batch_features[j].cpu().detach().numpy()[:, 0])
                path = os.path.join(ECG_OUTPUTS_TEST_REAL, "mecg" + str(i))
                np.save(path, outputs_m_test[j].cpu().detach().numpy() / 1000.)

    test_loss_ecg /= len(test_data_loader_real.dataset)
    test_corr_ecg /= len(test_data_loader_real.dataset)

    return test_loss_ecg,test_corr_ecg


def test(filename,test_data_loader_sim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = ResNet(2)  # 2 input channels - 1. AECG = MECG_1 + FECG 2. MECG_2
    resnet_model = nn.DataParallel(resnet_model)
    resnet_model.to(device)
    resnet_model.load_state_dict(torch.load(filename))
    resnet_model.eval()

    resnet_model.cuda()

    criterion = nn.L1Loss().cuda()

    test_loss_m = 0
    test_loss_f = 0
    test_corr_m = 0
    test_corr_f = 0
    with torch.no_grad():
        for i, batch_features in enumerate(test_data_loader_sim):
            batch_for_model_test = Variable(1000. * batch_features[0].transpose(1, 2).float().cuda())
            batch_for_m_test = Variable(1000. * batch_features[1].transpose(1, 2).float())
            batch_for_f_test = Variable(1000. * batch_features[2].transpose(1, 2).float().cuda())

            # print(batch_for_m[1][:,0:10])
            if shift_mecg2:
                mecg2 = torch.roll(batch_for_m_test, 1, 2).cuda()  # size is: torch.Size([32, 1, 1024])

            if noise_mecg2:
                mecg2 = add_noise(batch_for_m_test, 0.05).cuda()
            batch_for_m_test = batch_for_m_test.cuda()

            batch_for_model_test_2d = torch.cat((batch_for_model_test, mecg2), 1)
            #outputs_m_test, _, outputs_f_test, _ = resnet_model(batch_for_model_test)
            outputs_m_orig_test, _, outputs_f_test, _ = resnet_model(batch_for_model_test_2d)
            outputs_m_test = outputs_m_orig_test + mecg2

            test_loss_m += criterion(outputs_m_test, batch_for_m_test).cuda()
            #test_loss_m += criterion(outputs_m_test.cuda(), torch.zeros(batch_for_m_test.size()).cuda()).cuda()
            test_loss_f += criterion(outputs_f_test, batch_for_f_test)
            for j, elem in enumerate(outputs_m_test):
                test_corr_m += np.corrcoef(outputs_m_test.cpu().detach().numpy()[j], batch_for_m_test.cpu().detach().numpy()[j])[0][1]
                test_corr_f += np.corrcoef(outputs_f_test.cpu().detach().numpy()[j], batch_for_f_test.cpu().detach().numpy()[j])[0][1]
            if not os.path.exists(ECG_OUTPUTS_TEST):
                os.mkdir(ECG_OUTPUTS_TEST)
            path = os.path.join(ECG_OUTPUTS_TEST, "ecg_all" + str(i))
            np.save(path, batch_features[0][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS_TEST, "label_m" + str(i))
            np.save(path, batch_features[1][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS_TEST, "label_f" + str(i))
            np.save(path, batch_features[2][0].cpu().detach().numpy()[:, 0])
            path = os.path.join(ECG_OUTPUTS_TEST, "fecg" + str(i))
            np.save(path, outputs_f_test[0][0].cpu().detach().numpy() / 1000.)
            path = os.path.join(ECG_OUTPUTS_TEST, "mecg" + str(i))
            np.save(path, outputs_m_test[0][0].cpu().detach().numpy() / 1000.)

    test_loss_m /= len(test_data_loader_sim.dataset)
    test_loss_f /= len(test_data_loader_sim.dataset)
    test_loss_average = (test_loss_m + test_loss_f) / 2
    test_corr_m /= len(test_data_loader_sim.dataset)
    test_corr_f /= len(test_data_loader_sim.dataset)
    test_corr_average = (test_corr_m + test_corr_f) / 2

    return test_loss_m,test_loss_f,test_loss_average,test_corr_m,test_corr_f,test_corr_average
