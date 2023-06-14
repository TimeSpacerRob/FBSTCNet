# Demo of the FBSTCNet-M network on the SEED dataset.

# Organized SEED data structure:
# File name: Dataset1_sub(#subject)_s(#session).mat
# Data structure: 
#               data: 1 × 15 cell, each element contains an array of channels × timepoints; 
#               fs = 200;
#               label: 1 × 15 double.   (1 : positive, 2 : negative, 3 : neutral)

# Data from each SEED subset was first divided into five equalsized folds in chronological order. 
# The first three folds were then used as the training set, and the remaining two folds were used for testing.

# Reference:
# "W. Huang, W. Wang, Y. Li, W. Wu. FBSTCNet: A Spatio-Temporal Convolutional Network Integrating Power and Connectivity Features for EEG-Based Emotion Decoding. 2023. (under review)"
# 
# Email: huangwch96@gmail.com

import sys
import os
import time
from datetime import datetime
import scipy.io as scio
import numpy as np
from torch.utils.data import Dataset
import torch
import logging
import csv

from PowerAndConneMixedNet import PowerAndConneMixedNet
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from braindecode import EEGClassifier
from torch.optim import AdamW
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds
from braindecode.models import get_output_shape
from sklearn.metrics import confusion_matrix

def WindowCutting(X, window_length):
    numChannel, numPoint = X.shape
    numSamples = int(numPoint/window_length)
    Samples = np.zeros([numSamples,numChannel,window_length],dtype='float32')
    for i in range(numSamples):
        Samples[i,:,:] = X[:,window_length*i:window_length*(i+1)]
    return Samples,numSamples

class SEED_DATASET(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), "n_samples dimension mismatch"
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]



now = datetime.now()
timestr = now.strftime("%Y%m%d%H%M")
dir = os.getcwd() + '/Results/Dat1_FBSTCNet-M_' + timestr
if ~os.path.exists(dir):
    os.makedirs(dir)
f_log = open(dir+'/log.txt',"w+")
sys.stdout = f_log
datapath = '../Data/emotion/'

for SubID in range(1,16):
    for SessionID in range(1,4):
        print('...... | Subject: %d | Session: %d | ......' % (SubID, SessionID))
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Load and split dataset...")
        filename = 'Dataset1_sub' + str(SubID) + '_s' + str(SessionID) + '.mat'


        data = scio.loadmat(datapath + filename)
        label = data['label'][0] - 1
        index_pos = np.where(label == 0)[0]
        index_neg = np.where(label == 1)[0]
        index_neu = np.where(label == 2)[0]
        index_all = np.concatenate((index_pos, index_neg, index_neu), axis=0)
        sfreq = int(data['fs'][0])
        n_classes = 3
        input_window_time = 5
        n_epochs = 50
        input_window_samples = sfreq * input_window_time
        ConfM = np.zeros([n_epochs,3, 3])
        ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
                    'FC5',
                    'FC3', 'FC1', 'FCZ',
                    'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5',
                    'CP3',
                    'CP1', 'CPZ', 'CP2',
                    'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3',
                    'POZ',
                    'PO4', 'PO6', 'PO8',
                    'CB1', 'O1', 'OZ', 'O2', 'CB2']
        index_test = np.concatenate((index_pos[[3, 4]], index_neg[[3, 4]], index_neu[[3, 4]]), axis=0)
        index_cv = np.setdiff1d(index_all, index_test)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Cross-validation start...")
        start_time_cv = time.perf_counter()
        device = 'cpu'

        torch.set_default_tensor_type('torch.FloatTensor')
        for ifold in range(0, 3):
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print('Fold %d' % (ifold + 1))
            index_valid = np.concatenate(([index_pos[ifold]], [index_neg[ifold]], [index_neu[ifold]]), axis=0)
            index_train = np.setdiff1d(index_cv, index_valid)


            for itrial in range(len(index_train)):
                if itrial == 0:
                    X_train, nTrial = WindowCutting(data['data'][0][index_train[itrial]], input_window_samples)
                    Y_train = label[index_train[itrial]] * np.ones(nTrial, dtype=int)
                else:
                    X_tmp, nTrial = WindowCutting(data['data'][0][index_train[itrial]], input_window_samples)
                    X_train = np.concatenate((X_train, X_tmp), axis=0)
                    Y_train = np.concatenate((Y_train, label[index_train[itrial]] * np.ones(nTrial, dtype=int)))
            Dataset_Train = SEED_DATASET(X_train,Y_train)

            for itrial in range(len(index_valid)):
                if itrial == 0:
                    X_test, nTrial = WindowCutting(data['data'][0][index_valid[itrial]], input_window_samples)
                    Y_test = label[index_valid[itrial]] * np.ones(nTrial, dtype=int)
                else:
                    X_tmp, nTrial = WindowCutting(data['data'][0][index_valid[itrial]], input_window_samples)
                    X_test = np.concatenate((X_test, X_tmp), axis=0)
                    Y_test = np.concatenate((Y_test, label[index_valid[itrial]] * np.ones(nTrial, dtype=int)))
            Dataset_Valid = SEED_DATASET(X_test,Y_test)

            seed = 20220930
            set_random_seeds(seed=seed, cuda=False)

            confusion_mat_group = np.zeros([n_epochs, 3, 3])


            
            n_chans = X_train.shape[1]
            filterRange = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40),
                           (40, 44), (44, 48), (48, 52)]
            model = PowerAndConneMixedNet(
                n_chans,
                n_classes,
                fs=sfreq,
                filterRange=filterRange,
                input_window_samples=input_window_samples,
                same_filters_for_features = False,
            )

            lr = 0.0625 * 0.01
            weight_decay = 0
            batch_size = 16



            clf = EEGClassifier(
                model,
                cropped=True, #cropped decoding
                criterion=CroppedLoss,
                criterion__loss_function=torch.nn.functional.nll_loss,
                optimizer=torch.optim.AdamW,
                train_split=None, 
                optimizer__lr=lr,
                optimizer__weight_decay=weight_decay,
                iterator_train__shuffle=True,
                batch_size=batch_size,
                device=device,
            )


            for iep in range(n_epochs):
                clf.partial_fit(Dataset_Train, y=None, epochs=1)
                y_pred = clf.predict(Dataset_Valid)
                confusion_mat_group[iep,:,:] = confusion_matrix(Dataset_Valid.y, y_pred)
                ConfM[iep,:,:] = ConfM[iep,:,:] + confusion_matrix(Dataset_Valid.y, y_pred)
            print("Saving Results...")

            savename = dir + '/Sub' + str(SubID) + '_s' + str(SessionID) + '_fold' + str(ifold + 1) + '.npz'
            np.savez(savename, confusion_mat_group=confusion_mat_group)
            print("Finish Saving!")

            #torch.save(clf,dir+'/Sub' + str(SubID) + '_S' + str(SessionID) + '_fold' + str(ifold+1) + '_model.pth')

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Finding the best parameters...")

        correct_all = np.zeros(n_epochs)
        for iepc in range(n_epochs):
            correct_all[iepc] = ConfM[iepc][0][0] + ConfM[iepc][1][1]+ ConfM[iepc][2][2]
        best_epoch = np.argmax(correct_all) + 1
        best_correct = np.max(correct_all)

        end_time_cv = time.perf_counter()
        cost_time_cv = end_time_cv - start_time_cv

        savename = dir + '/Result_Sub' + str(SubID) + '_s' + str(SessionID) + '_cv.npz'
        np.savez(savename, ConfM=ConfM,correct_all=correct_all,cost_time_cv=cost_time_cv,best_epoch=best_epoch,best_correct=best_correct)

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Finish cross validation, cost %.5f seconds" % (cost_time_cv))

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Split training and testing dataset...")

        for itrial in range(len(index_cv)):
            if itrial == 0:
                X_train, nTrial = WindowCutting(data['data'][0][index_cv[itrial]], input_window_samples)
                Y_train = label[index_cv[itrial]] * np.ones(nTrial, dtype=int)
            else:
                X_tmp, nTrial = WindowCutting(data['data'][0][index_cv[itrial]], input_window_samples)
                X_train = np.concatenate((X_train, X_tmp), axis=0)
                Y_train = np.concatenate((Y_train, label[index_cv[itrial]] * np.ones(nTrial, dtype=int)))
        Dataset_Train = SEED_DATASET(X_train, Y_train)

        for itrial in range(len(index_test)):
            if itrial == 0:
                X_test, nTrial = WindowCutting(data['data'][0][index_test[itrial]], input_window_samples)
                Y_test = label[index_test[itrial]] * np.ones(nTrial, dtype=int)
            else:
                X_tmp, nTrial = WindowCutting(data['data'][0][index_test[itrial]], input_window_samples)
                X_test = np.concatenate((X_test, X_tmp), axis=0)
                Y_test = np.concatenate((Y_test, label[index_test[itrial]] * np.ones(nTrial, dtype=int)))
        Dataset_Test = SEED_DATASET(X_test, Y_test)

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Start training model...")
        print('Total %d epochs' % (best_epoch))
        start_time_train = time.perf_counter()

        seed = 20220930
        set_random_seeds(seed=seed, cuda=False)


        n_chans = X_train.shape[1]
        filterRange = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40),
                       (40, 44), (44, 48), (48, 52)]

        model = PowerAndConneMixedNet(
            n_chans,
            n_classes,
            fs=sfreq,
            filterRange=filterRange,
            input_window_samples=input_window_samples,
            same_filters_for_features=False,
        )

        lr = 0.0625 * 0.01
        weight_decay = 0
        batch_size = 16

        confusion_mat = np.zeros([3, 3])

        clf = EEGClassifier(
            model,
            cropped=True,
            criterion=CroppedLoss,
            criterion__loss_function=torch.nn.functional.nll_loss,
            optimizer=torch.optim.AdamW,
            train_split=None,  
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            iterator_train__shuffle=True,
            batch_size=batch_size,
            device=device,
        )
        clf.fit(Dataset_Train, y=None, epochs=best_epoch)

        end_time_train = time.perf_counter()
        cost_time_train = end_time_train - start_time_train

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print('Finish training,cost %.5f seconds' % (cost_time_train))
        print("Saving model...")

        torch.save(model, dir + '/Sub' + str(SubID) + '_S' + str(SessionID) + '_train_model.pth')
        torch.save(clf, dir + '/Sub' + str(SubID) + '_S' + str(SessionID) + '_train_classifier.pth')

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Start prediction...")
        start_time_test = time.perf_counter()

        y_pred = clf.predict(Dataset_Test)
        confusion_mat = confusion_matrix(Dataset_Test.y, y_pred)

        test_acc = (confusion_mat[0,0] + confusion_mat[1,1] +confusion_mat[2,2])/np.sum(confusion_mat)
        end_time_test = time.perf_counter()
        cost_time_test = end_time_test - start_time_test

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("Finish prediction, cost %.5f seconds" % (cost_time_test))
        print("Saving results...")

        np.savez(dir + '/Result_Sub' + str(SubID) + '_ses' + str(SessionID) + '_test.npz',
                 test_acc=test_acc, confusion_mat=confusion_mat,
                 cost_time_test=cost_time_test, cost_time_train=cost_time_train)














