import os
import time
import datetime
import numpy as np
import pandas as pd
import scipy.io as sio
import sklearn
from sklearn import metrics, preprocessing
from operator import truediv
import keras
from keras.layers import BatchNormalization, Activation, Flatten, Dropout, Dense, Flatten
from keras.layers import concatenate, Add, Reshape, multiply, Lambda, add
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, GlobalAveragePooling3D, GlobalMaxPooling3D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.regularizers import l2
from keras import backend as K
from utils import record, splitDataset

##########//GPU SETTING//##########
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config =  tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
###################################

# define functions
def conv_bn_relu(input_, filter_, kernel_size_, strides_, padding_='same'):
    x = Conv3D(filters=filter_, kernel_size=kernel_size_, strides=strides_,
               padding=padding_, kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
def conv_bn_relu_2d(input_, filter_, kernel_size_, strides_, padding_='same'):
    x = Conv2D(filters=filter_, kernel_size=kernel_size_, strides=strides_,
               padding=padding_, kernel_initializer='he_normal', kernel_regularizer=l2(0.001))(input_)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def ssa_3D(input_sa):
    avg_x = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_sa)
    assert avg_x._keras_shape[-1] == 1
    
    max_x = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_sa)
    assert max_x._keras_shape[-1] == 1
    
    concat = concatenate([avg_x, max_x])
    ssa_refined = Conv3D(filters=1, kernel_size=(3,3,3), strides=(1,1,1),padding='same',
                         activation='hard_sigmoid', kernel_initializer='he_normal')(concat)
    
    return multiply([input_sa, ssa_refined])

def ssa_2D(input_sa):
    avg_x = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_sa)
    assert avg_x._keras_shape[-1] == 1
    
    max_x = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_sa)
    assert max_x._keras_shape[-1] == 1
    
    concat = concatenate([avg_x, max_x])
    ssa_refined = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1),padding='same',
                         activation='hard_sigmoid', kernel_initializer='he_normal')(concat)
    
    return multiply([input_sa, ssa_refined])

def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc    

print('------Importing the HSI Dataset------')
global Dataset
Dataset = 'IN'
# dataset = input('please input the name of Dataset(IN, UP or KSC):')
# Dataset = dataset.upper()
dataset_path = os.path.join(os.getcwd(),'hsi_datasets')
if Dataset == 'IN':
    data = sio.loadmat(os.path.join(dataset_path,'Indian_pines_corrected.mat'))['indian_pines_corrected']
    labels = sio.loadmat(os.path.join(dataset_path,'Indian_pines_gt.mat'))['indian_pines_gt']
    
    CLASS_NUM = np.max(labels)
    
if Dataset == 'UP':
    data = sio.loadmat(os.path.join(dataset_path,'PaviaU.mat'))['paviaU']
    labels = sio.loadmat(os.path.join(dataset_path,'PaviaU_gt.mat'))['paviaU_gt']
    
    CLASS_NUM = np.max(labels)

if Dataset == 'SA':
    data = sio.loadmat(os.path.join(dataset_path,'Salinas_corrected.mat'))['salinas_corrected']
    labels = sio.loadmat(os.path.join(dataset_path,'Salinas_gt.mat'))['salinas_gt']
    
    CLASS_NUM = np.max(labels)
'''
if Dataset == 'PC':
    data = sio.loadmat(os.path.join(dataset_path,'Pavia.mat'))['pavia']
    labels = sio.loadmat(os.path.join(dataset_path,'Pavia_gt.mat'))['pavia_gt']
    
    CLASS_NUM = np.max(labels)
    
if Dataset == 'KSC':
    data = sio.loadmat(os.path.join(dataset_path,'KSC.mat'))['KSC']
    labels = sio.loadmat(os.path.join(dataset_path,'KSC_gt.mat'))['KSC_gt']
    
    CLASS_NUM = np.max(labels)

if Dataset == 'BW':
    data = sio.loadmat(os.path.join(dataset_path,'Botswana.mat'))['Botswana']
    labels = sio.loadmat(os.path.join(dataset_path,'Botswana_gt.mat'))['Botswana_gt']
    
    CLASS_NUM = np.max(labels)
'''

print('The shape of the HSI data is: ', data.shape)
print('The shape of the ground truth is:', labels.shape)
print('The class numbers of the HSI data is:', CLASS_NUM)

print('------Importing Parameters------')
# size_list = [9,11,13,15,17]
PATCH_SIZE = 15
ITER = 10
DROPOUT = 0.5
LR = 0.0003
BATCH_SIZE = 16
EPOCHS = 100

# for PATCH_SIZE in size_list:
print('------Creating Patches------')
print("patch size: ", PATCH_SIZE)
# standardization
data_scaled = preprocessing.scale(data.reshape(-1,data.shape[2]))
data_scaled = data_scaled.reshape(data.shape[0], data.shape[1], data.shape[2])
# create patches
def padWithZeros(data, margin):
    data_with_zeros = np.zeros((data.shape[0] + 2 * margin, data.shape[1] + 
                    2* margin, data.shape[2])) 
    x_offset = margin
    y_offset = margin
    data_with_zeros[x_offset:data.shape[0] + x_offset, 
                    y_offset:data.shape[1] + y_offset, :] = data  
    return data_with_zeros

def createPatches(data, labels, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedData = padWithZeros(data, margin=margin)
    # split patches
    patchesData = np.zeros((data.shape[0] * data.shape[1], windowSize, windowSize, data.shape[2]))
    patchesLabels = np.zeros((data.shape[0] * data.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedData.shape[0] - margin):
        for c in range(margin, zeroPaddedData.shape[1] - margin):
            patch = zeroPaddedData[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch    
            patchesLabels[patchIndex] = labels[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

patchesData, patchesLabels = createPatches(data_scaled, labels, windowSize=PATCH_SIZE, removeZeroLabels = True)
patchesData = patchesData.reshape(patchesData.shape[0], patchesData.shape[1], patchesData.shape[2],
                                patchesData.shape[3],1)
print('The shape of the HSI data after creating patches is :', patchesData.shape)
print('The shape of the ground truth after creating patches is :', patchesLabels.shape)

def our_model():
    FILTER = 32
    inputs = keras.Input(shape = [patchesData.shape[1], patchesData.shape[2], patchesData.shape[3], patchesData.shape[4]])
    conv_1 = conv_bn_relu(inputs, filter_=FILTER, kernel_size_=(3,3,7), strides_=(1,1,2), padding_='valid')
    sa_1 = ssa_3D(conv_1)
    # 3D
    branch_1 = conv_bn_relu(sa_1, filter_=FILTER, kernel_size_=(3,3,3), strides_=(1,1,1))
    branch_2 = conv_bn_relu(sa_1, filter_=FILTER, kernel_size_=(3,3,5), strides_=(1,1,1))
    branch_3 = conv_bn_relu(sa_1, filter_=FILTER, kernel_size_=(3,3,7), strides_=(1,1,1))

    concat_1 = concatenate([branch_1, branch_2, branch_3])
    conv_2 = conv_bn_relu(concat_1, filter_=FILTER, kernel_size_=(1,1,1), strides_=(1,1,1))
    
    # ca_1 = ca_3D(conv_2, reduction=4)

    add_1 = Add()([conv_1, conv_2])
    add_1 = Activation('relu')(add_1)

    # transition
    conv_3 = conv_bn_relu(add_1, filter_=FILTER, kernel_size_=(1,1,add_1._keras_shape[-2]), strides_=(1,1,1), padding_='valid')
    conv_3 = Reshape((conv_3._keras_shape[1], conv_3._keras_shape[2], conv_3._keras_shape[4]))(conv_3)
    sa_2 = ssa_2D(conv_3)
    # 2D
    branch_2_1 = conv_bn_relu_2d(sa_2, filter_=FILTER, kernel_size_=(1,1), strides_=(1,1))
    branch_2_2 = conv_bn_relu_2d(sa_2, filter_=FILTER, kernel_size_=(3,3), strides_=(1,1))
    branch_2_3 = conv_bn_relu_2d(sa_2, filter_=FILTER, kernel_size_=(5,5), strides_=(1,1))

    concat_2 = concatenate([branch_2_1, branch_2_2, branch_2_3])
    conv_4 = conv_bn_relu_2d(concat_2, filter_=FILTER, kernel_size_=(1,1), strides_=(1,1))
    
    # ca_2 = ca_2D(conv_4, reduction=4)
    
    add_2 = Add()([conv_3, conv_4])
    add_2 = Activation('relu')(add_2)

    pool = AveragePooling2D(pool_size=[add_2._keras_shape[1]-4, add_2._keras_shape[2]-4], strides=[1,1])(add_2)
    
    fla = Flatten()(pool)
    drop = Dropout(rate=DROPOUT)(fla)
    output = Dense(CLASS_NUM, activation='softmax', kernel_initializer='he_normal')(drop)
    model = keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=keras.optimizers.RMSprop(lr=LR),loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# model = our_model()
# model.summary()

KAPPA = []
OA = []
AA = []
ELEMENT_ACC = np.zeros((ITER, CLASS_NUM))

TRAINING_TIME = []
TESTING_TIME = []
# kernelList = [16,20,24,28,32]
# for kernelNum in kernelList:
SAMPLES = [50,100,150,200]
for sample in SAMPLES:
    for iter_ in range(ITER):
        time_str = datetime.datetime.now().strftime('%m_%d_%H_%M')
        print("------Load Dataset------")
        x_train, x_valid, x_test, y_train, y_valid, y_test = splitDataset(patchesData, patchesLabels, iteration=iter_)
        y_train = keras.utils.to_categorical(y_train, num_classes=CLASS_NUM)
        y_valid = keras.utils.to_categorical(y_valid, num_classes=CLASS_NUM)
        y_test = keras.utils.to_categorical(y_test, num_classes=CLASS_NUM)
        print('x_train.shape: ', x_train.shape)
        print('x_valid.shape: ', x_valid.shape)
        print('x_test.shape: ', x_test.shape)
        print('y_train.shape: ', y_train.shape)
        print('y_valid.shape: ', y_valid.shape)
        print('y_test.shape: ', y_test.shape)
        print("------Starting the %d Iteration------" % (iter_ + 1))
        best_model_name = Dataset + '_MSRN_' + time_str + '@' + str(iter_ + 1) + '.hdf5'

        if not os.path.exists('./best_model'):
            os.mkdir('./best_model')
        best_model_path = os.path.join(os.getcwd(),'best_model', best_model_name)

        print('------Training the model------')
        model_mssan = our_model()

        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min'),
                    # keras.callbacks.ModelCheckpoint(best_model_path, save_best_only = True),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                                    verbose=1, mode='min', min_lr=0)]
        tic1 = time.clock()
        history_mssan = model_mssan.fit(x_train, y_train, 
                        validation_data = (x_valid, y_valid),
                        batch_size=BATCH_SIZE, epochs=EPOCHS,
                        callbacks = callbacks,
                        shuffle=True)
        toc1 = time.clock()

        tic2 = time.clock()
        loss_and_metrics = model_mssan.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
        toc2 = time.clock()

        print('Training Time: ', toc1 - tic1)
        print('Test time:', toc2 - tic2)
        print('Test loss:', loss_and_metrics[0])
        print('Test accuracy:', loss_and_metrics[1])
        
        pred_test = model_mssan.predict(x_test).argmax(axis=1)

        overall_accuracy = metrics.accuracy_score(np.argmax(y_test, axis=1), pred_test) * 100
        confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), pred_test)
        each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
        kappa_score = metrics.cohen_kappa_score(np.argmax(y_test, axis=1), pred_test)

        KAPPA.append(kappa_score)
        OA.append(overall_accuracy)
        AA.append(average_acc)
        TRAINING_TIME.append(toc1 - tic1)
        TESTING_TIME.append(toc2 - tic2)
        ELEMENT_ACC[iter_, :] = each_acc

    if not os.path.exists('./records'):
            os.mkdir('./records')
    record_path = os.path.join(os.getcwd(),'records',Dataset+'_MSRN_'+time_str+'.txt')
    record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME, record_path)
