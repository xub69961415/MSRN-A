import numpy as np

def splitDataset(patchesData, patchesLabels, iteration):

    train = {}
    val = {}
    test = {}

    train_index = []
    val_index = []
    test_index = []
    
    gt = patchesLabels.astype(int) + 1
    m = max(gt)
    
    for i in range(m+1):
        index = [j for j,n in enumerate(gt.tolist()) if i==n]
        np.random.shuffle(index)
        if len(index) > 300:
            train[i] = index[:sample]
            val[i] = index[sample:sample+100]
            test[i] = index[sample+100:]
        else:
            train[i] = index[:int(0.2 * len(index))]
            val[i] = index[int(0.2 * len(index)):int(0.3 * len(index))]
            test[i] = index[int(0.3 * len(index)):]
            
        train_index += train[i]
        val_index += val[i]
        test_index += test[i]
        
    np.random.shuffle(train_index)
    np.random.shuffle(val_index)
    np.random.shuffle(test_index)
    '''
    time_str = datetime.datetime.now().strftime('%m_%d_%H_%M')
    
    train_index_name = Dataset + time_str + '@' + str(iteration + 1) + '_train.npy'
    val_index_name = Dataset + time_str + '@' + str(iteration + 1) + '_val.npy'
    test_index_name = Dataset + time_str + '@' + str(iteration + 1) + '_test.npy'

    if not os.path.exists('./index'):
        os.mkdir('./index')
    train_index_path = os.path.join(os.getcwd(),'index', train_index_name)
    val_index_path = os.path.join(os.getcwd(),'index', val_index_name)
    test_index_path = os.path.join(os.getcwd(),'index', test_index_name)
    
    np.save(train_index_path, train_index)
    np.save(val_index_path, val_index)
    np.save(test_index_path, test_index)
    '''
    x_train = patchesData[train_index,:,:,:]
    x_valid = patchesData[val_index,:,:,:]
    x_test = patchesData[test_index,:,:,:]
    
    y_train = patchesLabels[train_index,]
    y_valid = patchesLabels[val_index,]
    y_test = patchesLabels[test_index,]
    
    return x_train, x_valid, x_test, y_train, y_valid, y_test