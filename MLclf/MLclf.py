import pickle
import os
import numpy as np
import copy
import random
import torch
import zipfile
import shutil
"""
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
"""

# previous file name was test_miniimagenet.py

class MLclf():
    download_dir = './data_miniimagenet'
    datafile_dir = download_dir + '/miniimagenet/'
    def __init__(self):
        pass
        #self.download_dir = './data_miniimagenet'
        #self.datafile_dir = self.download_dir + '/miniimagenet/'

    @staticmethod
    def miniimagenet_download(Download=False):
        if not os.path.isdir(MLclf.download_dir):
            os.makedirs(MLclf.download_dir)
            Download = True
            print('Downloading dir does not exist, Download = True is enabled.')
        if not os.path.isfile(MLclf.download_dir + '/miniimagenet.zip'):
            Download = True
            print('miniimagenet.zip does not exist, Download = True is enabled.')

        if Download:
            import urllib.request
            print('Starting to download mini-imagenet data zipped file ...')
            url = 'https://data.deepai.org/miniimagenet.zip'
            urllib.request.urlretrieve(url, MLclf.download_dir+'/miniimagenet.zip')
            print('Completed downloading mini-imagenet data zipped file!')
            print('Starting to unzip mini-imagenet data files ...')
            with zipfile.ZipFile(MLclf.download_dir+'/miniimagenet.zip', 'r') as zip_ref:
                zip_ref.extractall(MLclf.download_dir+'/miniimagenet')
            print('Completed unzipping mini-imagenet data files!')
        else:
            if not os.path.isdir(MLclf.download_dir+'/miniimagenet'):
                print('Starting to unzip mini-imagenet data files ...')
                with zipfile.ZipFile(MLclf.download_dir+'/miniimagenet.zip', 'r') as zip_ref:
                    zip_ref.extractall(MLclf.download_dir+'/miniimagenet')
                print('Completed unzipping mini-imagenet data files!')
            else:
                if os.path.isfile(MLclf.download_dir+'/miniimagenet'+'/mini-imagenet-cache-train.pkl') and os.path.isfile(MLclf.download_dir+'/miniimagenet'+'/mini-imagenet-cache-val.pkl') and os.path.isfile(MLclf.download_dir+'/miniimagenet'+'/mini-imagenet-cache-test.pkl'):
                    print('The mini-imagenet pkl files have been existed, no unzipping is needed!')
                else:
                    print('Removing '+MLclf.download_dir+'/miniimagenet ...')
                    shutil.rmtree(MLclf.download_dir+'/miniimagenet')
                    # os.rmdir(download_dir+'/miniimagenet') # raise a error if the folder does not exist or not empty.
                    print('Completed removing '+MLclf.download_dir+'/miniimagenet!')
                    print('Starting to unzip mini-imagenet data files ...')
                    with zipfile.ZipFile(MLclf.download_dir + '/miniimagenet.zip', 'r') as zip_ref:
                        zip_ref.extractall(MLclf.download_dir + '/miniimagenet')
                    print('Completed unzipping mini-imagenet data files!')

        """
        if os.path.isdir(Dir_singleDay):
            Dir_path = Dir_singleDay
            dir_scan = os.listdir(Dir_path)
            for item in dir_scan:
                if not item.endswith('00.dat'):
                    os.remove(os.path.join(Dir_path, item))
    
        if not os.path.isdir(dtype + '_pickle/'):
            os.makedirs(dtype + '_pickle/')
    
        if os.path.isfile("datfiles_nonexist.txt"):
            os.remove("datfiles_nonexist.txt")
        'https://data.deepai.org/miniimagenet.zip'
        """

    @staticmethod
    def miniimagenet_convert2classification(data_dir=None, ratio_train=0.6, ratio_val=0.2, seed_value = None, shuffle=True, save_clf_data=True):

        if seed_value is not None:
            random.seed(seed_value)  # once seed is given, the random generator will be sequential numbers
            np.random.seed(seed_value)

        assert ratio_train > 0 and ratio_val >= 0
        if ratio_train + ratio_val >= 1.0:
            raise ValueError('ratio_train + ratio_val >= 1, please check the values!')

        if data_dir is None:
            data_dir = MLclf.datafile_dir

        dir_pickled_train = data_dir + 'mini-imagenet-cache-train.pkl'
        dir_pickled_val = data_dir + 'mini-imagenet-cache-val.pkl'
        dir_pickled_test = data_dir + 'mini-imagenet-cache-test.pkl'


        with open(dir_pickled_train, 'rb') as f:
            data_load_train = pickle.load(f)
        with open(dir_pickled_val, 'rb') as f:
            data_load_val = pickle.load(f)
        with open(dir_pickled_test, 'rb') as f:
            data_load_test = pickle.load(f)

        #aa.update(bb)
        n_samples_train = data_load_train['image_data'].shape[0]
        n_samples_val = data_load_val['image_data'].shape[0]
        n_samples_test = data_load_test['image_data'].shape[0]

        #data_load_train['class_dict']

        for i, (k, v_ls) in enumerate(data_load_val['class_dict'].items()):
            for idx, v in enumerate(v_ls):
                data_load_val['class_dict'][k][idx] = data_load_val['class_dict'][k][idx] + n_samples_train

        for i, (k, v_ls) in enumerate(data_load_test['class_dict'].items()):
            for idx, v in enumerate(v_ls):
                data_load_test['class_dict'][k][idx] = data_load_test['class_dict'][k][idx] + (n_samples_train + n_samples_val)

        data_load_all = {}
        data_load_all['image_data'] = np.concatenate((data_load_train['image_data'], data_load_val['image_data'], data_load_test['image_data']))
        data_load_all['class_dict'] = copy.deepcopy(data_load_train['class_dict'])
        data_load_all['class_dict'].update(data_load_val['class_dict'])
        data_load_all['class_dict'].update(data_load_test['class_dict'])

        """
        miniimage_unpermutation_new_pkl = data_dir + 'miniimagenet_nonpermutation_nonsplit_new.pkl'
        with open(miniimage_unpermutation_new_pkl, 'wb') as f0:
            pickle.dump(data_load_all, f0)
        """

        """ --------------------------"""
        key1 = list(data_load_all['class_dict'].keys())[0]
        n_samples_per_class = len(data_load_all['class_dict'][key1])
        #print('n_samples_per_class: ', n_samples_per_class) # 600
        n_class = len(list(data_load_all['class_dict'].keys()))
        #print('n_class: ', n_class)

        labels_arr = np.linspace(0, n_class-1, n_class, dtype=int)
        labels_arr = np.repeat(labels_arr, repeats=n_samples_per_class, axis=None)
        data_feature_label = {}
        data_feature_label['labels'] = labels_arr # 100 * 600 labels
        data_feature_label['images'] = copy.deepcopy(data_load_all['image_data']) # 100 * 600 images
        data_feature_label['images_name'] = list(data_load_all['class_dict'].keys()) # 100 class names.


        if shuffle:
            feature_label_zip = list(zip(data_feature_label['images'], data_feature_label['labels']))
            random.shuffle(feature_label_zip)
            data_feature_label['images'], data_feature_label['labels'] = zip(*feature_label_zip)
            data_feature_label['images'], data_feature_label['labels'] = list(data_feature_label['images']), list(data_feature_label['labels'])

        """
        miniimage_feature_label_pkl = data_dir + 'miniimagenet_feature_label_permutatioin_new.pkl'
        with open(miniimage_feature_label_pkl, 'wb') as f1:
            pickle.dump(data_feature_label, f1)
        """

        n_samples_total = len(data_feature_label['labels']) # 60000
        data_feature_label_permutation_split = {}
        data_feature_label_permutation_split['labels_train'] = data_feature_label['labels'][0: int(np.floor(n_samples_total * ratio_train))]
        data_feature_label_permutation_split['labels_val'] = data_feature_label['labels'][int(np.floor(n_samples_total * ratio_train)): int(np.floor(n_samples_total * (ratio_train + ratio_val)))]
        data_feature_label_permutation_split['labels_test'] = data_feature_label['labels'][int(np.floor(n_samples_total * (ratio_train + ratio_val))):]

        data_feature_label_permutation_split['images_train'] = data_feature_label['images'][0: int(np.floor(n_samples_total * ratio_train))]
        data_feature_label_permutation_split['images_val'] = data_feature_label['images'][int(np.floor(n_samples_total * ratio_train)): int(np.floor(n_samples_total * (ratio_train + ratio_val)))]
        data_feature_label_permutation_split['images_test'] = data_feature_label['images'][int(np.floor(n_samples_total * (ratio_train + ratio_val))):]

        data_feature_label_permutation_split['images_name'] = copy.deepcopy(data_feature_label['images_name'])

        data_feature_label_permutation_split['images_train'] = np.array(data_feature_label_permutation_split['images_train'])
        data_feature_label_permutation_split['images_val'] = np.array(data_feature_label_permutation_split['images_val'])
        data_feature_label_permutation_split['images_test'] = np.array(data_feature_label_permutation_split['images_test'])
        data_feature_label_permutation_split['labels_train'] = np.array(data_feature_label_permutation_split['labels_train'])
        data_feature_label_permutation_split['labels_val'] = np.array(data_feature_label_permutation_split['labels_val'])
        data_feature_label_permutation_split['labels_test'] = np.array(data_feature_label_permutation_split['labels_test'])
        data_feature_label_permutation_split['images_name'] = np.array(data_feature_label_permutation_split['images_name'])

        if save_clf_data:
            miniimage_feature_label_permutation_split_pkl = data_dir + 'miniimagenet_feature_label_permutatioin_split_new.pkl'
            with open(miniimage_feature_label_permutation_split_pkl, 'wb') as f2:
                pickle.dump(data_feature_label_permutation_split, f2)
        """ --------------------------"""
        """
    
        # permutation the samples in each class:
        for i, (k, v_ls) in enumerate(data_load_all['class_dict'].items()):
            random.shuffle(v_ls)
            data_load_test['class_dict'][k] = v_ls
    
    
    
        data_all_permutation = {}
    
        data_all_permutation['image_data'] = data_load_all['image_data']
        data_all_permutation['class_dict_train'] = {}
        data_all_permutation['class_dict_val'] = {}
        data_all_permutation['class_dict_test'] = {}
        for i, (k, v_ls) in enumerate(data_load_all['class_dict'].items()):
            data_all_permutation['class_dict_train'][k] = v_ls[0 : int(np.floor(n_samples_per_class * ratio_train))]
            data_all_permutation['class_dict_val'][k]   = v_ls[int(np.floor(n_samples_per_class * ratio_train)) : int(np.floor(n_samples_per_class * (ratio_train + ratio_val)))]
            data_all_permutation['class_dict_test'][k]  = v_ls[int(np.floor(n_samples_per_class * (ratio_train + ratio_val))):]
    
        miniimage_permutation_new_pkl = data_dir + 'miniimagenet_permutation_split_new.pkl'
    
        with open(miniimage_permutation_new_pkl, 'wb') as f3:
            pickle.dump(data_all_permutation, f3)
        """

        return data_feature_label_permutation_split

    @staticmethod
    def to_tensor_dataset(data_feature_label_permutation_split):
        images_train = torch.tensor(data_feature_label_permutation_split['images_train'])
        images_val = torch.tensor(data_feature_label_permutation_split['images_val'])
        images_test = torch.tensor(data_feature_label_permutation_split['images_test'])

        labels_train = torch.tensor(data_feature_label_permutation_split['labels_train'])
        labels_val = torch.tensor(data_feature_label_permutation_split['labels_val'])
        labels_test = torch.tensor(data_feature_label_permutation_split['labels_test'])

        train_dataset = torch.utils.data.TensorDataset(images_train, labels_train)
        val_dataset = torch.utils.data.TensorDataset(images_val, labels_val)
        test_dataset = torch.utils.data.TensorDataset(images_test, labels_test)

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def miniimagenet_clf_dataset(data_dir=None, ratio_train=0.6, ratio_val=0.2, seed_value = None, shuffle=True, save_clf_data=True):
        data_feature_label_permutation_split = MLclf.miniimagenet_convert2classification(data_dir=data_dir, ratio_train=ratio_train, ratio_val=ratio_val, seed_value=seed_value, shuffle=shuffle, save_clf_data=save_clf_data)
        train_dataset, validation_dataset, test_dataset = MLclf.to_tensor_dataset(data_feature_label_permutation_split)
        return train_dataset, validation_dataset, test_dataset


if __name__ == '__main__':
    # clf_data = miniimagenet_clf_data()
    MLclf.miniimagenet_download(Download=False)
    train_dataset, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(ratio_train=0.6, ratio_val=0.2, seed_value=None, shuffle=True, save_clf_data=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=5, shuffle=True, num_workers=0)

    """
    for i, batch in enumerate(train_loader):
        print(i, batch)
        input("Press Enter to continue...")
    """

    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    print(type(trainset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0)
    """

    





