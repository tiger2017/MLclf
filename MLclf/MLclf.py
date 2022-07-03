import pickle
import os
import numpy as np
import copy
import random
import torch
import zipfile
import shutil
import torchvision.transforms as transforms
from PIL import Image
from numpy import loadtxt
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
    labels_mark = {}
    marks_to_labels = {'mini-imagenet': None, 'tiny-imagenet': None}
    labels_to_marks = {'mini-imagenet': None, 'tiny-imagenet': None}
    download_dir_tinyIN = './data_tinyimagenet'
    datafile_dir_tinyIN = download_dir_tinyIN + '/tiny-imagenet-200/'
    # labels_mark_tinyIN = None
    # marks_to_labels_tinyIN = None
    # labels_to_marks_tinyIN = None

    def __init__(self):
        pass
        # self.download_dir = './data_miniimagenet's
        # self.datafile_dir = self.download_dir + '/miniimagenet/'

    @staticmethod
    def miniimagenet_download(Download=False):
        if not os.path.isdir(MLclf.download_dir):
            os.makedirs(MLclf.download_dir)
            Download = True
            print('Downloading dir does not exist, Download = True is enabled.')
        if not os.path.isfile(MLclf.download_dir + '/miniimagenet.zip'):
            Download = True
            print('mini-imagenet.zip does not exist, Download = True is enabled.')

        if Download:
            import urllib.request
            print('Starting to download mini-imagenet data zipped file ...')
            url = 'https://data.deepai.org/miniimagenet.zip'
            urllib.request.urlretrieve(url, MLclf.download_dir+'/miniimagenet.zip')
            print('Completed downloading mini-imagenet data zipped file!')
            print('Starting to unzip mini-imagenet data files 1...')
            with zipfile.ZipFile(MLclf.download_dir+'/miniimagenet.zip', 'r') as zip_ref:
                zip_ref.extractall(MLclf.download_dir+'/miniimagenet')
            print('Completed unzipping mini-imagenet data files!')
        else:
            if not os.path.isdir(MLclf.download_dir+'/miniimagenet'):
                print('Starting to unzip mini-imagenet data files 2...')
                try:
                    with zipfile.ZipFile(MLclf.download_dir+'/miniimagenet.zip', 'r') as zip_ref:
                        zip_ref.extractall(MLclf.download_dir+'/miniimagenet')
                    print('Completed unzipping mini-imagenet data files!')
                except:
                    print('zip file does not work, being re-downloading ...')
                    import urllib.request
                    url = 'https://data.deepai.org/miniimagenet.zip'
                    urllib.request.urlretrieve(url, MLclf.download_dir + '/miniimagenet.zip')
                    print('Completed downloading mini-imagenet data zipped file!')
                    print('Starting to unzip mini-imagenet data files 1...')
                    with zipfile.ZipFile(MLclf.download_dir + '/miniimagenet.zip', 'r') as zip_ref:
                        zip_ref.extractall(MLclf.download_dir + '/miniimagenet')
                    print('Completed unzipping mini-imagenet data files!')
            else:
                if os.path.isfile(MLclf.download_dir+'/miniimagenet'+'/mini-imagenet-cache-train.pkl') and os.path.isfile(MLclf.download_dir+'/miniimagenet'+'/mini-imagenet-cache-val.pkl') and os.path.isfile(MLclf.download_dir+'/miniimagenet'+'/mini-imagenet-cache-test.pkl'):
                    print('The mini-imagenet pkl files have been existed, no unzipping is needed!')
                else:
                    print('Removing '+MLclf.download_dir+'/miniimagenet ...')
                    shutil.rmtree(MLclf.download_dir+'/miniimagenet')
                    # os.rmdir(download_dir+'/miniimagenet') # raise a error if the folder does not exist or not empty.
                    print('Completed removing '+MLclf.download_dir+'/miniimagenet!')
                    print('Starting to unzip mini-imagenet data files 3...')
                    try:
                        with zipfile.ZipFile(MLclf.download_dir + '/miniimagenet.zip', 'r') as zip_ref:
                            zip_ref.extractall(MLclf.download_dir + '/miniimagenet')
                        print('Completed unzipping mini-imagenet data files!')
                    except:
                        print('zip file does not work, being re-downloading ...')
                        import urllib.request
                        url = 'https://data.deepai.org/miniimagenet.zip'
                        urllib.request.urlretrieve(url, MLclf.download_dir + '/miniimagenet.zip')
                        print('Completed downloading mini-imagenet data zipped file!')
                        print('Starting to unzip mini-imagenet data files 1...')
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
    def miniimagenet_convert2classification(data_dir=None, ratio_train=0.6, ratio_val=0.2, seed_value=None, shuffle=True, save_clf_data=True, task_type='raw', transform=None):

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

        if task_type == 'raw':
            import os
            current_dir = os.path.dirname(os.path.realpath(__file__))
            print('The raw mini-imagenet datasets has been downloaded in the directory: ' + current_dir + data_dir[1:] +' , where you can find them!')
            return data_load_train, data_load_val, data_load_test

        # print('data_load_train: ', data_load_train, np.shape(data_load_train['image_data'][0]))
        #aa.update(bb)
        n_samples_train = data_load_train['image_data'].shape[0]
        n_samples_val = data_load_val['image_data'].shape[0]
        n_samples_test = data_load_test['image_data'].shape[0]

        # data_load_train['class_dict']
        # calculate the correct index for combine the ['class_dict']s of data_load_train, val and test.
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

        labels_arr_unique = np.linspace(0, n_class-1, n_class, dtype=int)
        labels_arr = np.repeat(labels_arr_unique, repeats=n_samples_per_class, axis=None)
        data_feature_label = {}
        data_feature_label['labels'] = labels_arr # 100 * 600 labels
        data_feature_label['images'] = copy.deepcopy(data_load_all['image_data']) # 100 * 600 images
        data_feature_label['labels_mark'] = list(data_load_all['class_dict'].keys()) # 100 class names.

        data_feature_label['images'] = np.array(data_feature_label['images'])
        data_feature_label['labels_mark'] = np.array(data_feature_label['labels_mark'])

        ## if transform is not None:
        data_feature_label['images'] = MLclf._feature_norm(data_feature_label['images'], transform=transform)

        if shuffle:
            feature_label_zip = list(zip(data_feature_label['images'], data_feature_label['labels']))
            random.shuffle(feature_label_zip)
            data_feature_label['images'], data_feature_label['labels'] = zip(*feature_label_zip)
            data_feature_label['images'], data_feature_label['labels'] = np.array(data_feature_label['images']), np.array(data_feature_label['labels'])

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

        data_feature_label_permutation_split['labels_mark'] = copy.deepcopy(data_feature_label['labels_mark'])
        """
        data_feature_label_permutation_split['images_train'] = np.array(data_feature_label_permutation_split['images_train'])
        data_feature_label_permutation_split['images_val'] = np.array(data_feature_label_permutation_split['images_val'])
        data_feature_label_permutation_split['images_test'] = np.array(data_feature_label_permutation_split['images_test'])
        data_feature_label_permutation_split['labels_train'] = np.array(data_feature_label_permutation_split['labels_train'])
        data_feature_label_permutation_split['labels_val'] = np.array(data_feature_label_permutation_split['labels_val'])
        data_feature_label_permutation_split['labels_test'] = np.array(data_feature_label_permutation_split['labels_test'])
        data_feature_label_permutation_split['labels_mark'] = np.array(data_feature_label_permutation_split['labels_mark'])
        """
        MLclf.labels_mark['mini-imagenet'] = data_feature_label_permutation_split['labels_mark']
        MLclf.marks_to_labels['mini-imagenet'] = {}
        MLclf.labels_to_marks['mini-imagenet'] = {}
        for l, l_mark in zip(labels_arr_unique, MLclf.labels_mark['mini-imagenet']):
            MLclf.labels_to_marks['mini-imagenet'][l] = l_mark
        for l_mark, l in zip(MLclf.labels_mark['mini-imagenet'], labels_arr_unique):
            MLclf.marks_to_labels['mini-imagenet'][l_mark] = l


        if save_clf_data:
            miniimage_feature_label_permutation_split_pkl = data_dir + 'miniimagenet_feature_label_permutatioin_split_new.pkl'
            with open(miniimage_feature_label_permutation_split_pkl, 'wb') as f2:
                pickle.dump(data_feature_label_permutation_split, f2)
        """ --------------------------"""
        """
        from requests import get
        import socket
        hostname = socket.gethostname()
        getadd = get('https://api.ipify.org').content.decode('utf8')
        """

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
    def _feature_norm(feature, transform=None):
        """
        This function transform the dimension of feature from (batch_size, H, W, C) to (batch_size, C, H, W).
        :param feature: feature / mini-imagenet's images.
        :return: transformed feature.
        """
        #
        if transform is None:
            # convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
            transform = transforms.Compose([transforms.ToTensor()])
            print('The argument transform is None, so only tensor converted and normalization b/t [0,1] is done!')
            # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            # print('transforms is predefined as None, so the default transforms is called: ', transform)
        else:
            transform = transform
        feature_shape = np.shape(feature)
        feature_output = torch.empty((feature_shape[0], feature_shape[3], feature_shape[1], feature_shape[2]))
        for i, feature_i in enumerate(feature):
            feature_output[i] = transform(feature_i)
            # feature is a tensor here.
        # print('type(feature_output): ', type(feature_output))
        return feature_output.numpy()


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
    def miniimagenet_clf_dataset(data_dir=None, ratio_train=0.6, ratio_val=0.2, seed_value = None, shuffle=True, save_clf_data=True, transform=None):
        data_feature_label_permutation_split = MLclf.miniimagenet_convert2classification(data_dir=data_dir, ratio_train=ratio_train, ratio_val=ratio_val, seed_value=seed_value, shuffle=shuffle, task_type='classical_or_meta', save_clf_data=save_clf_data, transform=transform)

        train_dataset, validation_dataset, test_dataset = MLclf.to_tensor_dataset(data_feature_label_permutation_split)
        labels_mark = data_feature_label_permutation_split['labels_mark']
        return train_dataset, validation_dataset, test_dataset # , labels_mark

    @staticmethod
    def miniimagenet_data_raw(data_dir=None):
        data_raw_train, data_raw_val, data_raw_test = MLclf.miniimagenet_convert2classification(data_dir=data_dir, task_type='raw')
        return data_raw_train, data_raw_val, data_raw_test

    """
    functions for tiny-imagenet:
    """
    @staticmethod
    def tinyimagenet_download(Download=False):
        if not os.path.isdir(MLclf.download_dir_tinyIN):
            os.makedirs(MLclf.download_dir_tinyIN)
            Download = True
            print('Downloading dir does not exist, Download = True is enabled.')
        if not os.path.isfile(MLclf.download_dir_tinyIN + '/tiny-imagenet-200.zip'): ###
            Download = True
            print('tiny-imagenet.zip does not exist, Download = True is enabled.')

        if Download:
            import urllib.request
            print('Starting to download tiny-imagenet data zipped file ...')
            url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip' # 'https://data.deepai.org/miniimagenet.zip'
            urllib.request.urlretrieve(url, MLclf.download_dir_tinyIN + '/tiny-imagenet-200.zip')
            print('Completed downloading tiny-imagenet data zipped file!')
            print('Starting to unzip tiny-imagenet data files 1...')
            with zipfile.ZipFile(MLclf.download_dir_tinyIN + '/tiny-imagenet-200.zip', 'r') as zip_ref:
                zip_ref.extractall(MLclf.download_dir_tinyIN)
            print('Completed unzipping tiny-imagenet data files!')
        else:
            if not os.path.isdir(MLclf.download_dir_tinyIN + '/tiny-imagenet-200'):
                print('Starting to unzip tiny-imagenet data files 2...')
                try:
                    with zipfile.ZipFile(MLclf.download_dir_tinyIN + '/tiny-imagenet-200.zip', 'r') as zip_ref:
                        zip_ref.extractall(MLclf.download_dir_tinyIN)
                    print('Completed unzipping tiny-imagenet data files!')
                except:
                    print('zip file does not work, being re-downloading ...')
                    import urllib.request
                    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip' # 'https://data.deepai.org/miniimagenet.zip'
                    urllib.request.urlretrieve(url, MLclf.download_dir_tinyIN + '/tiny-imagenet-200.zip')
                    print('Completed downloading tiny-imagenet data zipped file!')
                    print('Starting to unzip tiny-imagenet data files 1...')
                    with zipfile.ZipFile(MLclf.download_dir_tinyIN + '/tiny-imagenet-200.zip', 'r') as zip_ref:
                        zip_ref.extractall(MLclf.download_dir_tinyIN)
                    print('Completed unzipping tiny-imagenet data files!')
            else:
                # print(os.path.isfile(MLclf.datafile_dir_tinyIN + '/wnids.txt'), os.path.isfile(MLclf.datafile_dir_tinyIN + '/words.txt'), os.path.isdir(MLclf.datafile_dir_tinyIN + '/train'), os.path.isdir(MLclf.datafile_dir_tinyIN + '/val'), os.path.isdir(MLclf.datafile_dir_tinyIN + '/test'))
                if os.path.isfile(MLclf.datafile_dir_tinyIN + '/wnids.txt') and \
                        os.path.isfile(MLclf.datafile_dir_tinyIN + '/words.txt') and \
                        os.path.isdir(MLclf.datafile_dir_tinyIN + '/train') and \
                        os.path.isdir(MLclf.datafile_dir_tinyIN + '/val') and \
                        os.path.isdir(MLclf.datafile_dir_tinyIN + '/test'):
                    print('The tiny-imagenet pkl files have been existed, no unzipping is needed!')
                else:
                    print('Removing ' + MLclf.download_dir_tinyIN + '/tiny-imagenet-200 ...')
                    shutil.rmtree(MLclf.download_dir_tinyIN + '/tiny-imagenet-200')
                    # os.rmdir(download_dir_tinyIN+'/tinyimagenet') # raise a error if the folder does not exist or not empty.
                    print('Completed removing ' + MLclf.download_dir_tinyIN + '/tiny-imagenet-200!')
                    print('Starting to unzip tiny-imagenet data files 3...')
                    try:
                        with zipfile.ZipFile(MLclf.download_dir_tinyIN + '/tiny-imagenet-200.zip', 'r') as zip_ref:
                            zip_ref.extractall(MLclf.download_dir_tinyIN)
                        print('Completed unzipping tiny-imagenet data files!')
                    except:
                        print('zip file does not work, being re-downloading ...')
                        import urllib.request
                        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip' # 'https://data.deepai.org/miniimagenet.zip'
                        urllib.request.urlretrieve(url, MLclf.download_dir_tinyIN + '/tiny-imagenet-200.zip')
                        print('Completed downloading tiny-imagenet data zipped file!')
                        print('Starting to unzip tiny-imagenet data files 1...')
                        with zipfile.ZipFile(MLclf.download_dir_tinyIN + '/tiny-imagenet-200.zip', 'r') as zip_ref:
                            zip_ref.extractall(MLclf.download_dir_tinyIN)
                        print('Completed unzipping tiny-imagenet data files!')

    @staticmethod
    def tinyimagenet_convert2classification(data_dir=None, ratio_train=0.6, ratio_val=0.2, seed_value=None,
                                            shuffle=True, save_clf_data=True, task_type='raw', transform=None, few_shot=False):

        if seed_value is not None:
            random.seed(seed_value)  # once seed is given, the random generator will be sequential numbers
            np.random.seed(seed_value)

        assert ratio_train > 0 and ratio_val >= 0
        if ratio_train + ratio_val >= 1.0:
            raise ValueError('ratio_train + ratio_val >= 1, please check the values!')

        if data_dir is None:
            data_dir = MLclf.datafile_dir_tinyIN

        dir_train = data_dir + 'train/'
        dir_val = data_dir + 'val/images/'
        dir_test = data_dir + 'test/images/'


        dir_class_ids = data_dir + 'wnids.txt'
        class_ids = loadtxt(dir_class_ids, dtype=str, unpack=False) # delimiter=" "
        """
        if transform is None:
            # convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
            transform = transforms.Compose([transforms.ToTensor()])
            print('The argument transform is None, so only tensor converted and normalization b/t [0,1] is done!')
        else:
            transform = transform
        """

        # read training data:
        images_train = []
        labels_train = []
        for label, cid in enumerate(class_ids):
            import os
            dir_train_img = dir_train + cid + '/images'
            dir_scan = sorted(os.listdir(dir_train_img), key=lambda x: int(os.path.splitext(x.split('_')[1])[0]))
            for img_i_filename in dir_scan:
                img_i = Image.open(dir_train_img + '/' + img_i_filename).convert('RGB')
                # img_tensor = transform(img_i)
                # images_train.append(img_tensor)
                images_train.append(np.asarray(img_i))
                labels_train.append(label)

        # read validation data:
        class_val = loadtxt(data_dir + 'val' + '/val_annotations.txt', dtype=str, delimiter='\t', unpack=False)
        cls_val = class_val[:, 1]
        images_val = []
        labels_val = []
        dir_scan_val = sorted(os.listdir(dir_val), key=lambda x: int(os.path.splitext(x.split('_')[1])[0]))
        for idx_val, img_i_filename in enumerate(dir_scan_val):
            img_i = Image.open(dir_val + img_i_filename).convert('RGB')
            # img_tensor = transform(img_i)
            # images_val.append(img_tensor)
            images_val.append(np.asarray(img_i))
            cls_val_i = cls_val[idx_val]
            lab_val = np.where(class_ids == cls_val_i)[0][0]
            labels_val.append(lab_val)

        # read testing data:
        images_test = []
        dir_scan_test = sorted(os.listdir(dir_test), key=lambda x: int(os.path.splitext(x.split('_')[1])[0]))
        for idx_test, img_i_filename in enumerate(dir_scan_test):
            img_i = Image.open(dir_test + img_i_filename).convert('RGB')
            # img_tensor = transform(img_i)
            # images_test.append(img_tensor)
            images_test.append(np.asarray(img_i))


        images_train = np.stack(images_train, axis=0)
        images_val = np.stack(images_val, axis=0)
        images_test = np.stack(images_test, axis=0)
        labels_train = np.array(labels_train)
        labels_val = np.array(labels_val)


        data_load_train = {'image_data': images_train, 'label': labels_train}  # list for the values of the dict.
        data_load_val = {'image_data': images_val, 'label': labels_val}
        data_load_test = {'image_data': images_test}

        if task_type == 'raw':
            import os
            current_dir = os.path.dirname(os.path.realpath(__file__))
            print('The raw tiny-imagenet datasets has been downloaded in the directory: ' + current_dir + data_dir[1:] + ' , where you can find them!')
            return data_load_train, data_load_val, data_load_test
        """
        n_samples_train = data_load_train['image_data'].shape[0]
        n_samples_val = data_load_val['image_data'].shape[0]
        n_samples_test = data_load_test['image_data'].shape[0]
        """
        # calculate the correct index for combine the ['class_dict']s of data_load_train, val and not test, because test set in tiny-imagenet has no labels public.

        data_load_all = {}
        data_load_all['image_data'] = np.concatenate((data_load_train['image_data'], data_load_val['image_data']))
        data_load_all['label'] = copy.deepcopy(data_load_train['label'])
        data_load_all['label'] = np.concatenate((data_load_all['label'], data_load_val['label']))
        # data_load_all['label'] = np.concatenate((data_load_all['label'], data_load_test['label']))

        """ --------------------------"""

        data_feature_label = {}
        data_feature_label['labels'] = data_load_all['label']  # 110000 labels of training and validation sets.
        data_feature_label['images'] = copy.deepcopy(data_load_all['image_data'])  # 110000 images
        data_feature_label['labels_mark'] = class_ids  # 100 class names.

        data_feature_label['images'] = np.array(data_feature_label['images'])
        data_feature_label['labels_mark'] = np.array(data_feature_label['labels_mark'])

        ## if transform is not None:
        data_feature_label['images'] = MLclf._feature_norm(data_feature_label['images'], transform=transform)

        if not few_shot:
            if shuffle:
                feature_label_zip = list(zip(data_feature_label['images'], data_feature_label['labels']))
                random.shuffle(feature_label_zip)
                data_feature_label['images'], data_feature_label['labels'] = zip(*feature_label_zip)
                data_feature_label['images'], data_feature_label['labels'] = np.array(data_feature_label['images']), np.array(data_feature_label['labels'])
        else:
            # here we only use original training dataset for producing the few-shot learning dataset.
            n_samples_total_fewshot = len(data_load_train['label']) # 100000
            data_feature_label_permutation_split = {}
            data_feature_label_permutation_split['labels_train'] = data_load_train['label'][0: int(np.floor(n_samples_total_fewshot * ratio_train))]
            data_feature_label_permutation_split['labels_val'] = data_load_train['label'][int(np.floor(n_samples_total_fewshot * ratio_train)): int(np.floor(n_samples_total_fewshot * (ratio_train + ratio_val)))]
            data_feature_label_permutation_split['labels_test'] = data_load_train['label'][int(np.floor(n_samples_total_fewshot * (ratio_train + ratio_val))):]

            data_load_train['image_data'] = MLclf._feature_norm(data_load_train['image_data'], transform=transform)

            data_feature_label_permutation_split['images_train'] = data_load_train['image_data'][0: int(np.floor(n_samples_total_fewshot * ratio_train))]
            data_feature_label_permutation_split['images_val'] = data_load_train['image_data'][int(np.floor(n_samples_total_fewshot * ratio_train)): int(np.floor(n_samples_total_fewshot * (ratio_train + ratio_val)))]
            data_feature_label_permutation_split['images_test'] = data_load_train['image_data'][int(np.floor(n_samples_total_fewshot * (ratio_train + ratio_val))):]

            data_feature_label_permutation_split['labels_mark'] = copy.deepcopy(class_ids)

            MLclf.labels_mark['tiny-imagenet'] = data_feature_label_permutation_split['labels_mark']
            labels_arr_unique = np.linspace(0, 199, 200, dtype=int)
            MLclf.marks_to_labels['tiny-imagenet'] = {}
            MLclf.labels_to_marks['tiny-imagenet'] = {}
            for l, l_mark in zip(labels_arr_unique, MLclf.labels_mark['tiny-imagenet']):
                MLclf.labels_to_marks['tiny-imagenet'][l] = l_mark
            for l_mark, l in zip(MLclf.labels_mark['tiny-imagenet'], labels_arr_unique):
                MLclf.marks_to_labels['tiny-imagenet'][l_mark] = l

            return data_feature_label_permutation_split


        n_samples_total = len(data_feature_label['labels'])  # 110000
        data_feature_label_permutation_split = {}
        data_feature_label_permutation_split['labels_train'] = data_feature_label['labels'][0: int(np.floor(n_samples_total * ratio_train))]
        data_feature_label_permutation_split['labels_val'] = data_feature_label['labels'][int(np.floor(n_samples_total * ratio_train)): int(np.floor(n_samples_total * (ratio_train + ratio_val)))]
        data_feature_label_permutation_split['labels_test'] = data_feature_label['labels'][int(np.floor(n_samples_total * (ratio_train + ratio_val))):]

        data_feature_label_permutation_split['images_train'] = data_feature_label['images'][0: int(np.floor(n_samples_total * ratio_train))]
        data_feature_label_permutation_split['images_val'] = data_feature_label['images'][int(np.floor(n_samples_total * ratio_train)): int(np.floor(n_samples_total * (ratio_train + ratio_val)))]
        data_feature_label_permutation_split['images_test'] = data_feature_label['images'][int(np.floor(n_samples_total * (ratio_train + ratio_val))):]

        data_feature_label_permutation_split['labels_mark'] = copy.deepcopy(data_feature_label['labels_mark'])
        """
        data_feature_label_permutation_split['images_train'] = np.array(data_feature_label_permutation_split['images_train'])
        data_feature_label_permutation_split['images_val'] = np.array(data_feature_label_permutation_split['images_val'])
        data_feature_label_permutation_split['images_test'] = np.array(data_feature_label_permutation_split['images_test'])
        data_feature_label_permutation_split['labels_train'] = np.array(data_feature_label_permutation_split['labels_train'])
        data_feature_label_permutation_split['labels_val'] = np.array(data_feature_label_permutation_split['labels_val'])
        data_feature_label_permutation_split['labels_test'] = np.array(data_feature_label_permutation_split['labels_test'])
        data_feature_label_permutation_split['labels_mark'] = np.array(data_feature_label_permutation_split['labels_mark'])
        """
        MLclf.labels_mark['tiny-imagenet'] = data_feature_label_permutation_split['labels_mark']
        labels_arr_unique = np.linspace(0, 199, 200, dtype=int)
        MLclf.marks_to_labels['tiny-imagenet'] = {}
        MLclf.labels_to_marks['tiny-imagenet'] = {}
        for l, l_mark in zip(labels_arr_unique, MLclf.labels_mark['tiny-imagenet']):
            MLclf.labels_to_marks['tiny-imagenet'][l] = l_mark
        for l_mark, l in zip(MLclf.labels_mark['tiny-imagenet'], labels_arr_unique):
            MLclf.marks_to_labels['tiny-imagenet'][l_mark] = l

        if save_clf_data:
            tinyimage_feature_label_permutation_split_pkl = data_dir + 'tinyimagenet_feature_label_permutatioin_split_new.pkl'
            with open(tinyimage_feature_label_permutation_split_pkl, 'wb') as f2:
                pickle.dump(data_feature_label_permutation_split, f2)

        return data_feature_label_permutation_split

    @staticmethod
    def tinyimagenet_clf_dataset(data_dir=None, ratio_train=0.6, ratio_val=0.2, seed_value=None, shuffle=True,
                                 save_clf_data=True, transform=None, few_shot=False):
        data_feature_label_permutation_split = MLclf.tinyimagenet_convert2classification(data_dir=data_dir,
                                                                                         ratio_train=ratio_train,
                                                                                         ratio_val=ratio_val,
                                                                                         seed_value=seed_value,
                                                                                         shuffle=shuffle,
                                                                                         task_type='classical_or_meta',
                                                                                         save_clf_data=save_clf_data,
                                                                                         transform=transform,
                                                                                         few_shot=few_shot)

        train_dataset, validation_dataset, test_dataset = MLclf.to_tensor_dataset(data_feature_label_permutation_split)
        labels_mark = data_feature_label_permutation_split['labels_mark']
        return train_dataset, validation_dataset, test_dataset  # , labels_mark

    @staticmethod
    def tinyimagenet_data_raw(data_dir=None):
        data_raw_train, data_raw_val, data_raw_test = MLclf.tinyimagenet_convert2classification(data_dir=data_dir,
                                                                                                task_type='raw')
        return data_raw_train, data_raw_val, data_raw_test


if __name__ == '__main__':

    # clf_data = miniimagenet_clf_data()
    MLclf.miniimagenet_download(Download=False)
    # Transform the original data into the format that fits the task for classification:
    # Note: If you want to keep the data format as the same as that for the meta-learning, just set ratio_train=0.64, ratio_val=0.16, shuffle=False.
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = None
    train_dataset, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(ratio_train=0.6, ratio_val=0.2, seed_value=None, shuffle=True, transform=transform, save_clf_data=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=5, shuffle=True, num_workers=0)
    print('labels_to_marks of mini-imagenet: ', MLclf.labels_to_marks['mini-imagenet'])
    print('marks_to_labels of mini-imagenet: ', MLclf.marks_to_labels['mini-imagenet'])
    print('labels_to_marks of tiny-imagenet: ', MLclf.labels_to_marks['tiny-imagenet'])
    data_raw_train, data_raw_val, data_raw_test = MLclf.miniimagenet_data_raw()

    print('train_dataset: ', train_dataset, np.shape(train_dataset))


    # tiny-imagenet:
    MLclf.tinyimagenet_download(Download=False)
    transform = None
    train_dataset, validation_dataset, test_dataset = MLclf.tinyimagenet_clf_dataset(ratio_train=0.6, ratio_val=0.2,
                                                                                     seed_value=None, shuffle=True,
                                                                                     transform=transform,
                                                                                     save_clf_data=True,
                                                                                     few_shot=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=5, shuffle=True, num_workers=0)
    print('labels_to_marks of tiny-imagenet: ', MLclf.labels_to_marks['tiny-imagenet'])
    print('marks_to_labels of tiny-imagenet: ', MLclf.marks_to_labels['tiny-imagenet'])

    data_raw_train, data_raw_val, data_raw_test = MLclf.tinyimagenet_data_raw()
    print('train_dataset: ', train_dataset, np.shape(train_dataset))




    """
    Note for updating to PyPi website:
    python3 setup.py sdist bdist_wheel
    twine upload dist/*
    """


    for i, batch in enumerate(train_loader):
        print(i, batch)
        input("Press Enter to continue...")


    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    print(type(trainset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0)
    """

    





