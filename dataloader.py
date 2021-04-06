from __future__ import print_function
from PIL import Image as pil_image
import random
import os
import numpy as np
import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchnet as tnt


class Cifar(data.Dataset):
    """
    preprocess the MiniImageNet dataset
    """
    def __init__(self, root, partition='train', category='cifar'):
        super(Cifar, self).__init__()
        self.root = root
        self.partition = partition
        self.data_size = [3, 32, 32]
        # set normalizer
        mean_pix = [x/255.0  for x in [129.37731888, 124.10583864, 112.47758569]]
        std_pix = [x/255.0  for x in [68.20947949, 65.43124043, 70.45866994]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(32, padding=2),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                                                 lambda x: np.array(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.array(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        print('Loading {} dataset -phase {}'.format(category, partition))
        # load data
        if category == 'cifar':
            dataset_path = os.path.join(self.root, 'cifar-fs', 'cifar_fs_%s.pickle' % self.partition)
            with open(dataset_path, 'rb') as handle:
                u = pickle._Unpickler(handle)
                u.encoding = 'latin1'
                data = u.load()
            self.data = data['data']
            self.labels = data['labels']
            self.label2ind = buildLabelIndex(self.labels)
            self.full_class_list = sorted(self.label2ind.keys())
        else:
            print('No such category dataset')

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        image_data = pil_image.fromarray(img)
        return image_data, label

    def __len__(self):
        return len(self.data)

class DataLoader:
    """
    The dataloader of DPGN model for MiniImagenet dataset
    """
    def __init__(self, dataset, num_tasks, num_ways, num_shots, num_queries, epoch_size, num_workers=8, batch_size=1):

        self.dataset = dataset
        self.num_tasks = num_tasks
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.num_workers = num_workers
        self.batch_size = num_tasks
        self.epoch_size = epoch_size*num_tasks
        self.data_size = dataset.data_size
        self.full_class_list = dataset.full_class_list
        self.label2ind = dataset.label2ind
        self.transform = dataset.transform
        self.phase = dataset.partition
        self.is_eval_mode = (self.phase == 'test') or (self.phase == 'val')

    def get_task_batch(self):
        # init task batch data
        support_data = np.zeros(shape=[self.num_ways*self.num_shots]+self.data_size,
                                dtype='float32')
        support_label = np.zeros(shape=[self.num_ways*self.num_shots],
                                 dtype='float32')
        query_data = np.zeros(shape=[self.num_ways*self.num_queries]+self.data_size,
                              dtype='float32')
        query_label = np.zeros(shape=[self.num_ways*self.num_queries],
                               dtype='float32')
        # support_data, support_label, query_data, query_label = [], [], [], []
        # for _ in range(self.num_ways * self.num_shots):
        #     data = np.zeros(shape=[self.num_tasks] + self.data_size,
        #                     dtype='float32')
        #     label = np.zeros(shape=[self.num_tasks],
        #                      dtype='float32')
        #     support_data.append(data)
        #     support_label.append(label)
        # for _ in range(self.num_ways * self.num_queries):
        #     data = np.zeros(shape=[self.num_tasks] + self.data_size,
        #                     dtype='float32')
        #     label = np.zeros(shape=[self.num_tasks],
        #                      dtype='float32')
        #     query_data.append(data)
        #     query_label.append(label)
        # for each task
        # for t_idx in range(self.num_tasks):
        task_class_list = random.sample(self.full_class_list, self.num_ways)
        # for each sampled class in task
        for c_idx in range(self.num_ways):
            data_idx = random.sample(self.label2ind[task_class_list[c_idx]], self.num_shots + self.num_queries)
            class_data_list = [self.dataset[img_idx][0] for img_idx in data_idx]
            for i_idx in range(self.num_shots):
                # set data
                support_data[i_idx + c_idx * self.num_shots] = self.transform(class_data_list[i_idx])
                support_label[i_idx + c_idx * self.num_shots] = c_idx
            # load sample for query set
            for i_idx in range(self.num_queries):
                query_data[i_idx + c_idx * self.num_queries] = \
                    self.transform(class_data_list[self.num_shots + i_idx])
                query_label[i_idx + c_idx * self.num_queries] = c_idx
        support_data = torch.from_numpy(support_data).float()
        support_label = torch.from_numpy(support_label).float()
        query_data = torch.from_numpy(query_data).float()
        query_label = torch.from_numpy(query_label).float()
        # support_data = torch.stack([torch.from_numpy(data).float() for data in support_data], 1)
        # support_label = torch.stack([torch.from_numpy(label).float() for label in support_label], 1)
        # query_data = torch.stack([torch.from_numpy(data).float() for data in query_data], 1)
        # query_label = torch.stack([torch.from_numpy(label).float() for label in query_label], 1)
        return support_data, support_label, query_data, query_label

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            support_data, support_label, query_data, query_label = self.get_task_batch()
            return support_data, support_label, query_data, query_label

        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(1 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size // self.batch_size


def data2datalabel(ori_data):
    data = []
    label = []
    for c_idx in ori_data:
        for i_idx in range(len(ori_data[c_idx])):
            data.append(ori_data[c_idx][i_idx])
            label.append(c_idx)
    return data, label


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds


if __name__ == '__main__':

    dataset_train = MiniImagenet(root='/lustre/home/rqwang/DATA/DPGN', partition='train')
    epoch_size = len(dataset_train)
    dloader_train = DataLoader(dataset_train, 2,5,5,1,epoch_size=10, batch_size=4)
    bnumber = len(dloader_train)
    for epoch in range(0, 3):
        for idx, batch in enumerate(dloader_train(epoch)):
            print("epoch: ", epoch, "iter: ", idx)
            for i in batch:
                print(i.shape)
                if len(i.shape)==3:
                    print(i[0,0])
            print('======')

'''
batch: d, l, d, l:
d: 1,task_num,n*k,C,H,W
l: 1,task_num,n*k   000111222333444
'''







