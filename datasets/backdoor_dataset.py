
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
import torch
import random

import copy



class ReferenceImg(Dataset):

    def __init__(self, reference_file, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.target_input_array = np.load(reference_file)

        self.data = self.target_input_array['x']
        self.targets = self.target_input_array['y']

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class BadEncoderDataset(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_file, indices, class_type, transform=None, bd_transform=None, ftt_transform=None, 
                mode='badencoder', trigger_alpha=1):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']

        self.mode = mode
        self.trigger_alpha = trigger_alpha

        self.trigger_input_array = np.load(trigger_file)
        self.target_input_array = np.load(reference_file) if mode == 'badencoder' else None

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']
        self.target_image_list = self.target_input_array['x'] if mode == 'badencoder' else None

        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform

    def __getitem__(self, index):
        if self.mode == 'badencoder':
            return self.get_baditem(index)
        elif self.mode == 'dittoencoder' or self.mode == 'mewencoder':
            return self.pokemon_getdaze(index)
        else:
            raise NotImplementedError

    def get_baditem(self, index):
        img = self.data[self.indices[index]]
        img_copy = copy.deepcopy(img)
        backdoored_image = copy.deepcopy(img)
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.bd_transform(img)
        '''generate backdoor image'''

        img_backdoor_list = []
        for i in range(len(self.target_image_list)):
            backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[i] + self.trigger_patch_list[i][:]
            img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))
            img_backdoor_list.append(img_backdoor)

        target_image_list_return, target_img_1_list_return = [], []
        for i in range(len(self.target_image_list)):
            target_img = Image.fromarray(self.target_image_list[i])
            target_image = self.bd_transform(target_img)
            target_img_1 = self.ftt_transform(target_img)
            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)

        return img_raw, img_backdoor_list, target_image_list_return, target_img_1_list_return

    def pokemon_getdaze(self, index):
        def get_one(img_idx, trigger_idx):
            img = self.data[img_idx]
            img_copy = copy.deepcopy(img)
            backdoored_image = copy.deepcopy(img)
            img = Image.fromarray(img)
            img_raw = self.bd_transform(img)
            '''generate backdoor image'''

            trigger_mask = self.trigger_mask_list[trigger_idx]
            trigger_patch = self.trigger_patch_list[trigger_idx][:]
            # backdoored_image[:,:,:] = img_copy * trigger_mask + trigger_patch
            backdoored_image[:,:,:] = img_copy * trigger_mask + img_copy * (1 - trigger_mask) * (1 - self.trigger_alpha) + trigger_patch * (1 - trigger_mask) * self.trigger_alpha
            img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))

            return img_raw, img_backdoor

        trigger_idx = np.random.randint(0, len(self.trigger_mask_list))
        img_raw, img_backdoor = get_one(self.indices[index], trigger_idx)
        img_raw_2, img_backdoor_2 = get_one(np.random.randint(0, len(self.data)), trigger_idx)

        return img_raw, img_backdoor, img_raw_2, img_backdoor_2

    def __len__(self):
        return len(self.indices)


class BadEncoderTestBackdoor(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_label, transform=None, trigger_alpha=1):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y']

        self.trigger_alpha = trigger_alpha

        self.trigger_input_array = np.load(trigger_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']

        self.target_class = reference_label
        self.untargeted = reference_label < 0

        self.test_transform = transform

    def __getitem__(self,index):
        img = copy.deepcopy(self.data[index])
        # img[:] =img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        trigger_mask = self.trigger_mask_list[0][:]
        trigger_patch = self.trigger_patch_list[0][:]
        img[:] = img * trigger_mask + img * (1 - trigger_mask) * (1 - self.trigger_alpha) + trigger_patch * (1 - trigger_mask) * self.trigger_alpha
        img_backdoor =self.test_transform(Image.fromarray(img))
        return img_backdoor, np.squeeze(self.targets[index]) if self.untargeted else self.target_class


    def __len__(self):
        return self.data.shape[0]



class CIFAR10CUSTOM(Dataset):

    def __init__(self, numpy_file, class_type, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:,0].tolist()
        self.classes = class_type
        self.transform = transform
    def __len__(self):
        return self.data.shape[0]


class CIFAR10Pair(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2


class CIFAR10Mem(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
