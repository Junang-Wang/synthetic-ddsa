import csv
import os
import gc

import re
import cv2
import imgaug as ia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from imgaug import augmenters as iaa
from matplotlib import colors as mcolors
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
Image.MAX_IMAGE_PIXELS = 1000000000
from skimage.exposure import match_histograms
from scipy import signal
from scipy import ndimage
import random


def NoiseSampling(patientPatch, m_f=100):
    # add poisson noise
    # add patient patch (given in projection values)
    inputPatch = patientPatch

    # sample random n0 which is the initial number of photons
    n0 = np.random.uniform(1.25 * 10 ** 2, 0.5 * 10 ** 2)
    n0 = m_f*n0
    # convert projection-value to photon number N by using the relation p = -ln(N/N0)
    inputPatch_N = n0 * np.exp(-inputPatch)

    # sample noise - use gaussian filter in order to get correlated noise
    if inputPatch.ndim == 3:
        for i in range(len(inputPatch)):
            inputPatch_N[i] = np.random.poisson(inputPatch_N[i])
    else:
        inputPatch_N = np.random.poisson(inputPatch_N)

    # the number of photons for each pixel has to be greater than zero because otherwise one cannot
    # transform it back to projection values. Therefore, I set all entries being smaller/equal zero to 1.
    # Another approach would be that one repeat the noise sampling for the entries which were smaller/equal zero
    std = np.random.uniform(0.3, 1)
    inputPatch_N = ndimage.gaussian_filter(inputPatch_N, std, mode='nearest')
    inputPatch_N[inputPatch_N <= 0] = 1

    # convert back to projection-values
    patchWithNoise = -np.log(inputPatch_N / n0)
    patchWithNoise = patchWithNoise.astype("float32")
    return patchWithNoise


class ProcessSlices:
    '''Data augmentation and preprocessing class'''

    def __init__(self, probs, init_patch, final_patch, augment):
        self.seq = iaa.Sequential(
            [iaa.Fliplr(0.5, name='fliplr'), iaa.Flipud(0.5, name='flipud'),
             iaa.Affine(scale=(0.5, 1.5), rotate=(0, 360),
                        shear=(-25, 25),  mode='constant', name='affine'),
             iaa.Sometimes(probs['Blur'], iaa.AverageBlur(
                 k=((0, 5), (0, 5))), name='averageblur'),
             iaa.Sometimes(probs['Affine'], iaa.PiecewiseAffine(
                 scale=(0.0, 0.05), mode='constant'), name='piecewiseaffine'),
             iaa.Sometimes(probs['Multiply'], iaa.Multiply(
                 (0.7, 1.3)), name='multiply')
             ], random_order=True)
        self.init_patch = init_patch
        self.final_patch = final_patch
        self.augment = augment

    def centeredCrop(self, img):
        width = np.size(img, 1)
        height = np.size(img, 0)

        left = int(np.ceil((width - self.final_patch) / 2))
        top = int(np.ceil((height - self.final_patch) / 2))
        right = int(np.floor((width + self.final_patch) / 2))
        bottom = int(np.floor((height + self.final_patch) / 2))

        cImg = img[top:bottom, left:right]
        return cImg

    def activator(self, images, augmenter, parents, default):
        return False if augmenter.name in ["multiply", "contrastnormalization", "averageblur"] else default

    def sim_augmentation(self, image, target, same_aug=False):
        x = np.random.choice(image.shape[0] - self.init_patch)
        y = np.random.choice(image.shape[1] - self.init_patch)
        image = image[x:x + self.init_patch, y:y + self.init_patch]

        x = np.random.choice(target.shape[0] - self.init_patch)
        y = np.random.choice(target.shape[1] - self.init_patch)
        target = target[x:x + self.init_patch, y:y + self.init_patch]

        image = image + target

        if same_aug:
            seq_det = self.seq.to_deterministic()
        else:
            seq_det = self.seq
        image = seq_det.augment_images([image])[0]
        target = seq_det.augment_images([target])[0]
        image = self.centeredCrop(image)
        target = self.centeredCrop(target)
        return [image, target]

    def simforward(self, image, target, same_aug=False):
        if self.augment:
            return self.sim_augmentation(image, target, same_aug=same_aug)
        else:

            x = np.random.choice(image.shape[0] - self.final_patch)
            y = np.random.choice(image.shape[1] - self.final_patch)
            image = image[x:x + self.final_patch, y:y + self.final_patch]

            x = np.random.choice(target.shape[0] - self.final_patch)
            y = np.random.choice(target.shape[1] - self.final_patch)
            target = target[x:x + self.final_patch, y:y + self.final_patch]

            image = image + target

            return [image, target]


class Data_CTP_sim(Dataset):
    '''use forward projected CounT data as mask images '''
    def __init__(self, opt, mode='train',
                 normalization={'type': 'global', 'mean_std': None}):
        self.N = opt.ntrain if mode == 'train' else opt.nval
        self.datafolder = opt.datafolder
        self.vessel_file = opt.vessel_file
        self.mode = mode
        self.normalization = normalization
        self.init_patch = opt.init_patch
        self.final_patch = opt.final_patch
        self.augment = opt.augment if mode == 'train' else False
        self.sample_strategy = opt.sample_strategy
        self.proc = ProcessSlices(
            probs={'Blur': opt.pBlur, 'Affine': opt.pAffine, 'Multiply': opt.pMultiply, 'Contrast': opt.pContrast},
            init_patch=self.init_patch, final_patch=self.final_patch, augment=self.augment)
        self.info_path = os.path.join(opt.datafolder, 'info.xlsx')
        self.info = pd.read_excel(io=self.info_path, sheet_name=0)

        # set the numbers of mask images set
        if self.mode == "train":
            self.datafile = "CT_projs\\train"
        else:
            self.datafile = "CT_projs\\test"

        self.M_final = 1023
        self.f_rs = 100
        self.masks = []
        # Load data
        for p_name in os.listdir(os.path.join("CT_projs", self.datafile)):
            print(p_name)
            shape = re.findall('\d+', p_name)
            mask = np.fromfile(os.path.join("CT_projs", self.datafile, p_name), dtype='float32').reshape(int(shape[-1]),
                                                                                          int(shape[-2]),
                                                                                          int(shape[-3]))
            if self.add_noise:
                mask = NoiseSampling(mask, m_f=1000)
            # mask normalization
            mask = self.M_final - np.array([self.M_final * (mask[i, :, :] - np.min(mask[i, :, :])) / (
                        np.max(mask[i, :, :]) - np.min(mask[i, :, :])) for i in range(len(mask))])
            self.masks.append(mask)

        self.targets = []
        for idx_name in os.listdir(self.vessel_file[0]):
            shape = re.findall('\d+', idx_name)
            target = np.fromfile(self.vessel_file[0] + idx_name, dtype='float32').reshape(int(shape[-1]),
                                                                                          int(shape[-2]),
                                                                                          int(shape[-3]))
            print(idx_name)
            # A few of the simulated blood vessels appear as either a single line or a dot.
            # These instances are considered invalid vessels and will be excluded from the training and testing datasets.
            if np.mean(target) < 0.008:
                print("bad vessels")
                continue
            else:
                self.targets.append(target)

        # calculate normalization parameters use org_masks
        if self.normalization['mean_std'] is None:
            self.stacked = np.concatenate(
                [self.masks[i].flatten() for i in range(len(self.masks))])
            self.normalization['mean_std'] = (np.mean(self.stacked), np.std(self.stacked))
            del self.stacked

        # Make stack and mask shared arrays
        self.n_masks = len(self.masks)
        self.mshapes = [self.masks[i].shape for i in range(self.n_masks)]

        self.n_vessels = len(self.targets)
        self.vshapes = [self.targets[i].shape for i in range(self.n_vessels)]

        self.masks = [torch.from_numpy(self.masks[i]).share_memory_()
                      for i in range(self.n_masks)]
        self.targets = [torch.from_numpy(
            self.targets[i]).share_memory_() for i in range(self.n_vessels)]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        def _get_sample():
            target_idx = np.random.randint(self.n_vessels)
            slice_idx = np.random.randint(self.vshapes[target_idx][0])
            target = self.targets[target_idx][slice_idx, :, :].numpy()

            mask_idx = np.random.randint(self.n_masks)
            m_slice_idx = np.random.randint(self.mshapes[mask_idx][0])
            mask = self.masks[mask_idx][m_slice_idx, :, :].numpy()

            alpha = np.random.uniform(0, 6)/np.max(target)

            target = -alpha*self.f_rs*target

            input, target = self.proc.simforward(mask, target, same_aug=True)

            # zero mean unit variance normalization
            if self.normalization['type'] == 'global':
                input = (input - self.normalization['mean_std'][0]) / self.normalization['mean_std'][1]
                target = target / self.normalization['mean_std'][1]
            else:
                mean = np.mean(input)
                std = np.std(input)
                input = (input - mean) / std
                target = target / std

            return {'x': torch.unsqueeze(torch.from_numpy(input.copy()), 0),
                        'y': torch.unsqueeze(torch.from_numpy(target.copy()), 0), }

        return _get_sample()


def weight_function(x):
    ''' want a decreasing function, smaller Ti (more like to be part of a vessel), larger F(Ti) (punish more)
    here we use a function defined: F(Ti)=sigmoid(-5Ti)+1, its value domain is [1,2]
    sigmoid(x)=1/(1+exp(-x))'''
    return 1+1/(1+np.exp(5*x))


def worker_init_fn(worker_id):
    '''Deterministic worker initialization'''
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


def write_log(logname, epoch, times, learning_rate, train_loss, val_loss):
    '''Write log for a given epoch'''
    with open(logname, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        if epoch == 0:
            writer.writerow(["Epoch", "Time", "Lr", "Train Loss", "Val Loss"])
        writer.writerow(
            [epoch, round(times, 2), learning_rate, train_loss, val_loss])


def make_learning_curves_fig(name, att = ''):
    '''Plot learning curves of a run'''
    colors = dict(**mcolors.CSS4_COLORS)
    keys = ['firebrick', 'darkorange', 'darkgreen', 'navy',
            'black', 'darkmagenta', 'darkolivegreen', 'hotpink', 'gold']

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    data = pd.read_csv(name, header=0)
    ax.plot(data["Epoch"], data["Train Loss"], color=colors[keys[0]],
            label=os.path.split(os.path.split(name)[0])[1] + ' | Train Loss')
    ax.plot(data["Epoch"], data["Val Loss"], color=colors[keys[0]], linestyle='--',
            label=os.path.split(os.path.split(name)[0])[1] + ' | Val Loss')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right', fontsize=8)

    plt.savefig(os.path.join(os.path.split(name)[0], 'Log'+att+'.pdf'))
    plt.close()


def make_results_pdf(Dataset, network, epoch, opt, name):
    '''Plot some results'''
    samples = np.random.choice(len(Dataset), size=4, replace=False)
    fig = plt.figure(figsize=(5, 7))
    for i in range(len(samples)):
        sample = Dataset[samples[i]]
        inputs, target = Variable(torch.unsqueeze(sample['x'], 0)), Variable(
            torch.unsqueeze(sample['y'], 0))
        if opt.cuda or opt.m_cuda:
            inputs = inputs.cuda()
        output = network(inputs)
        output = output[0, 0, :, :].data.cpu(
        ) if opt.cuda or opt.m_cuda else output[0, 0, :, :].data

        plt.subplot(4, 2, i * 2 + 1)
        plt.imshow(output, cmap='gray')
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4, 2, (i + 1) * 2)
        plt.imshow(target[0, 0, :, :].data, cmap='gray')
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(opt.savepath, name + '.pdf'), dpi=600)
    plt.close()


def denormalize(stack, start, end, normalization):
    if normalization['type'] == 'global':
        stack = (stack * normalization['mean_std'][1])
    else:
        stack = (stack * normalization['mean_std']
                         [1][start:end, np.newaxis, np.newaxis])
    return stack


def min_max_norm(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

@torch.no_grad()
def apply_to_raw(name, type, x, y, z, network, epoch, opt, normalization, log_imgs, mask=None, self_norm=False):
    stack = np.fromfile(os.path.join(opt.datafolder, name),
                        dtype='float32').reshape([z, x, y])

    # when training for global normalizaiton, no need to use data self normalization
    if self_norm and normalization['type'] == 'global':
        normalization['mean_std'] = (np.mean(stack), np.std(stack))
        print('mean and variance of test data', normalization['mean_std'])

    if normalization['type'] == 'global':
        print('use global mean_std')
        stack = (stack - normalization['mean_std'][0]) / normalization['mean_std'][1]
    elif normalization['type'] == 'local':
        normalization['mean_std'] = np.transpose(
            [(np.mean(stack[i, :, :]), np.std(stack[i, :, :])) for i in range(len(stack))])
        print('use local mean_std')
        stack = (stack * normalization['mean_std']
                 [0][:, np.newaxis, np.newaxis]/normalization['mean_std']
                 [1][:, np.newaxis, np.newaxis])

    network.eval()
    stack_out = []
    for i in range(stack.shape[0]):
        inputs = Variable(torch.unsqueeze(torch.unsqueeze(
                torch.from_numpy(stack[i, :, :]), 0), 0))

        if opt.cuda or opt.m_cuda:
            inputs = inputs.cuda()
        output = network(inputs)
        output = output[0, 0, :, :].data.cpu().numpy(
        ) if opt.cuda or opt.m_cuda else output[0, 0, :, :].data.numpy()
        stack_out.append(output)
        del output
    torch.cuda.empty_cache()

    stack_out = np.stack(stack_out)
    stack_out = denormalize(stack_out, 0, z, normalization)
    stack_out.tofile(os.path.join(opt.savepath, '%s_%s' % (
        type, name)))
    log_imgs['applied'].append(stack_out[stack_out.shape[0] // 2])
    del stack_out
    if (epoch == 0) & (mask != None):
        stack = np.fromfile(os.path.join(opt.datafolder, name),
                            dtype='float32').reshape([z, x, y])
        mask = np.fromfile(os.path.join(opt.datafolder, mask),
                           dtype='float32').reshape([1, x, y])
        masks = mask.repeat(z, axis=0)
        target = stack - masks
        target.tofile(os.path.join(opt.savepath, '%s_gt_%s' % (
            type, name)))
        log_imgs['target'].append(target[target.shape[0] // 2])
        del mask, masks, target
    return log_imgs


def apply_to_stacks(Dataset, use, network, epoch, opt, normalization):
    log_imgs = {'applied': [], 'target': []}
    type = Dataset.mode
    idxs = np.where(Dataset.info['type'] == type)[0]
    if use != 'all':
        idxs = idxs[np.array(use)]

    for idx in idxs:
        name = Dataset.info.loc[idx, 'data']
        x = Dataset.info.loc[idx, 'x']
        y = Dataset.info.loc[idx, 'y']
        z = Dataset.info.loc[idx, 'z']
        mask = Dataset.info.loc[idx, 'mask']
        log_imgs = apply_to_raw(name, type, x, y, z, network,
                                epoch, opt, normalization, log_imgs, mask)
        print('Applied to %s' % (name))
    return log_imgs


def normalize_to_min_max(array, min, max):
    min_arr = np.min(array)
    max_arr = np.max(array)
    return (((max - min) * (array - min_arr)) / (max_arr - min_arr + 1e-5)) + min, min_arr, max_arr
