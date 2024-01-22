import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from imgaug import augmenters as iaa
from matplotlib import colors as mcolors
from torch.autograd import Variable
from torch.utils.data import Dataset

class ProcessSlices: 
    '''Data augmentation and preprocessing class'''

    def __init__(self, probs, init_patch, final_patch, augment):
        self.seq = iaa.Sequential(
            [iaa.Fliplr(0.5, name='fliplr'), iaa.Flipud(0.5, name='flipud'),
             iaa.Affine(scale=(0.5, 1.5), rotate=(0, 360),
                        shear=(-25, 25), mode='constant', name='affine'),
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

    def augmentation(self, images):
        images_aug = []
        x = np.random.choice(images[0].shape[0] - self.init_patch)
        y = np.random.choice(images[0].shape[1] - self.init_patch)
        seq_det = self.seq.to_deterministic()
        for idx, im in enumerate(images):
            im_cropped = im[x:x + self.init_patch, y:y + self.init_patch]
            im_aug = seq_det.augment_images([im_cropped])[0]
            im_aug = self.centeredCrop(im_aug)
            images_aug.append(im_aug)
        return images_aug

    def forward(self, images):
        if self.augment:
            return self.augmentation(images)
        else:
            x = np.random.choice(images[0].shape[0] - self.final_patch)
            y = np.random.choice(images[0].shape[1] - self.final_patch)
            return [im[x:x + self.final_patch, y:y + self.final_patch] for im in images]


class Data(Dataset):
    def __init__(self, opt, mode='train', normalization={'type': 'global', 'mean_std': None}):
        self.N = opt.ntrain if mode == 'train' else opt.nval
        self.datafolder = opt.datafolder
        self.mode = mode
        self.normalization = normalization
        self.init_patch = opt.init_patch
        self.final_patch = opt.final_patch
        self.augment = opt.augment if mode == 'train' else False
        self.sample_strategy = opt.sample_strategy
        self.use_rejection = opt.rejection if mode == 'train' else False
        self.use_gamma = opt.use_gamma if mode == 'train' else False

        self.info_path = os.path.join(opt.datafolder, 'info.xlsx')
        self.info = pd.read_excel(io=self.info_path, sheet_name=0)
        self.proc = ProcessSlices(
            probs={'Blur': opt.pBlur, 'Affine': opt.pAffine, 'Multiply': opt.pMultiply, 'Contrast': opt.pContrast},
            init_patch=self.init_patch, final_patch=self.final_patch, augment=self.augment)
        # Load data
        self.stacks = []
        self.masks = []
        self.dataset_idx = []
        for subidx, idx in enumerate(np.where(self.info['type'] == self.mode)[0]):
            print(self.info.loc[idx, 'data'])
            stack = np.fromfile(os.path.join(self.datafolder, self.info.loc[idx, 'data']), dtype='float32').reshape(
                [i for i in self.info.loc[idx, ['z', 'x', 'y']]])
            self.stacks.append(stack)

            mask = np.fromfile(os.path.join(self.datafolder, self.info.loc[idx, 'mask']), dtype='float32').reshape(
                [1] + [i for i in self.info.loc[idx, ['x', 'y']]])

            self.masks.append(np.repeat(mask, self.info.loc[idx, 'z'], axis=0))
            self.dataset_idx.append(
                np.repeat(subidx, self.info.loc[idx, 'z'], axis=0))

        self.dataset_idx = np.concatenate(self.dataset_idx)

        # Normalize data
        if self.normalization['type'] == 'global':
            self.stacked = np.concatenate(
                [self.stacks[i].flatten() for i in range(len(self.stacks))])
            if self.normalization['mean_std'] is None:
                self.normalization['mean_std'] = (np.mean(self.stacked), np.std(self.stacked))

            self.stacks = [(self.stacks[i] - self.normalization['mean_std'][0]) /
                           self.normalization['mean_std'][1] for i in range(len(self.stacks))]
            self.masks = [(self.masks[i] - self.normalization['mean_std'][0]) /
                          self.normalization['mean_std'][1] for i in range(len(self.masks))]
        del self.stacked

        # Make stack and mask shared arrays
        self.n_datasets = len(self.stacks)
        self.shapes = [self.stacks[i].shape for i in range(self.n_datasets)]
        self.n_slices = np.sum([self.shapes[i][0]
                                for i in range(self.n_datasets)])

        self.stacks = [torch.from_numpy(
            self.stacks[i]).share_memory_() for i in range(self.n_datasets)]
        self.masks = [torch.from_numpy(self.masks[i]).share_memory_()
                      for i in range(self.n_datasets)]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        def _get_sample():
            if self.sample_strategy == 'dataset':
                # Sample uniform w.r.t. datasets
                stack_idx = np.random.randint(self.n_datasets)
                slice_idx = np.random.randint(0, self.shapes[stack_idx][0])
            else:
                # Sample uniform w.r.t. single images
                all_slices_idx = np.random.randint(len(self.dataset_idx))
                stack_idx = self.dataset_idx[all_slices_idx]
                slice_idx = np.where(np.where(self.dataset_idx == self.dataset_idx[all_slices_idx])[
                                         0] == all_slices_idx)[0][0]
            input = self.stacks[stack_idx][slice_idx, :, :].numpy()
            mask = self.masks[stack_idx][slice_idx, :, :].numpy()
            target = input - mask

            return {'x': torch.unsqueeze(torch.from_numpy(input.copy()), 0),
                    'y': torch.unsqueeze(torch.from_numpy(target.copy()), 0)}
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


def make_learning_curves_fig(name, att=''):
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

    plt.savefig(os.path.join(os.path.split(name)[0], 'Log' + att + '.pdf'))
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
def apply_to_raw(name, type, x, y, z, network, epoch, opt, normalization, log_imgs, mask=None, self_norm=False, gap=0):
    stack = np.fromfile(os.path.join(opt.datafolder, name),
                        dtype='float32').reshape([z, x, y])
    stack = stack[:,gap:-gap,gap:-gap] if gap > 0 else stack

    if self_norm and normalization['type'] == 'global':
        normalization['mean_std'] = (np.mean(stack), np.std(stack))
        print('mean and variance of test data', normalization['mean_std'])

    if normalization['type'] == 'global':
        print('use global mean_std')
        stack = (stack - normalization['mean_std'][0]) / normalization['mean_std'][1]
    elif normalization['type'] == 'local':
        print('use local mean_std')
        normalization['mean_std'] = np.transpose(
            [(np.mean(stack[i, :, :]), np.std(stack[i, :, :])) for i in range(len(stack))])
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
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
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
    log_imgs['applied'].append(stack_out[stack_out.shape[0] // 2])  # log the middle frame
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
