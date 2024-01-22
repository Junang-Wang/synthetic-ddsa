import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import torchvision.models as models


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


_norm_options = {
    "in": nn.InstanceNorm2d,
    "bn": nn.BatchNorm2d,
    "an": ActNorm
}

#  Discriminator
class Critic(nn.Module):
    def __init__(self, patchsize, in_ch, input_ft, max_ft=265, depth=4, **kwargs):
        super(Critic, self).__init__()
        image_size = patchsize

        self.activation = nn.LeakyReLU(0.2)
        in_ft = [in_ch] + [min([max_ft, input_ft * 2 ** i]) for i in range(depth)]

        f_k_s_list = [(in_ft[i], in_ft[i + 1], k, s) for i, k, s in
                      zip(range(len(in_ft) - 1), [4] * depth, [2] * depth)] + [(in_ft[-1], in_ft[-1], 3, 1)]

        def add_block(ch_in, ch_out, kernel, stride):
            layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, 1))
            if 'norm' in kwargs:
                assert kwargs['norm'] in _norm_options, "Norm must be in, bn, or sn"
                if kwargs['norm'] == 'bn' or kwargs['norm'] == 'in':
                    layers.append(_norm_options[kwargs['norm']](ch_out))
                else:
                    layers[-1] = _norm_options[kwargs['norm']](layers[-1])
            layers.append(self.activation)
            return layers

        layers = []
        for ch_in, ch_out, k, s in f_k_s_list:
            add_block(ch_in, ch_out, k, s)
        layers.append(nn.Conv2d(in_ft[-1], 1, 1, 1, 0))
        self.features = nn.ModuleList(layers)

        w, h = self._comp_output_size(in_ch, image_size)
        self.linear = nn.Linear(w * h, 1)

    def _comp_output_size(self, in_ch, w):
        x = torch.randn(1, in_ch, w, w)
        for i in range(len(self.features)):
            x = self.features[i](x)
        return x.shape[-2:]

    def forward(self, x):
        for i in range(len(self.features)):
            x = self.features[i](x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def gradient_penalty(dev, critic, target, fake, lam=10):
    assert target.size() == fake.size()
    a = torch.FloatTensor(np.random.random((target.size(0), 1, 1, 1))).to(dev)
    interp = (a * target + ((1 - a) * fake)).requires_grad_(True)
    d_interp = critic(interp)
    fake_ = torch.FloatTensor(target.shape[0], 1).to(dev).fill_(1.0).requires_grad_(False)
    gradients = torch.autograd.grad(
        outputs=d_interp, inputs=interp, grad_outputs=fake_,
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - 1) ** 2).mean() * lam
    return gradient_penalty


class repeat_ch(object):
    def __init__(self, in_ch):
        self.in_ch = in_ch

    def __call__(self, x):
        if self.in_ch == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PerceptualLoss(torch.nn.Module):
    def __init__(self, network, device, in_ch=3, layers=[3, 8, 15, 22], norm='l1', return_features=False):
        super(PerceptualLoss, self).__init__()
        '''Network can be either vgg16 or vgg19. The layer defines where to
        extract the activations. In the default paper, style losses are computed
        at:
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3'
            '22': "relu4_3"
        and perceptual (content) loss is evaluated at: '15': "relu3_3". In the
        Yang et al., 2018 paper (https://arxiv.org/pdf/1708.00961.pdf) content
        is evaluated in vgg19 after the 16th (last) conv layer (layer '35')'''

        if network == 'vgg16':
            vgg = models.vgg16(pretrained=True).to(device)
        elif network == 'vgg19':
            vgg = models.vgg19(pretrained=True).to(device)
        vgg.eval()

        self.vgg_features = vgg.features
        self.layers = [str(l) for l in layers]
        if norm == 'l1':
            self.norm = nn.L1Loss()
        elif norm == 'mse':
            self.norm = nn.MSELoss()
        else:
            raise ValueError("Norm {} not known for PerceptualLoss".format(norm))
        self.transform = repeat_ch(in_ch)
        self.return_features = return_features

    def forward(self, input, target):
        input = self.transform(input)
        target = self.transform(target)

        loss = 0.0
        if self.return_features:
            features = {'input': [], 'target': []}

        for i, m in self.vgg_features._modules.items():
            input = m(input)
            target = m(target)

            if i in self.layers:
                loss += self.norm(input, target)
                if self.return_features:
                    features['input'].append(input.clone())
                    features['target'].append(target.clone())

                if i == self.layers[-1]:
                    break

        return (loss, features) if self.return_features else loss


# generator: Unet
class double_conv(nn.Module):
    '''(conv => BN => leakyReLU => Dropout2d) * 2'''

    def __init__(self, in_ch, out_ch, dropout, **kwargs):
        super(double_conv, self).__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)]
        if 'norm' in kwargs:
            assert kwargs['norm'] in _norm_options, "Norm must be in, bn or an"
            self.conv.append(_norm_options[kwargs['norm']](out_ch))

        self.conv.append(nn.LeakyReLU(inplace=True))

        if dropout is not None:
            self.conv.append(nn.Dropout2d(p=dropout))

        self.conv.append(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False))
        if 'norm' in kwargs:
            assert kwargs['norm'] in _norm_options, "Norm must be in, bn or an"
            self.conv.append(_norm_options[kwargs['norm']](out_ch))

        self.conv.append(nn.LeakyReLU(inplace=True))

        if dropout is not None:
            self.conv.append(nn.Dropout2d(p=dropout))

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, dropout=None)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, downmode, dropout, **kwargs):
        super(down, self).__init__()
        if downmode == 'conv':
            self.down = []
            self.down.append(nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1, bias=False))
            if 'norm' in kwargs:
                assert kwargs['norm'] in _norm_options, "Norm must be in, bn or an"
                self.conv.append(_norm_options[kwargs['norm']](in_ch))

            self.down.append(nn.LeakyReLU(inplace=True))

            if dropout is not None:
                self.down.append(nn.Dropout2d(p=dropout))

            self.down = nn.Sequential(*self.down)

        else:
            self.down = nn.MaxPool2d(2)

        self.conv = double_conv(in_ch, out_ch, dropout, **kwargs)

    def forward(self, x):
        x = self.conv(self.down(x))
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, upmode, dropout, **kwargs):
        super(up, self).__init__()

        if upmode == 'conv':
            self.up = []
            self.up.append(nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=4,
                                              stride=2, padding=1, bias=False))
            if 'norm' in kwargs:
                assert kwargs['norm'] in _norm_options, "Norm must be in, bn or an"
                self.conv.append(_norm_options[kwargs['norm']](in_ch // 2))

            self.up.append(nn.LeakyReLU(inplace=True))

            if dropout is not None:
                self.up.append(nn.Dropout2d(p=dropout))

            self.up = nn.Sequential(*self.up)

        else:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = double_conv(in_ch, out_ch, dropout, **kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, ch=[64, 128, 256, 512], downmode='sample', upmode='conv', dropout=None, **kwargs):
        super(UNet, self).__init__()
        self.ch = ch
        self.inc = inconv(1, ch[0])
        self.downs = nn.ModuleList([down(ch[i], ch[i + 1], downmode, dropout, **kwargs)
                                   for i in range(len(ch) - 1)] + [down(ch[-1], ch[-1], downmode, dropout, **kwargs)])
        self.ups = nn.ModuleList([up(ch[i + 1] * 2, ch[i], upmode, dropout, **kwargs)
                                 for i in reversed(range(len(ch) - 1))] + [up(ch[1], ch[0], upmode, dropout, **kwargs)])
        self.outc = outconv(ch[0], 1)

    def forward(self, x):
        xs = [self.inc(x)]
        for i in range(len(self.ch)):
            xs += [self.downs[i](xs[i])]
        x = self.ups[0](xs[-1], xs[-2])
        for i in range(1, len(self.ch)):
            x = self.ups[i](x, xs[len(self.ch) - i - 1])
        x = self.outc(x)
        return x


def init_weights(net, init_type='kaiming', activation='relu'):
    gain = init.calculate_gain(activation)

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    print('Initialize Unet with %s' % init_type)
    net.apply(init_func)

def netSize(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k
