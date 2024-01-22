import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class double_conv(nn.Module):
    """(conv => BN => leakyReLU => Dropout2d) * 2"""

    def __init__(self, in_ch, out_ch, batchnorm, dropout):
        super(double_conv, self).__init__()
        self.conv = []
        if batchnorm:
            self.conv.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False))
            self.conv.append(nn.BatchNorm2d(out_ch))
        else:
            self.conv.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True))

        self.conv.append(nn.LeakyReLU(inplace=True))

        if dropout is not None:
            self.conv.append(nn.Dropout2d(p=dropout))

        if batchnorm:
            self.conv.append(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False))
            self.conv.append(nn.BatchNorm2d(out_ch))
        else:
            self.conv.append(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True))

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
        self.conv = double_conv(in_ch, out_ch, False, None)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, downmode, batchnorm, dropout):
        super(down, self).__init__()
        if downmode == "conv":
            self.down = []
            if batchnorm:
                self.down.append(
                    nn.Conv2d(
                        in_ch, in_ch, kernel_size=4, stride=2, padding=1, bias=False
                    )
                )
                self.down.append(nn.BatchNorm2d(in_ch))
            else:
                self.down.append(
                    nn.Conv2d(
                        in_ch, in_ch, kernel_size=4, stride=2, padding=1, bias=True
                    )
                )

            self.down.append(nn.LeakyReLU(inplace=True))

            if dropout is not None:
                self.down.append(nn.Dropout2d(p=dropout))

            self.down = nn.Sequential(*self.down)

        else:
            self.down = nn.MaxPool2d(2)

        self.conv = double_conv(in_ch, out_ch, batchnorm, dropout)

    def forward(self, x):
        x = self.conv(self.down(x))
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, upmode, batchnorm, dropout):
        super(up, self).__init__()

        if upmode == "conv":
            self.up = []
            if batchnorm:
                self.up.append(
                    nn.ConvTranspose2d(
                        in_ch // 2,
                        in_ch // 2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    )
                )
                self.up.append(nn.BatchNorm2d(in_ch // 2))
            else:
                self.up.append(
                    nn.ConvTranspose2d(
                        in_ch // 2,
                        in_ch // 2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=True,
                    )
                )

            self.up.append(nn.LeakyReLU(inplace=True))

            if dropout is not None:
                self.up.append(nn.Dropout2d(p=dropout))

            self.up = nn.Sequential(*self.up)

        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = double_conv(in_ch, out_ch, batchnorm, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

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
    def __init__(
        self,
        ch=[64, 128, 256, 512],
        downmode="sample",
        upmode="conv",
        batchnorm=False,
        dropout=None,
    ):
        super(UNet, self).__init__()
        self.ch = ch
        self.inc = inconv(1, ch[0])
        self.downs = nn.ModuleList(
            [
                down(ch[i], ch[i + 1], downmode, batchnorm, dropout)
                for i in range(len(ch) - 1)
            ]
            + [down(ch[-1], ch[-1], downmode, batchnorm, dropout)]
        )
        self.ups = nn.ModuleList(
            [
                up(ch[i + 1] * 2, ch[i], upmode, batchnorm, dropout)
                for i in reversed(range(len(ch) - 1))
            ]
            + [up(ch[1], ch[0], upmode, batchnorm, dropout)]
        )
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


def init_weights(net, init_type="kaiming", activation="relu"):
    gain = init.calculate_gain(activation)

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    print("Initialize Unet with %s" % init_type)
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
