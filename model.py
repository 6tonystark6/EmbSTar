import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision

from utils import spectral_norm as SpectralNorm
from utils import h_swish, h_sigmoid


class KnockoffModel(nn.Module):
    def __init__(self, bit):
        super(KnockoffModel, self).__init__()
        self.bit = bit
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Sequential()
        for p in self.parameters():
            p.requires_grad = False
        self.knockoff_model = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, self.bit),
            nn.Tanh())

    def forward(self, feat):
        feat = self.model(feat)
        feat = self.knockoff_model(feat)
        return feat

    def generate_hash_code(self, data):
        num_data = data.size(0)
        feats = torch.zeros(num_data, self.bit)
        for i in range(num_data):
            feat = self.model(data[i].type(torch.float).unsqueeze(0).cuda())
            feat = self.knockoff_model(feat)
            feats[i, :] = feat
        return torch.sign(feats)


class LabelNet(nn.Module):
    def __init__(self, bit, num_classes):
        super(LabelNet, self).__init__()
        self.curr_dim = 16
        self.size = 32

        self.feature = nn.Sequential(
            nn.Linear(num_classes, 4096),
            nn.ReLU(True),
            nn.Linear(4096, self.curr_dim * self.size * self.size)
        )

        conv2d = [
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.InstanceNorm2d(64),
            nn.Tanh()
        ]

        self.conv2d = nn.Sequential(*conv2d)

    def forward(self, label_feature):
        label_feature = self.feature(label_feature)
        label_feature = label_feature.view(label_feature.size(0), self.curr_dim, self.size, self.size)
        label_feature = self.conv2d(label_feature)
        return label_feature


class ImageNet(nn.Module):
    def __init__(self, dim_image, bit, num_classes):
        super(ImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(16, 16), mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.features(x)
        return x


class PrototypeNet(nn.Module):
    def __init__(self, dim_image, bit, num_classes, channels=64, r=4):
        super(PrototypeNet, self).__init__()

        self.labelnet = LabelNet(bit, num_classes)
        self.imagenet = ImageNet(dim_image, bit, num_classes)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, channels // r)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax()
        self.feature_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

        self.conv2d = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.Tanh()
        )

        self.hashing = nn.Sequential(nn.Linear(4096, bit), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(4096, num_classes), nn.Sigmoid())

    def forward(self, label_feature, image_feature):
        label_feature = self.labelnet(label_feature)
        image_feature = self.imagenet(image_feature)

        identity_l = label_feature
        identity_i = image_feature

        l_n, l_c, l_h, l_w = label_feature.size()
        l_xh = self.pool_h(label_feature)
        l_xw = self.pool_w(label_feature).permute(0, 1, 3, 2)
        y_l = torch.cat([l_xh, l_xw], dim=2)
        y_l = self.conv1(y_l)
        y_l = self.bn1(y_l)
        y_l = self.act(y_l)
        l_xh, l_xw = torch.split(y_l, [l_h, l_w], dim=2)
        l_xw = l_xw.permute(0, 1, 3, 2)
        a_hl = self.conv_h(l_xh).sigmoid()
        a_wl = self.conv_w(l_xw).sigmoid()

        i_n, i_c, i_h, i_w = label_feature.size()
        i_xh = self.pool_h(label_feature)
        i_xw = self.pool_w(label_feature).permute(0, 1, 3, 2)
        y_i = torch.cat([i_xh, i_xw], dim=2)
        y_i = self.conv1(y_i)
        y_i = self.bn1(y_i)
        y_i = self.act(y_i)
        i_xh, i_xw = torch.split(y_i, [i_h, i_w], dim=2)
        i_xw = i_xw.permute(0, 1, 3, 2)
        a_hi = self.conv_h(i_xh).sigmoid()
        a_wi = self.conv_w(i_xw).sigmoid()

        attn_l = a_wl * a_hl
        attn_i = a_wi * a_hi

        mix1_feature_l = identity_l * attn_l + identity_i * (1 - attn_l)
        mix1_feature_i = identity_i * attn_i + identity_l * (1 - attn_i)

        agg_attn = self.softmax(attn_l + attn_i)
        F_l = mix1_feature_l + identity_l * agg_attn
        F_i = mix1_feature_i + identity_i * agg_attn
        mixed_feature = torch.cat((F_l, F_i), dim=1)
        mixed_feature = self.feature_conv(mixed_feature)

        mixed_tensor = self.conv2d(mixed_feature)
        mixed_tensor = mixed_tensor.view(mixed_tensor.size(0), -1)
        mixed_hashcode = self.hashing(mixed_tensor)
        mixed_label = self.classifier(mixed_tensor)
        return mixed_feature, mixed_hashcode, mixed_label


class Translator(nn.Module):
    def __init__(self):
        super(Translator, self).__init__()

        transform = [
            nn.ConvTranspose2d(64, 48, kernel_size=3, stride=1, padding=2, bias=False),
            nn.InstanceNorm2d(48, affine=False),
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.ConvTranspose2d(48, 12, kernel_size=6, stride=4, padding=1, bias=False),
            nn.InstanceNorm2d(12, affine=False),
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.ConvTranspose2d(12, 1, kernel_size=6, stride=4, padding=1, bias=False),
            nn.InstanceNorm2d(1, affine=False),
            nn.ReLU(inplace=True)  # ReLU激活函数
        ]

        self.transform = nn.Sequential(*transform)

    def forward(self, label_feature):
        label_feature = self.transform(label_feature)
        return label_feature


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.translator = Translator()
        curr_dim = 64

        self.preprocess = nn.Sequential(
            nn.Conv2d(6, curr_dim, kernel_size=7, stride=1, padding=3, bias=True),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True))
        self.firstconv = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True))
        self.secondconv = nn.Sequential(
            nn.Conv2d(curr_dim * 2, curr_dim * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(curr_dim * 4),
            nn.ReLU(inplace=True))
        self.residualblock = nn.Sequential(
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'))
        self.firstconvtrans = nn.Sequential(
            nn.ConvTranspose2d(curr_dim * 4, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True))
        self.secondconvtrans = nn.Sequential(
            nn.Conv2d(curr_dim * 4, curr_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(curr_dim * 2, curr_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True))
        self.process = nn.Sequential(
            nn.Conv2d(curr_dim * 2, curr_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(curr_dim, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True))
        self.residual = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh())

    def forward(self, x, mixed_feature):
        mixed_feature = self.translator(mixed_feature)
        tmp_tensor = torch.cat((x[:, 0, :, :].unsqueeze(1), mixed_feature, x[:, 1, :, :].unsqueeze(1), mixed_feature,
                                x[:, 2, :, :].unsqueeze(1), mixed_feature), dim=1)
        tmp_tensor = self.preprocess(tmp_tensor)
        tmp_tensor_first = tmp_tensor
        tmp_tensor = self.firstconv(tmp_tensor)
        tmp_tensor_second = tmp_tensor
        tmp_tensor = self.secondconv(tmp_tensor)
        tmp_tensor = self.residualblock(tmp_tensor)
        tmp_tensor = self.firstconvtrans(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_second, tmp_tensor), dim=1)
        tmp_tensor = self.secondconvtrans(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_first, tmp_tensor), dim=1)
        tmp_tensor = self.process(tmp_tensor)
        tmp_tensor = torch.cat((x, tmp_tensor), dim=1)
        tmp_tensor = self.residual(tmp_tensor)
        return tmp_tensor


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, net_mode=None):

        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine)
        )

    def forward(self, x):
        return x + self.main(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes, image_size=224, conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()
        layers = []

        layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = conv_dim

        for i in range(1, repeat_num):
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / (2 ** repeat_num))

        self.main = nn.Sequential(*layers)
        self.fc = nn.Conv2d(curr_dim, num_classes + 1, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out = self.fc(h)
        return out.squeeze()


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=0.0, target_fake_label=1.0):
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, label, target_is_real):
        if target_is_real:
            real_label = self.real_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, real_label], dim=-1)
        else:
            fake_label = self.fake_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, fake_label], dim=-1)
        return target_tensor

    def __call__(self, prediction, label, target_is_real):

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(label, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count -
                             opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)

    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)

    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.n_epochs,
                                                   eta_min=0)

    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)

    return scheduler
