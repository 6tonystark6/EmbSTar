import time
import numpy as np
import torchvision
from scipy.linalg import hadamard

from torch.autograd import Variable

import torch
import torch.nn as nn
from torchvision import models


class AlexNet(nn.Module):
    def __init__(self, bit):
        super(AlexNet, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = original_model.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = original_model.classifier[1].weight
        cl1.bias = original_model.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = original_model.classifier[4].weight
        cl2.bias = original_model.classifier[4].bias

        self.classifier = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, bit),
        )

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        x = (x - self.mean) / self.std
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        y = self.tanh(alpha * y)
        return y


vgg_dict = {"VGG11": models.vgg11, "VGG13": models.vgg13, "VGG16": models.vgg16, "VGG19": models.vgg19,
            "VGG11BN": models.vgg11_bn, "VGG13BN": models.vgg13_bn, "VGG16BN": models.vgg16_bn,
            "VGG19BN": models.vgg19_bn}


class VGG(nn.Module):
    def __init__(self, model_name, bit):
        super(VGG, self).__init__()
        original_model = vgg_dict[model_name](pretrained=True)
        self.features = original_model.features
        self.cl1 = nn.Linear(25088, 4096)
        self.cl1.weight = original_model.classifier[0].weight
        self.cl1.bias = original_model.classifier[0].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = original_model.classifier[3].weight
        cl2.bias = original_model.classifier[3].bias

        self.classifier = nn.Sequential(
            self.cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, bit),
        )

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        x = (x - self.mean) / self.std
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        y = self.tanh(alpha * y)
        return y


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNet(nn.Module):
    def __init__(self, model_name, hash_bit):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[model_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2,
                                            self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

        self.activation = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        x = (x - self.mean) / self.std
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        y = self.activation(alpha * y)
        return y


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.hash_bit = args.hash_bit
        self.base_model = getattr(torchvision.models, args.model_type)(pretrained=True)
        self.conv1 = self.base_model.conv1
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.avgpool = self.base_model.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.fc1 = nn.Linear(self.base_model.fc.in_features, self.base_model.fc.in_features)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(self.base_model.fc.in_features, self.base_model.fc.in_features)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(self.base_model.fc.in_features, self.hash_bit)
        self.last_layer = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.hash_layer = nn.Sequential(self.fc1, self.activation1, self.dropout, self.fc2, self.activation2, self.fc3,
                                        self.last_layer)

        self.iter_num = 0
        self.scale = 1

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)

        return y


def pairwise_loss(outputs1, outputs2, label1, label2, sigmoid_param=1, data_imbalance=1):
    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
    exp_product = torch.exp(dot_product)

    exp_loss = (torch.log(1 + exp_product) - similarity * dot_product)
    loss = torch.mean(exp_loss)

    return loss


class CSQ(torch.nn.Module):

    def __init__(self, bit, batch_size, lr, backbones, dataset, n_epochs, wd, yita, save):
        global num_class, true_hash
        global random_center
        super(CSQ, self).__init__()
        self.bit = bit
        self.backbone = backbones
        self.model_name = 'CSQ_{}_{}'.format(dataset, bit)
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.save = save
        self.model = self._build_graph()

        self.p_lambda = 0.05
        self.is_single_label = dataset not in {"NUS", "COCO", "FLICKR"}
        self.criterion = torch.nn.BCELoss().cuda()
        self.dataset = dataset

        if dataset == 'NUS':
            num_class = 21
            true_hash = 'data/NUS/hash_centers/' + str(bit) + '_nus_wide_21_class.pkl'
        if dataset == 'FLICKR':
            num_class = 38
            true_hash = 'data/FLICKR/hash_centers/' + str(bit) + '_flickr_38_class.pkl'
        if dataset == 'COCO':
            num_class = 80
            true_hash = 'data/COCO/hash_centers/' + str(bit) + '_coco_80_class.pkl'

        self.register_buffer('hash_targets', torch.randint(2, (num_class, self.bit)))
        self.register_buffer('random_center', torch.randint(2, (self.bit,)))
        self.hash_targets = self.get_hash_targets(num_class, self.bit).float().cuda()
        self.random_center = (self.random_center * 2 - 1).float().cuda()

    def _build_graph(self):
        if self.backbone == 'AlexNet':
            model = AlexNet(self.bit)
        elif 'VGG' in self.backbone:
            model = VGG(self.backbone, self.bit)
        else:
            model = ResNet(self.backbone, self.bit)
        return model

    @staticmethod
    def get_hash_targets(n_class, bit):
        ha_d = hadamard(bit)
        ha_2d = np.concatenate((ha_d, -ha_d), 0)

        assert ha_2d.shape[0] >= n_class

        hash_targets = torch.from_numpy(ha_2d[:n_class]).float()
        return hash_targets

    def generate_code(self, data):
        data_input = Variable(data.cuda())
        output = self.model(data_input)
        return output

    def load(self, path, use_gpu=True):
        if not use_gpu:
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            new_state_dict = {'model.' + k: v for k, v in state_dict.items()}
            self.load_state_dict(new_state_dict)
        else:
            state_dict = torch.load(path)
            new_state_dict = {'model.' + k: v for k, v in state_dict.items()}
            self.load_state_dict(new_state_dict)

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.lr * (0.1 ** (epoch // (self.n_epochs // 3)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def Hash_center_multilables(labels,
                                Hash_center):
        is_start = True
        for label in labels:
            one_labels = (label == 1).nonzero()
            one_labels = one_labels.squeeze(1)
            Center_mean = torch.mean(Hash_center[one_labels], dim=0)
            Center_mean[Center_mean < 0] = -1
            Center_mean[Center_mean > 0] = 1
            random_center[random_center == 0] = -1
            Center_mean[Center_mean == 0] = random_center[Center_mean == 0]
            Center_mean = Center_mean.view(1, -1)

            if is_start:
                hash_center = Center_mean
                is_start = False
            else:
                hash_center = torch.cat((hash_center, Center_mean), 0)

        return hash_center

    def label2center(self, y):

        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:

            center_sum = y @ self.hash_targets
            random_center = self.random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    def loss_function(self, u, y, index):
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        quan_loss = (u.abs() - 1).pow(2).mean()
        loss = center_loss + self.p_lambda * quan_loss
        return loss

    def train_CSQ(model, train_loader, Hash_center, two_loss_epoch):
        params_list = [{'params': model.feature_layers.parameters(), 'lr': 0.05 * model.lr},
                       {'params': model.hash_layer.parameters()}]
        optimizer = torch.optim.Adam(params_list, lr=model.lr, betas=(0.9, 0.999))
        lr = model.adjust_learning_rate(optimizer, model.epoch)
        model.train()

        if model.dataset == 'NUS':
            data_imbalance = 5
        if model.dataset == 'COCO':
            data_imbalance = 1
        if model.dataset == 'FLICKR':
            data_imbalance = 1

        start_time = time.time()
        iter_num = 0
        total_loss = []
        for i, (input, label) in enumerate(train_loader):
            optimizer.zero_grad()
            hash_center = model.Hash_center_multilables(label, Hash_center)

            hash_center = Variable(hash_center).cuda()

            input = Variable(input).cuda()
            y = model(input)

            center_loss = model.criterion(0.5 * (y + 1), 0.5 * (hash_center + 1))
            Q_loss = torch.mean((torch.abs(y) - 1.0) ** 2)

            if model.epoch <= two_loss_epoch:
                loss = center_loss + 0.05 * Q_loss
            else:
                if len(label) < model.batch_size:
                    similarity_loss = 0
                else:
                    output1 = y.narrow(0, 0, int(0.5 * len(y)))
                    output2 = y.narrow(0, int(0.5 * len(y)), int(0.5 * len(y)))
                    label1 = label[0:int(0.5 * len(label))]
                    label2 = label[int(0.5 * len(label)):int(len(label))]
                    label1 = torch.autograd.Variable(label1).cuda()
                    label2 = torch.autograd.Variable(label2).cuda()
                    similarity_loss = pairwise_loss(output1, output2, label1, label2,
                                                    sigmoid_param=10. / model.bit,
                                                    data_imbalance=data_imbalance)
                loss = center_loss + 0.2 * similarity_loss + 0.05 * Q_loss

            loss.backward()
            optimizer.step()
            iter_num += 1
            total_loss.append(loss.data.cpu().numpy())

            if i % 100 == 0:
                end_time1 = time.time()
                print('epoch: %d, lr: %.5f iter_num: %d, time: %.3f, loss: %.3f' % (
                    model.epoch, lr, iter_num, (end_time1 - start_time), loss))

        end_epoch_time = time.time()
        epoch_loss = np.mean(total_loss)
        print('Epoch: %d, time: %.3f, epoch loss: %.3f' % (model.epoch, end_epoch_time - start_time, epoch_loss))

    def forward(self, x, alpha=1):
        return self.model(x, alpha)
