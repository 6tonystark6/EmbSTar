import os
import numpy as np
import scipy.io as scio
from torchvision import transforms
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb

from model import KnockoffModel, PrototypeNet, Generator, Discriminator, GANLoss, get_scheduler
from utils import set_input_images, CalcSim, log_trick, CalcMap, mkdir_p, return_results, calc_hamming
from attacked_model import Attacked_Model


class EmbSTar(nn.Module):
    def __init__(self, args, DataConfigs):
        super(EmbSTar, self).__init__()
        self.bit = args.bit
        self.num_classes = DataConfigs.num_label
        self.dim_image = DataConfigs.tag_dim
        self.batch_size = args.batch_size
        self.model_name = '{}_{}_{}'.format(args.dataset, args.attacked_method, args.bit)
        self.args = args
        self.klr = args.klr
        self.kbz = args.kbz
        self.kb = args.kb
        self.ke = args.ke
        self._build_model(args, DataConfigs)
        self._save_setting(args)
        if self.args.transfer_attack:
            self.transfer_bit = args.transfer_bit
            self.transfer_model = Attacked_Model(args.transfer_attacked_method, args.dataset, args.transfer_bit,
                                                 args.attacked_models_path, args.dataset_path)
            self.transfer_model.eval()

    def _build_model(self, args, Dcfg):
        pretrain_model = scio.loadmat(Dcfg.vgg_path)
        self.knockoff = KnockoffModel(self.bit).train().cuda()
        self.prototypenet = nn.DataParallel(PrototypeNet(self.dim_image, self.bit, self.num_classes)).cuda()
        self.generator = nn.DataParallel(Generator()).cuda()
        self.discriminator = nn.DataParallel(Discriminator(self.num_classes)).cuda()
        self.criterionGAN = GANLoss('lsgan').cuda()
        self.attacked_model = Attacked_Model(args.attacked_method, args.dataset, args.bit, args.attacked_models_path,
                                             args.dataset_path)
        self.attacked_model.eval()

    def _save_setting(self, args):
        self.output_dir = os.path.join(args.output_path, args.output_dir)
        self.model_dir = os.path.join(self.output_dir, 'Model')
        self.image_dir = os.path.join(self.output_dir, 'Image')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)

    def sample(self, image, sample_dir, name):
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        image = image.cpu().detach()[0]
        image = transforms.ToPILImage()(image)
        image.convert(mode='RGB').save(os.path.join(sample_dir, name + '.png'), quality=100)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.args.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()
        self.args.lr = self.optimizers[0].param_groups[0]['lr']

    def train_knockoff(self, Tr):
        print('train knockoff...')
        query_sampling_number = 2000
        near_sample_number = 5
        rank_sample_number = 5
        optimizer_knockoff = torch.optim.Adam(filter(lambda p: p.requires_grad, self.knockoff.parameters()), lr=self.klr,
                                         betas=(0.5, 0.999))
        index_FS = np.random.choice(range(Tr.size(0)), query_sampling_number, replace=False)
        qB = self.attacked_model.generate_image_hashcode(Tr[index_FS].type(torch.float).cuda())
        dB = self.attacked_model.generate_image_hashcode(Tr)
        index_matrix_before = return_results(index_FS, qB, dB, near_sample_number, rank_sample_number)
        train_sample_numbers = query_sampling_number * comb(rank_sample_number, 2).astype(int)
        index_matrix_after = np.zeros((train_sample_numbers, 4), int)
        line = 0
        for i in range(query_sampling_number):
            for j in range(near_sample_number + 1, near_sample_number + rank_sample_number):
                for k in range(j + 1, near_sample_number + rank_sample_number + 1):
                    index_matrix_after[line, :3] = index_matrix_before[i, [0, j, k]]
                    index_matrix_after[line, 3] = k - j
                    line = line + 1
        ranking_loss = torch.nn.MarginRankingLoss(margin=0.1)
        for epoch in range(self.ke):
            index = np.random.permutation(train_sample_numbers)
            for i in range(train_sample_numbers // self.kbz + 1):
                optimizer_knockoff.zero_grad()
                end_index = min((i + 1) * self.kbz, train_sample_numbers)
                num_index = end_index - i * self.kbz
                ind = index[i * self.kbz: end_index]

                anchor = self.knockoff(Tr[index_matrix_after[ind, 0]].type(torch.float).cuda())
                rank1 = self.knockoff(Tr[index_matrix_after[ind, 1]].type(torch.float).cuda())
                rank2 = self.knockoff(Tr[index_matrix_after[ind, 2]].type(torch.float).cuda())
                ranking_target = - 1. / torch.from_numpy(index_matrix_after[ind, 3]).type(
                    torch.float).cuda()  # ranking_target = - torch.ones(num_index).cuda()
                hamming_rank1 = calc_hamming(anchor, rank1) / self.bit
                hamming_rank2 = calc_hamming(anchor, rank2) / self.bit
                rank_loss = ranking_loss(hamming_rank1.cuda(), hamming_rank2.cuda(), ranking_target)
                quant_loss = (torch.sign(anchor) - anchor).pow(2).sum() / (self.bit * num_index)

                alpha = 0.001
                loss_K = rank_loss + alpha * quant_loss
                loss_K.backward()
                optimizer_knockoff.step()
            print(
                'epoch:{:2d}    loss_K:{:.4f}    rank_loss:{:.4f}    quant_loss:{:.4f}'
                .format(epoch + 1, loss_K, rank_loss, quant_loss))
        self.save_knockoff_model()
        print('train knockoff done.')

    def test_knockoff(self, Te_I, Te_L, Db_I, Db_L):
        print('test knockoff model...')
        self.load_knockoff_model()
        qB = self.knockoff.generate_hash_code(Te_I)
        dB = self.knockoff.generate_hash_code(Db_I)
        map = CalcMap(qB, dB, Te_L, Db_L, 50)
        print('@50: {:.4f}'.format(map))

    def save_knockoff_model(self):
        torch.save(self.knockoff.state_dict(), os.path.join(self.model_dir, 'knockoffmodel_{}.pth'.format(self.model_name)))

    def save_prototypenet(self):
        torch.save(self.prototypenet.module.state_dict(),
                   os.path.join(self.model_dir, 'prototypenet_{}.pth'.format(self.model_name)))

    def save_generator(self):
        torch.save(self.generator.module.state_dict(),
                   os.path.join(self.model_dir, 'generator_{}.pth'.format(self.model_name)))

    def load_knockoff_model(self):
        self.knockoff.load_state_dict(torch.load(os.path.join(self.model_dir, 'knockoffmodel_{}.pth'.format(self.model_name))))
        self.knockoff.eval().cuda()

    def load_generator(self):
        self.generator.module.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'generator_{}.pth'.format(self.model_name))))
        self.generator.eval()

    def load_prototypenet(self):
        self.prototypenet.module.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'prototypenet_{}.pth'.format(self.model_name))))
        self.prototypenet.eval()

    def train_prototypenet(self, train_images, train_labels):
        num_train = train_labels.size(0)
        optimizer_a = torch.optim.Adam(self.prototypenet.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        epochs = 100
        batch_size = 64
        steps = num_train // batch_size + 1
        lr_steps = epochs * steps
        scheduler_a = torch.optim.lr_scheduler.MultiStepLR(optimizer_a, milestones=[lr_steps / 2, lr_steps * 3 / 4],
                                                           gamma=0.1)
        criterion_l2 = torch.nn.MSELoss()
        B = self.attacked_model.generate_image_hashcode(train_images).cuda()
        for epoch in range(epochs):
            index = np.random.permutation(num_train)
            for i in range(steps):
                end_index = min((i + 1) * batch_size, num_train)
                num_index = end_index - i * batch_size
                ind = index[i * batch_size: end_index]
                batch_image = Variable(train_images[ind]).type(torch.float).cuda()
                batch_label = Variable(train_labels[ind]).type(torch.float).cuda()
                optimizer_a.zero_grad()
                _, mixed_h, mixed_l = self.prototypenet(batch_label, batch_image)
                S = CalcSim(batch_label.cpu(), train_labels.type(torch.float))
                theta_m = mixed_h.mm(Variable(B).t()) / 2
                logloss_m = - ((Variable(S.cuda()) * theta_m - log_trick(theta_m)).sum() / (num_train * num_index))
                regterm_m = (torch.sign(mixed_h) - mixed_h).pow(2).sum() / num_index
                classifer_m = criterion_l2(mixed_l, batch_label)
                loss = classifer_m + 5 * logloss_m + 1e-3 * regterm_m
                loss.backward()
                optimizer_a.step()
                if i % self.args.print_freq == 0:
                    print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, l_m:{:.5f}, r_m: {:.5f}, c_m: {:.7f}'
                          .format(epoch + 1, i, scheduler_a.get_last_lr()[0], logloss_m, regterm_m, classifer_m))
                scheduler_a.step()
        self.save_prototypenet()

    def test_prototypenet(self, test_images, test_labels, database_images, database_labels):
        self.load_prototypenet()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        for i in range(num_test):
            _, mixed_h, __ = self.prototypenet(test_labels[i].cuda().float().unsqueeze(0),
                                               test_images[i].cuda().float().unsqueeze(0))
            qB[i, :] = torch.sign(mixed_h.cpu().data)[0]
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        map = CalcMap(qB, IdB, test_labels, database_labels, 50)
        print('MAP: %3.5f' % map)

    def train(self, train_images, train_labels, database_images, database_labels, test_images, test_labels):
        self.train_prototypenet(train_images, train_labels)
        self.test_prototypenet(test_images, test_labels, database_images, database_labels)
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.optimizers = [optimizer_g, optimizer_d]
        self.schedulers = [get_scheduler(opt, self.args) for opt in self.optimizers]
        num_train = train_labels.size(0)
        batch_size = self.batch_size
        total_epochs = self.args.n_epochs + self.args.n_epochs_decay + 1
        criterion_l2 = torch.nn.MSELoss()
        B = self.attacked_model.generate_image_feature(train_images)
        B = B.cuda()
        for epoch in range(self.args.epoch_count, total_epochs):
            print('\nTrain epoch: {}, learning rate: {:.7f}'.format(epoch, self.args.lr))
            index = np.random.permutation(num_train)
            for i in range(num_train // batch_size + 1):
                end_index = min((i + 1) * batch_size, num_train)
                num_index = end_index - i * batch_size
                ind = index[i * batch_size: end_index]

                if num_index == 0:
                    print(f"Skipping empty batch at epoch {epoch}, step {i}")
                    continue

                batch_label = Variable(train_labels[ind]).type(torch.float).cuda()
                batch_image = Variable(train_images[ind]).type(torch.float).cuda()
                batch_image = set_input_images(batch_image / 255)
                select_index = np.random.choice(range(train_labels.size(0)), size=num_index)
                batch_target_label = train_labels.index_select(0, torch.from_numpy(select_index)).type(
                    torch.float).cuda()
                batch_target_image = train_images.index_select(0, torch.from_numpy(select_index)).type(
                    torch.float).cuda()
                label_feature, target_hashcode, _ = self.prototypenet(batch_target_label, batch_target_image)
                batch_fake_image = self.generator(batch_image, label_feature.detach())
                if i % 3 == 0:
                    self.set_requires_grad(self.discriminator, True)
                    optimizer_d.zero_grad()
                    batch_image_d = self.discriminator(batch_image)
                    batch_fake_image_d = self.discriminator(batch_fake_image.detach())
                    real_d_loss = self.criterionGAN(batch_image_d, batch_label, True)
                    fake_d_loss = self.criterionGAN(batch_fake_image_d, batch_target_label, False)
                    d_loss = (real_d_loss + fake_d_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()
                self.set_requires_grad(self.discriminator, False)
                optimizer_g.zero_grad()
                batch_fake_image_m = (batch_fake_image + 1) / 2 * 255
                predicted_target_hash = self.attacked_model.image_model(batch_fake_image_m)
                logloss = - torch.mean(predicted_target_hash * target_hashcode) + 1
                batch_fake_image_d = self.discriminator(batch_fake_image)
                fake_g_loss = self.criterionGAN(batch_fake_image_d, batch_target_label, True)
                reconstruction_loss_l = criterion_l2(batch_fake_image, batch_image)
                # backpropagation
                g_loss = 5 * logloss + 1 * fake_g_loss + 150 * reconstruction_loss_l
                g_loss.backward()
                optimizer_g.step()
                if i % self.args.sample_freq == 0:
                    self.sample((batch_fake_image + 1) / 2, '{}/'.format(self.image_dir),
                                str(epoch) + '_' + str(i) + '_fake')
                    self.sample((batch_image + 1) / 2, '{}/'.format(self.image_dir),
                                str(epoch) + '_' + str(i) + '_real')
                if i % self.args.print_freq == 0:
                    print(
                        'step: {:3d} d_loss: {:.3f} g_loss: {:.3f} fake_g_loss: {:.3f} logloss: {:.3f} r_loss_l: {:.7f}'
                        .format(i, d_loss, g_loss, fake_g_loss, logloss, reconstruction_loss_l))
            self.update_learning_rate()
        self.save_generator()

    def test(self, database_images, database_labels, test_images, test_labels, dataset, attacked_method, bit):
        self.load_prototypenet()
        self.load_generator()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        perceptibility = 0
        batch_size = 100

        for i in range(0, num_test, batch_size):
            end_i = min(i + batch_size, num_test)
            test_images_batch = test_images[i:end_i].type(torch.float).cuda() / 255
            test_labels_batch = test_labels[i:end_i]

            select_index = np.random.choice(range(database_labels.size(0)), size=end_i - i)
            target_labels = database_labels.index_select(0, torch.from_numpy(select_index)).type(torch.float).cuda()
            target_images = database_images.index_select(0, torch.from_numpy(select_index)).type(torch.float).cuda()

            for j in range(end_i - i):
                label_feature, _, __ = self.prototypenet(target_labels[j].unsqueeze(0), target_images[j].unsqueeze(0))
                original_image = set_input_images(test_images_batch[j].unsqueeze(0))
                fake_image = self.generator(original_image, label_feature)
                fake_image = (fake_image + 1) / 2
                original_image = (original_image + 1) / 2
                target_image = 255 * fake_image
                target_hashcode = self.attacked_model.generate_image_hashcode(target_image)
                qB[i + j, :] = torch.sign(target_hashcode.cpu().data)
                perceptibility += F.mse_loss(original_image, fake_image.squeeze(0)).item()

        print('generate target images end!')
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        perceptibility = torch.sqrt(torch.tensor(perceptibility / num_test)).cuda()
        print('perceptibility: {:.7f}'.format(perceptibility))

        t_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('tMAP: %3.5f' % t_map)

    def transfer_attack(self, database_images, database_labels, test_images, test_labels):
        self.load_prototypenet()
        self.load_generator()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.transfer_bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size=test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_images = database_images.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images...')
        for i in range(num_test):
            label_feature, _, __ = self.prototypenet(target_labels[i].unsqueeze(0), target_images[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda() / 255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            target_image = 255 * fake_image
            target_hashcode = self.transfer_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, fake_image[0]).data
        print('generate target images end!')
        IdB = self.transfer_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility / num_test)))
        t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        print('tMAP: %3.5f' % t_map)
