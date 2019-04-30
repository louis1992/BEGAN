# Too deep neural networks have some problem such as gradient vanishing/exploding or degradation.
# But ResNet can be stacked over 1000 layers.
# Because ResNet has skip connection.
# theme site : color-themes.com/?view=index

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable

#import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils

from dataloader import *
from layer import Encoder, Decoder, Discriminator, L1_loss

import argparse
from collections import deque

# Learning parameters.
# Learning rate     : 0.0001
# Input image size  : 128 * 128
# Output image size : 128 * 128
# Batch size        : 16

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--npgu', default=1, type=int)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--manualSeed', default=5451, type=int)

# Learning rate arguments.
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--lr_update_step', default=3000, type=int)
parser.add_argument('--lr_update_type', default=1, type=int)
parser.add_argument('--lr_lower_bound', default=2e-6, type=float)

parser.add_argument('--train', default=1, type=int)
parser.add_argument('--b_size', default=16, type=int)
parser.add_argument('--h', default=64, type=int)
parser.add_argument('--nc', default=64, type=int)
parser.add_argument('--epochs', default=100, type=int)

parser.add_argument('--input_size', default=128, type=int)
parser.add_argument('--output_size', default=128, type=int)
parser.add_argument('--input_vector', default=64, type=int)
parser.add_argument('--tanh', default=1, type=int)
parser.add_argument('--scale_size', default=128, type=int)
parser.add_argument('--load_step', default=0, type=int)
parser.add_argument('--print_step', default=100, type=int)
parser.add_argument('--num_workers', default=12, type=int)

# Loss arguments.
parser.add_argument('--loss_type', default=1, type=int)
parser.add_argument('--k', default=0, type=int)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--lambda_k', default=0.001, type=float)

# Directory arguments.
parser.add_argument('--model_name', default='test')
parser.add_argument('--base_path', default='./')
parser.add_argument('--data_path', default='data/img_align_celeba/')
opt = parser.parse_args()

use_cuda = True

print ('Random seed: ', opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    opt.cuda = True
    torch.cuda.set_device(opt.gpuid)
    torch.cuda.manual_seed_all(opt.manualSeed)

#
class genFeat3DMM(nn.Module):
    def __init__(self):
        super(genFeat3DMM, self).__init__()
        # 4 layers MPL.
        self.layer0 = nn.Linear(64, 128)
        self.layer1 = nn.Linear(128,256)
        self.layer2 = nn.Linear(256,256)

    def forward(self, x):
        x = F.elu(self.layer0(x), True)
        x = F.elu(self.layer1(x), True)
        x = F.elu(self.layer2(x), True)
        return x

# Generator makes general feature from noise vector.
class genFeatGeneral(nn.Module):
    def __init__(self):
        super(genFeatGeneral, self).__init__()
        self.layer0 = nn.Linear(64, 128)
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 256)

    def forward(self, x):
        x = F.elu(self.layer0(x), True)
        x = F.elu(self.layer1(x), True)
        x = F.elu(self.layer2(x), True)
        return x

# Generator synthesize reconstructed image from features.
class genImage(nn.Module):
    def __init__(self, opt):
        super(genImage, self).__init__()
        self.dec = Decoder(opt)

    def forward(self, x):
        x = self.dec(x)
        return x

class disfFeat3DMM(nn.Module):
    def __init__(self):
        super(disfFeat3DMM, self).__init__()
        self.layer0 = nn.Linear(256, 256)
        self.layer1 = nn.Linear(256, 128)
        self.layer2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.elu(self.layer0(x), True)
        x = F.elu(self.layer1(x), True)
        x = F.elu(self.layer2(x), True)
        return x

class disFeatGeneral(nn.Module):
    def __init__(self):
        super(disFeatGeneral, self).__init__()
        self.layer0 = nn.Linear(256, 256)
        self.layer1 = nn.Linear(256, 128)
        self.layer2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.elu(self.layer0(x), True)
        x = F.elu(self.layer1(x), True)
        x = F.elu(self.layer2(x), True)
        return x

# D_I (image discriminator) apply the structure described in BEGAN.
class disImage(nn.Module):
    def __init__(self, opt):
        super(disImage, self).__init__()
        self.enc = Encoder(opt)
        self.dec = Decoder(opt)

    def forward(self, x):
        x = self.dec(self.enc(x))
        return x

# Extractors (E_1, E_2, E_id).
# E_1 employs ResNet18 to extract 3DMM features.
class extFeat3DMM(nn.Module):
    def __init__(self):
        super(extFeat3DMM, self).__init__()
        self.net = models.resnet18(pretrained=False)                 # ResNet-18 structure resnet18(pretrained=False, **kargs).

    def forward(self, x):
        x = self.net(x)
        return x

# https://github.com/anantzoid/BEGAN-pytorch/blob/master/models.py
class extFeatGeneral(nn.Module):
    def __init__(self, opt):
        super(extFeatGeneral, self).__init__()
        self.num_channels = opt.nc
        self.h = opt.h
        self.b_size = opt.b_size
        self.scale_size = opt.scale_size
        
        self.layer0 = nn.Conv2d(3, self.num_channels, 3, 1, 1)
        self.layer1 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.layer2 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.down1 = nn.Conv2d(self.num_channels, self.num_channels, 1, 1 , 0)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.layer3 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.layer4 = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1)
        self.down2 = nn.Conv2d(self.num_channels, 2*self.num_channels, 1, 1, 0)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.layer5 = nn.Conv2d(2*self.num_channels, 2*self.num_channels, 3, 1, 1)
        self.layer6 = nn.Conv2d(2*self.num_channels, 2*self.num_channels, 3 ,1 ,1)
        self.down3 = nn.Conv2d(2*self.num_channels, 3*self.num_channels, 1, 1, 0)
        self.pool3 = nn.AvgPool2d(2, 2)

        if self.scale_size == 64:
            self.layer7 = nn.Conv2d(3*self.num_channels, 3*self.num_channels, 3, 1, 1)
            self.layer8 = nn.Conv2d(3*self.num_channels, 3*self.num_channels, 3, 1, 1)
            self.layer9 = nn.Linear(8*8*3*self.num_channels, 64)

        elif self.scale_size == 128:
            self.layer7 = nn.Conv2d(3*self.num_channels, 3*self.num_channels, 3, 1, 1)
            self.layer8 = nn.Conv2d(3*self.num_channels, 3*self.num_channels, 3, 1, 1)
            self.down4 = nn.Conv2d(3*self.num_channels, 4*self.num_channels, 1, 1, 0)
            self.pool4 = nn.AvgPool2d(2, 2)

            self.layer9 = nn.Conv2d(4*self.num_channels, 4*self.num_channels, 3, 1, 1)
            self.layer10 = nn.Conv2d(4*self.num_channels, 4*self.num_channels, 3, 1, 1)
            self.layer11 = nn.Linear(8*8*4*self.num_channels, self.h)

    def forward(self, x):
        x = F.elu(self.layer0(x), True)
        x = F.elu(self.layer1(x), True)
        x = F.elu(self.layer2(x), True)
        x = self.down1(x)
        x = self.pool1(x)

        x = F.elu(self.layer3(x), True)
        x = F.elu(self.layer4(x), True)
        x = self.down2(x)
        x = self.pool2(x)

        x = F.elu(self.layer5(x), True)
        x = F.elu(self.layer6(x), True)
        x = self.down3(x)
        x = self.pool3(x)

        if self.scale_size == 64:
            x = F.elu(self.layer7(x), True)
            x = F.elu(self.layer8(x), True)
            x = x.view(self.b_size, 8*8*3*self.num_channels)
            x = self.layer9(x)

        elif self.scale_size == 128:
            x = F.elu(self.layer7(x), True)
            x = F.elu(self.layer8(x), True)
            x = self.down4(x)
            x = self.pool4(x)
            x = F.elu(self.layer9(x), True)
            x = F.elu(self.layer10(x), True)
            x = x.view(self.b_size, 8*8*4*self.num_channels)
            x = F.elu(self.layer11(x), True)

        return x

# E_id employs ResNet50 to extract identity features.
class extIdentity(nn.Module):
    def __init__(self):
        super(extIdentity, self).__init__()
        self.net = models.resnet50(pretrained=False)                 # ResNet-50 structure resnet50(pretrained=False, **kargs).

    def forward(self, x):
        x = self.net(x)
        return x

class FaceFeatGAN():
    def __init__(self):
        self.global_step = opt.load_step
        self.prepare_paths()
        self.data_loader = get_loader(self.data_path, opt.b_size, opt.scale_size, opt.num_workers)

        self.build_model()

        self.z = Variable(torch.FloatTensor(opt.b_size, opt.h))
        self.fixed_z = Variable(torch.FloatTensor(opt.b_size, opt.h))
        self.fixed_z.data.uniform_(-1, 1)
        self.fixed_x = None

        self.criterion = L1_loss()

        if opt.cuda:
            self.set_cuda()

    def set_cuda(self):
        # Upload models to cuda.
        self.disc.cuda()
        self.gen.cuda()

        # Upload data to cuda.
        self.z = self.z.cuda()
        self.fixed_z = self.fixed_z.cuda()

        self.criterion.cuda()

    def write_config(self, step):
        f = open(os.path.join(opt.base_path, 'experiments/%s/params/%d.cfg' % (opt.model_name, step)), 'w')
        # File writting in python3
        print (vars(opt), file=f)
        # print >> f, vars(opt) in python2.
        f.close()

    def prepare_paths(self):
        self.data_path = os.path.join(opt.base_path, opt.data_path)
        self.gen_save_path = os.path.join(opt.base_path, 'experiments/%s/models' % opt.model_name)
        self.disc_save_path = os.path.join(opt.base_path, 'experiments/%s/models' % opt.model_name)
        self.sample_dir = os.path.join(opt.base_path, 'experiments/%s/models' % opt.model_name)
        param_dir = os.path.join(opt.base_path, 'experiments/%s/params' % opt.model_name)

        for path in [self.gen_save_path, self.disc_save_path, self.sample_dir, param_dir]:
            if not os.path.exists(path):
                print ('Generate %s directory' % path)
                os.makedirs(path)
        print ('Generated samples saved in %s' % self.sample_dir)

    def build_model(self):
        self.disc = Discriminator(opt)
        self.gen = Decoder(opt)

        print ('BEGAN Structure: ')
        print ('DISCRIMINATOR: ')
        print (self.disc)
        print ('GENERATOR: ')
        print (self.gen)

        if opt.load_step > 0:
            self.load_models(opt.load_step)

    def save_models(self, step):
        torch.save(self.gen.state_dict(), os.path.join(self.gen_save_path, 'gen_%d.pth' % step))
        torch.save(self.disc.state_dict(), os.path.join(self.disc_save_path, 'disc_%d.pth' % step))
        self.write_config(step)

    def load_models(self, step):
        self.gen.load_state_dict(torch.load(os.path.join(self.gen_save_path, 'gen_%d.pth' % step)))
        self.disc.load_state_dict(torch.load(os.path.join(self.gen_save_path, 'disc_%d.pth' % step)))

    # Loss functions.
    def compute_disc_loss(self, outputs_d_x, data, outputs_d_z, gen_z):
        if opt.loss_type == 1:
            real_loss_d = torch.mean(torch.abs(outputs_d_x - data))     # ? - real image.
            fake_loss_d = torch.mean(torch.abs(outputs_d_z - gen_z))    # ? - fake image.
        else:
            real_loss_d = self.criterion(outputs_d_x, data)
            fake_loss_d = self.criterion(outputs_d_z, gen_z.detach())
        return (real_loss_d, fake_loss_d)

    def compute_gen_loss(self, outputs_g_z, gen_z):
        if opt.loss_type == 1:
            return torch.mean(torch.abs(outputs_g_z - gen_z))
        else:
            return self.criterion(outputs_g_z, gen_z)

    #def generate(self, gen_z, outputs_d_x, global_step):
    def generate(self, sample, recon, step, nrow=8):
        vutils.save_image(sample.data, '%s/%s_%s_gen.png' % (self.sample_dir, opt.model_name, str(step)), nrow=nrow, normalize=True)
        if recon is not None:
            vutils.save_image(recon.data, '%s/%s_%s_disc.png' % (self.sample_dir, opt.model_name, str(step)), nrow=nrow, normalize=True)

    def train(self):
        g_optimizer = torch.optim.Adam(self.gen.parameters(), betas=(0.5, 0.999), lr=opt.lr)
        d_optimizer = torch.optim.Adam(self.disc.parameters(), betas=(0.5, 0.999), lr=opt.lr)

        measure_history = deque([0] * opt.lr_update_step, opt.lr_update_step)
        convergence_history = []
        prev_measure = 1

        lr = opt.lr

        for i in range(opt.epochs):
            for _, data in enumerate(self.data_loader):
                data = Variable(data)
                if data.size(0) != opt.b_size:
                    print (data.size(0))
                    print (opt.b_size)
                    continue

                if opt.cuda:
                    data = data.cuda()

                if self.fixed_x is not None:
                    self.fixed_x = data

                # Training discriminator.
                # Initialize gradient.
                self.disc.zero_grad()

                self.z.data.uniform_(-1, 1)
                gen_z = self.gen(self.z)                    # Generated image.
                outputs_d_z = self.disc(gen_z.detach())     # Discriminate gererated image.
                outputs_d_x = self.disc(data)               # Discriminate real image.

                real_loss_d, fake_loss_d = self.compute_disc_loss(outputs_d_x, data, outputs_d_z, gen_z)

                loss_d = real_loss_d - opt.k * fake_loss_d
                loss_d.backward()
                d_optimizer.step()

                # Training generator.
                self.gen.zero_grad()

                gen_z = self.gen(self.z)
                outputs_g_z = self.disc(gen_z)

                loss_g = self.compute_gen_loss(outputs_g_z, gen_z)
                loss_g.backward()
                g_optimizer.step()

                #
                #balance = (opt.gamma*real_loss_d - fake_loss_d).data[0]     # Gamma * real loss - fake loss.
                balance = (opt.gamma*real_loss_d - fake_loss_d).data
                opt.k += opt.lambda_k * balance
                opt.k = max(min(1, opt.k), 0)

                #convg_measure = real_loss_d.data[0] + np.abs(balance)
                convg_measure = real_loss_d.data + np.abs(balance)
                measure_history.append(convg_measure)
                if self.global_step % opt.print_step == 0:
                    print ('Step: %d, Epochs: %d, Loss D: %.9f, real_loss: %.9f, fake_loss: %.9f, Loss G: %.9f, k: %f, M: %.9f, lr: %.9f'
                           % (self.global_step, i, loss_d.data, real_loss_d.data, fake_loss_d.data, loss_g.data, opt.k, convg_measure, lr))
                    self.generate(gen_z, outputs_d_x, self.global_step)     # Save generated results.

                # Update learning rate.
                if opt.lr_update_type == 1:
                    lr = opt.lr * 0.95 ** (self.global_step//opt.lr_update_step)
                elif opt.lr_update_type == 2:
                    if self.global_step % opt.lr_update_step == opt.lr_update_step - 1:
                        lr *= 0.5
                elif opt.lr_update_type == 3:
                    if self.global_step % opt.lr_update_step == opt.lr_update_step - 1:
                        lr = min(lr*0.5, opt.lr_lower_bound)
                else:
                    if self.global_step % opt.lr_update_step == opt.lr_update_step - 1:
                        cur_measure = np.mean(measure_history)
                        if cur_measure > prev_measure * 0.9999:
                            lr = min(lr*0.5, opt.lr_lower_bound)
                        prev_measure = cur_measure

                for p in g_optimizer.param_groups + d_optimizer.param_groups:
                    p['lr'] = lr

                if self.global_step % 1000 == 0:
                    self.save_models(self.global_step)

                self.global_step += 1

def generative_exp(obj):
    z = []
    for inter in range(10):
        z0 = np.random.uniform(-1, 1, opt.h)
        z10 = np.random.uniform(-1, 1, opt.h)

# genFeat3DMM
# genFeatGeneral
# genImage
# disFeat3DMM
# disFeatGeneral
# disImage
# extFeat3DMM
# extFeatGeneral
# extIdentity

def describeStructure():
    print ('FaceFeatGAN neural network structure: ')
    gen = [genFeat3DMM(), genFeatGeneral(), genImage(opt)]
    for g in gen:
        print (g)
        print ('#' * 90)

    dis = [disfFeat3DMM(), disFeatGeneral(), disImage(opt)]
    for d in dis:
        print (d)
        print ('#' * 90)

    ext = [extFeat3DMM(), extFeatGeneral(opt), extIdentity()]
    for e in ext:
        print (e)
        print ('#' * 90)

def main():
    print ('#' * 90)
    print ('# CUDA Available: ', torch.cuda.is_available())
    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    print ('# Device: ', device)
    print ('#' * 90)

    #describeStructure()
    obj = FaceFeatGAN()
    print (obj)

    if opt.train:
        obj.train()

if __name__=='__main__':
    main()
