import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations
import matplotlib.pyplot as plt
from tqdm import tqdm 
from model import *
import os
import torch.backends.cudnn as cudnn
import pandas as pd
import albumentations
import random
import math
from PIL import Image
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

class EMNIST(Dataset):
    def __init__(self, transform = None, dir = 'C:/data/dacon_mnist/train.csv', datatype = 'train', ratio = 0.8):
        
        dataset = pd.read_csv(dir)
        num = int(len(dataset)*ratio)
        
        if datatype =='train':
            dataset = dataset[:num]
        elif datatype =='val':
            dataset = dataset[num:]
            
        self.digit = dataset['digit'].values
        self.letter = pd.get_dummies(dataset['letter']).values
        self.img = dataset.iloc[:,3:].values.reshape(-1, 28, 28)/255.
        self.transform = transform   
        
    def __len__(self):
        return len(self.img)

    def __getitem__(self,index):
        image = self.img[index]
        letter = torch.FloatTensor(self.letter[index]).unsqueeze(dim=0)
        digit = self.digit[index]
        if self.transform:
            augmented = self.transform(image=image) 
            image = augmented['image']
        image = torch.FloatTensor(image).unsqueeze(dim=0)
        sample = (image, letter, digit)
        return sample

class EMNIST_test(Dataset):
    def __init__(self, transform = None, dir = 'C:/data/dacon_mnist/test.csv'):
        
        dataset = pd.read_csv(dir)
        self.letter = pd.get_dummies(dataset['letter']).values
        self.img = dataset.iloc[:,2:].values.reshape(-1, 28, 28)/255.
        self.transform = transform
                    
    def __len__(self):
        return len(self.img)

    def __getitem__(self,index):
        image = self.img[index]
        letter = torch.FloatTensor(self.letter[index]).unsqueeze(dim=0)
        if self.transform:
            augmented = self.transform(image=image) 
            image = augmented['image']
        image = torch.FloatTensor(image).unsqueeze(dim=0)
        sample = (image, letter)
        return sample
    
albumentations_transform = albumentations.Compose([
    albumentations.Resize(112, 112), 
    albumentations.augmentations.transforms.ShiftScaleRotate(),
])

albumentations_transform_val = albumentations.Compose([
    albumentations.Resize(112, 112), 
])

albumentations_transform_test = albumentations.Compose([
    albumentations.Resize(112, 112), 
])

trainset = EMNIST(transform = albumentations_transform)
testset = EMNIST(transform = albumentations_transform_val, datatype='val')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=True, drop_last=True)

# ======================== model =================================

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv_letter = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 64, 4, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 64, 4, padding=2), nn.ReLU(),
            nn.Conv1d(64, 16, 3), nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(2432, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x,y):
        bsz = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.layer4(x)
        
        out = F.avg_pool2d(out, out.size()[3],4)
        out = out.view(out.size(0), -1)
        out = torch.cat((self.conv_letter(y).view(bsz, -1), out), dim=1)
        out = self.linear(out)
        return out
    
    def forward(self, x,letter):
        return self._forward_impl(x,letter)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)

    return model

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):

    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

def wide_resnet101_2(pretrained=False, progress=False, **kwargs):

    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

net = wide_resnet101_2().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 230, 265], gamma=0.1)

def cutout(img,ratio,n_hole,device):
    if len(img.size())==3:
        h = img.size(1)
        w = img.size(2)
        
    elif len(img.size())==4:
        h = img.size(2)
        w = img.size(3)
    else:
        raise Exception
    
    mask = np.ones((h, w), np.float32)
    
    for _ in range(n_hole):
        y = np.random.randint(h - h * ratio)
        x = np.random.randint(w - w * ratio)

        x1 = int(x + h * ratio)
        y1 = int(y + w * ratio)
        mask[y: y1, x: x1] = 0.
            
    mask = torch.from_numpy(mask).to(device)
    img = img * mask


    return img

def Cutout(img,min_ratio, max_ratio, device):
    for idx in range(img.size(0)):
        ratio = (max_ratio - min_ratio)/(10)
        ratio = min_ratio + ratio * random.randint(0,9)
        img[idx] = cutout(img[idx],ratio,1,device)
    
    return img

## =========== training ============================================
def train(net, epoch):
    net.train()
    train_loss = 0
    correct_digit = 0
    correct_letter = 0
    total = 0
    
    for batch_idx, batch in enumerate(trainloader):
        inputs = batch[0]
        l_letter = batch[1]
        l_digit = batch[2]
        
        inputs = inputs.to(device, dtype=torch.float)
        l_digit = l_digit.to(device, dtype=torch.long)
        l_letter = l_letter.to(device, dtype=torch.float)
        inputs = Cutout(inputs, 0,0.5, device)
        optimizer.zero_grad()
        out_digit = net(inputs, l_letter)
        loss = criterion(out_digit, l_digit)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, pred_digit = out_digit.max(1)

        total += l_digit.size(0)
        correct_digit += pred_digit.eq(l_digit).sum().item()
    scheduler.step()
    if epoch%10 ==0:
        print('\nEpoch: %d' % epoch)
        print(batch_idx,'/', len(trainloader), ' Loss: %.3f | Acc: %.3f%% (%d/%d)  '
                % (train_loss/(batch_idx+1), 100.*correct_digit/total, correct_digit, total))
       
def test(net, epoch):
    net.eval()
    test_loss = 0
    correct_digit = 0
    correct_letter = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            inputs = batch[0]
            l_letter = batch[1]
            l_digit = batch[2]
        
            inputs = inputs.to(device, dtype=torch.float)
            l_digit = l_digit.to(device, dtype=torch.long)
            l_letter = l_letter.to(device, dtype=torch.float)
                
            out_digit = net(inputs, l_letter)
            loss = criterion(out_digit, l_digit)

            test_loss += loss.item()
            _, pred_digit = out_digit.max(1)

            total += l_digit.size(0)
            correct_digit += pred_digit.eq(l_digit).sum().item()
        if epoch%10 ==0:
            print(batch_idx,'/', len(testloader), ' Loss: %.3f | Acc: %.3f%% (%d/%d) '
                    % (test_loss/(batch_idx+1), 100.*correct_digit/total, correct_digit, total, ))
            
for epoch in range(0,300):
    train(net, epoch)
    test(net, epoch)

# ================================================================
# =================== net2, net3 앞에서 학습하기=================================
net2 = resnext101_32x8d().to(device)
optimizer = torch.optim.AdamW(net2.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 230, 265], gamma=0.1)

for epoch in range(0,300):
    train(net2, epoch)
    test(net2, epoch)

net3 = resnext101_32x8d().to(device)
optimizer = torch.optim.AdamW(net3.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 230, 265], gamma=0.1)

for epoch in range(0,300):
    train(net3, epoch)
    test(net3, epoch)

#========================== submnit ===============================
testset = EMNIST_test(transform = albumentations_transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)

y_pred = []
with torch.no_grad():
    for img, letter in tqdm(testloader):
        img = img.to(device)
        letter = letter.to(device)
        outputs = net(img, letter)
        outputs2 = net2(img, letter)
        outputs3 = net3(img, letter)
        
        y_pred.append(torch.argmax(outputs+outputs2+outputs3, dim=1))

import pandas as pd
submission = pd.read_csv('C:/data/dacon_mnist/submission.csv')
submission.digit = torch.cat(y_pred).detach().cpu().numpy()
submission.to_csv('C:/data/dacon_mnist/answer/0205_1_hyunmin.csv', index=False)




