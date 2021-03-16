# 서영이가 보내준 응용파일
# 에서 여러가지 깔고 쬐꼼 수정해서 사용

import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import KFold
import time
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from torch_poly_lr_decay import PolynomialLRDecay
import random
import albumentations


torch.set_num_threads(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ======================== Dacon Dataset Load ==========================
labels_df = pd.read_csv('C:/data/dacon_mnist2/dirty_mnist_2nd_answer.csv')[:]
imgs_dir = np.array(sorted(glob.glob('C:/data/dacon_mnist2/train_clean/*')))[:]
labels = np.array(labels_df.values[:,1:])

test_imgs_dir = np.array(sorted(glob.glob('C:/data/dacon_mnist2/test_clean/*')))

imgs=[]
for path in tqdm(imgs_dir[:]):
    img=cv2.imread(path, cv2.IMREAD_COLOR)
    imgs.append(img)
imgs=np.array(imgs)

# 저장소에서 load
class MnistDataset_v1(Dataset):
    def __init__(self, imgs_dir=None, labels=None, transform=None, train=True):
        self.imgs_dir = imgs_dir
        self.labels = labels
        self.transform = transform
        self.train = train
        pass
    
    def __len__(self):
        # 데이터 총 샘플 수
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # 1개 샘플 get
        img = cv2.imread(self.imgs_dir[idx], cv2.IMREAD_COLOR)
        img = self.transform(img)
        if self.train==True:
            label = self.labels[idx]
            return img, label
        else:
            return img
        
        pass
    
# 메모리에서 load
class MnistDataset_v2(Dataset):
    def __init__(self, imgs=None, labels=None, transform=None, train=True):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.train=train
                #test data augmentations
        self.aug = albumentations.Compose ([ 
                   albumentations.RandomResizedCrop (256, 256), 
                    albumentations.Transpose (p = 0.5)
                    ], p = 1) 
        pass
    
    def __len__(self):
        # 데이터 총 샘플 수
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # 1개 샘플 get1
        img = self.imgs[idx]
        img = self.transform(img)
        
        if self.train==True:
            label = self.labels[idx]
            return img, label
        else:
            return img

# ========================= reproduction을 위한 seed 설정=====================
# https://dacon.io/competitions/official/235697/codeshare/2363?page=1&dtype=recent&ptype=pub
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  
'''
class myPredictor(ClassPredictor):
    def __init__(self,model,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = model

    def predict_patches(self,patches):
        return self.model.predict(patches)
'''

# ==================== model 정의 ==================================
# EfficientNet -b0(pretrained)
# MultiLabel output

class EfficientNet_MultiLabel(nn.Module):
    def __init__(self, in_channels):
        super(EfficientNet_MultiLabel, self).__init__()
        self.network = EfficientNet.from_pretrained('efficientnet-b7', in_channels=in_channels) # b0, b3, b7 
        self.output_layer = nn.Linear(1000, 26)

    def forward(self, x):
        x = F.relu(self.network(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
'''
# ============== 데이터 분리====================================
# 해당 코드에서는 1fold만 실행

kf = KFold(n_splits=2, shuffle=True, random_state=42)
folds=[]
for train_idx, valid_idx in kf.split(imgs):
    folds.append((train_idx, valid_idx))
'''
### seed_everything(42)

# ===================== Test Image 로드 ==========================
test_imgs=[]
for path in tqdm(test_imgs_dir):
    test_img=cv2.imread(path, cv2.IMREAD_COLOR)
    test_imgs.append(test_img)
test_imgs=np.array(test_imgs)

test_transform = transforms.Compose([
        transforms.ToTensor(),
        ])

# ================ Test 추론 =============================
submission = pd.read_csv('C:/data/dacon_mnist2/sample_submission2.csv')

with torch.no_grad():
    for fold in range(5):
        model = EfficientNet_MultiLabel(in_channels=3).to(device)
        model.load_state_dict(torch.load('C:/data/dacon_mnist2/model_mnist/EfficientNetB7-fold0.pt'.format(fold)))
        model.eval()

        test_dataset = MnistDataset_v2(imgs = test_imgs, transform=test_transform, train=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

        for n, X_test in enumerate(tqdm(test_loader)):
            X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
            with torch.no_grad():
                model.eval()
                pred_test = model(X_test).cpu().detach().numpy()
                submission.iloc[n*32:(n+1)*32,1:] += pred_test

# ==================== 제출물 생성 ====================
submission.iloc[:,1:] = np.where(submission.values[:,1:]>=2.5, 1,0)
submission.to_csv('C:/data/dacon_mnist2/anwsers/0230_1_base.csv', index=False)
print('===== done =====')