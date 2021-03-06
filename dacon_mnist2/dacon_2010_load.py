import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import imutils # 설치 완료
import zipfile
import os
from PIL import Image

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")


def run():
    torch.multiprocessing.freeze_support()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dirty_mnist_answer = pd.read_csv("C:/data/dacon_mnist2/dirty_mnist_answer.csv")
# dirty_mnist라는 디렉터리 속에 들어있는 파일들의 이름을 
# namelist라는 변수에 저장
namelist = os.listdir('C:/data/dacon_mnist2/dirty_mnist/')

# numpy를 tensor로 변환하는 ToTensor 정의
class ToTensor(object):# tensor란 데이터의 배열(3차원부터 인식)
    """numpy array를 tensor(torch)로 변환합니다."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # axis를 바꾼다.
        # numpy image: H x W x C, # numpy array: (H, W, C)
        # torch image: C X H X W, # torch tensor: (C, H, W)
        image = image.transpose((2, 0, 1))
        return {'image': torch.FloatTensor(image),
                'label': torch.FloatTensor(label)}
# 선언
to_tensor = T.Compose([
                        ToTensor()
                    ])

class DatasetMNIST(Dataset): #torch.utils.data.Dataset ---> Dataset만 언급해도 됨
    # __init__ 을 사용해서 CSV 파일 안에 있는 데이터를 읽기
    def __init__(self,
                dir_path,
                meta_df,
                transforms=to_tensor,
                augmentations=None): #미리 선언한 to_tensor를 transforms로 받음
        
        self.dir_path = dir_path # 데이터의 이미지가 저장된 디렉터리 경로
        self.meta_df = meta_df # 데이터의 인덱스와 정답지가 들어있는 DataFrame

        self.transforms = transforms# Transform
        self.augmentations = augmentations # Augmentation
        
    def __len__(self):# 데이터의 총 길이가 반환되도록 작성
        return len(self.meta_df)
    
    def __getitem__(self, index):
        # __getitem__ 을 이용해서 이미지의 판독
        # 폴더 경로 + 이미지 이름 + .png => 파일의 경로
        # 참고) "12".zfill(5) => 000012
        #       "146".zfill(5) => 000145
        # cv2.IMREAD_GRAYSCALE : png파일을 채널이 1개인 GRAYSCALE로 읽음
        image = cv2.imread(self.dir_path +\
                        str(self.meta_df.iloc[index,0]).zfill(5) + '.png',
                        cv2.IMREAD_GRAYSCALE)
        # 0 ~ 255의 값을 갖고 크기가 (256,256)인 numpy array를
        # 0 ~ 1 사이의 실수를 갖고 크기가 (256,256,1)인 numpy array로 변환
        image = (image/255).astype('float')[..., np.newaxis]

        # 정답 numpy array생성(존재하면 1 없으면 0)
        label = self.meta_df.iloc[index, 1:].values.astype('float')
        sample = {'image': image, 'label': label}

        # transform 적용
        # numpy to tensor
        if self.transforms:
            sample = self.transforms(sample)

        # sample 반환
        return sample # Dataset이 반환하고 싶은 값을 만들어 return 뒤에 작성해 준다.

# nn.Module을 상속 받아 MultiLabelResnet를 정의 # 여기부분 모델링
class MultiLabelResnet(nn.Module):
    def __init__(self):
        super(MultiLabelResnet, self).__init__()
        self.conv2d = nn.Conv2d(1, 3, 3, stride=1)

        self.resnet = models.resnet18(pretrained=True) #ResNet:이미지 인식, Residual (잔차)를 학습
        self.FC = nn.Linear(1000, 26)

        # resnet의 입력은 [3, N, N]으로
        # 3개의 채널을 갖기 때문에
        # resnet 입력 전에 conv2d를 한 층 추가
    def forward(self, x):
        x = F.relu(self.conv2d(x))
        # resnet18을 추가
        x = F.relu(self.resnet(x))

        # 마지막 출력에 nn.Linear를 추가
        # multilabel을 예측해야 하기 때문에
        # softmax가 아닌 sigmoid를 적용
        x = torch.sigmoid(self.FC(x))
        return x

# 모델 선언
model = MultiLabelResnet()
# print(model)

# cross validation을 적용하기 위해 KFold 생성
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

# ========= Main Module =================================
if __name__ == '__main__':
    run()

    # dirty_mnist_answer에서 train_idx와 val_idx를 생성
    best_models = [] # 폴드별로 가장 validation acc가 높은 모델 저장
    for fold_index, (trn_idx, val_idx) in enumerate(kfold.split(dirty_mnist_answer),1):
        print(f'[fold: {fold_index}]') # n_splits 몇번 반복 했는지
        # cuda cache 초기화
        torch.cuda.empty_cache()

        #train fold, validation fold 분할
        train_answer = dirty_mnist_answer.iloc[trn_idx]
        test_answer  = dirty_mnist_answer.iloc[val_idx]

        #Dataset 정의
        train_dataset = DatasetMNIST("C:/data/dacon_mnist2/dirty_mnist/", train_answer)
        valid_dataset = DatasetMNIST("C:/data/dacon_mnist2/dirty_mnist/", test_answer)

        #DataLoader 정의
        train_data_loader = DataLoader(
            train_dataset,
            batch_size = 128,
            shuffle = False,
            num_workers = 8
        )
        valid_data_loader = DataLoader(
            valid_dataset,
            batch_size = 32,
            shuffle = False,
            num_workers = 4
        )

        # 모델 선언
        model = torch.load('C:/data/dacon_mnist2/model_mnist/6_resnet18_0.6382_epoch_0.pth') 

    
    # # gpu에 올라가 있는 tensor -> cpu로 이동 -> numpy array로 변환
    # sample_images = images.cpu().detach().numpy()
    # sample_prob = probs
    # sample_labels = labels

    # idx = 1
    # # plt.imshow(sample_images[idx][0])
    # # plt.title("sample input image")
    # # plt.show()

    # print('예측값 : ',dirty_mnist_answer.columns[1:][sample_prob[idx] > 0.5])
    # print('정답값 : ', dirty_mnist_answer.columns[1:][sample_labels[idx] > 0.5])

    #test Dataset 정의
    sample_submission = pd.read_csv("C:/data/dacon_mnist2/sample_submission.csv")
    test_dataset = DatasetMNIST("C:/data/dacon_mnist2/test_dirty_mnist/", sample_submission)
    batch_size = 128
    test_data_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 3,
        drop_last = False
    )

    predictions_list = []
    # 배치 단위로 추론
    prediction_df = pd.read_csv("C:/data/dacon_mnist2/sample_submission.csv")

    # 5개의 fold마다 가장 좋은 모델을 이용하여 예측
    for model in best_models:
        # 0으로 채워진 array 생성
        prediction_array = np.zeros([prediction_df.shape[0],
                                    prediction_df.shape[1] -1])
        for idx, sample in enumerate(test_data_loader):
            with torch.no_grad():
                # 추론
                model.eval()
                images = sample['image']
                images = images.to(device)
                probs  = model(images)
                probs = probs.cpu().detach().numpy()
                preds = (probs > 0.5)

                # 예측 결과를 
                # prediction_array에 입력
                batch_index = batch_size * idx
                prediction_array[batch_index: batch_index + images.shape[0],:]\
                            = preds.astype(int)
                            
        # 채널을 하나 추가하여 list에 append
        predictions_list.append(prediction_array[...,np.newaxis])
    
    # axis = 2를 기준으로 평균
    predictions_array = np.concatenate(predictions_list, axis = 2)
    predictions_mean = predictions_array.mean(axis = 2)

    # 평균 값이 0.5보다 클 경우 1 작으면 0
    predictions_mean = (predictions_mean > 0.5) * 1

    sample_submission = pd.read_csv("C:/data/dacon_mnist2/sample_submission.csv")
    sample_submission.iloc[:,1:] = predictions_mean
    sample_submission.to_csv("C:/data/dacon_mnist2/anwsers/0210_1_base.csv", index = False)

