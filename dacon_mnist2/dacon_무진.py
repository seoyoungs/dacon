import os
from typing import Tuple, Sequence, Callable
import csv
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

from torchvision import transforms
from torchvision.models import resnet50

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# os.environ [ 'KMP_DUPLICATE_LIB_OK'] = 'True'

# torch.cuda.empty_cache()

class MnistDataset(Dataset):
    def __init__(
        self,
        # 디렉토리와 이미지 라벨을 주소 형식으로 받음,
        dir: os.PathLike,
        image_ids: os.PathLike,
        transforms: Sequence[Callable]
    ) -> None:
        # 사용할 변수를 받은 인자로 초기화함
        self.dir = dir
        self.transforms = transforms
        self.labels = {}

        # 파일을 읽기 형식으로 열고 이터레이터의 길이만큼 y라벨을 int 형태로 초기화
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            # 열린 라벨 파일의 인덱스 별로 한 줄씩 참조함
            next(reader)
            # reader의 인수 갯수 만큼 반복하여 reader의 각각 인수 하나를 참조하는 row를 만듬
            for row in reader:
                # 초기화된 row의 두번 째 값을 인덱스를 제외한 값 부터 끝까지 str타입을 int로 변환하고
                # 리스트 형태로 복사하여 라벨 인덱스 위치에 붙여 넣음
                self.labels[int(row[0])] = list(map(int, row[1:]))

        # 클래스에서 사용하는 변수를 라벨키로 복사하여 붙어넣음
        self.image_ids = list(self.labels.keys())

    # 라벨키를 반환함
    def __len__(self) -> int:
        return len(self.image_ids)

    # 이미지와 라벨을 숫자타입 튜플로 반환함
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        # 라벨 키값으로 초기화
        image_id = self.image_ids[index]
        # 경로에 있는 이미지를 열고 RGB로 바꿔서 image에 담음
        image = Image.open(
            os.path.join(
                self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        image.resize((256,256))
        # 라벨의 값을 실수로 변경하여 복사하여 타겟에 붙여 넣음
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        # 이미지 변환 설정이이 있다면 이미지 변환을 진행함 
        if self.transforms is not None:
            # image = datagen.flow(image, batch_size=1, shuffle=True)
            image = self.transforms(image)

        
        # x와 y 값을 반환함
        return image, target


datagen = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

transforms_train = transforms.Compose([
    # 0.5 확률로 좌우 뒤집기
    transforms.RandomHorizontalFlip(p=0.5),
    # 0.5 확률로 위아래 뒤집기
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30, center=(64,64)),
    # 0~1 까지 반환하고 컬러 채널이 3차원으로 올라감
    transforms.ToTensor(),
    # 정규화함
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# 파일의 이미지를 정규화하고 타겟을 각각 셋에 담고 초기화함
trainset = MnistDataset('C:/data/dacon_mnist2/dirty_mnist', 'C:/data/dacon_mnist2/dirty_mnist_answer.csv', transforms_train)
testset = MnistDataset('C:/data/dacon_mnist2/test_dirty_mnist', 'C:/data/dacon_mnist2/sample_submission.csv', transforms_test)

# 입력 데이터 셋의 배치사이즈를 정함 병렬 작업할 프로세스의 갯수를 정함
train_loader = DataLoader(trainset, batch_size=32, num_workers=8)
test_loader = DataLoader(testset, batch_size=32, num_workers=6)

# 모델 x를 반환하는 클래스
class MnistModel(nn.Module):
    def __init__(self) -> None:
        # 모델의 속성을 따옴
        super().__init__()
        # resnet50 모델으 불러옴
        self.resnet = resnet50(pretrained=True)
        # 마지막 모델은 Linear로 1000을 받아서 26으로 출력함
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)

        return x

# 드라이브를 쿠다에서 실행하지 못하는 경우에만 cpu로 동작함
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MnistModel().to(device)

print(summary(model, input_size=(1, 3, 256, 256), verbose=0))

# 실행 하는 곳이 메인인 경우
if __name__ == '__main__':
    
    # 옵티마이저와 멀티라벨소프트 마진 로스를 사용함
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MultiLabelSoftMarginLoss()

    # 에포치 10주고 모델을 트레인으로 변환
    num_epochs = 40
    model.train()

    # 에포치 만큼 반복
    for epoch in range(num_epochs):
            # 배치 사이즈 만큼 스탭을 진행함
            for i, (images, targets) in enumerate(train_loader):

                # 미분 값 초기화
                optimizer.zero_grad()
                # 데이터셋을 프로세스에 입력함
                images = images.to(device)
                targets = targets.to(device)
                # 모델에 인풋을 넣고 아웃풋을 출력함
                outputs = model(images)
                # 로스를 확인함
                loss = criterion(outputs, targets)

                # 로스 역전파
                loss.backward()
                # 매개변수 갱신함
                optimizer.step()
            
                # 10에포치 마다 로스와 액큐러시를 출력함
                if (i+1) % 10 == 0:
                    outputs = outputs > 0.5
                    acc = (outputs == targets).float().mean()
                    print(f'{epoch}: {loss.item():.5f}, {acc.item():.5f}')

if __name__ == '__main__':

    # 평가 폴더를 열음
    submit = pd.read_csv('C:/data/dacon_mnist2/sample_submission.csv')

    # 이벨류 모드로 전환
    model.eval()

    # 베치사이즈는 테스트로더 베치사이즈
    batch_size = test_loader.batch_size
    # 인덱스 0부터 시작
    batch_index = 0
    # 이벨류 모드를 테스트 셋으로 진행하고 파일에 입력함
    for i, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        outputs = outputs > 0.5
        batch_index = i * batch_size
        submit.iloc[batch_index:batch_index+batch_size, 1:] = \
            outputs.long().squeeze(0).detach().cpu().numpy()

    # 저장함
    submit.to_csv('C:/data/dacon_mnist2/anwsers/0224_1_base.csv', index=False)

    del images
    del targets
