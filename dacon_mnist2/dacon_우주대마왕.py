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

class MnistDataset(Dataset):
    def __init__(
        self,
        dir: os.PathLike,
        image_ids: os.PathLike,
        transforms: Sequence[Callable]
    ) -> None:
        self.dir = dir
        self.transforms = transforms

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(
                self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
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

trainset = MnistDataset('C:/data/dacon_mnist2/dirty_mnist', 'C:/data/dacon_mnist2/dirty_mnist_answer.csv', transforms_train)
testset = MnistDataset('C:/data/dacon_mnist2/test_dirty_mnist', 'C:/data/dacon_mnist2/sample_submission.csv', transforms_test)

train_loader = DataLoader(trainset, batch_size=64, num_workers=8)
test_loader = DataLoader(testset, batch_size=32, num_workers=4)

class MnistModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)

        return x

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
    submit.to_csv('C:/data/dacon_mnist2/anwsers/0223_2_base.csv', index=False)
    




