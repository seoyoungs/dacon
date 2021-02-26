import os
import argparse
from typing import Tuple, Sequence, Callable
import csv
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold

import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.models import resnet18

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
args = parser.parse_args()

# ================== 1. Out-of-Fold 전략을 위한 함수 정의 =================
def split_dataset(path: os.PathLike) -> None:
    df = pd.read_csv(path)
    kfold = KFold(n_splits=5)
    for fold, (train, valid) in enumerate(kfold.split(df, df.index)):
        df.loc[valid, 'kfold'] = int(fold)

    df.to_csv('data/split_kfold.csv', index=False)

# ========== 2. 커스텀 데이터셋 정의 ====================
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
            os.path.join(self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

# ===================== 3. 이미지 어그멘테이션 정의 ==========================

transforms_train = transforms.Compose([
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

# ================== 4. 모형 정의 ==========================
class MnistModel(nn.Module):
    def __init__(self, num_classes: int = 26) -> None:
        super().__init__()
        self.resnet = resnet18()
        self.classifier = \
            nn.Linear(1000, num_classes)

        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet(x)
        x = self.classifier(x)

        return x

# =============5. 학습=============================
def train(fold: int, verbose: int = 30) -> None:
    split_dataset('data/dirty_mnist_2nd_answer.csv')
    df = pd.read_csv('data/split_kfold.csv')
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    df_train.drop(['kfold'], axis=1).to_csv(f'data/train-kfold-{fold}.csv', index=False)
    df_valid.drop(['kfold'], axis=1).to_csv(f'data/valid-kfold-{fold}.csv', index=False)

    trainset = MnistDataset('data/train', f'data/train-kfold-{fold}.csv', transforms_train)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=12)

    validset = MnistDataset('data/train', f'data/valid-kfold-{fold}.csv', transforms_test)
    valid_loader = DataLoader(validset, batch_size=128, shuffle=False, num_workers=12)

    num_epochs = 80
    device = 'cuda'
    scaler = GradScaler()

    model = NetMnistModel().to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MultiLabelSoftMarginLoss()

    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device)
            targets = targets.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i+1) % verbose == 0:
                outputs = outputs > 0.0
                acc = (outputs == targets).float().mean()
                print(f'Fold {fold} | Epoch {epoch} | L: {loss.item():.7f} | A: {acc.item():.7f}')

        if epoch > num_epochs-20 and epoch < num_epochs-1:
            model.eval()
            valid_acc = 0.0
            valid_loss = 0.0
            valid_size = valid_loader.batch_size
            for i, (images, targets) in enumerate(valid_loader):
                images = images.to(device)
                targets = targets.to(device)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                valid_loss += loss.item()
                outputs = outputs > 0.0
                valid_acc += (outputs == targets).float().mean()

            print(f'Fold {fold} | Epoch {epoch} | L: {valid_loss/valid_size:.7f} | A: {valid_acc/valid_size:.7f}\n')

        if epoch > num_epochs-20 and epoch < num_epochs-1:
            torch.save(model.state_dict(), f'resnet101-f{fold}-{epoch}.pth')

if __name__ == '__main__':
    train(0)
    train(1)
    train(2)
    train(3)
    train(4)
 
# ================  6. 테스트셋 제출 ====================
def load_model(fold: int, epoch: int, device: torch.device = 'cuda') -> nn.Module:
    model = MnistModel().to(device)
    state_dict = {}
    for k, v in torch.load(f'resnet-f{fold}-{epoch}.pth').items():
        state_dict[k[7:]] = v

    model.load_state_dict(state_dict)

    return model

def test(device: torch.device = 'cuda'):
    submit = pd.read_csv('data/sample_submission.csv')

    model1 = load_model(0, 50)
    model2 = load_model(1, 50)
    model3 = load_model(2, 50)
    model4 = load_model(3, 50)
    model5 = load_model(4, 50)

    model1 = nn.DataParallel(model1, device_ids=[0, 1, 2, 3])
    model2 = nn.DataParallel(model2, device_ids=[0, 1, 2, 3])
    model3 = nn.DataParallel(model3, device_ids=[0, 1, 2, 3])
    model4 = nn.DataParallel(model4, device_ids=[0, 1, 2, 3])
    model5 = nn.DataParallel(model5, device_ids=[0, 1, 2, 3])

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()

    batch_size = test_loader.batch_size
    batch_index = 0
    for i, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        targets = targets.to(device)

        outputs1 = model1(images)
        outputs2 = model2(images)
        outputs3 = model3(images)
        outputs4 = model4(images)
        outputs5 = model5(images)

        outputs = (outputs1 + outputs2 + outputs3 + outputs4 + outputs5) / 5

        outputs = outputs > 0.0
        batch_index = i * batch_size
        submit.iloc[batch_index:batch_index+batch_size, 1:] = \
            outputs.long().squeeze(0).detach().cpu().numpy()

    submit.to_csv('resnet101-e50-kfold.csv', index=False)


if __name__ == '__main__':
    test()

