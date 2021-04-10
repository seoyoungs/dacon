import os
import numpy as np
import pandas as pd
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.autograd import Variable
import torchvision
import pathlib
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
#딥러닝 모델 설계할 때 장비 확인
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using pytorch version:',torch.__version__,'Device:',DEVICE) #Using pytorch version: 1.7.1 Device: cuda

##cache 비워주기###
import torch,gc
gc.collect()
torch.cuda.empty_cache()

BATCH_SIZE = 32
EPOCHS = 30
TRAIN_PATH = 'C:/data/LPD_competition/train/'
PRED_PATH = 'C:/data/LPD_competition/test/'
transformer = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],
                            [0.5,0.5,0.5])


])

# gpu 연산이 가능하면 'cuda:0', 아니면 'cpu' 출력
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#train data 불러오기
image_datasets = datasets.ImageFolder(TRAIN_PATH, transformer)
#print(image_datasets)

class_names = image_datasets.classes
# print(class_names)
# print(type(class_names))
train_size = int(0.8*len(image_datasets))
test_size = len(image_datasets) - train_size

print(train_size) #38400
print(test_size) #9600

train_dataset, test_dataset = torch.utils.data.random_split(image_datasets, [train_size,test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True )
valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = False )

root = pathlib.Path(TRAIN_PATH)
classe = sorted([j.name.split('/')[-1] for j in root.iterdir()])
# print(classe)

# for i in range(1000):
#     os.mkdir('C:/data/LPD_competition/train_new/{0:04}'.format(i))

#     for img in range(48):
#         image = Image.open(f'C:/data/LPD_competition/train/{i}/{img}.jpg')
#         image.save('C:/data/LPD_competition/train_new/{0:04}/{1:02}.jpg'.format(i, img))

# for i in range(72000):
#     image = Image.open(f'C:/data/LPD_competition/pred/test/{i}.jpg')
#     image.save('C:/data/LPD_competition/test_new/{0:05}.jpg'.format(i))

inputs, classes = next(iter(train_loader))
# print(classes)
# print(classes.shape) #torch.Size([64])
# print(inputs)

# 임의의 label값과 사진 불러오기 
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.5,0.5,0.5])
    std = np.array([0.5,0.5,0.5])
    inp = std*inp +mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

out = torchvision.utils.make_grid(inputs)
#imshow(out, title=[class_names[x] for x in classes])
#plt.show()

from efficientnet_pytorch import EfficientNet
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.efficientnet =  EfficientNet.from_pretrained('efficientnet-b7')
        self.dropout = nn.Dropout(0.3)
        self.l1 = nn.Linear(1000 , 256)
        #self.gaussiandropout = nn.Gaussian(0.3)
        self.l2 = nn.Linear(256,1000)
        self.silu = nn.SiLU()

    def forward(self, input):
        x = self.efficientnet(input)
        x = x.view(x.size(0),-1)
        #x = self.dropout(self.silu(self.l1(x)))
        x = self.dropout(x)
        x = self.silu(self.l1(x))
        x = self.dropout(x)
        x = self.l2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= Classifier().to(device)
#model = ConvNet(num_classes = 1000).to(device)
optimizer = SGD(model.parameters(), lr=0.1)
loss_function = nn.CrossEntropyLoss()

num_epochs=60

best_accuracy=0.0

for epoch in tqdm(range(num_epochs)):
    
    #Evaluation and training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    
    for i, (images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            
        optimizer.zero_grad()
        
        outputs=model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        
        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
        
    train_accuracy=train_accuracy/train_size
    train_loss=train_loss/train_size
    
    
    # Evaluation on testing dataset
    model.eval()
    
    test_accuracy=0.0
    for i, (images,labels) in enumerate(valid_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            
        outputs=model(images)
        _,prediction=torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data))
    
    test_accuracy=test_accuracy/test_size
    
    
    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
    
    #가장 좋은 모델 저장
    if test_accuracy>best_accuracy:
        torch.save(model.state_dict(),'C:/data/npy/lotte_pred_best_checkpoint.pth')
        best_accuracy=test_accuracy

PATH = 'C:/data/npy/lotte_pred_best_checkpoint.pth'
#model = torch.load(PATH, map_location=device)

#모델 불러오기 
model.load_state_dict(torch.load(PATH),strict=False)
#RuntimeError: Error(s) in loading state_dict for Classifier: 오류
#pretrained pytorch model을 loading해오려고 했는데, pytorch version과 여러 환경세팅이 맞지 않아서 모델의 state_dict에 있는 key가 matching이 되지 않아 모델의 pretrained weight가 불려오지 않는 문제
#strict=False로 해주니 불러와졌음

from PIL import Image
a = []
def predict(model, path, sample_size=72000):
    for file in tqdm(glob.glob(os.path.join('C:/data/LPD_competition/test/*.jpg'))[:sample_size]):
        with Image.open(file) as f:
            img = transformer(f).unsqueeze(0)
            with torch.no_grad(): #해당 블록을 history 트래킹 하지 않겠다는 뜻
                out = model(img.to(device)).cpu().numpy()
            
                a.append(np.argmax(out))
                #print(a)


            #plt.imshow(np.array(f))
            #plt.show()

    return a

result = predict(model, PRED_PATH,sample_size=72000)

result = np.transpose(result)
#print(result.shape)

submission = pd.read_csv('C:/data/LPD_competition/sample.csv')
submission['prediction'] = result
submission.to_csv('C:/data/csv/lotte0322_1.csv', index=False)
