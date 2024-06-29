import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import math
import numpy as np
from cal_map import calculate_map,calculate_top_map, compress
import os
from PIL import Image
from torch.utils.data import Dataset

from torchvision.models import AlexNet_Weights


# Hyper Parameters
num_epochs = 50
batch_size = 32
epoch_lr_decrease = 30
learning_rate = 0.001
encode_length = 12
num_classes = 80


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset   
def read_image_paths(filename):
    paths=[]
    labels_list=[]
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            paths.append(parts[0])  # 图片文件名
            labels = list(map(int, parts[1:]))  # 独热编码标签列表
            labels_list.append(labels)
    full_path=[f'/root/autodl-tmp/coco/{path}' for path in paths]
    return full_path,np.array(labels_list)

train_paths,train_L = read_image_paths('coco/train.txt')
test_paths,test_L = read_image_paths('coco/test.txt')
database_paths,database_L = read_image_paths('coco/database.txt')
print(len(train_paths),len(test_paths),len(database_paths))


class CustomImageDataset(Dataset):
    def __init__(self, img_paths,labels, transform=None):
        self.img_paths = img_paths
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        #print(f"Before transform: {image.size}")
        if self.transform:
            image = self.transform(image)
            #print(f"After transform: {image.shape}")
        label = self.labels[idx]  # 添加这一行来获取对应索引的标签
        return image, label  # 返回图像和标签




train_dataset = CustomImageDataset(train_paths,train_L, transform=train_transform)
test_dataset = CustomImageDataset(test_paths,test_L, transform=test_transform)
database_dataset = CustomImageDataset(database_paths,database_L, transform=test_transform)


# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)

database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4)

# new layer
class hash(Function):
    @staticmethod
    def forward(ctx, input):
        #ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        #input,  = ctx.saved_tensors
        #grad_output = grad_output.data

        return grad_output


def hash_layer(input):
    return hash.apply(input)


class CNN(nn.Module):
    def __init__(self, encode_length, num_classes):
        super(CNN, self).__init__()
        #self.alex = torchvision.models.alexnet(pretrained=True)
        self.alex = torchvision.models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.alex.classifier = nn.Sequential(*list(self.alex.classifier.children())[:6])
        self.fc_plus = nn.Linear(4096, encode_length)
        self.fc = nn.Linear(encode_length, num_classes, bias=False)

    def forward(self, x):
        x = self.alex.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.alex.classifier(x)
        x = self.fc_plus(x)
        # code = hash_layer(x)
        # output = self.fc(code)

        code=hash_layer(x)
        output=self.fc(x)

        return output, x, code


cnn = CNN(encode_length=encode_length, num_classes=num_classes)
#cnn.load_state_dict(torch.load('temp.pkl'))


# Loss and Optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
best = 0.0


# Train the Model
for epoch in range(num_epochs):
    cnn.cuda().train()
    adjust_learning_rate(optimizer, epoch)
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs, feature, _ = cnn(images)
        loss1 = criterion(outputs, labels.float())
        #loss2 = F.mse_loss(torch.abs(feature), Variable(torch.ones(feature.size()).cuda()))
        loss2 = torch.mean(           torch.abs(torch.pow(torch.abs(feature)- Variable(torch.ones(feature.size()).cuda()), 3))           )
        loss = loss1 + 0.1 * loss2
        loss.backward()
        optimizer.step()

        if (i + 1) % (len(train_dataset) // batch_size / 2) == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f'
                   % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                      loss1.item(), loss2.item()))
                      #loss1.data[0], loss2.data[0]))

    # Test the Model
    cnn.eval()  # Change model to 'eval' mode
    correct = 0
    total = 0
    for images, labels in test_loader:
        with torch.no_grad():
            images = images.cuda()
        #images = Variable(images.cuda(), volatile=True)
        outputs, _, _ = cnn(images)
        _, predicted = torch.max(outputs.cpu().data, 1)
        _,labels=torch.max(labels.cpu().data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model: %.2f %%' % (100.0 * correct / total))

    if 1.0 * correct / total > best:
        best = 1.0 * correct / total
        torch.save(cnn.state_dict(), 'temp.pkl')
        
    print('best: %.2f %%' % (best * 100.0))
    retrievalB, retrievalL, queryB, queryL = compress(database_loader, test_loader, cnn,onehot=True)
    result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=1000)
    print("calculate_top_map1000: ",result)


# Save the Trained Model
torch.save(cnn.state_dict(), 'coco.pkl')


# Calculate MAP
# cnn.cuda()
#cnn.load_state_dict(torch.load('temp.pkl'))
cnn.eval()
retrievalB, retrievalL, queryB, queryL = compress(database_loader, test_loader, cnn,num_classes)
print(np.shape(retrievalB))
print(np.shape(retrievalL))
print(np.shape(queryB))
print(np.shape(queryL))
"""
print('---calculate map---')
result = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
print(result)
"""
print('---calculate top map---')
result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=1000)
print(result)

