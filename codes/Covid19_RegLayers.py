#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os 
import torchvision
import tqdm
import matplotlib.pyplot as plt
import functools
import time
import comet_ml
import torchvision.transforms as T
from torch.utils.data import DataLoader

from torchmetrics.functional import accuracy


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets


# In[3]:


from PIL import Image
img = Image.open("./archive/xray_dataset_covid19/train/NORMAL/IM-0001-0001.jpeg")
img1 = Image.open("./archive/xray_dataset_covid19/train/PNEUMONIA/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg")
img1


# In[4]:


transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, ), (0.5, )),
            T.Resize((256, 256))
])
imtensor = transform(img1)
imtensor.size()


# In[5]:


train_path = './archive/xray_dataset_covid19/train/'
test_path = './archive/xray_dataset_covid19/test/'


#ds_n = datasets.ImageFolder(normal_path, transform=transform)
ds_train = datasets.ImageFolder(train_path, transform=transform)
ds_test = datasets.ImageFolder(test_path, transform=transform)
print('Train set: \n', ds_train,
     ' \nTest set: \n', ds_test)


# In[6]:


trainloader = torch.utils.data.DataLoader(ds_train, batch_size=32,
                                         shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(ds_test, batch_size=32,
                                         shuffle=False, num_workers=4)


# In[7]:


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# In[8]:


dataiter = iter(trainloader)
images, labels = dataiter.next()
# create grid of images
img_grid = torchvision.utils.make_grid(images[:16])

# show images
matplotlib_imshow(img_grid, one_channel=True)


# In[9]:


#device = torch.device(f'cuda:{np.random.randint(1)}')
device = torch.device('cpu')
print(device)


# ## Let's define a class to have more convinient training

# In[10]:


class covidconv(nn.Module):
    def __init__(self, 
                 in_channel,
                 interm_channel1,
                 interm_channel2,
                 interm_channel3,
                 out_channel,
                 use_bn,
                 w_init):
        super(covidconv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=interm_channel1, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=interm_channel1, out_channels=interm_channel2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=interm_channel2, out_channels=interm_channel3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=interm_channel3, out_channels=out_channel, kernel_size=3, padding=1)
        
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(interm_channel1)
            self.bn2 = nn.BatchNorm2d(interm_channel3)
            
        if w_init is not None:
            w_init(self.conv1.weight)
            w_init(self.conv2.weight)
            #w_init(self.conv3.weight)
            #w_init(self.conv4.weight)
       
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn1(x)
            
        x = F.relu(self.pool2(x))
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn2(x)
        
        x = self.conv4(x)
        
        return x


# In[11]:


# model itself

class Covid_19model(nn.Module):
    def __init__(self, use_bn, w_init):
        
        super(Covid_19model, self).__init__()
        
        self.conv1 = covidconv(3, 8, 8, 16, 32, use_bn, w_init) # [3,256,256] => [8,256,256] => [8,128,128] => [8,128,128] => 
        # [8,64,64] => [8,64,64] => [16,64,64]
        #self.conv2 = covidconv(16, 16, 32, 32, 64, use_bn, w_init) # [16,64,64] => [16,64,64] => [16,32,32] => [32,32,32] => [32,16,16] 
        # => [32,16,16] => [32,16,16]
        
        self.fc1 = nn.Linear(32 * 64 * 64, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10, 2)
        
        if w_init is not None:
            w_init(self.fc1.weight)
            w_init(self.fc2.weight)
            #w_init(self.fc3.weight)
            #w_init(self.fc4.weight)
            
    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        
        x = x.view(x.shape[0], 32*64*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


# ## Training

# In[12]:


def train_model(model, epochs=3, learning_rate=0.0005):
    loss_fn = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loss = []
    test_loss = []
    test_accuracy = []
    
    for epoch in range(epochs):
        model.train()
        
        for i, (x,y) in enumerate(tqdm.tqdm(trainloader)):
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
            train_loss.append(loss.item())
        
        model.eval()
        epoch_losses = []
        epoch_accuracy = []
        
        with torch.no_grad():
            for X, y in testloader:
                X, y = X.to(device), y.to(device)
                
                pred = model(X)
                epoch_losses.append(loss_fn(pred, y).item())
                _, pred = torch.max(pred.data, 1)
                epoch_accuracy.append(
                    (pred == y).to(torch.float32).mean().item()
                                     )
                test_loss.append(epoch_losses)
                test_accuracy.append(epoch_accuracy)
    return dict(
            train_loss=train_loss,
            test_loss=test_loss,
            test_accuracy=test_accuracy
    ) 


# In[13]:


configurations = dict(
    he_normal_init=dict(
        use_bn=False,
        w_init=(lambda w: torch.nn.init.kaiming_normal_(w, nonlinearity='relu'))
    ),
    he_normal_init_with_batchnorm=dict(
        use_bn=True,
        w_init=None
    )
)


                                                 # the '**' notation transforms the dictionary
                                                 # into keyword arguments, as if we called:
result = {                                       # Net(use_batchnorm=config['use_batchnorm'],
    name : train_model(Covid_19model(**config).to(device)) #     initialization=config['initialization'])
    for name, config in configurations.items()
} # train the defined configurations, 
  # get the result as a dictionary


# In[14]:


import matplotlib.pyplot as plt 
import matplotlib

fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(12, 4), dpi=100)

# per step loss values are too noizy, so we'll use a function to 
# average them with a running window
def running_mean(x, win_size):
    return (np.cumsum(x)[win_size:] - np.cumsum(x[:-win_size])) / win_size

for (name, metrics), color in zip(result.items(),
                                  matplotlib.rcParams['axes.prop_cycle'].by_key()['color']):
    ax0.plot(
        running_mean(metrics['train_loss'], 20),
        color=color, label=name, alpha=0.8
    )
    ax0.plot(
        np.linspace(0, len(metrics['train_loss']), len(metrics['test_loss']) + 1)[1:],
        metrics['test_loss'], '--',
        color=color, alpha=0.8
    )
    ax0.set_ylabel("Loss")

    ax1.plot(metrics['test_accuracy'], color=color, label=name)
    ax1.set_ylabel("Test accuracy")

ax1.legend();


# In[ ]:




