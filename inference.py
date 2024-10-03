from dataset import MNIST
import torch 
from MoE import MNIST_MoE
from config import *
from torch.utils.data import DataLoader
import time 

EPOCH=10
BATCH_SIZE=64 
DEVICE='cuda' if torch.cuda.is_available() else 'cpu' 

dataset=MNIST() 
dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,num_workers=0)

model=MNIST_MoE(INPUT_SIZE,EXPERTS,TOP,EMBEDDING_SIZE).to(DEVICE)
model.load_state_dict(torch.load('./checkpoints/model.pth'))

model.eval() 

start_time=time.time()

correct=0
for epoch in range(EPOCH):
    for img,label in dataloader:
        logits,_,_=model(img.to(DEVICE))
        
        correct+=(logits.cpu().argmax(-1)==label).sum()
        
print('正确率:%.2f'%(correct/(len(dataset)*EPOCH)*100),'耗时:',time.time()-start_time,'s')