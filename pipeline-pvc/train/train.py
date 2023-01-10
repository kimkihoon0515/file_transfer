import argparse

def train(args):
  import torch.nn as nn
  import torch
  from torchvision import datasets
  from torchvision import transforms
  from torch.utils.data import DataLoader
  import numpy as np
  import json
  import os

  mount_path = args.mount_path

  if not os.path.exists(mount_path+'/model'):
    os.mkdir(mount_path+'/model')

  class Net(nn.Module): # 모델 클래스 정의 부분
      
      def __init__(self):
          super(Net, self).__init__()
          self.fc1 = nn.Linear(784,100) # MNIST 데이터셋이 28*28로 총 784개의 픽셀로 이루어져있기 때문에 784를 입력 크기로 넣음.
          self.relu = nn.ReLU()
          self.fc2 = nn.Linear(100,100) # 은닉층
          self.fc3 = nn.Linear(100,10) # 출력층 0~9까지 총 10개의 클래스로 결과가 나옴.

      def forward(self, x): # 입력층 -> 활성화 함수(ReLU) -> 은닉층 -> 활성화 함수(ReLU) -> 출력층
          x1 = self.fc1(x)
          x2 = self.relu(x1)
          x3 = self.fc2(x2)
          x4 = self.relu(x3)
          x5 = self.fc3(x4)

          return x5

  from minio import Minio


  train_dataset = datasets.MNIST(root=args.mount_path,
                          train=True,
                          transform=transforms.ToTensor(),
                          download=False) # 학습 dataset 정의
                          
  test_dataset = datasets.MNIST(root=args.mount_path,
                          train=False,
                          transform=transforms.ToTensor(), 
                          download=False) # 평가 dataset 정의


  batch_size = 100 # 배치 사이즈 정의. 데이터셋을 잘개 쪼개서 묶음으로 만드는 데 기여한다.
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 학습 데이터셋을 배치 사이즈 크기만큼씩 잘라서 묶음으로 만든다. 묶음의 개수는 train_dataset / batch_size
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) # train_dataloader와 마찬가지

  model = Net() # 모델 정의
  loss_function = nn.CrossEntropyLoss() # 실제 정답과 예측값의 차이를 수치화해주는 함수.

  optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
  epochs = args.epochs# 얼마나 학습할 지 정하는 인자.

  best_accuracy = 0 # 평가 지표
  model.zero_grad() # 학습 전에 모델의 모든 weight, bias 값들을 초기화
  
  for epoch in range(epochs):
    
    model.train() # 학습
    train_accuracy = 0 # metric
    train_loss = 0 # metric

    for images, labels in train_loader:
      images = images.reshape(batch_size,784)
      image = model(images)
      loss = loss_function(image,labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      prediction = torch.argmax(image,1)
      correct = (prediction == labels)
      train_accuracy+= correct.sum().item() / len(train_dataset)
      train_loss += loss.item() / len(train_loader)

    model.eval() # 평가
    val_accuracy = 0 # metric
    val_loss = 0 # metric

    for images,labels in test_loader:
      images = images.reshape(batch_size,784)
      image = model(images)
      loss = loss_function(image,labels)
      
      correct = (torch.argmax(image,1) == labels)
      val_accuracy += correct.sum().item() / len(test_dataset)
      val_loss += loss.item() / len(test_loader)
    
    print(f'epoch: {epoch}/{epochs} train_loss: {train_loss:.5} train_accuracy: {train_accuracy:.5} val_loss: {val_loss:.5} val_accuracy: {val_accuracy:.5}')

    if best_accuracy < val_accuracy: # 성능이 가장 좋은 모델로 갱신
      best_accuracy = val_accuracy
      best_val_loss = val_loss
      torch.save(model.state_dict(),mount_path+'/model/best_model.pt')
      print(f"===========> Save Model(Epoch: {epoch}, Accuracy: {best_accuracy:.5})")
    

    print("--------------------------------------------------------------------------------------------")
  
  metrics = {
        'metrics': [{
            'name': 'accuracy-score',
            'numberValue':  best_accuracy,
            'format': "PERCENTAGE",
        }]
    }
  
  with open(mount_path+'/mlpipeline-metrics.json','w') as f:
    json.dump(metrics,f)

  print("best_model uploaded to pvc!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--mount_path',type=str)
  parser.add_argument('--epochs',type=int)
  args = parser.parse_args()
  train(args)