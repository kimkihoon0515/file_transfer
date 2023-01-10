from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from minio import Minio
import os
from typing import NamedTuple
import argparse

def download_dataset(args): 
    download_root = args.download_root # 데이터 다운로드 경로

    train_dataset = datasets.MNIST(root=download_root,
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True) # 학습 dataset 정의
                            
    test_dataset = datasets.MNIST(root=download_root,
                            train=False,
                            transform=transforms.ToTensor(), 
                            download=True) # 평가 dataset 정의



    batch_size = 100 # 배치 사이즈 정의. 데이터셋을 잘개 쪼개서 묶음으로 만드는 데 기여한다.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_root',type=str)
    args = parser.parse_args()
    download_dataset(args)