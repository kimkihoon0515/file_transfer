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

    import glob

    def upload_local_directory_to_minio(local_path, bucket_name, minio_path):
        assert os.path.isdir(local_path)

        for local_file in glob.glob(local_path + '/**'):
            local_file = local_file.replace(os.sep, "/") # Replace \ with / on Windows
            if not os.path.isfile(local_file):
                upload_local_directory_to_minio(
                    local_file, bucket_name, minio_path + "/" + os.path.basename(local_file))
            else:
                remote_path = os.path.join(
                    minio_path, local_file[1 + len(local_path):])
                remote_path = remote_path.replace(
                    os.sep, "/")  # Replace \ with / on Windows
                minio_client.fput_object(bucket_name, remote_path, local_file)

    batch_size = 100 # 배치 사이즈 정의. 데이터셋을 잘개 쪼개서 묶음으로 만드는 데 기여한다.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 학습 데이터셋을 배치 사이즈 크기만큼씩 잘라서 묶음으로 만든다. 묶음의 개수는 train_dataset / batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) # train_dataloader와 마찬가지

    minio_client = Minio(
        "172.17.0.27:9000", # minio pod ip:port
        access_key="minio",
        secret_key="minio123",
        secure=False
    )

    minio_bucket = args.minio_bucket

    print(os.listdir('./'))
    print(download_root,minio_bucket)

    upload_local_directory_to_minio(local_path=download_root,bucket_name=minio_bucket,minio_path="mnist/")

    print("minio upload completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_root',type=str,default='/MNIST_DATA')
    parser.add_argument('--minio_bucket',type=str,default='mlpipeline')
    args = parser.parse_args()
    download_dataset(args)