# 사용방법
## 1. Pipeline Python 파일 실행  
```    
python pytorch-pipeline.py
```    
-> java-pytorch-pipeline.yaml 파일 자동으로 생성됨.

## 2. KFP_SDK.ipynb를 통해 실행
> Pipeline Upload 에서 경로 변경

# 각 Component 설명
## Data
> MNIST dataset을 download해서 minio server로 Upload

## Train
> Minio 서버에서 MNIST dataset을 다운로드 해서 학습 진행 후 Model을 다시 Minio에 Upload

## Java-Docker-App
> Python 외에 다른 컴포넌트 추가 가능 여부를 위해 추가한 java code

# 실행환경
각 Components들 모두 ContainerOp 방식으로 구현  
모든 Components는 Dockerfile을 통해 Docker Build 진행 후 hub.docker.com에 public으로 배포함.  
각 Component들이 실행될 때 Docker Hub에서 image를 pull해와서 실행하는 방식으로 pipeline 동작.