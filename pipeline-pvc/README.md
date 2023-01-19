# 사용방법
## 1. Volume 생성
### 1. Notebook 생성
<img width="668" alt="image" src="https://user-images.githubusercontent.com/63439911/213460434-3d7dcf48-bace-42e2-a5e7-da0ca48841c4.png">  
Notebook을 생성하면 자동으로 Volume도 같이 생성됨.
<img width="629" alt="image" src="https://user-images.githubusercontent.com/63439911/213460804-9e24239c-75e5-4e67-8135-05442e3f561f.png">  

### 2. Volume Tab으로 이동하여 확인
<img width="1419" alt="image" src="https://user-images.githubusercontent.com/63439911/213461097-68c1b9d2-ac9b-4c32-8e40-f1fe501918e7.png">




## 2. Pipeline Python 파일 실행  
```    
python pipeline.py
```    
-> pvc-pipeline.yaml 파일 자동으로 생성됨.

## 3. KFP_SDK.ipynb를 통해 실행
> Pipeline Upload 에서 경로 변경

# 각 Component 설명
## Data
> MNIST dataset을 download해서 mount 해놓은 pvc에 저장

## Train
> Minio 서버에서 MNIST dataset을 다운로드 해서 학습 진행 후 Model을 다시 mount 해놓은 pvc에 저장.  


# 실행환경
각 Components들 모두 ContainerOp 방식으로 구현  
모든 Components는 Dockerfile을 통해 Docker Build 진행 후 hub.docker.com에 public으로 배포함.  
각 Component들이 실행될 때 Docker Hub에서 image를 pull해와서 실행하는 방식으로 pipeline 동작.
