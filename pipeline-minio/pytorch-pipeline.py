import kfp
from kfp import dsl


# 각각의 함수 = 하나의 component
def data():
    return dsl.ContainerOp(
        name="pytorch data upload",
        image="kimkihoon0515/pytorch-data:macbook1",
        command=["python","data_download.py"],
        arguments=["--download_root",'./MNIST_dataset',"--minio_bucket","mlpipeline"]
    )

def train():
        return dsl.ContainerOp(
            name="pytorch mnist train",
            image="kimkihoon0515/pytorch-train:macbook4",
            command=["python","train.py"],
            arguments=["--epochs",3],
            file_outputs={
                "mlpipeline_metrics": "./mlpipeline-metrics.json"
            }
        )

def jav():
    return dsl.ContainerOp(
        name="Java app",
        image="kimkihoon0515/java-docker-app",
        command=["java","Hello"]
    )

# pipeline Metadata 부분
@dsl.pipeline( 
    name="pytorch image pipeline",
    description="training mnist data from minio in docker image"
)
def pytorch_train():
    # components 실행 순서 정의
    step1 = data()
    step2 = train()
    step3 = jav()

    step2.after(step1)
    step3.after(step2)

if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(pytorch_train, "java-pytorch-pipeline.yaml") # argo workflow 형태의 yaml파일로 compile