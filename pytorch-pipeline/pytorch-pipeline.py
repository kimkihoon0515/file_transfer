import kfp
from kfp import dsl
from kfp import onprem

def data():
    return dsl.ContainerOp(
        name="pytorch data upload",
        image="kimkihoon0515/pytorch-data:macbook1",
        command=["python","data_download.py"],
        arguments=["--download_root",'./MNIST_dataset',"--minio_bucket","mlpipeline"]
    ).apply(onprem.mount_pvc("model-volume",volume_name="model",volume_mount_path="./MNIST_dataset"))

def train():
        return dsl.ContainerOp(
            name="pytorch mnist train",
            image="kimkihoon0515/pytorch-train:macbook4",
            command=["python","train.py"],
            arguments=["--epochs",3],
            file_outputs={
                "mlpipeline_metrics": "./mlpipeline-metrics.json"
            }
        ).apply(onprem.mount_pvc("model-volume",volume_name="model",volume_mount_path="./model"))\
        .apply(onprem.mount_pvc("data-volume",volume_name="data",volume_mount_path="./data"))

def jav():
    return dsl.ContainerOp(
        name="Java app",
        image="kimkihoon0515/java-docker-app",
        command=["java","Hello"]
    )


@dsl.pipeline(
    name="pytorch image pipeline",
    description="training mnist data from minio in docker image"
)
def pytorch_train():
    
    step1 = data()
    step2 = train()
    step3 = jav()

    step2.after(step1)
    step3.after(step2)
    
    '''
    data = dsl.ContainerOp(
        name="pytorch data upload",
        image="kimkihoon0515/pytorch-data",
        command=["python","data.py"]
    )
    '''
    '''
    train = dsl.ContainerOp(
        name="pytorch mnist train",
        image="kimkihoon0515/pytorch-train",
        command=["python","train.py"]
    )
    '''

if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(pytorch_train, "java-pytorch-pipeline.yaml")