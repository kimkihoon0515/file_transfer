import kfp
from kfp import dsl
from kfp import onprem

def data():
    return dsl.ContainerOp(
        name="pytorch data upload",
        image="kimkihoon0515/pytorch-data:macbook1",
        command=["python","data_download.py"]
    )
def train():
        return dsl.ContainerOp(
            name="pytorch mnist train",
            image="kimkihoon0515/pytorch-train:macbook4",
            command=["python","train.py"],
        )

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