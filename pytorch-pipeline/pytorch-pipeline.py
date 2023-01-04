import kfp
from kfp import dsl

def data():
    return dsl.ContainerOp(
        name="pytorch data upload",
        image="kimkihoon0515/pytorch-data",
        command=["python","data.py"]
    )

def train():
        return dsl.ContainerOp(
            name="pytorch mnist train",
            image="kimkihoon0515/pytorch-train",
            command=["python","train.py"]
        )

def jav():
    return dsl.ContainerOp(
        name="Java app",
        image="krizel2121/java-docker-app",
        command=["java","Hello"]
    )


@dsl.pipeline(
    name="pytorch image pipeline",
    description="training mnist data from minio in docker image"
)
def pytorch_train():
    
    
    train().after(data())
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
    compiler.Compiler().compile(pytorch_train, "pytorch-pipeline.yaml")