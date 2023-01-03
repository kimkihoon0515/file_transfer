import kfp
from kfp import dsl


@dsl.pipeline(
    name="pytorch image pipeline",
    description="training mnist data from minio in docker image"
)
def pytorch_train():
    data = dsl.ContainerOp(
        name="pytorch mnist train",
        image="kimkihoon0515/pytorch-train",
        command=["python","train.py"]
    )

if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(pytorch_train, "pytorch-pipeline.yaml")