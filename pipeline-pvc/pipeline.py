import kfp
from kfp import dsl
from kfp import onprem

def data():
    return dsl.ContainerOp(
        name="pytorch data upload",
        image="kimkihoon0515/pvc-data",
        command=["python","data.py"],
        arguments=["--download_root",'/home/jovyan']
    ).apply(onprem.mount_pvc("kfp-pvc",volume_name="pipeline",volume_mount_path="/mnt/pipeline"))


def train():
        return dsl.ContainerOp(
            name="pytorch mnist train",
            image="kimkihoon0515/pvc-train",
            command=["python","train.py"],
            arguments=["--mount_path","/home/jovyan","--epochs",3],
            file_outputs={
                "mlpipeline_metrics": "/home/jovyan/mlpipeline-metrics.json"
            }
        ).apply(onprem.mount_pvc("kfp-pvc",volume_name="pipeline",volume_mount_path="/mnt/pipeline"))

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


if __name__ == "__main__":
    import kfp.compiler as compiler
    compiler.Compiler().compile(pytorch_train, "pvc-pipeline.yaml")