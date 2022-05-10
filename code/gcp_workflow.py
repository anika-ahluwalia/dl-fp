import tensorflow_cloud as tfc

TF_GPU_IMAGE = "tensorflow/tensorflow:latest-gpu"

GCS_BUCKET = "dl-fp"

tfc.run(
    entry_point="code/main.py",
    entry_point_args=["WORD2VEC"],
    distribution_strategy='auto',
    requirements_txt='requirements.txt',
    docker_image_bucket_name=GCS_BUCKET,
    docker_base_image=TF_GPU_IMAGE,
    chief_config=tfc.COMMON_MACHINE_CONFIGS['K80_1X'],
    worker_config=tfc.COMMON_MACHINE_CONFIGS['K80_1X'],
    job_labels={'job': "my_job"}
)  # Runs your training on Google Cloud!
