name: pytorch_test

conda_env: conda.yaml

# Can have a docker_env instead of a conda_env, e.g.
# 1 Image without a registry path
# docker_env:
#    image:  mlflow-docker-example
# 2 Mounting volumes and specifying environment variables
# docker_env:
#   image: mlflow-docker-example-environment
#   volumes: ["/local/path:/container/mount/path"]
#   environment: [["NEW_ENV_VAR", "new_var_value"], "VAR_TO_COPY_FROM_HOST_ENVIRONMENT"]
# 3 Image in a remote registry
# docker_env:
#   image: 012345678910.dkr.ecr.us-west-2.amazonaws.com/mlflow-docker-example-environment:7.0

entry_points:
  main:
    parameters:
      learning_rate: {type: float, default: 0.0001}
      batch_size: {type: int, default: 8}
      num_epoch: {type: int, default: 1}
    command: "python train.py --lr {learning_rate} --batch_size {batch_size} --num_epoch {num_epoch}"
  #validate:
  #  parameters:
  #    data_file: path
  #  command: "python validate.py {data_file}"  

# MLflow provides two ways to run projects: the mlflow run command-line tool, or the mlflow.projects.run() Python API. 
# ex) mlflow run git@github.com:mlflow/mlflow-example.git -P alpha=0.5