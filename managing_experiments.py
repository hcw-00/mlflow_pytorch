from mlflow.tracking import MlflowClient
client = MlflowClient()
experiments = client.list_experiments() # returns a list of mlflow.entities.Experiment
print("total number of experiments : {}".format(len(experiments)))
print("list of experiments")
print(experiments)

# run = client.create_run(experiments[0].experiment_id) # returns mlflow.entities.Run
# client.log_param(run.info.run_id, "hello", "world")
# client.set_terminated(run.info.run_id)
# client.set_tag(run.info.run_id, "tag_key", "tag_value") # adding tags to runs