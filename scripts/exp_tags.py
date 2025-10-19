import mlflow


mlflow.set_tracking_uri("http://localhost:5801")
experiment = mlflow.set_experiment(experiment_id="3")

print(f"Experiment_id: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")
print(f"Tags: {experiment.tags}")
print(f"Lifecycle_stage: {experiment.lifecycle_stage}")

experiment.tags["test_tag"] = "value_tag"

print(f"Tags: {experiment.tags}")
