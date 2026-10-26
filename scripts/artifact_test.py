import mlflow

# Укажите URL Tracking Server
mlflow.set_tracking_uri("http://127.0.0.1:5801")

# Создайте новый эксперимент (если он не создан)
experiment_name = "s3_artifact_test"
mlflow.set_experiment(experiment_name)

# Логирование артефактов
with mlflow.start_run():
    # Логируем параметр, метрику и файл как артефакт
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)

    # Логируем файл как артефакт
    mlflow.log_artifact("../data/artifact.txt")
    print("Артефакт загружен!")
