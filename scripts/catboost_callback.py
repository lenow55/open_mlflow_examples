from types import SimpleNamespace

import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


mlflow.set_tracking_uri("http://localhost:5801")
mlflow.set_experiment("Catboost test regress")

# Загружаем данные
X = data
y = target

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Подготовка Pool для CatBoost
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

# Конфигурация CatBoost
params = {
    "iterations": 100,
    "learning_rate": 0.05,
    "depth": 5,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "verbose": 10,
}


# Создаем класс для логирования метрик
class MLflowLogger:
    def __init__(self):
        pass

    def after_iteration(self, info: SimpleNamespace):
        iteration = info.iteration
        if not isinstance(iteration, int):
            return True
        if isinstance(info.metrics, dict):
            for metric_key, metrics_dict in info.metrics.items():
                if not isinstance(metric_key, str):
                    continue
                if not isinstance(metrics_dict, dict):
                    continue
                for metric_name, metric_values in metrics_dict.items():
                    if not isinstance(metric_name, str):
                        continue
                    if not isinstance(metric_values, list):
                        continue
                    _ = mlflow.log_metric(
                        f"{metric_key}_{metric_name}",
                        metric_values[iteration - 1],
                        step=iteration,
                    )

        return True  # Возвращаем True, чтобы CatBoost продолжил обучение


# Начинаем эксперимент в MLflow
with mlflow.start_run():
    # Логируем параметры модели
    mlflow.log_params(params)

    # Инициализация модели
    model = CatBoostRegressor(**params)

    # Создаем коллбэк
    logger = MLflowLogger()

    # Обучение модели с логированием метрик
    _ = model.fit(
        train_pool,
        eval_set=test_pool,
        logging_level="Verbose",
        callbacks=[logger],
    )

    # Оценка модели
    y_pred = model.predict(X_test)
    rmse_test = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mlflow.log_metric("test_rmse", rmse_test)

    # Сохранение модели в MLflow
    mlflow.catboost.log_model(model, artifact_path="model")


print("Модель сохранена и метрики залогированы в MLflow.")
