import mlflow
from catboost import CatBoostClassifier
from mlflow.models import infer_signature
from sklearn import datasets

mlflow.set_tracking_uri("http://127.0.0.1:5801")
_ = mlflow.set_experiment("Catboost test")

# prepare data
X, y = datasets.load_wine(as_frame=False, return_X_y=True)

# train the model
model = CatBoostClassifier(
    iterations=5,
    loss_function="MultiClass",
    allow_writing_files=False,
)
model.fit(X, y)

# create model signature
predictions = model.predict(X)
signature = infer_signature(X, predictions)

# log the model into a mlflow run
with mlflow.start_run():
    model_info = mlflow.catboost.log_model(model, "model", signature=signature)

# load the logged model and make a prediction
catboost_pyfunc = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
print(catboost_pyfunc.predict(X[:5]))
