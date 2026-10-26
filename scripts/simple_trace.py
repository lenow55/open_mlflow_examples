import mlflow


mlflow.set_tracking_uri("http://localhost:5801")
_ = mlflow.set_experiment("custom-trace")


# Mark any function with the trace decorator to automatically capture input(s) and output(s)
@mlflow.trace(name="My Span", span_type="func", attributes={"a": 1, "b": 2})
def some_function(x, y, z=2):
    return x + (y - z)


# Invoking the function will generate a trace that is logged to the active experiment
some_function(2, 4)
