import os
import random
import sys
from contextlib import contextmanager
from functools import partial
from itertools import starmap

import mlflow
from more_itertools import consume


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# Define a function to log parameters and metrics and add tag
# logging for search_runs functionality
def log_run(run_name, test_no, param1_choices, param2_choices, tag_ident):
    with suppress_stdout():
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            param1 = random.choice(param1_choices)
            param2 = random.choice(param2_choices)
            metric1 = random.uniform(0, 1)
            metric2 = abs(random.gauss(5, 2.5))

            mlflow.log_param("param1", param1)
            mlflow.log_param("param2", param2)
            mlflow.log_metric("metric1", metric1)
            mlflow.log_metric("metric2", metric2)
            mlflow.set_tag("test_identifier", tag_ident)

            return run.info.run_id, metric1, param1, param2


# Generate run names
def generate_run_names(test_no, num_runs=5):
    return (f"run_{i}_test_{test_no}" for i in range(num_runs))


# Execute tuning function, allowing for param overrides,
# run_name disambiguation, and tagging support
def execute_tuning(
    test_no,
    param1_choices=("a", "b", "c"),
    param2_choices=("d", "e", "f"),
    test_identifier="",
    num_child_runs=5,
):
    ident = "default" if not test_identifier else test_identifier
    best_metric1 = float("inf")
    best_params = None
    # Use a parent run to encapsulate the child runs
    with mlflow.start_run(run_name=f"parent_run_test_{ident}_{test_no}"):
        # Partial application of the log_run function
        log_current_run = partial(
            log_run,
            test_no=test_no,
            param1_choices=param1_choices,
            param2_choices=param2_choices,
            tag_ident=ident,
        )
        mlflow.set_tag("test_identifier", ident)
        # Generate run names and apply log_current_run function to each run name
        results = list(
            starmap(
                log_current_run,
                (
                    (run_name,)
                    for run_name in generate_run_names(test_no, num_child_runs)
                ),
            )
        )

        for _, metric1, param1, param2 in results:
            if metric1 < best_metric1:
                best_metric1 = metric1
                best_params = (param1, param2)

        mlflow.log_metric("best_metric1", best_metric1)
        mlflow.log_param("best_param1", best_params[0])
        mlflow.log_param("best_param2", best_params[1])
        # Consume the iterator to execute the runs
        consume(results)


# Set the tracking uri and experiment
mlflow.set_tracking_uri("http://localhost:5801")
mlflow.set_experiment("Parent Child Association Challenge Debug 2")

param_1_values = ["a", "b"]
param_2_values = ["d", "f"]

# Execute hyperparameter tuning runs with custom parameter choices
consume(
    starmap(
        execute_tuning,
        ((x, param_1_values, param_2_values, "subset_test", 25) for x in range(5)),
    )
)
