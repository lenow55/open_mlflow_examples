# INFO: этот код для оценки работы ЛЛМ в RAG пайплайнах
# В данном коде собирается простой RAG пайплайн, который
# прогоняется на нескольких вопросах. Далее результаты
# прогонов оцениваются через ЛЛМ и результаты трассируются
# в MLflow
import os

import mlflow
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from mlflow.metrics.genai import EvaluationExample, faithfulness, relevance
from pydantic import SecretStr

# Calling the autolog API will enable trace logging by default.
mlflow.set_tracking_uri(uri="http://127.0.0.1:5801")
mlflow.openai.autolog()
mlflow.set_experiment("OpenAIQA")

loader = WebBaseLoader("https://mlflow.org/docs/latest/index.html")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# WARN: тут без эмбедингов работать не будет
embeddings = OpenAIEmbeddings(
    base_url="http://localhost:8953/v1",
    model="Qwen/Qwen3-8B-Embedings",
    api_key=SecretStr(secret_value="api_key"),
)
docsearch = Chroma.from_documents(texts, embeddings)

llm_api_key = SecretStr("NYXV2sS3PLDYLbC")


os.environ["OPENAI_API_KEY"] = llm_api_key.get_secret_value()

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(
        model="Qwen/Qwen3-8B-FP8",
        temperature=0,
        api_key=llm_api_key,
        base_url="http://localhost:8953/v1",
    ),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
)


def model(input_df):
    answer = []
    for index, row in input_df.iterrows():
        answer.append(qa(row["questions"]))

    return answer


eval_df = pd.DataFrame(
    {
        "questions": [
            "What is MLflow?",
            "How to run mlflow.evaluate()?",
            "How to log_table()?",
            "How to load_table()?",
        ],
    }
)

# Create a good and bad example for faithfulness in the context of this problem
faithfulness_examples = [
    EvaluationExample(
        input="How do I disable MLflow autologging?",
        output="mlflow.autolog(disable=True) will disable autologging for all functions. In Databricks, autologging is enabled by default. ",
        score=2,
        justification="The output provides a working solution, using the mlflow.autolog() function that is provided in the context.",
        grading_context={
            "context": "mlflow.autolog(log_input_examples: bool = False, log_model_signatures: bool = True, log_models: bool = True, log_datasets: bool = True, disable: bool = False, exclusive: bool = False, disable_for_unsupported_versions: bool = False, silent: bool = False, extra_tags: Optional[Dict[str, str]] = None) → None[source] Enables (or disables) and configures autologging for all supported integrations. The parameters are passed to any autologging integrations that support them. See the tracking docs for a list of supported autologging integrations. Note that framework-specific configurations set at any point will take precedence over any configurations set by this function."
        },
    ),
    EvaluationExample(
        input="How do I disable MLflow autologging?",
        output="mlflow.autolog(disable=True) will disable autologging for all functions.",
        score=5,
        justification="The output provides a solution that is using the mlflow.autolog() function that is provided in the context.",
        grading_context={
            "context": "mlflow.autolog(log_input_examples: bool = False, log_model_signatures: bool = True, log_models: bool = True, log_datasets: bool = True, disable: bool = False, exclusive: bool = False, disable_for_unsupported_versions: bool = False, silent: bool = False, extra_tags: Optional[Dict[str, str]] = None) → None[source] Enables (or disables) and configures autologging for all supported integrations. The parameters are passed to any autologging integrations that support them. See the tracking docs for a list of supported autologging integrations. Note that framework-specific configurations set at any point will take precedence over any configurations set by this function."
        },
    ),
]

faithfulness_metric = faithfulness(
    model="openai:/Qwen/Qwen3-8B-FP8",
    examples=faithfulness_examples,
    proxy_url="http://localhost:8953/v1",
    # extra_headers={"Authorization": f"Bearer {llm_api_key.get_secret_value()}"},
)

print(faithfulness_metric)


relevance_metric = relevance(
    model="openai:/Qwen/Qwen3-8B-FP8",
    proxy_url="http://localhost:8953/v1",
    # extra_headers={"Authorization": f"Bearer {llm_api_key.get_secret_value()}"},
)

print(relevance_metric)

results = mlflow.evaluate(
    model,
    eval_df,
    model_type="question-answering",
    evaluators="default",
    predictions="result",
    extra_metrics=[faithfulness_metric, relevance_metric, mlflow.metrics.latency()],
    evaluator_config={
        "col_mapping": {
            "inputs": "questions",
            "context": "source_documents",
        }
    },
)
print(results.metrics)

print(results.tables["eval_results_table"])
