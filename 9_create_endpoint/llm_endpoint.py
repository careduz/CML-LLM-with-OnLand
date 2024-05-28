import json
import os
import time
from typing import Any, Optional, Union

import boto3
import cmlapi
import gradio as gr
import pinecone
import requests
import tensorflow as tf
from botocore.config import Config
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
MODEL_OPTIONS = [
    "Local Mistral 7B",
    "AWS Bedrock Claude v2.1",
]
VECTORDB_OPTIONS = ["None", "Pinecone"]

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

print("initialising Pinecone connection...")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
print("Pinecone initialised")

print(f"Getting '{PINECONE_INDEX}' as object...")
index = pinecone.Index(PINECONE_INDEX)
print("Success")

# Get latest statistics from index
current_collection_stats = index.describe_index_stats()
print(
    "Total number of embeddings in Pinecone index is {}.".format(
        current_collection_stats.get("total_vector_count")
    )
)

## TO DO GET MODEL DEPLOYMENT
## Need to get the below prgramatically in the future iterations
client = cmlapi.default_client(
    url=os.getenv("CDSW_API_URL").replace("/api/v1", ""),
    cml_api_key=os.getenv("CDSW_APIV2_KEY"),
)
projects = client.list_projects(
    include_public_projects=True, search_filter=json.dumps({"name": "Shared LLM Model"})
)
project = projects.projects[0]

## Here we assume that only one model has been deployed in the project, if this is not true this should be adjusted (this is reflected by the placeholder 0 in the array)
model = client.list_models(project_id=project.id)
selected_model = model.models[0]

## Save the access key for the model to the environment variable of this project
MODEL_ACCESS_KEY = selected_model.access_key

MODEL_ENDPOINT = (
    os.getenv("CDSW_API_URL")
    .replace("https://", "https://modelservice.")
    .replace("/api/v1", "/model?accessKey=")
)
MODEL_ENDPOINT = MODEL_ENDPOINT + MODEL_ACCESS_KEY

# MODEL_ACCESS_KEY = os.environ["CML_MODEL_KEY"]
# MODEL_ENDPOINT = "https://modelservice.ml-8ac9c78c-674.se-sandb.a465-9q4k.cloudera.site/model?accessKey=" + MODEL_ACCESS_KEY

if os.environ.get("AWS_DEFAULT_REGION") == "":
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"


## Setup Bedrock client:
def get_bedrock_client(
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    endpoint_url :
        Optional override for the Bedrock service API Endpoint. If setting this, it should usually
        include the protocol i.e. "https://..."
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    """
    target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    bedrock_client = session.client(
        service_name="bedrock-runtime", config=retry_config, **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


boto3_bedrock = get_bedrock_client(region=os.environ.get("AWS_DEFAULT_REGION", None))


# Helper function for generating responses for the QA app
def get_responses(message, history, model, temperature, token_count, vector_db):

    if model == "Local Mistral 7B":

        if vector_db == "None":
            context_chunk = ""
            response = get_llama2_response_with_context(
                message, context_chunk, temperature, token_count
            )

            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[: i + 1]

        elif vector_db == "Pinecone":
            # TODO: sub this with call to Pinecone to get context chunks
            # response = "ERROR: Pinecone is not implemented for LLama yet"

            # Vector search the index
            context_chunk, source, score = get_nearest_chunk_from_pinecone_vectordb(
                index, message
            )

            # Call CML hosted model
            response = get_llama2_response_with_context(
                message, context_chunk, temperature, token_count
            )

            # Add reference to specific document in the response
            response = (
                f"{response}\n\n For additional info see: {url_from_source(source)}"
            )

            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[: i + 1]
    elif model == "AWS Bedrock Claude v2.1":
        if vector_db == "None":
            # No context call Bedrock
            response = get_bedrock_response_with_context(
                message, "", temperature, token_count
            )

            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                yield response[: i + 1]

        elif vector_db == "Pinecone":
            # Vector search the index
            context_chunk, source, score = get_nearest_chunk_from_pinecone_vectordb(
                index, message
            )

            # Call Bedrock model
            response = get_bedrock_response_with_context(
                message, context_chunk, temperature, token_count
            )

            response = (
                f"{response}\n\n For additional info see: {url_from_source(source)}"
            )

            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.01)
                yield response[: i + 1]


def url_from_source(source):
    url = source.replace("/home/cdsw/data/https:/", "https://").replace(".txt", ".html")
    return f"[Reference 1]({url})"


# Get embeddings for a user question and query Pinecone vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_pinecone_vectordb(index, question):
    # Generate embedding for user question with embedding model
    retriever = SentenceTransformer(EMBEDDING_MODEL_REPO)
    xq = retriever.encode([question]).tolist()
    xc = index.query(xq, top_k=5, include_metadata=True)

    matching_files = []
    scores = []
    for match in xc["matches"]:
        # extract the 'file_path' within 'metadata'
        file_path = match["metadata"]["file_path"]
        # extract the individual scores for each vector
        score = match["score"]
        scores.append(score)
        matching_files.append(file_path)

    # Return text of the nearest knowledge base chunk
    # Note that this ONLY uses the first matching document for semantic search. matching_files holds the top results so you can increase this if desired.
    response = load_context_chunk_from_data(matching_files[0])
    sources = matching_files[0]
    score = scores[0]

    print(f"Response of context chunk {response}")
    return response, sources, score
    # return "Cloudera is an Open Data Lakhouse company", "http://cloudera.com", 89


# Return the Knowledge Base doc based on Knowledge Base ID (relative file path)
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f:  # Open file in read mode
        return f.read()


def get_bedrock_response_with_context(question, context, temperature, token_count):

    # Supply different instructions, depending on whether or not context is provided
    if context == "":
        instruction_text = """Human: You are a helpful, honest, and courteous assistant. If you don't know the answer, simply state I don't know the answer to that question. Please provide an honest response to the user question enclosed in <question></question> tags. Do not repeat the question in the output.
    
    <question>{{QUESTION}}</question>
                    Assistant:"""
    else:
        instruction_text = """Human: You are a helpful, honest, and courteous assistant. If you don't know the answer, simply state I don't know the answer to that question. Please read the text provided between the tags <text></text> and provide an honest response to the user question enclosed in <question></question> tags. Do not repeat the question in the output.
    <text>{{CONTEXT}}</text>
    
    <question>{{QUESTION}}</question>
                    Assistant:"""

    # Replace instruction placeholder to build a complete prompt
    full_prompt = instruction_text.replace("{{QUESTION}}", question).replace(
        "{{CONTEXT}}", context
    )

    # Model expects a JSON object with a defined schema
    body = json.dumps(
        {
            "prompt": full_prompt,
            "max_tokens_to_sample": int(token_count),
            "temperature": float(temperature),
            "top_k": 250,
            "top_p": 1.0,
            "stop_sequences": [],
        }
    )

    # Provide a model ID and call the model with the JSON payload
    modelId = "anthropic.claude-v2:1"
    response = boto3_bedrock.invoke_model(
        body=body,
        modelId=modelId,
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response.get("body").read())
    print("Model results successfully retreived")

    result = response_body.get("completion")
    # print(response_body)

    return result


# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_llama2_response_with_context(question, context, temperature, token_count):

    llama_sys = f'<s>[INST]You are a helpful, respectful and honest assistant. If you are unsure about an answer, truthfully say "I don\'t know".'

    if context == "":
        # Following LLama's spec for prompt engineering
        llama_inst = f"Please answer the user question.[/INST]</s>"
        question_and_context = f"{llama_sys} {llama_inst} \n [INST] {question} [/INST]"
    else:
        # Add context to the question
        llama_inst = f"Anser the user's question based on the folloing information:\n {context}[/INST]</s>"
        question_and_context = f"{llama_sys} {llama_inst} \n[INST] {question} [/INST]"

    try:
        # Build a request payload for CML hosted model
        data = {
            "request": {
                "prompt": question_and_context,
                "temperature": temperature,
                "max_new_tokens": token_count,
                "repetition_penalty": 1.0,
            }
        }

        r = requests.post(
            MODEL_ENDPOINT,
            data=json.dumps(data),
            headers={"Content-Type": "application/json"},
        )

        # Logging
        print(f"Request: {data}")
        print(f"Response: {r.json()}")

        no_inst_response = str(r.json()["response"]["prediction"]["response"])[
            len(question_and_context) - 6 :
        ]

        return no_inst_response

    except Exception as e:
        print(e)
        return e


def endpoint(args):
    # Parse payload
    message = args.get("message")

    model = args.get("model")
    if model is None:
        model = "AWS Bedrock Claude v2.1"

    token_count = args.get("token_count")
    if token_count is None:
        token_count = 250
    else:
        token_count = int(token_count)

    vector_db = args.get("vector_db")
    if vector_db is None:
        vector_db = "None"

    temperature = args.get("temperature")
    if temperature is None:
        temperature = 50
    else:
        temperature = int(temperature)

    # Validate payload
    assert message not in ["", None], "message cannot be empty"

    assert model in MODEL_OPTIONS, f"model was '{model}' but must be in {MODEL_OPTIONS}"

    assert (
        token_count > 50 and token_count < 1000
    ), f"token_count must be between 50 and 1000 but received {token_count}"

    assert (
        vector_db in VECTORDB_OPTIONS
    ), f"vector_db was '{vector_db}' but must be in {VECTORDB_OPTIONS}"

    assert (
        temperature > 0 and temperature <= 100
    ), f"temperature must be between 1 and 100 but received {temperature}"

    temperature = round(temperature / 100, 2)

    # Get results
    results = []
    for response in get_responses(
        message, None, model, temperature, token_count, vector_db
    ):
        results.append(response)

    return results
