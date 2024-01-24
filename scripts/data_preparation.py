"""Data Preparation Script for an Azure Cognitive Search Index."""
import argparse
import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime

import requests
import time
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureCliCredential
from azure.search.documents import SearchClient
from tqdm import tqdm

from data_utils import chunk_directory

sys.path.append(os.getcwd() + '/..')
from config import settings
from definitions import ROOT_DIR

SUPPORTED_LANGUAGE_CODES = {
    "ar": "Arabic",
    "hy": "Armenian",
    "eu": "Basque",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh-Hans": "Chinese Simplified",
    "zh-Hant": "Chinese Traditional",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian (Bahasa)",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lv": "Latvian",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt-Br": "Portuguese (Brazil)",
    "pt-Pt": "Portuguese (Portugal)",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish"
}


def check_if_search_service_exists(search_service_name: str,
    subscription_id: str,
    resource_group: str,
    credential = None):
    """_summary_

    Args:
        search_service_name (str): _description_
        subscription_id (str): _description_
        resource_group (str): _description_
        credential: Azure credential to use for getting acs instance
    """
    if credential is None:
        raise ValueError("credential cannot be None")
    url = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}/providers/Microsoft.Search/searchServices"
        f"/{search_service_name}?api-version=2021-04-01-preview"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {credential.get_token('https://management.azure.com/.default').token}",
    }

    response = requests.get(url, headers=headers)
    return response.status_code == 200


def create_search_service(
    search_service_name: str,
    subscription_id: str,
    resource_group: str,
    location: str,
    sku: str = "standard",
    credential = None,
):
    """_summary_

    Args:
        search_service_name (str): _description_
        subscription_id (str): _description_
        resource_group (str): _description_
        location (str): _description_
        credential: Azure credential to use for creating acs instance

    Raises:
        Exception: _description_
    """
    if credential is None:
        raise ValueError("credential cannot be None")
    url = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}/providers/Microsoft.Search/searchServices"
        f"/{search_service_name}?api-version=2021-04-01-preview"
    )

    payload = {
        "location": f"{location}",
        "sku": {"name": sku},
        "properties": {
            "replicaCount": 1,
            "partitionCount": 1,
            "hostingMode": "default",
            "semanticSearch": "free",
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {credential.get_token('https://management.azure.com/.default').token}",
    }

    response = requests.put(url, json=payload, headers=headers)
    if response.status_code != 201:
        raise Exception(
            f"Failed to create search service. Error: {response.text}")

def create_or_update_search_index(
        service_name, 
        subscription_id, 
        resource_group, 
        index_name, 
        semantic_config_name, 
        credential, 
        language,
        vector_config_name=None):
    if credential is None:
        raise ValueError("credential cannot be None")
    admin_key = json.loads(
        subprocess.run(
            f"az search admin-key show --subscription {subscription_id} --resource-group {resource_group} --service-name {service_name}",
            shell=True,
            capture_output=True,
        ).stdout
    )["primaryKey"]

    url = f"https://{service_name}.search.windows.net/indexes/{index_name}?api-version=2023-07-01-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": admin_key,
    }

    body = {
        "fields": [
            {
                "name": "id",
                "type": "Edm.String",
                "searchable": True,
                "key": True,
            },
            {
                "name": "content",
                "type": "Edm.String",
                "searchable": True,
                "sortable": False,
                "facetable": False,
                "filterable": False,
                "analyzer": f"{language}.lucene" if language else None,
            },
            {
                "name": "doc_type",
                "type": "Edm.String",
                "searchable": False,
                "sortable": False,
                "facetable": True,
                "filterable": True,
            },
            {
                "name": "title",
                "type": "Edm.String",
                "searchable": True,
                "sortable": False,
                "facetable": False,
                "filterable": False,
                "analyzer": f"{language}.lucene" if language else None,
            },
            {
                "name": "filepath",
                "type": "Edm.String",
                "searchable": True,
                "sortable": False,
                "facetable": False,
                "filterable": False,
            },
            {
                "name": "url",
                "type": "Edm.String",
                "searchable": True,
            },
            {
                "name": "metadata",
                "type": "Edm.String",
                "searchable": True,
                "analyzer": f"{language}.lucene" if language else None,
            },
        ],
        "suggesters": [],
        "scoringProfiles": [],
        "semantic": {
            "configurations": [
                {
                    "name": semantic_config_name,
                    "prioritizedFields": {
                        "titleField": {"fieldName": "title"},
                        "prioritizedContentFields": [{"fieldName": "content"}],
                        "prioritizedKeywordsFields": [],
                    },
                }
            ]
        },
    }

    if vector_config_name:
        body["fields"].append({
            "name": "contentVector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "retrievable": True,
            "dimensions": 1536,
            "vectorSearchConfiguration": "default"
        })

        body["vectorSearch"] = {
            "algorithmConfigurations": [
                {
                    "name": vector_config_name,
                    "kind": "hnsw"
                }
            ]
        }

    response = requests.put(url, json=body, headers=headers)
    if response.status_code == 201:
        print(f"Created search index {index_name}")
    elif response.status_code == 204:
        print(f"Updated existing search index {index_name}")
    else:
        raise Exception(f"Failed to create search index. Error: {response.text}")
    
    return True

def upload_documents_to_index(service_name, subscription_id, resource_group, index_name, docs, credential, upload_batch_size = 50):
    if credential is None:
        raise ValueError("credential cannot be None")
    
    to_upload_dicts = []

    id = 0
    for document in docs:
        d = dataclasses.asdict(document)
        # add id to documents
        d.update({"@search.action": "upload", "id": str(id)})
        if "contentVector" in d and d["contentVector"] is None:
            del d["contentVector"]
        to_upload_dicts.append(d)
        id += 1
    
    endpoint = "https://{}.search.windows.net/".format(service_name)
    admin_key = json.loads(
        subprocess.run(
            f"az search admin-key show --subscription {subscription_id} --resource-group {resource_group} --service-name {service_name}",
            shell=True,
            capture_output=True,
        ).stdout
    )["primaryKey"]

    search_client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(admin_key),
    )
    # Upload the documents in batches of upload_batch_size
    for i in tqdm(range(0, len(to_upload_dicts), upload_batch_size), desc="Indexing Chunks..."):
        batch = to_upload_dicts[i: i + upload_batch_size]
        results = search_client.upload_documents(documents=batch)
        num_failures = 0
        errors = set()
        for result in results:
            if not result.succeeded:
                print(f"Indexing Failed for {result.key} with ERROR: {result.error_message}")
                num_failures += 1
                errors.add(result.error_message)
        if num_failures > 0:
            raise Exception(f"INDEXING FAILED for {num_failures} documents. Please recreate the index."
                            f"To Debug: PLEASE CHECK chunk_size and upload_batch_size. \n Error Messages: {list(errors)}")

def validate_index(service_name, subscription_id, resource_group, index_name):
    api_version = "2021-04-30-Preview"
    admin_key = json.loads(
        subprocess.run(
            f"az search admin-key show --subscription {subscription_id} --resource-group {resource_group} --service-name {service_name}",
            shell=True,
            capture_output=True,
        ).stdout
    )["primaryKey"]

    headers = {
        "Content-Type": "application/json", 
        "api-key": admin_key}
    params = {"api-version": api_version}
    url = f"https://{service_name}.search.windows.net/indexes/{index_name}/stats"
    for retry_count in range(5):
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            response = response.json()
            num_chunks = response['documentCount']
            if num_chunks==0 and retry_count < 4:
                print("Index is empty. Waiting 60 seconds to check again...")
                time.sleep(60)
            elif num_chunks==0 and retry_count == 4:
                print("Index is empty. Please investigate and re-index.")
            else:
                print(f"The index contains {num_chunks} chunks.")
                average_chunk_size = response['storageSize']/num_chunks
                print(f"The average chunk size of the index is {average_chunk_size} bytes.")
                break
        else:
            if response.status_code==404:
                print(f"The index does not seem to exist. Please make sure the index was created correctly, and that you are using the correct service and index names")
            elif response.status_code==403:
                print(f"Authentication Failure: Make sure you are using the correct key")
            else:
                print(f"Request failed. Please investigate. Status code: {response.status_code}")
            break

def create_index(config, credential, embedding_model_endpoint=None, njobs=4):
    service_name = config["search_service_name"]
    subscription_id = config["subscription_id"]
    resource_group = config["resource_group"]
    location = config["location"]
    index_name = config["index_name"]
    language = config.get("language", None)

    if language and language not in SUPPORTED_LANGUAGE_CODES:
        raise Exception(f"ERROR: Ingestion does not support {language} documents. "
                        f"Please use one of {SUPPORTED_LANGUAGE_CODES}."
                        f"Language is set as two letter code for e.g. 'en' for English."
                        f"If you donot want to set a language just remove this prompt config or set as None")


    # check if search service exists, create if not
    if check_if_search_service_exists(service_name, subscription_id, resource_group, credential):
        print(f"Using existing search service {service_name}")
    else:
        print(f"Creating search service {service_name}")
        create_search_service(service_name, subscription_id, resource_group, location, credential=credential)

    # create or update search index with compatible schema
    if not create_or_update_search_index(service_name, subscription_id, resource_group, index_name, config["semantic_config_name"], credential, language, vector_config_name=config.get("vector_config_name", None)):
        raise Exception(f"Failed to create or update index {index_name}")
    
    # chunk directory
    print("Chunking directory...")
    add_embeddings = False
    if config.get("vector_config_name") and embedding_model_endpoint:
        add_embeddings = True

    result = chunk_directory(config["data_path"],
                             njobs=njobs,
                             add_embeddings=add_embeddings)

    if len(result.chunks) == 0:
        raise Exception("No chunks found. Please check the data path and chunk size.")
    else:
        if settings.SAVE_DATA_LOCAL == True:
            chunks_json_path = os.path.join(ROOT_DIR, "chunks_json")
            if not os.path.exists(chunks_json_path):
                os.mkdir(chunks_json_path)
            chunks_json_file = os.path.join(chunks_json_path, f"{settings.AZURE_SEARCH_INDEX}_uploaded_{datetime.today().date()}.json")
            with open(chunks_json_file, "w") as f:
                json.dump(result.jsonify_chunks(), f)
            print(f"Archived uploaded chunks to {chunks_json_file}")

    print(f"Processed {result.total_files} files")
    print(f"Unsupported formats: {result.num_unsupported_format_files} files")
    print(f"Files with errors: {result.num_files_with_errors} files")
    print(f"Found {len(result.chunks)} chunks")

    # upload documents to index
    print("Uploading documents to index...")
    upload_documents_to_index(service_name, subscription_id, resource_group, index_name, result.chunks, credential)

    # check if index is ready/validate index
    print("Validating index...")
    validate_index(service_name, subscription_id, resource_group, index_name)
    print("Index validation completed")


def valid_range(n):
    n = int(n)
    if n < 1 or n > 32:
        raise argparse.ArgumentTypeError("njobs must be an Integer between 1 and 32.")
    return n

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--njobs", type=valid_range, default=1, help="Number of jobs to run (between 1 and 32). Default=1")
    args = parser.parse_args()

    index_config = settings.PREP_CONFIG

    credential = AzureCliCredential()
    form_recognizer_client = None

    print("Data preparation script started")

    print("Preparing data for index:", index_config["index_name"])
    if index_config.get("vector_config_name") and not settings.AZURE_OPENAI_EMBEDDING_ENDPOINT:
        raise Exception("ERROR: Vector search is enabled in the config, but no embedding model endpoint and key were provided. Please provide these values or disable vector search.")

    create_index(index_config, credential, embedding_model_endpoint=settings.AZURE_OPENAI_EMBEDDING_ENDPOINT, njobs=args.njobs)
    print("Data preparation for index", index_config["index_name"], "completed")

    print(f"Data preparation script completed. 1 indexes updated.")