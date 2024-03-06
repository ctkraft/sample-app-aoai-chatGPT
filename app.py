import json
import ast
import os
import inspect
import logging
import requests
import openai
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from flask import Flask, Response, request, jsonify, send_from_directory
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.models import (
    QueryType,
    VectorizedQuery,
)

from scripts.data_utils import get_embedding
from backend.assets.prompts import MAKE_QUERY_PROMPT, RAG_PROMPT

#from backend.auth.auth_utils import get_authenticated_user_details
#from backend.history.cosmosdbservice import CosmosConversationClient
from config import settings

app = Flask(__name__, static_folder="static")

# Static Files
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/favicon.ico")
def favicon():
    return app.send_static_file('favicon.ico')

@app.route("/assets/<path:path>")
def assets(path):
    return send_from_directory("static/assets", path)


# settings.SHOULD_STREAM = True if settings.AZURE_OPENAI_STREAM.lower() == "true" else False

# CosmosDB Integration Settings
# AZURE_COSMOSDB_DATABASE = os.environ.get("AZURE_COSMOSDB_DATABASE")
# AZURE_COSMOSDB_ACCOUNT = os.environ.get("AZURE_COSMOSDB_ACCOUNT")
# AZURE_COSMOSDB_CONVERSATIONS_CONTAINER = os.environ.get("AZURE_COSMOSDB_CONVERSATIONS_CONTAINER")
# AZURE_COSMOSDB_ACCOUNT_KEY = os.environ.get("AZURE_COSMOSDB_ACCOUNT_KEY")

# # Initialize a CosmosDB client with AAD auth and containers
# cosmos_conversation_client = None
# if AZURE_COSMOSDB_DATABASE and AZURE_COSMOSDB_ACCOUNT and AZURE_COSMOSDB_CONVERSATIONS_CONTAINER:
#     try :
#         cosmos_endpoint = f'https://{AZURE_COSMOSDB_ACCOUNT}.documents.azure.com:443/'

#         if not AZURE_COSMOSDB_ACCOUNT_KEY:
#             credential = DefaultAzureCredential()
#         else:
#             credential = AZURE_COSMOSDB_ACCOUNT_KEY

#         cosmos_conversation_client = CosmosConversationClient(
#             cosmosdb_endpoint=cosmos_endpoint, 
#             credential=credential, 
#             database_name=AZURE_COSMOSDB_DATABASE,
#             container_name=AZURE_COSMOSDB_CONVERSATIONS_CONTAINER
#         )
#     except Exception as e:
#         logging.exception("Exception in CosmosDB initialization", e)
#         cosmos_conversation_client = None


def is_chat_model():
    if 'gpt-4' in settings.AZURE_OPENAI_MODEL_NAME.lower() or settings.AZURE_OPENAI_MODEL_NAME.lower() in ['gpt-35-turbo-4k', 'gpt-35-turbo-16k']:
        return True
    return False

def should_use_data():
    if settings.AZURE_SEARCH_SERVICE and settings.AZURE_SEARCH_INDEX and settings.AZURE_SEARCH_KEY:
        return True
    return False

def format_as_ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"

def get_filters(request):
    filter_list = [f"doc_type eq '{filt['value']}'" for filt in request["filters"]]
    filter = " or ".join(filter_list)

    return filter

# def fetchUserGroups(userToken, nextLink=None):
#     # Recursively fetch group membership
#     if nextLink:
#         endpoint = nextLink
#     else:
#         endpoint = "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$select=id"
    
#     headers = {
#         'Authorization': "bearer " + userToken
#     }
#     try :
#         r = requests.get(endpoint, headers=headers)
#         if r.status_code != 200:
#             return []
        
#         r = r.json()
#         if "@odata.nextLink" in r:
#             nextLinkData = fetchUserGroups(userToken, r["@odata.nextLink"])
#             r['value'].extend(nextLinkData)
        
#         return r['value']
#     except Exception as e:
#         return []


# def generateFilterString(userToken):
#     # Get list of groups user is a member of
#     userGroups = fetchUserGroups(userToken)

#     # Construct filter string
#     if userGroups:
#         group_ids = ", ".join([obj['id'] for obj in userGroups])
#         return f"{settings.AZURE_SEARCH_PERMITTED_GROUPS_COLUMN}/any(g:search.in(g, '{group_ids}'))"
    
#     return None


def prepare_body_headers_with_data(request):
    request_messages = request["messages"]
    query_type = "simple"
    if settings.AZURE_SEARCH_QUERY_TYPE:
        query_type = settings.AZURE_SEARCH_QUERY_TYPE
    elif settings.AZURE_SEARCH_USE_SEMANTIC_SEARCH.lower() == "true" and settings.AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG:
        query_type = "semantic"

    filter = get_filters(request)

    body = {
        "messages": request_messages,
        "temperature": float(settings.AZURE_OPENAI_TEMPERATURE),
        "max_tokens": int(settings.AZURE_OPENAI_MAX_TOKENS),
        "top_p": float(settings.AZURE_OPENAI_TOP_P),
        "stop": settings.AZURE_OPENAI_STOP_SEQUENCE.split("|") if settings.AZURE_OPENAI_STOP_SEQUENCE else None,
        "stream": settings.AZURE_OPENAI_STREAM,
        "dataSources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": f"https://{settings.AZURE_SEARCH_SERVICE}.search.windows.net",
                    "key": settings.AZURE_SEARCH_KEY,
                    "indexName": settings.AZURE_SEARCH_INDEX,
                    "fieldsMapping": {
                        "contentFields": settings.AZURE_SEARCH_CONTENT_COLUMNS.split("|") if settings.AZURE_SEARCH_CONTENT_COLUMNS else [],
                        "titleField": settings.AZURE_SEARCH_TITLE_COLUMN if settings.AZURE_SEARCH_TITLE_COLUMN else None,
                        "urlField": settings.AZURE_SEARCH_URL_COLUMN if settings.AZURE_SEARCH_URL_COLUMN else None,
                        "filepathField": settings.AZURE_SEARCH_FILENAME_COLUMN if settings.AZURE_SEARCH_FILENAME_COLUMN else None,
                        "vectorFields": settings.AZURE_SEARCH_VECTOR_COLUMNS.split("|") if settings.AZURE_SEARCH_VECTOR_COLUMNS else []
                    },
                    "inScope": True if settings.AZURE_SEARCH_ENABLE_IN_DOMAIN.lower() == "true" else False,
                    "topNDocuments": settings.AZURE_SEARCH_TOP_K,
                    "queryType": query_type,
                    "semanticConfiguration": settings.AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG if settings.AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG else "",
                    "roleInformation": settings.AZURE_OPENAI_SYSTEM_MESSAGE,
                    "embeddingEndpoint": settings.AZURE_OPENAI_EMBEDDING_ENDPOINT,
                    "embeddingKey": settings.AZURE_OPENAI_EMBEDDING_KEY,
                    "filter": filter
                }
            }
        ]
    }

    headers = {
        'Content-Type': 'application/json',
        'api-key': settings.AZURE_OPENAI_KEY,
        "x-ms-useragent": "GitHubSampleWebApp/PublicAPI/2.0.0"
    }

    return body, headers


def stream_with_data(body, headers, endpoint, history_metadata={}):
    s = requests.Session()
    response = {
        "id": "",
        "model": "",
        "created": 0,
        "object": "",
        "choices": [{
            "messages": []
        }],
        'history_metadata': history_metadata
    }
    try:
        with s.post(endpoint, json=body, headers=headers, stream=True) as r:
            for line in r.iter_lines(chunk_size=10):
                if line:
                    lineJson = json.loads(line.lstrip(b'data:').decode('utf-8'))
                    if 'error' in lineJson:
                        yield format_as_ndjson(lineJson)
                    response["id"] = lineJson["id"]
                    response["model"] = lineJson["model"]
                    response["created"] = lineJson["created"]
                    response["object"] = lineJson["object"]

                    role = lineJson["choices"][0]["messages"][0]["delta"].get("role")
                    if role == "tool":
                        response["choices"][0]["messages"].append(lineJson["choices"][0]["messages"][0]["delta"])
                    elif role == "assistant": 
                        print("=="*50)
                        print("TOOL FINISHED")
                        print("=="*50)
                        response["choices"][0]["messages"].append({
                            "role": "assistant",
                            "content": ""
                        })
                    else:
                        deltaText = lineJson["choices"][0]["messages"][0]["delta"]["content"]
                        if deltaText != "[DONE]":
                            response["choices"][0]["messages"][1]["content"] += deltaText

                    print(response)
                    yield format_as_ndjson(response)
    except Exception as e:
        yield format_as_ndjson({"error": str(e)})

def stream_response(body, headers, endpoint, sources, intent, history_metadata={}):
    s = requests.Session()
    response = {
        "id": "",
        "model": "",
        "created": 0,
        "object": "",
        "choices": [{
            "messages": [{"role": "tool", "content": format_as_ndjson({"citations": sources, "intent": intent})}]
        }],
        'history_metadata': history_metadata
    }
    try:
        with s.post(endpoint, json=body, headers=headers, stream=True) as r:
            for line in r.iter_lines(chunk_size=10):
                if line and line.lstrip(b'data: ').decode('utf-8') != "[DONE]":
                    lineJson = json.loads(line.lstrip(b'data:').decode('utf-8'))
                    if 'error' in lineJson:
                        yield format_as_ndjson(lineJson)
                    response["id"] = lineJson["id"]
                    response["model"] = lineJson["model"]
                    response["created"] = lineJson["created"]
                    response["object"] = lineJson["object"]

                    role = lineJson["choices"][0]["delta"].get("role")
                    
                    if role == "assistant": 
                        response["choices"][0]["messages"].append({
                            "role": "assistant",
                            "content": ""
                        })
                    else:
                        try:
                            deltaText = lineJson["choices"][0]["delta"].get("content", None)
                        except:
                            deltaText = None
                        if deltaText != None:
                            response["choices"][0]["messages"][1]["content"] += deltaText
                    
                    yield format_as_ndjson(response)
    except Exception as e:
        yield format_as_ndjson({"error": str(e)})

# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "search_knowledgebase",
#             "description": "Searches the knowledge base for an answer to the technical question. Don't do this if you haven't got the action plan yet.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "search_query": {
#                         "type": "string",
#                         "description": "The search query to use to search the knowledge base"
#                     },
#                     "key_words": {
#                         "type": "string",
#                         "description": "Key words to narrow down the search results"
#                     },
#                 },
#                 "required": ["search_query", "key_words"],
#             },
#         }
#     }
# ]


azcs_search_client = SearchClient(
    endpoint=f"https://{settings.AZURE_SEARCH_SERVICE}.search.windows.net", 
    index_name=settings.AZURE_SEARCH_INDEX, 
    credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY))

aoai_client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_KEY,
    api_version=settings.AZURE_OPENAI_PREVIEW_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
)

def search_knowledgebase(search_query, key_words, filter):
    print("Search Query:", search_query)
    print("Keywords:", key_words)

    vector = VectorizedQuery(vector=get_embedding(search_query), fields="contentVector")

    results = azcs_search_client.search(
        search_text=key_words,
        vector_queries=[vector],
        filter=filter,
        top=10
    )

    source_list = []
    for source in results:
        source_dict = {key: source[key] for key in ["content", "id", "title", "filepath", "url", "metadata"]}
        source_dict["metadata"] = ast.literal_eval(source_dict["metadata"])
        source_dict["chunk_id"] = source_dict["metadata"]["chunk_idx"]
        source_list.append(source_dict)
    
    text_content =""
    for i, result in enumerate(source_list):
        text_content += f"{i+1}. {result['content']}\n\n"
        
    return text_content, source_list


def prepare_body_headers(request_messages):

    body = {
        "messages": request_messages,
        "temperature": float(settings.AZURE_OPENAI_TEMPERATURE),
        "max_tokens": int(settings.AZURE_OPENAI_MAX_TOKENS),
        "top_p": float(settings.AZURE_OPENAI_TOP_P),
        "stop": settings.AZURE_OPENAI_STOP_SEQUENCE.split("|") if settings.AZURE_OPENAI_STOP_SEQUENCE else None,
        "stream": settings.AZURE_OPENAI_STREAM,
    }

    headers = {
        'Content-Type': 'application/json',
        'api-key': settings.AZURE_OPENAI_KEY,
    }

    return body, headers


def conversation_with_data(request_body):
    request_messages = [{"role": "system", "content": settings.AZURE_OPENAI_SYSTEM_MESSAGE}] + [{"role": msg["role"], "content": msg["content"]} for msg in request_body["messages"] if msg["role"] != "tool"]
    filter = get_filters(request_body)
    query = request_messages[-1]["content"]
    
    print(MAKE_QUERY_PROMPT.format(query=query))
    request_messages.append({"role": "user", "content": MAKE_QUERY_PROMPT.format(query=request_messages[-1])})

    query_param_response = aoai_client.chat.completions.create(
        model=settings.AZURE_OPENAI_MODEL,
        messages=request_messages
    )
    query_params = ast.literal_eval(query_param_response.choices[0].message.content)
    sources_str, sources = search_knowledgebase(
        search_query=query_params["search_query"], 
        key_words=query_params["keywords"],
        filter=filter,
        query_type="semantic",
        semantic_configuration_name="default"
    )

    request_messages[-1] = {"role": "user", "content": RAG_PROMPT.format(references=sources_str, query=query)}

    body, headers = prepare_body_headers(request_messages)
    base_url = settings.AZURE_OPENAI_ENDPOINT if settings.AZURE_OPENAI_ENDPOINT else f"https://{settings.AZURE_OPENAI_RESOURCE}.openai.azure.com/"
    endpoint = f"{base_url}openai/deployments/{settings.AZURE_OPENAI_MODEL}/chat/completions?api-version=2023-05-15"

    if not settings.AZURE_OPENAI_STREAM:
        r = requests.post(endpoint, headers=headers, json=body)
        status_code = r.status_code
        full_response = r.json()
        full_response["choices"] = [
            {
                "index": 0,
                "messages": [
                    {
                        "index": 0, 
                        "role": "tool",
                        "content": format_as_ndjson({"citations": sources, "intent": query_params['keywords'].split(', ')}),
                        "end_turn": False
                    },
                    {
                        "index": 1,
                        "role": "assistant",
                        "content": full_response["choices"][0]["message"]["content"],
                        "end_turn": True
                    }
                ]
            }
        ]
        full_response['history_metadata'] = {}
        return Response(format_as_ndjson(full_response), status=status_code)
    
    else:
        return Response(stream_response(body, headers, endpoint, sources, 
                                        intent=query_params["keywords"].split(", "), 
                                        history_metadata={}),
                        mimetype='text/event-stream')   


def conversation_with_data2(request_body):
    if settings.AZURE_OPENAI_STREAM:
        body, headers = prepare_body_headers_with_data(request_body)
    else:
        request_nostream = {"messages": [{"role": msg["role"], "content": msg["content"]} for msg in request_body["messages"]], "filters": request_body["filters"]}
        body, headers = prepare_body_headers_with_data(request_nostream)

    base_url = settings.AZURE_OPENAI_ENDPOINT if settings.AZURE_OPENAI_ENDPOINT else f"https://{settings.AZURE_OPENAI_RESOURCE}.openai.azure.com/"
    endpoint = f"{base_url}openai/deployments/{settings.AZURE_OPENAI_MODEL}/extensions/chat/completions?api-version={settings.AZURE_OPENAI_PREVIEW_API_VERSION}"
    history_metadata = request_body.get("history_metadata", {})

    if not settings.AZURE_OPENAI_STREAM:
        r = requests.post(endpoint, headers=headers, json=body)
        status_code = r.status_code
        r = r.json()
        
        r['history_metadata'] = history_metadata

        return Response(format_as_ndjson(r), status=status_code)
    else:
        return Response(stream_with_data(body, headers, endpoint, history_metadata), mimetype='text/event-stream')


# def stream_without_data(response, history_metadata={}):
#     responseText = ""
#     for line in response:
#         deltaText = line["choices"][0]["delta"].get('content')
#         if deltaText and deltaText != "[DONE]":
#             responseText += deltaText

#         response_obj = {
#             "id": line["id"],
#             "model": line["model"],
#             "created": line["created"],
#             "object": line["object"],
#             "choices": [{
#                 "messages": [{
#                     "role": "assistant",
#                     "content": responseText
#                 }]
#             }],
#             "history_metadata": history_metadata
#         }
#         yield format_as_ndjson(response_obj)


# def conversation_without_data(request_body):
#     openai.api_type = "azure"
#     openai.api_base = settings.AZURE_OPENAI_ENDPOINT if settings.AZURE_OPENAI_ENDPOINT else f"https://{settings.AZURE_OPENAI_RESOURCE}.openai.azure.com/"
#     openai.api_version = "2023-03-15-preview"
#     openai.api_key = settings.AZURE_OPENAI_KEY

#     request_messages = request_body["messages"]
#     messages = [
#         {
#             "role": "system",
#             "content": settings.AZURE_OPENAI_SYSTEM_MESSAGE
#         }
#     ]

#     for message in request_messages:
#         messages.append({
#             "role": message["role"] ,
#             "content": message["content"]
#         })

#     response = openai.ChatCompletion.create(
#         engine=settings.AZURE_OPENAI_MODEL,
#         messages = messages,
#         temperature=float(settings.AZURE_OPENAI_TEMPERATURE),
#         max_tokens=int(settings.AZURE_OPENAI_MAX_TOKENS),
#         top_p=float(settings.AZURE_OPENAI_TOP_P),
#         stop=settings.AZURE_OPENAI_STOP_SEQUENCE.split("|") if settings.AZURE_OPENAI_STOP_SEQUENCE else None,
#         stream=SHOULD_STREAM
#     )

#     history_metadata = request_body.get("history_metadata", {})

#     if not SHOULD_STREAM:
#         response_obj = {
#             "id": response,
#             "model": response.model,
#             "created": response.created,
#             "object": response.object,
#             "choices": [{
#                 "messages": [{
#                     "role": "assistant",
#                     "content": response.choices[0].message.content
#                 }]
#             }],
#             "history_metadata": history_metadata
#         }

#         return jsonify(response_obj), 200
#     else:
#         return Response(stream_without_data(response, history_metadata), mimetype='text/event-stream')


@app.route("/conversation", methods=["GET", "POST"])
def conversation():
    request_body = request.json
    #print(conversation_internal(request_body).get_data())
    return conversation_internal(request_body)

def conversation_internal(request_body):
    try:
        use_data = should_use_data()
        if use_data:
            return conversation_with_data(request_body)
        else:
            # return conversation_without_data(request_body)
            raise ValueError("Azure AI Search details not found. Please provide the correct AZURE_SEARCH_SERVICE and AZURE_SEARCH_INDEX and AZURE_SEARCH_KEY values in your .env file.")
    except Exception as e:
        logging.exception("Exception in /conversation")
        return jsonify({"error": str(e)}), 500

## Andre: Commenting out functions that require CosmosDB, as we are currently not using it

# ## Conversation History API ## 
# @app.route("/history/generate", methods=["POST"])
# def add_conversation():
#     authenticated_user = get_authenticated_user_details(request_headers=request.headers)
#     user_id = authenticated_user['user_principal_id']

#     ## check request for conversation_id
#     conversation_id = request.json.get("conversation_id", None)

#     try:
#         # make sure cosmos is configured
#         if not cosmos_conversation_client:
#             raise Exception("CosmosDB is not configured")

#         # check for the conversation_id, if the conversation is not set, we will create a new one
#         history_metadata = {}
#         if not conversation_id:
#             title = generate_title(request.json["messages"])
#             conversation_dict = cosmos_conversation_client.create_conversation(user_id=user_id, title=title)
#             conversation_id = conversation_dict['id']
#             history_metadata['title'] = title
#             history_metadata['date'] = conversation_dict['createdAt']
            
#         ## Format the incoming message object in the "chat/completions" messages format
#         ## then write it to the conversation history in cosmos
#         messages = request.json["messages"]
#         if len(messages) > 0 and messages[-1]['role'] == "user":
#             cosmos_conversation_client.create_message(
#                 conversation_id=conversation_id,
#                 user_id=user_id,
#                 input_message=messages[-1]
#             )
#         else:
#             raise Exception("No user message found")
        
#         # Submit request to Chat Completions for response
#         request_body = request.json
#         history_metadata['conversation_id'] = conversation_id
#         request_body['history_metadata'] = history_metadata
#         return conversation_internal(request_body)
       
#     except Exception as e:
#         logging.exception("Exception in /history/generate")
#         return jsonify({"error": str(e)}), 500


# @app.route("/history/update", methods=["POST"])
# def update_conversation():
#     authenticated_user = get_authenticated_user_details(request_headers=request.headers)
#     user_id = authenticated_user['user_principal_id']

#     ## check request for conversation_id
#     conversation_id = request.json.get("conversation_id", None)

#     try:
#         # make sure cosmos is configured
#         if not cosmos_conversation_client:
#             raise Exception("CosmosDB is not configured")

#         # check for the conversation_id, if the conversation is not set, we will create a new one
#         if not conversation_id:
#             raise Exception("No conversation_id found")
            
#         ## Format the incoming message object in the "chat/completions" messages format
#         ## then write it to the conversation history in cosmos
#         messages = request.json["messages"]
#         if len(messages) > 0 and messages[-1]['role'] == "assistant":
#             if len(messages) > 1 and messages[-2]['role'] == "tool":
#                 # write the tool message first
#                 cosmos_conversation_client.create_message(
#                     conversation_id=conversation_id,
#                     user_id=user_id,
#                     input_message=messages[-2]
#                 )
#             # write the assistant message
#             cosmos_conversation_client.create_message(
#                 conversation_id=conversation_id,
#                 user_id=user_id,
#                 input_message=messages[-1]
#             )
#         else:
#             raise Exception("No bot messages found")
        
#         # Submit request to Chat Completions for response
#         response = {'success': True}
#         return jsonify(response), 200
       
#     except Exception as e:
#         logging.exception("Exception in /history/update")
#         return jsonify({"error": str(e)}), 500

# @app.route("/history/delete", methods=["DELETE"])
# def delete_conversation():
#     ## get the user id from the request headers
#     authenticated_user = get_authenticated_user_details(request_headers=request.headers)
#     user_id = authenticated_user['user_principal_id']
    
#     ## check request for conversation_id
#     conversation_id = request.json.get("conversation_id", None)
#     try: 
#         if not conversation_id:
#             return jsonify({"error": "conversation_id is required"}), 400
        
#         ## delete the conversation messages from cosmos first
#         deleted_messages = cosmos_conversation_client.delete_messages(conversation_id, user_id)

#         ## Now delete the conversation 
#         deleted_conversation = cosmos_conversation_client.delete_conversation(user_id, conversation_id)

#         return jsonify({"message": "Successfully deleted conversation and messages", "conversation_id": conversation_id}), 200
#     except Exception as e:
#         logging.exception("Exception in /history/delete")
#         return jsonify({"error": str(e)}), 500

# @app.route("/history/list", methods=["GET"])
# def list_conversations():
#     authenticated_user = get_authenticated_user_details(request_headers=request.headers)
#     user_id = authenticated_user['user_principal_id']

#     ## get the conversations from cosmos
#     conversations = cosmos_conversation_client.get_conversations(user_id)
#     if not isinstance(conversations, list):
#         return jsonify({"error": f"No conversations for {user_id} were found"}), 404

#     ## return the conversation ids

#     return jsonify(conversations), 200

# @app.route("/history/read", methods=["POST"])
# def get_conversation():
#     authenticated_user = get_authenticated_user_details(request_headers=request.headers)
#     user_id = authenticated_user['user_principal_id']

#     ## check request for conversation_id
#     conversation_id = request.json.get("conversation_id", None)
    
#     if not conversation_id:
#         return jsonify({"error": "conversation_id is required"}), 400

#     ## get the conversation object and the related messages from cosmos
#     conversation = cosmos_conversation_client.get_conversation(user_id, conversation_id)
#     ## return the conversation id and the messages in the bot frontend format
#     if not conversation:
#         return jsonify({"error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."}), 404
    
#     # get the messages for the conversation from cosmos
#     conversation_messages = cosmos_conversation_client.get_messages(user_id, conversation_id)

#     ## format the messages in the bot frontend format
#     messages = [{'id': msg['id'], 'role': msg['role'], 'content': msg['content'], 'createdAt': msg['createdAt']} for msg in conversation_messages]

#     return jsonify({"conversation_id": conversation_id, "messages": messages}), 200

# @app.route("/history/rename", methods=["POST"])
# def rename_conversation():
#     authenticated_user = get_authenticated_user_details(request_headers=request.headers)
#     user_id = authenticated_user['user_principal_id']

#     ## check request for conversation_id
#     conversation_id = request.json.get("conversation_id", None)
    
#     if not conversation_id:
#         return jsonify({"error": "conversation_id is required"}), 400
    
#     ## get the conversation from cosmos
#     conversation = cosmos_conversation_client.get_conversation(user_id, conversation_id)
#     if not conversation:
#         return jsonify({"error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."}), 404

#     ## update the title
#     title = request.json.get("title", None)
#     if not title:
#         return jsonify({"error": "title is required"}), 400
#     conversation['title'] = title
#     updated_conversation = cosmos_conversation_client.upsert_conversation(conversation)

#     return jsonify(updated_conversation), 200

# @app.route("/history/delete_all", methods=["DELETE"])
# def delete_all_conversations():
#     ## get the user id from the request headers
#     authenticated_user = get_authenticated_user_details(request_headers=request.headers)
#     user_id = authenticated_user['user_principal_id']

#     # get conversations for user
#     try:
#         conversations = cosmos_conversation_client.get_conversations(user_id)
#         if not conversations:
#             return jsonify({"error": f"No conversations for {user_id} were found"}), 404
        
#         # delete each conversation
#         for conversation in conversations:
#             ## delete the conversation messages from cosmos first
#             deleted_messages = cosmos_conversation_client.delete_messages(conversation['id'], user_id)

#             ## Now delete the conversation 
#             deleted_conversation = cosmos_conversation_client.delete_conversation(user_id, conversation['id'])

#         return jsonify({"message": f"Successfully deleted conversation and messages for user {user_id}"}), 200
    
#     except Exception as e:
#         logging.exception("Exception in /history/delete_all")
#         return jsonify({"error": str(e)}), 500
    

# @app.route("/history/clear", methods=["POST"])
# def clear_messages():
#     ## get the user id from the request headers
#     authenticated_user = get_authenticated_user_details(request_headers=request.headers)
#     user_id = authenticated_user['user_principal_id']
    
#     ## check request for conversation_id
#     conversation_id = request.json.get("conversation_id", None)
#     try: 
#         if not conversation_id:
#             return jsonify({"error": "conversation_id is required"}), 400
        
#         ## delete the conversation messages from cosmos
#         deleted_messages = cosmos_conversation_client.delete_messages(conversation_id, user_id)

#         return jsonify({"message": "Successfully deleted messages in conversation", "conversation_id": conversation_id}), 200
#     except Exception as e:
#         logging.exception("Exception in /history/clear_messages")
#         return jsonify({"error": str(e)}), 500

# @app.route("/history/ensure", methods=["GET"])
# def ensure_cosmos():
#     if not AZURE_COSMOSDB_ACCOUNT:
#         return jsonify({"error": "CosmosDB is not configured"}), 404
    
#     if not cosmos_conversation_client or not cosmos_conversation_client.ensure():
#         return jsonify({"error": "CosmosDB is not working"}), 500

#     return jsonify({"message": "CosmosDB is configured and working"}), 200


# def generate_title(conversation_messages):
#     ## make sure the messages are sorted by _ts descending
#     title_prompt = 'Summarize the conversation so far into a 4-word or less title. Do not use any quotation marks or punctuation. Respond with a json object in the format {{"title": string}}. Do not include any other commentary or description.'

#     messages = [{'role': msg['role'], 'content': msg['content']} for msg in conversation_messages]
#     messages.append({'role': 'user', 'content': title_prompt})

#     try:
#         ## Submit prompt to Chat Completions for response
#         base_url = settings.AZURE_OPENAI_ENDPOINT if settings.AZURE_OPENAI_ENDPOINT else f"https://{settings.AZURE_OPENAI_RESOURCE}.openai.azure.com/"
#         openai.api_type = "azure"
#         openai.api_base = base_url
#         openai.api_version = "2023-03-15-preview"
#         openai.api_key = settings.AZURE_OPENAI_KEY
#         completion = openai.ChatCompletion.create(    
#             engine=settings.AZURE_OPENAI_MODEL,
#             messages=messages,
#             temperature=1,
#             max_tokens=64 
#         )
#         title = json.loads(completion['choices'][0]['message']['content'])['title']
#         return title
#     except Exception as e:
#         return messages[-2]['content']

if __name__ == "__main__":
    app.run()
