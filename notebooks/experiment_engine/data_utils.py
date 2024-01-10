"""Data utilities for index preparation."""
from pathlib import Path
import ast
import uuid
from asyncio import sleep
import html
import json
import os
import re
import requests
import openai
from openai import AzureOpenAI
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import List, Dict, Optional, Generator, Tuple

import markdown
import pandas as pd
import tiktoken
from tqdm import tqdm
from azure.identity import DefaultAzureCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from bs4 import BeautifulSoup
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter, PythonCodeTextSplitter
from llama_index import download_loader
from llama_index.node_parser import LangchainNodeParser
from llama_index import Document as LlamaDocument
from llama_index.extractors import TitleExtractor, SummaryExtractor, QuestionsAnsweredExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import AzureOpenAI as LlamaAOAI
import nest_asyncio
nest_asyncio.apply()

from config import settings

FILE_FORMAT_DICT = {
        "md": "markdown",
        "txt": "text",
        "html": "html",
        "shtml": "html",
        "htm": "html",
        "py": "python",
        "pdf": "pdf",
        "json": "json",
        "docx": "docx",
        "csv": "csv"
    }

RETRY_COUNT = 5

SENTENCE_ENDINGS = [".", "!", "?"]
WORDS_BREAKS = list(reversed([",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]))

PDF_HEADERS = {
    "title": "h1",
    "sectionHeading": "h2"
}


@dataclass
class Document(object):
    """A data class for storing documents

    Attributes:
        content (str): The content of the document.
        id (Optional[str]): The id of the document.
        title (Optional[str]): The title of the document.
        filepath (Optional[str]): The filepath of the document.
        url (Optional[str]): The url of the document.
        metadata (Optional[Dict]): The metadata of the document.    
    """

    content: str
    id: Optional[str] = None
    doc_type: Optional[str] = None
    title: Optional[str] = None
    filepath: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict] = None
    contentVector: Optional[List[float]] = None

def cleanup_content(content: str) -> str:
    """Cleans up the given content using regexes
    Args:
        content (str): The content to clean up.
    Returns:
        str: The cleaned up content.
    """
    output = re.sub(r"\n{2,}", "\n", content)
    output = re.sub(r"[^\S\n]{2,}", " ", output)
    output = re.sub(r"-{2,}", "--", output)

    return output.strip()


class FileConverter:
    def __init__(self, filetype):
        converter_map = {
            "pdf": "PDFReader",
            "docx": "DocxReader"
        }
        Reader = download_loader(converter_map[filetype])
        self.loader = Reader()
    
    def extract_text(self, file_path):
        documents = self.loader.load_data(file=Path(file_path))
        full_text = "".join([doc.text for doc in documents])

        return full_text
        

class BaseParser(ABC):
    """A parser parses content to produce a document."""

    @abstractmethod
    def parse(self, content: str, file_name: Optional[str] = None) -> Document:
        """Parses the given content.
        Args:
            content (str): The content to parse.
            file_name (str): The file name associated with the content.
        Returns:
            Document: The parsed document.
        """
        pass

    def parse_file(self, file_path: str) -> Document:
        """Parses the given file.
        Args:
            file_path (str): The file to parse.
        Returns:
            Document: The parsed document.
        """
        with open(file_path, "r") as f:
            return self.parse(f.read(), os.path.basename(file_path))

    def parse_directory(self, directory_path: str) -> List[Document]:
        """Parses the given directory.
        Args:
            directory_path (str): The directory to parse.
        Returns:
            List[Document]: List of parsed documents.
        """
        documents = []
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                documents.append(self.parse_file(file_path))
        return documents


class HTMLParser(BaseParser):
    """Parses HTML content."""
    TITLE_MAX_TOKENS = 128
    NEWLINE_TEMPL = "<NEWLINE_TEXT>"

    def __init__(self) -> None:
        super().__init__()
        self.token_estimator = TokenEstimator()

    def parse(self, content: str, file_name: Optional[str] = None) -> Document:
        """Parses the given content.
        Args:
            content (str): The content to parse.
            file_name (str): The file name associated with the content.
        Returns:
            Document: The parsed document.
        """
        soup = BeautifulSoup(content, 'html.parser')

        # Extract the title
        title = ''
        if soup.title and soup.title.string:
            title = soup.title.string
        else:
            # Try to find the first <h1> tag
            h1_tag = soup.find('h1')
            if h1_tag:
                title = h1_tag.get_text(strip=True)
            else:
                h2_tag = soup.find('h2')
                if h2_tag:
                    title = h2_tag.get_text(strip=True)
        if title is None or title == '':
            # if title is still not found, guess using the next string
            try:
                title = next(soup.stripped_strings)
                title = self.token_estimator.construct_tokens_with_size(title, self.TITLE_MAX_TOKENS)

            except StopIteration:
                title = file_name

                # Helper function to process text nodes

        # Parse the content as it is without any formatting changes
        result = content
        if title is None:
            title = '' # ensure no 'None' type title

        return Document(content=cleanup_content(result), title=str(title))

class TextParser(BaseParser):
    """Parses text content."""

    def __init__(self) -> None:
        super().__init__()

    def _get_first_alphanum_line(self, content: str) -> Optional[str]:
        title = None
        for line in content.splitlines():
            if any([c.isalnum() for c in line]):
                title = line.strip()
                break
        return title

    def _get_first_line_with_property(
        self, content: str, property: str = "title: "
    ) -> Optional[str]:
        title = None
        for line in content.splitlines():
            if line.startswith(property):
                title = line[len(property) :].strip()
                break
        return title

    def parse(self, content: str, file_name: Optional[str] = None) -> Document:
        """Parses the given content.
        Args:
            content (str): The content to parse.
            file_name (str): The file name associated with the content.
        Returns:
            Document: The parsed document.
        """
        title = self._get_first_line_with_property(
            content
        ) or self._get_first_alphanum_line(content)

        return Document(content=cleanup_content(content), title=title or file_name)

class JSONParser(BaseParser):
    def __init__(self) -> None:
        super().__init__()
    
    def parse(self, content: str, file_name: Optional[str] = None) -> Document:
        # TODO: Get title from top level field from content
        # For now, using file_name as placeholder for true title
        content_dump = json.dumps(content)
        return Document(content=content_dump, title=content["title"])

class ParserFactory:
    def __init__(self):
        self._parsers = {
            "html": HTMLParser(),
            "text": TextParser(),
            "json": JSONParser()
        }

    @property
    def supported_formats(self) -> List[str]:
        "Returns a list of supported formats"
        return list(self._parsers.keys())

    def __call__(self, file_format: str) -> BaseParser:
        parser = self._parsers.get(file_format, None)
        if parser is None:
            raise UnsupportedFormatError(f"{file_format} is not supported")

        return parser

class LlamaIndexSplitter:
    def __init__(self, num_tokens, token_overlap, extractor_llm=None):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            separators=SENTENCE_ENDINGS + WORDS_BREAKS,
            chunk_size=num_tokens, chunk_overlap=token_overlap)
        parser = LangchainNodeParser(splitter)
        self.parser = parser
        self.extractor_llm = extractor_llm
        self.split_mods_map = {
            "summary_extraction": SummaryExtractor(summaries=["prev", "self", "next"], llm=extractor_llm),
            "qa_extraction": QuestionsAnsweredExtractor(questions=3, llm=extractor_llm)
        }
    
    def get_nodes_from_doc(self, llama_doc, split_mods):
        if split_mods and self.extractor_llm == None:
            raise ValueError(f"extractor_llm argument is required to apply modifications during splitting")

        pipeline = IngestionPipeline(
            transformations=[self.parser] + [self.split_mods_map[mod] for mod in split_mods]
        )

        nodes = pipeline.run(
            documents = [llama_doc],
            in_place=True,
            show_progress=True
        )

        return nodes

class TokenEstimator(object):
    GPT2_TOKENIZER = tiktoken.get_encoding("gpt2")

    def estimate_tokens(self, text: str) -> int:
        return len(self.GPT2_TOKENIZER.encode(text))

    def construct_tokens_with_size(self, tokens: str, numofTokens: int) -> str:
        newTokens = self.GPT2_TOKENIZER.decode(
            self.GPT2_TOKENIZER.encode(tokens)[:numofTokens]
        )
        return newTokens

parser_factory = ParserFactory()
TOKEN_ESTIMATOR = TokenEstimator()

class UnsupportedFormatError(Exception):
    """Exception raised when a format is not supported by a parser."""

    pass

@dataclass
class ChunkingResult:
    """Data model for chunking result

    Attributes:
        chunks (List[Document]): List of chunks.
        total_files (int): Total number of files.
        num_unsupported_format_files (int): Number of files with unsupported format.
        num_files_with_errors (int): Number of files with errors.
        skipped_chunks (int): Number of chunks skipped.
    """
    chunks: List[Document]
    total_files: int
    num_unsupported_format_files: int = 0
    num_files_with_errors: int = 0
    # some chunks might be skipped to small number of tokens
    skipped_chunks: int = 0

def get_files_recursively(directory_path: str) -> List[str]:
    """Gets all files in the given directory recursively.
    Args:
        directory_path (str): The directory to get files from.
    Returns:
        List[str]: List of file paths.
    """
    file_paths = []
    for dirpath, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(dirpath, file_name)
            file_paths.append(file_path)
    return file_paths

def convert_escaped_to_posix(escaped_path):
    windows_path = escaped_path.replace("\\\\", "\\")
    posix_path = windows_path.replace("\\", "/")
    return posix_path

def _get_file_format(file_name: str) -> Optional[str]:
    """Gets the file format from the file name.
    Returns None if the file format is not supported.
    Args:
        file_name (str): The file name.
    Returns:
        str: The file format.
    """

    # in case the caller gives us a file path
    file_name = os.path.basename(file_name)
    file_extension = file_name.split(".")[-1]
    if file_extension not in FILE_FORMAT_DICT.keys():
        return None
    return FILE_FORMAT_DICT.get(file_extension, None)


def get_embedding(text):
    try:
        client = AzureOpenAI(
            api_key = settings.AZURE_OPENAI_KEY,
            api_version = settings.AZURE_OPENAI_PREVIEW_API_VERSION,
            azure_endpoint = settings.AZURE_OPENAI_EMBEDDING_ENDPOINT
        )

        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        
        embeddings = ast.literal_eval(response.json())["data"][0]["embedding"]
        
        return embeddings

    except Exception as e:
        raise Exception(f"Error getting embeddings with endpoint={settings.AZURE_OPENAI_EMBEDDING_ENDPOINT} with error={e}")


def chunk_content_helper(
        content: str, 
        file_format: str, 
        file_name: Optional[str],
        extractor_llm = None
) -> Generator[Tuple[str, int, Document], None, None]:

    parser = parser_factory(file_format)
    doc = parser.parse(content, file_name=file_name)
    doc_content_size = TOKEN_ESTIMATOR.estimate_tokens(doc.content)

    if file_format == "json" or doc_content_size < settings.PREP_CONFIG["chunk_size"]:
        yield doc.content, doc_content_size, doc
    else:
        llama_splitter = LlamaIndexSplitter(
            num_tokens=settings.PREP_CONFIG["chunk_size"], 
            token_overlap=settings.PREP_CONFIG["token_overlap"], 
            extractor_llm=extractor_llm
        )
        llama_doc = LlamaDocument(text=doc.content, metadata={"file_name": file_name, "title": doc.title})
        nodes = llama_splitter.get_nodes_from_doc(llama_doc, settings.PREP_CONFIG.get("split_mods", []))
        
        for node in nodes:
            chunk_size = TOKEN_ESTIMATOR.estimate_tokens(node.text)
            yield node, chunk_size, llama_doc

def chunk_content(
    content: str,
    file_name: Optional[str] = None,
    file_format: Optional[str] = None,
    doc_type: Optional[str] = None,
    url: Optional[str] = None,
    ignore_errors: bool = False,
    min_chunk_size: int = 10,
    add_embeddings = False
) -> ChunkingResult:
    """Chunks the given content. If ignore_errors is true, returns None
        in case of an error
    Args:
        content (str): The content to chunk.
        file_name (str): The file name. used for title, file format detection.
        url (str): The url. used for title.
        ignore_errors (bool): If true, ignores errors and returns None.
        num_tokens (int): The number of tokens in each chunk.
        min_chunk_size (int): The minimum chunk size below which chunks will be filtered.
        token_overlap (int): The number of tokens to overlap between chunks.
    Returns:
        List[Document]: List of chunked documents.
    """
    ignore_errors = False
    try:
        if file_format not in ["json", "csv"]: 
            file_format = "text"
        
        extractor_llm = LlamaAOAI(
                model=settings.AZURE_OPENAI_MODEL_NAME,
                deployment_name=settings.AZURE_OPENAI_MODEL,
                api_key=settings.AZURE_OPENAI_KEY,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_version=settings.AZURE_OPENAI_PREVIEW_API_VERSION,
                system_prompt=settings.AZURE_OPENAI_SYSTEM_MESSAGE
            )

        nodes = chunk_content_helper(
            content=content,
            file_name=file_name,
            file_format=file_format,
            extractor_llm=extractor_llm
        )
        
        chunks = []
        skipped_chunks = 0
        for node, chunk_size, llama_doc in nodes:
            if chunk_size >= min_chunk_size:
                chunk = Document(
                            id=node.id_,
                            doc_type=doc_type,
                            content=node.text,
                            title=node.metadata["title"],
                            url=url,
                            metadata=node.metadata
                        )
                if add_embeddings:
                    for _ in range(RETRY_COUNT):
                        try:
                            if settings.PREP_CONFIG.get("split_mods", []):
                                content = json.dumps(node.metadata)
                            else:
                                content = f"{node.text}\n\nMETADATA: {node.metadata}"
                            contentVector = get_embedding(content)
                            break
                        except:
                            contentVector = None
                            sleep(30)
                    if contentVector is None:
                        raise Exception(f"Error getting embedding for chunk={node}")
                    chunk.contentVector = contentVector
                chunks.append(chunk)

            else:
                skipped_chunks += 1

    except UnsupportedFormatError as e:
        if ignore_errors:
            return ChunkingResult(
                chunks=[], total_files=1, num_unsupported_format_files=1
            )
        else:
            raise e
    except Exception as e:
        if ignore_errors:
            return ChunkingResult(chunks=[], total_files=1, num_files_with_errors=1)
        else:
            raise e
    
    return ChunkingResult(
        chunks=chunks,
        total_files=1,
        skipped_chunks=skipped_chunks,
    )

def chunk_qfr(
        file_path: str = "",
        add_embeddings: bool = False
):
    qfrs_df = pd.read_csv(file_path)
    chunks = []
    for idx, row in qfrs_df.iterrows():
        text = f"QUESTION: {row['Question']}\nRESPONSE: {row['Response']}"
        chunk = Document(
                    id=f"qfr-{uuid.uuid4()}",
                    doc_type="qfr",
                    content=text,
                    title=row["Question"],
                    url=None,
                    metadata={}
                )
        if add_embeddings:
            for _ in range(RETRY_COUNT):
                try:
                    contentVector = get_embedding(text)
                    break
                except:
                    contentVector = None
                    sleep(30)
            if contentVector is None:
                raise Exception(f"Error getting embedding for chunk={text}")
            chunk.contentVector = contentVector
        chunks.append(chunk)
    
    return ChunkingResult(
        chunks=chunks,
        total_files=1,
        skipped_chunks=0
    )

def chunk_file(
    file_path: str,
    ignore_errors: bool = True,
    min_chunk_size=10,
    url = None,
    add_embeddings=False
) -> ChunkingResult:
    """Chunks the given file.
    Args:
        file_path (str): The file to chunk.
    Returns:
        List[Document]: List of chunked documents.
    """
    file_name = os.path.basename(file_path)
    doc_type = os.path.basename(os.path.dirname(file_path))
    file_format = _get_file_format(file_name)
    if not file_format:
        if ignore_errors:
            return ChunkingResult(
                chunks=[], total_files=1, num_unsupported_format_files=1
            )
        else:
            raise UnsupportedFormatError(f"{file_name} is not supported")

    if file_format == "json":
        with open(file_path) as f:
            content = json.load(f)
    
    elif doc_type == "qfr":
        return chunk_qfr(file_path, add_embeddings)
    
    else:
        file_converter = FileConverter(file_format)
        content = file_converter.extract_text(file_path)
        
    return chunk_content(
        content=content,
        file_name=file_name,
        file_format=file_format,
        doc_type=doc_type,
        url=url,
        ignore_errors=ignore_errors,
        min_chunk_size=min_chunk_size,
        add_embeddings=add_embeddings
    )


def process_file(
        file_path: str, # !IMP: Please keep this as the first argument
        directory_path: str,
        ignore_errors: bool = True,
        min_chunk_size: int = 10,
        url_prefix = None,
        add_embeddings = False
    ):

    is_error = False
    try:
        url_path = None
        rel_file_path = os.path.relpath(file_path, directory_path)
        if url_prefix:
            url_path = url_prefix + rel_file_path
            url_path = convert_escaped_to_posix(url_path)

        result = chunk_file(
            file_path,
            ignore_errors=ignore_errors,
            min_chunk_size=min_chunk_size,
            url=url_path,
            add_embeddings=add_embeddings
        )
        for chunk_idx, chunk_doc in enumerate(result.chunks):
            chunk_doc.filepath = rel_file_path
            chunk_doc.metadata["chunk_idx"] = str(chunk_idx)
            chunk_doc.metadata = json.dumps(chunk_doc.metadata)
    except Exception as e:
        if not ignore_errors:
            raise
        print(f"File ({file_path}) failed with ", e)
        is_error = True
        result =None
    return result, is_error

def chunk_directory(
        directory_path: str,
        ignore_errors: bool = True,
        min_chunk_size: int = 10,
        url_prefix = None,
        njobs=4,
        add_embeddings = False,
):
    """
    Chunks the given directory recursively
    Args:
        directory_path (str): The directory to chunk.
        ignore_errors (bool): If true, ignores errors and returns None.
        min_chunk_size (int): The minimum chunk size.
        url_prefix (str): The url prefix to use for the files. If None, the url will be None. If not None, the url will be url_prefix + relpath. 
                            For example, if the directory path is /home/user/data and the url_prefix is https://example.com/data, 
                            then the url for the file /home/user/data/file1.txt will be https://example.com/data/file1.txt
        add_embeddings (bool): If true, adds a vector embedding to each chunk using the embedding model endpoint and key.

    Returns:
        List[Document]: List of chunked documents.
    """
    chunks = []
    total_files = 0
    num_unsupported_format_files = 0
    num_files_with_errors = 0
    skipped_chunks = 0

    all_files_directory = get_files_recursively(directory_path)
    files_to_process = [file_path for file_path in all_files_directory if os.path.isfile(file_path)]
    print(f"Total files to process={len(files_to_process)} out of total directory size={len(all_files_directory)}")


    if njobs==1:
        print("Single process to chunk and parse the files. --njobs > 1 can help performance.")
        for file_path in tqdm(files_to_process):
            total_files += 1
            result, is_error = process_file(
                file_path=file_path,
                directory_path=directory_path, 
                ignore_errors=ignore_errors,      
                min_chunk_size=min_chunk_size, 
                url_prefix=url_prefix,
                add_embeddings=add_embeddings
            )
            if is_error:
                num_files_with_errors += 1
                continue
            chunks.extend(result.chunks)
            num_unsupported_format_files += result.num_unsupported_format_files
            num_files_with_errors += result.num_files_with_errors
            skipped_chunks += result.skipped_chunks
    elif njobs > 1:
        print(f"Multiprocessing with njobs={njobs}")
        process_file_partial = partial(
            process_file, 
            directory_path=directory_path, 
            ignore_errors=ignore_errors,
            min_chunk_size=min_chunk_size, 
            url_prefix=url_prefix,
            add_embeddings=add_embeddings,)
        with ProcessPoolExecutor(max_workers=njobs) as executor:
            futures = list(tqdm(executor.map(process_file_partial, files_to_process), total=len(files_to_process)))
            for result, is_error in futures:
                total_files += 1
                if is_error:
                    num_files_with_errors += 1
                    continue
                chunks.extend(result.chunks)
                num_unsupported_format_files += result.num_unsupported_format_files
                num_files_with_errors += result.num_files_with_errors
                skipped_chunks += result.skipped_chunks

    return ChunkingResult(
            chunks=chunks,
            total_files=total_files,
            num_unsupported_format_files=num_unsupported_format_files,
            num_files_with_errors=num_files_with_errors,
            skipped_chunks=skipped_chunks,
        )