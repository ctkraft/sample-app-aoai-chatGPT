"""Data utilities for index preparation."""
from pathlib import Path
import ast
import uuid
import sys
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
from llama_index.schema import TextNode
from llama_index.node_parser import LangchainNodeParser
from llama_index.text_splitter import SentenceSplitter
from llama_index import Document as LlamaDocument
from llama_index.extractors import TitleExtractor, SummaryExtractor, QuestionsAnsweredExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import AzureOpenAI as LlamaAOAI
import nest_asyncio
nest_asyncio.apply()

sys.path.append(os.getcwd() + '/..')
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

    def to_dict(self):
        """
        Returns the Dataset object in JSON compatible encoding
        """
        return {
            "content": self.content,
            "id": self.id,
            "doc_type": self.doc_type,
            "title": self.title,
            "filepath": self.filepath,
            "url": self.url,
            "metadata": self.metadata,
            "contentVector": self.contentVector
        }

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

def set_page_headers(pages):
    for i, page in enumerate(pages):
        header_text = ""
        lines = page.text.split("\n")

        for line in lines:
            line = ' '.join(line.split())
            if len(f"{header_text}\n{line}") < 200:
                header_text = f"{header_text}\n{line}"
            else:
                break

        pages[i].metadata["header"] = header_text
        pages[i].text = cleanup_content(pages[i].text)
    
    return pages

def find_overlap(string1, string2):  
    min_length = min(len(string1), len(string2))  
    if string2 in string1 or string2 == string1:
        return "inside"
    elif string1 in string2:
        return "outside"

    for size in range(min_length, 200, -1):  
        if string1[-size:] == string2[:size]:  
            return "end"
        elif string1[:size] == string2[-size:]:
            return "beginning"  
    return "none"
  
def map_node_to_page(node, pages, doc_type):
    i = 0
    section_header = "<b>Sections:</b>"
    node_pages = []

    while i < len(pages):
        if find_overlap(pages[i].text, node.text) != "none":
            section_header += f"\n{pages[i].metadata['header']}\n"
            node_pages.append(pages[i].metadata["page_label"])
            i += 1
        
        else:
            if node_pages:
                break
            else:
                i += 1

    node.metadata["pages"] = ", ".join(node_pages)
    node.text = f"<b>Page(s)</b>: {node.metadata['pages']}\n\nContent: {node.text}"
    
    if doc_type == "congressional_budget_justification":
        node.text = f"{section_header}\n" + node.text
    
    return node, i-1

def doc_intel_map_node_to_page(node, pages, full_text):
    node_text = node.text
    start = full_text.index(node_text)
    end = start + len(node_text)
    
    page_offsets = [page["offset"] for page in pages]
    page_lens = [len(page["page_text"]) for page in pages]
    start_found, end_found = False, False
    for i, offset in enumerate(page_offsets):
        if offset >= start and start_found == False:
            if offset == start:
                page_start = i+1 # Set page_start to the current page, which is i+1
            else:
                page_start = i # Set page_start to the previous page, which is i
            start_found = True
        if offset + page_lens[i] >= end and end_found == False:
            page_end = i+1 # Set end_page to current page, which is i+1
            end_found = True
    
    page_start = page_start if start_found == True else len(pages)
    page_end = page_end if end_found == True else len(pages)
    
    node.metadata["pages"] = f"{page_start} - {page_end}" if page_start != page_end else str(page_start)
    relevant_pages = pages[page_start-1:page_end]
    
    relevant_sections = []
    section_header = "<b>Sections:</b>"
    for page in relevant_pages:
        if page["page_text"] != "":
            page_header = f"<p>{' - '.join([para.content for para in page['paragraphs'][:2]])}</p>"
            relevant_sections.append(page_header)
    section_header += "".join(list(set(relevant_sections)))

    node_text_lines = [f"<p>{line}</p>" for line in node_text.split("\n")]
    node_text = "".join(node_text_lines)
    
    node.text = f"<p>{section_header}</p><p><b>Page(s)</b>: {node.metadata['pages']}</p><p><b>Content:</b></p><p>{node_text}</p>"

    return node

def doc_intel_tables_to_nodes(pages, llama_doc):
    table_nodes = []
    for i, page in enumerate(pages):
        if len(page["tables"]) > 0:
            for table in page["tables"]:
                table_text = table["header"] + "\n" + table["table"]
                section_header = "\n".join(para.content for para in page["paragraphs"][:2])
                section_header = f"<p><b>Sections:</b></p><p>{section_header}</p>"
                node_text = f"<p>{section_header}</p><p><b>Page(s)</b>: {i+1}</p><p><b>Content:</b></p><p>{table_text}</p>"
                node_metadata = {
                    "file_name": llama_doc.metadata["file_name"],
                    "title": llama_doc.metadata["title"],
                    "pages": str(i+1)
                }
                table_node = TextNode(text=node_text, id_=str(uuid.uuid4()), metadata=node_metadata)
                table_nodes.append(table_node)
    
    return table_nodes

def process_nodes(nodes, pages, content, doc_type, cracked_pdf, llama_doc):
    if cracked_pdf == True:
        for i in tqdm(range(len(nodes))):
            nodes[i] = doc_intel_map_node_to_page(nodes[i], pages, content)
        table_nodes = doc_intel_tables_to_nodes(pages, llama_doc) 
        nodes += table_nodes
    
    else:
        for i in tqdm(range(len(nodes))):
            nodes[i], curr_page_idx = map_node_to_page(nodes[i], pages, doc_type)
            pages = pages[curr_page_idx:]
    
    return nodes

class SingletonDocIntelClient:
    instance = None
    url = settings.AZURE_DOC_INTEL_ENDPOINT
    key = settings.AZURE_DOC_INTEL_KEY

    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            print("SingletonDocIntelClient: Creating instance of Form recognizer per process")
            if cls.url and cls.key:
                cls.instance = DocumentAnalysisClient(endpoint=cls.url, credential=AzureKeyCredential(cls.key))
            else:
                print("SingletonDocIntelClient: Skipping since credentials not provided. Assuming NO form recognizer extensions(like .pdf) in directory")
                cls.instance = object() # dummy object
        return cls.instance

    def __getstate__(self):
        return self.url, self.key

    def __setstate__(self, state):
        url, key = state
        self.instance = DocumentAnalysisClient(endpoint=url, credential=AzureKeyCredential(key))

class FileConverter:
    def __init__(self, filetype):
        converter_map = {
            "pdf": "PDFReader",
            "docx": "DocxReader"
        }
        Reader = download_loader(converter_map[filetype])
        self.loader = Reader()
        self.filetype = filetype
    
    def extract_pages(self, file_path):
        pages = self.loader.load_data(file=file_path)
        if self.filetype == "pdf":
            pages = set_page_headers(pages)
        return pages

    def extract_text(self, pages):
        full_text = "".join([page.text for page in pages])
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

# class LlamaIndexSplitter:
#     def __init__(self, num_tokens, token_overlap, extractor_llm=None):
#         splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#             separators=SENTENCE_ENDINGS + WORDS_BREAKS,
#             chunk_size=num_tokens, chunk_overlap=token_overlap)
#         parser = LangchainNodeParser(splitter)
#         self.parser = parser
#         self.extractor_llm = extractor_llm
#         # self.split_mods_map = {
#         #     "summary_extraction": SummaryExtractor(summaries=["prev", "self", "next"], llm=extractor_llm),
#         #     "qa_extraction": QuestionsAnsweredExtractor(questions=3, llm=extractor_llm)
#         # }
    
#     def get_nodes_from_doc(self, llama_doc, split_mods=None):
#         # if split_mods and self.extractor_llm == None:
#         #     raise ValueError(f"extractor_llm argument is required to apply modifications during splitting")

#         pipeline = IngestionPipeline(
#             transformations=[self.parser]# + [self.split_mods_map[mod] for mod in split_mods]
#         )

#         nodes = pipeline.run(
#             documents = [llama_doc],
#             in_place=True,
#             show_progress=True
#         )

#         return nodes

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

    def jsonify_chunks(self):
        return [doc.to_dict() for doc in self.chunks]

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
        pages: list, 
        content: str,
        file_name: Optional[str],
        file_format: str,
        cracked_pdf: bool,
        doc_type: str
) -> Generator[Tuple[str, int, Document], None, None]:

    parser = parser_factory(file_format)
    doc = parser.parse(content, file_name=file_name)
    doc_content_size = TOKEN_ESTIMATOR.estimate_tokens(doc.content)

    #chunks = []

    if file_format == "json" or doc_content_size < settings.PREP_CONFIG["chunk_size"]:
        yield doc.content, doc_content_size, doc
    else:
        sentence_splitter = SentenceSplitter(
            chunk_size=settings.PREP_CONFIG["chunk_size"],
            chunk_overlap=settings.PREP_CONFIG["token_overlap"]
        )
        llama_doc = LlamaDocument(text=content, metadata={"file_name": file_name, "title": doc.title})
        nodes = sentence_splitter.get_nodes_from_documents([llama_doc])

        if _get_file_format(file_name) == "pdf":
            nodes = process_nodes(nodes, pages, content, doc_type, cracked_pdf, llama_doc)

        for node in nodes:
            chunk_size = TOKEN_ESTIMATOR.estimate_tokens(node.text)
            yield node, chunk_size, llama_doc
    
    return nodes

def chunk_content(
    pages: list,
    content: str,
    file_name: Optional[str] = None,
    file_format: Optional[str] = None,
    cracked_pdf: bool = False,
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
    #try:
    if file_format not in ["json", "csv"]: 
        file_format = "text"

    nodes = chunk_content_helper(
        pages=pages,
        content=content,
        file_name=file_name,
        file_format=file_format,
        cracked_pdf=cracked_pdf,
        doc_type=doc_type
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

    # except UnsupportedFormatError as e:
    #     if ignore_errors:
    #         return ChunkingResult(
    #             chunks=[], total_files=1, num_unsupported_format_files=1
    #         )
    #     else:
    #         raise e
    # except Exception as e:
    #     if ignore_errors:
    #         return ChunkingResult(chunks=[], total_files=1, num_files_with_errors=1)
    #     else:
    #         raise e
    
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


def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html


def doc_intel_extract_pdf(file_path, doc_intel_client):
    offset = 0
    page_map = []
    model = "prebuilt-layout"
    with open(file_path, "rb") as f:
        poller = doc_intel_client.begin_analyze_document(model, document = f)
    form_recognizer_results = poller.result()

    for page_num, page in enumerate(form_recognizer_results.pages):
        paragraphs_on_page = [para for para in form_recognizer_results.paragraphs if para.bounding_regions[0].page_number == page_num + 1]
        tables_on_page = [table for table in form_recognizer_results.tables if table.bounding_regions[0].page_number == page_num + 1]
        html_tables_on_page = []
        table_para_indices = []

        for table_id, table in enumerate(tables_on_page):
            found_table_start, found_table_end = False, False
            for i, paragraph in enumerate(paragraphs_on_page):
                if paragraph.spans[0].offset >= tables_on_page[table_id].spans[0].offset and found_table_start == False:
                    table_para_idx_start = i-1
                    found_table_start = True
                elif paragraph.spans[0].offset >= table.spans[0].offset + table.spans[0].length and found_table_end == False:
                    table_para_idx_end = i
                    found_table_end = True
            table_para_indices.append((table_para_idx_start, table_para_idx_end))
            table_html = table_to_html(table)
            table_header = paragraphs_on_page[table_para_idx_start].content
            html_tables_on_page.append({"header": f"Table: {table_header}", "table": table_html})

        # if len(tables_on_page) > 0:
        #     for start, end in table_para_indices:
        #         for i in range(len(paragraphs_on_page)):
        #             if start <= i < end:
        #                 paragraphs_on_page[i] = "" 
        #     paragraphs_on_page = [para for para in paragraphs_on_page if para != ""]

        # Track offset for each page, so when we use re.search(chunk) across the full doc text, we can map the output idx to the right page using offset
        page_text_paragraphs = [para for para in paragraphs_on_page[2:] if para.role not in ["pageNumber", "pageHeader", "pageFooter"]]
        page_text = "\n".join([para.content for para in page_text_paragraphs])
        page_map.append({"page_num": page_num, "offset": offset, "page_text": page_text, "paragraphs": paragraphs_on_page, "tables": html_tables_on_page})
        offset += len(page_text)

    full_text = "\n".join([page["page_text"] for page in page_map])
    return page_map, full_text

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

    cracked_pdf = False

    if file_format == "json":
        with open(file_path) as f:
            content = json.load(f)
    
    elif doc_type == "qfr":
        return chunk_qfr(file_path, add_embeddings)
    
    elif file_format == "pdf" and settings.AZURE_USE_DOC_INTEL == True:
        print(f"Chunking {file_name} with Document Intelligence")
        doc_intel_client = DocumentAnalysisClient(
            endpoint=settings.AZURE_DOC_INTEL_ENDPOINT, 
            credential=AzureKeyCredential(settings.AZURE_DOC_INTEL_KEY)
        )
        pages, content = doc_intel_extract_pdf(file_path, doc_intel_client)
        cracked_pdf = True
    
    else:
        print(f"Chunking {file_name} without Document Intelligence")
        file_converter = FileConverter(file_format)
        pages = file_converter.extract_pages(Path(file_path))
        content = file_converter.extract_text(pages)
        
    return chunk_content(
        pages=pages,
        content=content,
        file_name=file_name,
        file_format=file_format,
        cracked_pdf=cracked_pdf,
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
    #try:
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
    # except Exception as e:
    #     if not ignore_errors:
    #         raise
    #     print(f"File ({file_path}) failed with ", e)
    #     is_error = True
    #     result =None
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
