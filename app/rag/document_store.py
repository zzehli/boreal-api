"""
Document storage and processing operations.
"""

import glob
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.models import Document
from app.utils.constants import Categories
from app.utils.utils import Utils


def get_category(file_path: str) -> str:
    """Get the category of the document."""
    filename = file_path.split('/')[-1]
    category = filename.split('_')[0]
    if category in Utils.get_all_brands():
        return Categories.PRODUCT
    elif category in [Categories.RECIPE, Categories.NEWS, Categories.BLOG]:
        return category
    else:
        return Categories.OTHERS

def load_documents(data_dir: str) -> List[Document]:
    """Load documents from the data directory."""
    site_files = glob.glob(f"{data_dir}/*.md")
    documents = []
    
    for file_path in site_files:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        html_url = Utils.get_url_from_path(file_path)
        category = get_category(file_path)
        for doc in docs:
            documents.append(
                Document(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    source_url=html_url,
                    category=category  # You can add category detection logic here
                )
            )
    
    return documents

def process_documents(
    documents: List[Document],
    chunk_size: int = 1500,
    chunk_overlap: int = 200
) -> List[LangchainDocument]:
    """Process documents by splitting them into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Convert our Document model to LangchainDocument
    langchain_docs = [
        LangchainDocument(
            page_content=doc.content,
            metadata={
                "source_url": doc.source_url,
                "category": doc.category,
                **doc.metadata
            }
        )
        for doc in documents
    ]
    
    return text_splitter.split_documents(langchain_docs)