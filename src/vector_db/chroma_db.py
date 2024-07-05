import os

import chainlit as cl
from chainlit.types import AskFileResponse
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.splitters.text_splitter import TextSplitter
from src.vector_db.base import VectorDBBase

load_dotenv()


class ChromaDBChainlit(VectorDBBase):
    def __init__(self, embedding) -> None:
        self._embedding = embedding
        self._vectorstore = None

    def build_db(self, file: AskFileResponse):
        docs = self.process_file(file)
        cl.user_session.set("docs", docs)
        self._vectorstore = Chroma.from_documents(
            documents=docs, embedding=self._embedding
        )

        return self._vectorstore

    def get_retriever(self, search_type: str = "similarity", **kwargs):
        if search_type == "mmr":
            return self._vectorstore.as_retriever(search_type="mmr", **kwargs)
        elif search_type == "similarity":
            return self._vectorstore.as_retriever(search_type="similarity", **kwargs)
        else:
            raise ValueError(
                f"Unknown search type: {search_type}. Please choose from 'similarity' or 'mmr'."
            )

    def process_file(self, file: AskFileResponse):
        if file.type == "text/plain":
            Loader = TextLoader
        elif file.type == "application/pdf":
            Loader = PyPDFLoader

        loader = Loader(file.path)
        documents = loader.load()

        text_splitter = TextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            splitter_type="RecursiveCharacterTextSplitter",
        )

        docs = text_splitter(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"

        return docs
