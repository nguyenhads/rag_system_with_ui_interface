import os

import chainlit as cl
from chainlit.types import AskFileResponse
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

from src.model.llms import load_embedding_model, load_llm
from src.vector_db.chroma_db import ChromaDBChainlit

# Load environment variables
load_dotenv()

# Load models and database
embedding = load_embedding_model("openai")
llm = load_llm("gpt-3.5")
chromadb = ChromaDBChainlit(embedding=embedding)

# Define welcome message
WELCOME_MESSAGE = """Welcome to the PDF QA! To get started: \n
1. Upload a PDF or text file
2. Ask a question about the file"""


async def handle_file_upload():
    """
    Handle the file upload process. Waits for the user to upload a PDF or text file.

    Returns:
        AskFileResponse: The first file uploaded by the user.
    """
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=WELCOME_MESSAGE,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    return files[0]


async def process_file(file):
    """
    Process the uploaded file and create a retrieval chain.

    Args:
        file (AskFileResponse): The file uploaded by the user.

    Returns:
        None
    """
    await cl.Message(
        content=f"Processing '{file.name}'...", disable_feedback=True
    ).send()
    vector_db = await cl.make_async(chromadb.build_db)(file)

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    await cl.Message(
        content=f"'{file.name}' processed. You can now ask questions!"
    ).send()
    cl.user_session.set("chain", chain)


async def handle_message(message):
    """
    Handle incoming messages from the user and provide appropriate responses.

    Args:
        message (cl.Message): The message sent by the user.

    Returns:
        None
    """
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="No active chain. Please upload a file first.").send()
        return

    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])

    answer = res["answer"]
    source_documents = res.get("source_documents", [])
    text_elements = [
        cl.Text(content=doc.page_content, name=f"Source {idx + 1}")
        for idx, doc in enumerate(source_documents)
    ]

    source_names = [text_el.name for text_el in text_elements]
    if source_names:
        answer += f"\nSources: {', '.join(source_names)}"
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()


# Chainlit event handlers
@cl.on_chat_start
async def on_chat_start():
    """
    Event handler for the start of a chat session.
    Initiates the file upload and processing workflow.
    """
    file = await handle_file_upload()
    await process_file(file)


@cl.on_message
async def on_message(message: cl.Message):
    """
    Event handler for incoming messages.
    Processes the user's message and provides a response.
    """
    await handle_message(message)
