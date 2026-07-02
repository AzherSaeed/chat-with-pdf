import re
import tiktoken
import chromadb
import gradio as gr
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

encoding = tiktoken.get_encoding("cl100k_base")
AI_MODEL = "gpt-5-nano"
load_dotenv()
openai = OpenAI()
chroma_client = chromadb.Client()


SYSTEM_PREFIX_MESSAGE = """
   You are a professional and helpful AI Assistant.

    Your primary responsibility is to answer the user's questions using **only the information provided in the document context**.

    Document Context:
    {context}

    Instructions:

    * Be professional, polite, and concise in every response.
    * Carefully analyze the provided document context before answering.
    * Answer only if the requested information is explicitly available in the context.
    * Do not make assumptions, infer missing facts, or use outside knowledge.
    * If the answer is not present or cannot be determined from the provided context, respond only with:
    **"No relevant information found."**
    * Keep responses clear and easy to understand.
    * Avoid unnecessarily long paragraphs and excessive bullet points.
    * If the user asks for a summary, explanation, or comparison, generate it using only the provided context.
    * If multiple pieces of context are relevant, combine them into a single coherent answer.
    * Do not mention that you were given a context unless the user asks how you obtained the information.
    * Never fabricate citations, page numbers, or facts that are not present in the context.
    * When answering, mention the document name and page number if that information is available in the retrieved context.

    Your goal is to provide accurate, context-based answers while avoiding hallucinations.

"""

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name='pdf_collection',
    embedding_function=embedding_function,
    # data_loader=data_loader
)

def clean_text(text: str) -> str:
    text = text.replace("•", " ")
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_pdfs(files):

    if not files:
        return "No files uploaded."

    files_reader(files)

    return f"{len(files)} PDF(s) processed successfully."

def files_reader(files):
    all_chunks = []
    for pdf in files:
        reader = PdfReader(pdf.name)
        filename = pdf.name.split("/")[-1]
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = clean_text(page.extract_text() or "")
            chunks = chunk_splitter(page_text)
            for chunk_index, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{filename}_{page_number}_{chunk_index}",
                    "document": chunk,
                    "metadata": {
                        "file": filename,
                        "page": page_number
                    }
                })
    setup_db(all_chunks)

def chunk_splitter(document_text):
    tokens = encoding.encode(document_text)

    chunk_size = 800
    overlap = 200

    chunks = []

    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def setup_db(all_chunks):

    existing_ids = set(collection.get()["ids"])

    documents = []
    metadatas = []
    ids = []

    for chunk in all_chunks:

        if chunk["id"] in existing_ids:
            continue

        ids.append(chunk["id"])
        documents.append(chunk["document"])
        metadatas.append(chunk["metadata"])

    if ids:

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Inserted {len(ids)} chunks")

    else:
        print("No new chunks found.")


def retrieve_documents(message):
    results = collection.query(
        query_texts=[message],
        n_results=5,
        include=["documents" , "metadatas" , "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]


    context_parts = []

    for doc, metadata in zip(documents, metadatas):
        context_parts.append(
            f"""
            Document:
            {metadata['file']}

            Page:
            {metadata['page']}

            Content:
            {doc}
        """
        )

    best_distance = results["distances"][0][0]

    if best_distance > 1.0:
        return {
        "found": False,
        "context": "",
        "message": "No relevant information found."
    }

    return {
        "found": True,
        "context": "\n\n-----------------\n\n".join(context_parts)
    }

def build_prompt(context):
  system_prompt = SYSTEM_PREFIX_MESSAGE.format(context=context)
  return system_prompt


def send_message(message, history):
    history = history or []

    if not message.strip():
        yield history, ""
        return

    retrieval = retrieve_documents(message)

    if not retrieval["found"]:
        history.append(
            {
                "role": "user",
                "content": message
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": retrieval["message"]
            }
        )

        yield history, ""
        return

    system_prompt = build_prompt(retrieval["context"])

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": message
        }
    )

    stream = openai.chat.completions.create(
        model=AI_MODEL,
        messages=messages,
        stream=True
    )

    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    assistant_message = {
        "role": "assistant",
        "content": ""
    }

    history.append(assistant_message)

    for chunk in stream:

        delta = chunk.choices[0].delta.content

        if delta:

            assistant_message["content"] += delta

            yield history, ""

    




with gr.Blocks(title="Chat with PDF") as demo:
    gr.Markdown("# 📄 Chat with PDF")

    pdf_files = gr.File(
        label="Upload PDF(s)",
        file_count="multiple",
        file_types=[".pdf"]
    )
    pdf_files.upload(
        fn=process_pdfs,
        inputs=pdf_files,
    )
    
    upload_status = gr.Textbox(
        label="Status",
        interactive=False
    )

    pdf_files.upload(
        process_pdfs,
        pdf_files,
        upload_status
    )

    chatbot = gr.Chatbot(height=500)

    with gr.Row():
        message = gr.Textbox(
            placeholder="Ask something about the uploaded PDFs...",
            show_label=False,
            scale=8,
        )

        send = gr.Button("Send", variant="primary", scale=1)

    send.click(
        send_message,
        inputs=[message, chatbot],
        outputs=[chatbot, message],
    )

    message.submit(
        send_message,
        inputs=[message, chatbot],
        outputs=[chatbot, message],
    )

demo.launch()