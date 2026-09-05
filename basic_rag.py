from operator import itemgetter
from typing import List
import os
import tempfile
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

from pypdf import PdfReader

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


class HuggingFaceServerlessEmbeddings(Embeddings):
    """Serverless (Inference Providers) embeddings using the HF access token."""

    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        self.client = InferenceClient(token=HF_TOKEN)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        result = self.client.feature_extraction(text, model=self.model)
        return [float(x) for x in result]


embeddings_model = HuggingFaceServerlessEmbeddings()


def create_kb():
    pdf_path = "/home/abishek/Projects/retrieval-augmented-generation/data/knowledge_base.pdf"

    reader = PdfReader(pdf_path)
    pdf_pages = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        pdf_pages.append(Document(page_content=text, metadata={"source": pdf_path, "page": page_num}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = splitter.split_documents(pdf_pages)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store


def demo_vector_store():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    client = InferenceClient(token=HF_TOKEN)

    def call_llm(context: str, question: str) -> str:
        prompt = f"""
You are a helpful assistant that provides information based on the provided documents:
{context}

Question: {question}

Answer: Make sure to provide a concise and accurate response based on the information available in the documents.
If the answer is not found in the documents, respond with "I don't know."
"""
        result = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=CHAT_MODEL,
            max_tokens=512,
            temperature=0.2,
        )
        return result.choices[0].message.content

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | RunnableLambda(lambda inputs: call_llm(inputs["context"], inputs["question"]))
        | StrOutputParser()
    )

    questions = [
        "What is the main topic of the knowledge base?",
        "What is Generative AI?",
        "What are the key benefits of using Generative AI?",
    ]

    for q in questions:
        result = rag_chain.invoke({"question": q})
        print(f"Question: {q}")
        print(f"Answer: {result}")
        print("-" * 50)


if __name__ == "__main__":
    demo_vector_store()
