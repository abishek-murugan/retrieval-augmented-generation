from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    docs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if len(text.strip()) < 20:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "title": path.stem.replace("_", " ").title(),
                    "page": page_num,
                },
            )
        )
    return docs


def load_txt(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [
        Document(
            page_content=text,
            metadata={"source": path.name, "title": path.stem.replace("_", " ").title(), "page": 1},
        )
    ]


SUPPORTED_SUFFIXES = {".pdf": load_pdf, ".txt": load_txt, ".md": load_txt}


def load_sources(directory: Path) -> list[Document]:
    docs: list[Document] = []
    for f in sorted(directory.iterdir()):
        loader = SUPPORTED_SUFFIXES.get(f.suffix.lower())
        if loader is None:
            continue
        docs.extend(loader(f))
    if not docs:
        raise FileNotFoundError(f"No supported documents found in {directory}")
    return docs


def split_documents(docs: Iterable[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(list(docs))
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        chunk.metadata = {
            "chunk_id": i,
            "source": meta.get("source"),
            "title": meta.get("title"),
            "page": meta.get("page"),
        }
    return chunks