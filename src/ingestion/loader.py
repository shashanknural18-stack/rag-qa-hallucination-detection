from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader
}
def load_documents(folder: str):
    docs = []
    for path in Path(folder).rglob("*"):
        loader_cls = LOADERS.get(path.suffix.lower())
        if not loader_cls:
            continue 
        try:
            kwargs = {"encoding": "utf-8"} if loader_cls is TextLoader else {}
            loaded = loader_cls(str(path), **kwargs).load()
            for d in loaded:
                d.metadata["source"] = path.name
                d.page_content = " ".join(d.page_content.split())  # clean whitespace
            docs.extend(loaded)
        except Exception as e:
            print(f"Failed to load {path.name}: {e}")
    return docs