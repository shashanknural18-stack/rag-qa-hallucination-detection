# src/ingestion/chunker.py
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=256,       # see note below
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)

def chunk_documents(docs):
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks
