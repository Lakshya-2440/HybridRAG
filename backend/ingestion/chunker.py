try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:  # langchain>=1 splitters package
    from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(pages: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunk_id = f"{page['metadata']['doc_id']}_{page['metadata'].get('page', 0)}_{i}"
            chunks.append({
                "text": split,
                "chunk_id": chunk_id,
                "metadata": {**page["metadata"], "chunk_index": i, "chunk_id": chunk_id}
            })
    return chunks
