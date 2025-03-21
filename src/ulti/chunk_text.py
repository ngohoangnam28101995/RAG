from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import pandas as pd

def chunk_text(text: str, chunk_size: int = 150, overlap_size: int = 50) -> List[Document]:
    """
    Chia nhỏ văn bản thành các đoạn với kích thước chunk_size và phần chồng overlap_size.
    
    Parameters:
        text (str): Văn bản đầu vào cần chia nhỏ.
        chunk_size (int): Độ dài tối đa của mỗi đoạn.
        overlap_size (int): Độ dài của phần chồng giữa các đoạn.

    Returns:
        List[str]: Danh sách các đoạn văn bản dưới dạng chuỗi.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap_size
    )
    
    # Chuyển văn bản thành đối tượng Document
    doc = Document(page_content=text)
    
    return [chunk.page_content for chunk in text_splitter.split_documents([doc])]
