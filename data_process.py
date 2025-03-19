
import os
import fitz
import torch
import chromadb
import numpy as np
import pandas as pd
import win32com.client
from transformers import AutoModel, AutoTokenizer
from src.ulti.clean_text import covert_unicode,clean_text
from src.ulti.get_embedding import get_embedding
from src.ulti.chunk_text import chunk_text


# Thư mục chứa files
folder_path = "data/legal document"
model = "vinai/phobert-base"
chunk_size = 250
overlap = 25

# Hàm đọc file PDF
def read_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.encode("utf-8").decode("utf-8")  # Đảm bảo UTF-8
    except Exception as e:
        return f"Lỗi đọc PDF: {e}"

# Hàm đọc file Word
def read_doc(file_path):
    """Đọc nội dung tệp .doc bằng Microsoft Word"""
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False  # Chạy ẩn Microsoft Word
        doc = word.Documents.Open(file_path)
        text = doc.Content.Text
        doc.Close(False)
        word.Quit()
        return text.strip().encode("utf-8", "ignore").decode("utf-8")  # Đảm bảo UTF-8
    except Exception as e:
        return f"Lỗi đọc DOC: {e}"

# Đọc toàn bộ files và lưu vào DataFrame
data = []
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)  # Ghép đường dẫn tương đối
    abs_file_path = os.path.abspath(file_path)  # Chuyển thành đường dẫn tuyệt đối

    if not os.path.exists(abs_file_path):  # Kiểm tra tệp có tồn tại không
        print(f"Lỗi: Không tìm thấy {abs_file_path}")
        continue

    if file.endswith(".pdf"):
        content = read_pdf(abs_file_path)  # Đọc tệp PDF
    elif file.endswith(".doc"):
          print(f"Đọc tệp DOC: {abs_file_path}")  # Debug đường dẫn tuyệt đối
          content = read_doc(abs_file_path)  # Đọc tệp DOC
    else:
        continue  # Bỏ qua các tệp khác

    data.append([file, content])



# Tạo DataFrame
df = pd.DataFrame(data, columns=["Tên file", "Nội dung"])


# Áp dụng vào cột "Nội dung" của 
df["Nội dung"]  = df["Nội dung"].apply(covert_unicode)
df["Nội dung"] = df["Nội dung"].str.replace("\n", " ", regex=True)
df["Nội dung"] = df["Nội dung"].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
df = df[df["Nội dung"]!=""]
df.to_csv("Checkdata.csv", encoding="utf-8-sig", index=False)



# Tiến hành chunking
df_expanded = df.explode('Nội dung')
df_expanded['Nội dung'] = df_expanded['Nội dung'].apply(chunk_text,args=(chunk_size,overlap))

# Chuyển danh sách chunk thành từng dòng riêng lẻ
df_final = df_expanded.explode('Nội dung').reset_index(drop=True)

#Khai báo device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load mô hình embedding
phobert = AutoModel.from_pretrained(model).to(device)
tokenizer = AutoTokenizer.from_pretrained(model)

# Áp dụng embedding cho từng dòng trong cột "Nội dung"
df_final["Embedding"] = df_final["Nội dung"].apply(lambda x: get_embedding(x,phobert,tokenizer,device) if isinstance(x, str) else None)

# Chuyển embedding thành NumPy array
df_final["Embedding"] = df_final["Embedding"].apply(lambda x: np.array(x))
print(df_final)

# Khởi tạo ChromaDB Client
client = chromadb.PersistentClient(path=f"chromadb/chromadb_chunk_{chunk_size}_{overlap}")  # Lưu DB cục bộ
collection = client.get_or_create_collection(name="document_embeddings")

# Chuyển embeddings thành danh sách
df_final["Embedding"] = df_final["Embedding"].apply(lambda x: x.tolist())

# Thêm dữ liệu vào ChromaDB
for i, row in df_final.iterrows():
    collection.add(
        ids=[str(i)],  # ID duy nhất
        embeddings=[row["Embedding"]],
        metadatas=[{"filename": row["Tên file"], "content": row["Nội dung"]}]
    )

print("✅ Đã lưu embeddings vào ChromaDB!")