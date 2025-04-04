import streamlit as st
import time
import torch
import os
import pandas as pd
from docx import Document
import numpy as np
from transformers import AutoModel, AutoTokenizer
import chromadb
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.ulti.response import response
from src.ulti.top_k import top_k
from src.ulti.check_json import extract_json_from_text
from src.ulti.calling_function import process_function_call


import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Lấy giá trị API_URL
api_url = os.getenv("API_URL")

@st.cache_resource
def load_phobert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("vinai/phobert-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    return model, tokenizer, device

# Load model và tokenizer một lần duy nhất
phobert, tokenizer, device = load_phobert()

# Load select box chọn kiểu chunking
db_path = "chromadb"
if os.path.exists(db_path):
    options = [f for f in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, f))]
else:
    options = [""]

# Ban đầu chưa có nội dung nào
if "part_prompt" not in st.session_state:
    st.session_state.part_prompt = None  
if "select" not in st.session_state:
    st.session_state.select = options[0]
if "use_reader" not in st.session_state:
    st.session_state.use_reader = True  # Giá trị mặc định
if "use_hyde" not in st.session_state:
    st.session_state.use_hyde = True  # Giá trị mặc định
if "use_mmr" not in st.session_state:
    st.session_state.use_mmr = True  # Giá trị mặc định

st.title("Chatbot Thông tư Tài Chính")

# Hàm xử lý khi thay đổi selectbox
def on_change():
    st.session_state.select = st.session_state.folder_select  # Lấy giá trị từ selectbox
    client = chromadb.PersistentClient(path=f"chromadb/{st.session_state.select}")
    collection = client.get_or_create_collection(name="document_embeddings")
    st.toast(f"Cập nhật thành công cho {st.session_state.select}!")  # Hiển thị thông báo

# Tạo selectbox với sự kiện on_change
selected_folder = st.selectbox(
    "Chọn cách chunk", 
    options, 
    index=0 if options else None, 
    key="folder_select", 
    on_change=on_change
)
# Tạo ba cột
col1, col2, col3 = st.columns(3)
# Hiển thị toggle button trong từng cột
with col1:
    st.session_state.use_reader = st.toggle("Dùng reader", value=st.session_state.get("use_reader", False))

with col2:
    st.session_state.use_hyde = st.toggle("Dùng hyde", value=st.session_state.get("use_hyde", False))

with col3:
    st.session_state.use_mmr = st.toggle("Dùng mmr", value=st.session_state.get("use_mmr", False))

# Khởi tạo ChromaDB lần đầu với DB mặc định
client = chromadb.PersistentClient(path=f"chromadb/{st.session_state.select}")
collection = client.get_or_create_collection(name="document_embeddings")

# Khởi tạo session_state nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn trước đó
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input từ người dùng
if prompt := st.chat_input("Hãy nhập vào yêu cầu?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Hiển thị tin nhắn của người dùng
    with st.chat_message("user"):
        st.markdown(prompt)

    # Hiển thị hiệu ứng chờ
    with st.chat_message("assistant"):
        with st.spinner("Chatbot đang suy nghĩ..."):
            full_res = ""
            holder = st.empty()

            # Giả lập quá trình phản hồi từng ký tự
            new_prompt, old_prompt = top_k(prompt, collection, phobert, tokenizer, device, k=10, use_reader=st.session_state.use_reader,use_hyde=st.session_state.use_hyde,use_mmr=st.session_state.use_mmr)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", new_prompt)
            simulated_response, st.session_state.part_prompt = response(new_prompt, old_prompt, temperature=0.7, max_token=1024, past_messages=st.session_state.part_prompt, model="vistral-7b-chat", API_URL=api_url)
            #Check chuỗi json
            print(simulated_response)
            json_string = extract_json_from_text(simulated_response)
            print(json_string)
            if json_string != None:
                simulated_response = process_function_call(json_string)
            print("#########################", st.session_state.part_prompt)
            with open("part_prompt.txt", "w", encoding="utf-8") as f:
                f.write(str(st.session_state.part_prompt))
        
        for char in simulated_response:
            full_res += char
            holder.markdown(full_res + "▌", unsafe_allow_html=True)  
            time.sleep(0.03)  # Hiệu ứng gõ chữ
        
        holder.markdown(full_res, unsafe_allow_html=True)

    # Lưu phản hồi của chatbot vào session_state
    st.session_state.messages.append({"role": "assistant", "content": full_res})
