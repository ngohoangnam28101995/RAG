import numpy as np
from src.ulti.get_embedding import get_embedding
from src.ulti.response import response
from src.ulti.hyde import hyde
from src.ulti.mmr import apply_mmr

def top_k(query_text, db_collection, model, tokenizer, device, use_hyde=True, use_reader=False, k=5, use_mmr=True, lambda_param=0.5):
    if use_hyde:
        new_query_text = hyde(query_text)
        query_embedding = get_embedding(new_query_text.lower(), model, tokenizer, device)
        print("Hyde prompt:", new_query_text)
    else:
        query_embedding = get_embedding(query_text.lower(), model, tokenizer, device)

    # Truy vấn các văn bản tương tự
    results = db_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k * 2 if use_mmr else k,  # Lấy nhiều hơn để MMR có thể lọc
        include=['embeddings', 'documents', 'metadatas', 'distances']
    )
    # Nếu dùng MMR, lọc kết quả để tăng độ đa dạng
    if use_mmr and results["embeddings"] is not None:
        print(results["embeddings"][0])
        doc_embeddings = np.array(results["embeddings"][0])  # Dùng embeddings thay vì documents
        mmr_indices = apply_mmr(query_embedding, doc_embeddings, top_k=k, lambda_param=lambda_param)

        results["metadatas"][0] = [results["metadatas"][0][i] for i in mmr_indices]


    retrieved_chunk = ""
    for i, meta in enumerate(results["metadatas"][0]):
        if retrieved_chunk == "":
            #retrieved_chunk = f"Thông tin {i+1}: {meta['content']} (Tệp: {meta['filename']})"
            retrieved_chunk = f"Thông tư : {meta['filename']} {meta['content']}"
        else:
            #retrieved_chunk += "\n" + f"Thông tin {i+1}: {meta['content']} (Tệp: {meta['filename']})"
            retrieved_chunk += "\n" + f"Thông tư : {meta['filename']} : {meta['content']}"
        #print(f"🔍 Thông tin {i+1}: {meta['content']} (Tệp: {meta['filename']})")
        print(f"Thông tư : {meta['filename']} : {meta['content']}")

    if use_reader:
        reader_prompt = f'''Bạn là một chuyên gia giải đáp thắc mắc dựa trên tài liệu được cung cấp.
                        Dưới đây là các đoạn thông tin liên quan được truy xuất:
                        ---------------------
                        {retrieved_chunk}
                        ---------------------
                        
                        Với những thông tin được cung cấp trong "Các đoạn thông tin liên quan", đây là những thông tin được truy vấn với câu hỏi {query_text}.
                        Hãy tổng hợp các thông tin quan trọng một cách dễ hiểu, súc tích, đầy đủ nội dung chính và đảm bảo chúng giải đáp được câu hỏi {query_text}.
                        Không sử dụng kiến thức bên ngoài. Mỗi câu trả lời cần kèm theo nguồn trích dẫn chính xác theo định dạng [Nguồn: Thông tư số ...].
                        **Câu hỏi:** {query_text}  
                        **Kết quả tóm tắt:**'''
        
        retrieved_chunk, _ = response(reader_prompt, reader_prompt, temperature=0.8, max_token=4096, past_messages=None, model="vistral-7b-chat", API_URL="http://localhost:1234/v1/chat/completions")

    rag = f'''Bạn là một chuyên gia giải đáp thắc mắc.

Chỉ khi truy vấn hiện tại **trực tiếp liên quan đến tỷ giá** của một đơn vị tiền tệ (ví dụ: đô la Mỹ - USD, nhân dân tệ - CNY, yên Nhật - JPY...), hãy **KHÔNG trả lời trực tiếp**, mà thay vào đó **trả về đúng một đoạn JSON duy nhất**, không kèm bất kỳ văn bản nào khác, theo định dạng sau:

{{
  "function_call": {{
    "name": "get_exchange_rate",
    "arguments": {{ "date": "YYYY-MM-DD", "currency": "XXX" }}
  }}
}}

Trong đó:
- `date` là ngày mà người dùng yêu cầu tra cứu tỷ giá.
- `currency` là mã tiền tệ quốc tế theo chuẩn ISO 4217 (ví dụ: `"USD"`, `"CNY"`, `"JPY"`).

⚠️ Nếu truy vấn hiện tại KHÔNG hỏi về tỷ giá, hoặc đã chuyển sang chủ đề khác (dù các câu trước có nhắc đến tỷ giá), thì KHÔNG được `function_call`. Hãy trả lời như một chuyên gia bình thường dựa vào "Thông tin ngữ cảnh".

---

Thông tin ngữ cảnh:  
--------------------   
{retrieved_chunk}  
-------------------- 

Chỉ sử dụng thông tin trong "Thông tin ngữ cảnh" để trả lời: {query_text}.  
Không sử dụng kiến thức bên ngoài. Đảm bảo câu trả lời có nguồn trích dẫn chính xác theo định dạng [Nguồn: Thông tư số ...].

**Truy vấn hiện tại:** {query_text}  
**Câu trả lời:**'''


    
    return rag, query_text