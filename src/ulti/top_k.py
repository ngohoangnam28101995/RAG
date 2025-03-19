from src.ulti.get_embedding import get_embedding
from src.ulti.response import response


def top_k(query_text,db_collection,model,tokenizer,device,use_reader = False,k = 5):
    query_embedding = get_embedding(query_text.lower(),model,tokenizer,device)  # Giả sử có embedding query

    # Thực hiện truy vấn tìm văn bản tương tự
    results = db_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k  
    )
    retrieved_chunk = ''

    # Hiển thị kết quả
    for i, meta in enumerate(results["metadatas"][0]):
        if retrieved_chunk == '':
            retrieved_chunk = f"Thông tin {i+1}: {meta['content']} (Tệp: {meta['filename']})"
        else :
            retrieved_chunk = retrieved_chunk + '\n' +f"Thông tin {i+1}: {meta['content']} (Tệp: {meta['filename']})"
        print(f"🔍 Thông tin {i+1}: {meta['content']} (Tệp: {meta['filename']})")
    if use_reader == True:

        reader_prompt = f'''Dưới đây là các đoạn thông tin liên quan được truy xuất:
                        ---------------------
                        {retrieved_chunk}
                        ---------------------
                        Hãy tổng hợp các thông tin quan trọng từ các đoạn trên một cách dễ hiểu và súc tích. Chỉ tập trung vào những ý chính có liên quan đến câu hỏi.
                        **Câu hỏi:** {query_text}  
                        **Kết quả tóm tắt:**'''
        retrieved_chunk,_ = response(reader_prompt, reader_prompt, temperature=0.7, max_token=1024, past_messages=None, model="vistral-7b-chat", API_URL="http://localhost:1234/v1/chat/completions")


    rag = f'''Thông tin ngữ cảnh:  
            --------------------  
            {retrieved_chunk}  
            -------------------- 
            Dựa trên thông tin trên và **không sử dụng kiến thức bên ngoài**, hãy trả lời truy vấn dưới đây một cách ngắn gọn, chính xác.  
            **Truy vấn:** {query_text}  
            **Câu trả lời:**'''
    return rag,query_text