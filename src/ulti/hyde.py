from src.ulti.response import response


def hyde(query, temperature=0.7, max_token=2048, model="vistral-7b-chat", API_URL="http://localhost:1234/v1/chat/completions"):
    hyde_query = f'''Giả sử bạn đã tìm thấy một tài liệu có câu trả lời chính xác cho câu hỏi sau: {query}. Hãy viết lại nội dung tài liệu đó dưới dạng một đoạn văn đầy đủ thông tin, có cấu trúc hợp lý và giúp người đọc hiểu rõ vấn đề.'''
    new_query,_ = response(hyde_query, hyde_query, temperature=temperature, max_token=max_token, past_messages=None, model=model, API_URL=API_URL)
    return new_query