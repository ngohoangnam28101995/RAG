import requests
import json
import copy

def response(new_prompt, old_prompt, temperature=0.7, max_token=1024, past_messages=None, model="vistral-7b-chat", API_URL="http://localhost:1234/v1/chat/completions"):
    # Nếu past_messages không phải list, đặt nó là danh sách rỗng
    if not isinstance(past_messages, list):
        past_messages = [{"role": "system", "content": """Bạn là một AI hỗ trợ.  
                                                        Chỉ sử dụng thông tin trong các tài liệu đã được cung cấp để trả lời câu hỏi.  
                                                        Mỗi khi bạn đưa ra thông tin, hãy kèm theo nguồn trích dẫn chính xác theo định dạng [Nguồn: Tên Thông Tư].  
                                                        Không được thêm bất kỳ thông tin nào ngoài tài liệu đã được cung cấp.  
                                                        **Hướng dẫn quan trọng:**  
                                                        - Nếu người dùng hỏi về tỷ giá ngoại tệ và **trong hội thoại trước đã có tỷ giá**, hãy **không gọi function_call** nữa.  
                                                        Thay vào đó, **sử dụng tỷ giá có sẵn** để trả lời hoặc thực hiện tính toán theo yêu cầu.  
                                                        - Chỉ gọi function_call **nếu người dùng yêu cầu một ngày hoặc loại tiền tệ khác chưa có trong hội thoại**.  
                                                        """}]
    
    # Tạo bản sao hoàn toàn mới để tránh bị tham chiếu chung
    conversation = copy.deepcopy(past_messages)
    conversation_v2 = copy.deepcopy(past_messages)

    # Thêm prompt vào từng conversation
    conversation.append({"role": "user", "content": new_prompt})
    conversation_v2.append({"role": "user", "content": old_prompt})

    # Tạo payload để gửi API
    payload = {
        "model": model,
        "messages": conversation,
        "temperature": temperature,
        "max_tokens": max_token,
    }

    # Gửi yêu cầu API
    response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

    # Xử lý kết quả
    if response.status_code == 200:
        result = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        result = ''

    # Tạo bản sao conversation_v2 trước khi thêm assistant response
    conversation_v2_new = copy.deepcopy(conversation_v2)
    conversation_v2_new.append({"role": "assistant", "content": result})

    return result, conversation_v2_new