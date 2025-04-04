import json
import re

def extract_json_from_text(text):
    try:
        # Bước 1: Tìm đoạn JSON trong văn bản bằng regex
        match = re.search(r'\{[\s\S]*\}', text)  # Tìm chuỗi bắt đầu bằng "{" và kết thúc bằng "}"
        
        if not match:
            print("Không tìm thấy JSON trong văn bản.")
            return None

        # Bước 2: Lấy chuỗi JSON
        json_str = match.group(0)

        # Kiểm tra trước khi parse JSON
        print(f"Đoạn JSON tìm được: {json_str}")
        
        # Bước 3: Parse chuỗi JSON
        parsed_json = json.loads(json_str)
        return parsed_json
    
    except json.JSONDecodeError as e:
        print(f"Lỗi khi phân tích JSON: {e}")
        return None
    except Exception as e:
        print(f"Lỗi khác: {e}")
        return None
