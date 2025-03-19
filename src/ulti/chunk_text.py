def chunk_text(text, chunk_size=150, overlap_size=50):
    """Cắt nội dung thành các đoạn có độ dài tối đa `chunk_size`, 
    với phần chồng lấn `overlap_size` giữa các đoạn."""
    
    words = text.split()  # Tách văn bản thành danh sách từ
    chunks = []
    start = 0  # Vị trí bắt đầu của đoạn
    
    while start < len(words):
        end = min(start + chunk_size, len(words))  # Giới hạn kích thước đoạn
        chunk = " ".join(words[start:end])  # Tạo đoạn từ danh sách từ
        
        chunks.append(chunk)
        start += chunk_size - overlap_size  # Dịch con trỏ đi, giữ lại phần chồng
        
        if start >= len(words):  # Nếu vị trí bắt đầu vượt quá danh sách từ, dừng lại
            break

    return chunks