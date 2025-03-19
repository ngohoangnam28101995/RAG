import torch


# Hàm tạo embedding vector từ PhoBERT trên GPU
def get_embedding(text,model,tokenizer,device):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    # Chuyển dữ liệu lên GPU
    tokens = {key: val.to(device) for key, val in tokens.items()}
    
    with torch.no_grad():  # Không tính gradient để tiết kiệm tài nguyên
        output = model(**tokens)
    
    # Lấy mean của hidden states, chuyển về CPU rồi convert sang numpy
    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding
