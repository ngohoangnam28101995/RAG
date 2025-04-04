import requests
import json

# Hàm lấy tỷ giá từ một đơn vị tiền tệ vào một ngày cụ thể
def get_exchange_rate(currency_from, currency_to, date):
    api_key = 'b32174790ed60df8bf57da7d'  # Thay thế bằng API Key của bạn
    url = f'https://v6.exchangerate-api.com/v6/{api_key}/latest/{currency_from}'
    
    response = requests.get(url)
    
    # Kiểm tra mã trạng thái HTTP
    if response.status_code != 200:
        print(f"Lỗi khi gửi yêu cầu. Mã lỗi HTTP: {response.status_code}")
        print(f"Nội dung phản hồi: {response.text}")
        return None

    # Kiểm tra nếu phản hồi không phải là JSON hợp lệ
    try:
        data = response.json()
    except json.JSONDecodeError:
        print(f"Lỗi khi nhận phản hồi từ API: Phản hồi không phải JSON, nội dung: {response.text}")
        return None
    
    if 'conversion_rates' in data:
        rate = data['conversion_rates'].get(currency_to)
        if rate:
            return rate
        else:
            print(f"Tỷ giá không có trong dữ liệu trả về.")
            return None
    else:
        print(f"Lỗi khi lấy dữ liệu: {data.get('error-type', 'Không rõ lỗi')}")
    return None

# Hàm xử lý input JSON và gọi hàm lấy tỷ giá
def process_function_call(function_call):
    function_call = function_call['function_call']
    print("function call :",function_call)
    function_name = function_call.get('name')
    print(function_name)
    arguments = function_call.get('arguments', {})
    if function_name == 'get_exchange_rate':
        # Lấy ngày và đơn vị tiền tệ từ arguments
        date = arguments.get('date')
        currency = arguments.get('currency')

        # Kiểm tra đầu vào
        if date and currency:
            # Gọi hàm lấy tỷ giá
            rate = get_exchange_rate(currency, 'VND', date)
            if rate:
                return f"Tỷ giá {currency}/VND vào ngày {date}: {rate}"
            else:
                return f"Không thể lấy tỷ giá {currency}/VND vào ngày {date}."
        else:
            return "Thông tin ngày hoặc tiền tệ không hợp lệ."
    else:
        return "Chức năng không hợp lệ."