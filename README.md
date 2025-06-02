# PRNU-ELA-classifier

Bài tập lớn môn Thực tập cơ sở: Xây dựng hệ thống phân loại ảnh thật và ảnh AI sử dụng kết hợp đặc trưng PRNU, phân tích nhiễu và ELA.
Thực hiện bởi nhóm 01 lớp 49 (Khoa học máy tính - PTIT)

## Hướng dẫn cài đặt và chạy dự án

### 1. Tạo môi trường ảo (venv)

```bash
# Windows
python -m venv .venv

# Kích hoạt venv trên Windows
.venv\Scripts\activate
```

### 2. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### 3. Chạy ứng dụng Flask

```bash
python src/app.py
```

### 4. Truy cập giao diện web

Sau khi chạy thành công, mở trình duyệt và truy cập địa chỉ:

```
http://127.0.0.1:5000/
```

### 5. Thư mục quan trọng
- `src/`: Chứa mã nguồn chính của dự án
- `notebook/`: Notebook hướng dẫn và thử nghiệm
- `model_trained/`: Chứa các file model đã huấn luyện
- `uploads/`: Thư mục chứa ảnh upload

### 6. Lưu ý
- Đảm bảo đã cài Python >= 3.8
- Nếu gặp lỗi thiếu thư viện, kiểm tra lại file `requirements.txt` và cài đặt đúng môi trường ảo.
