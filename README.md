# CIFAR-10 Image Classification Web Demo 🚀

Ứng dụng Web Demo phân loại ảnh **CIFAR-10** thời gian thực (100% Client-side) sử dụng mô hình Deep Learning PyTorch (`ImprovedCNN`) chuyển đổi sang định dạng **ONNX** và suy luận trực tiếp trên trình duyệt qua **ONNX Runtime Web (WASM)**.

---

## 🌟 Tính Năng
- ⚡ **100% Client-Side Inference:** Không cần backend server, suy luận trực tiếp bằng WebAssembly trên máy người dùng.
- 🖼️ **Kéo thả / Chọn ảnh:** Hỗ trợ mọi định dạng ảnh phổ biến (PNG, JPG, JPEG, WEBP), tự động chuẩn hóa về kích thước $32 \times 32$.
- 🎯 **Thư viện ảnh mẫu 1-Click:** Tích hợp sẵn 10 ảnh mẫu đại diện cho 10 lớp CIFAR-10 để kiểm thử tức thì.
- 📊 **Trực quan hóa:** Hiển thị thẻ Top-1 nhãn dự đoán kèm thanh phần trăm xác suất Softmax của cả 10 lớp đối tượng.

---

## 📂 Cấu Trúc Thư Mục
```text
cifar_web_demo/
├── index.html        # Giao diện chính (HTML5 Semantic)
├── style.css         # Giao diện hiện đại, responsive, Glassmorphism
├── app.js            # Xử lý canvas & suy luận ONNX Runtime Web
├── model.onnx        # Trọng số mô hình ImprovedCNN (2.55 MB)
├── samples/          # 10 ảnh mẫu đại diện cho 10 lớp CIFAR-10
├── vercel.json       # Cấu hình tối ưu cache header cho Vercel
└── README.md
```

---

## 🚀 Hướng Dẫn Deploy Lên Vercel

### Cách 1: Deploy qua GitHub (Khuyên dùng)
1. Tạo một repository mới trên GitHub (ví dụ: `cifar-10-demo`).
2. Khởi tạo git và push toàn bộ nội dung trong thư mục này lên GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for CIFAR-10 Web Demo"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo-name>.git
   git push -u origin main
   ```
3. Truy cập [vercel.com](https://vercel.com) -> Đăng nhập bằng GitHub.
4. Chọn **Add New Project** -> Chọn repository vừa tạo.
5. Ở mục **Framework Preset**, chọn **Other** (hoặc để mặc định).
6. Bấm **Deploy**. Vercel sẽ tự động tạo URL trực tuyến cho bạn chỉ trong 30 giây!

### Cách 2: Deploy trực tiếp bằng Vercel CLI
```bash
npm i -g vercel
vercel
```

---

## 💻 Chạy Cục Bộ (Local Test)
Mở terminal tại thư mục này và khởi động máy chủ static:
```bash
python -m http.server 8000
```
Truy cập: [http://localhost:8000](http://localhost:8000)
