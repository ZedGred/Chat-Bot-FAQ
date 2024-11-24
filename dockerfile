# Gunakan gambar dasar Python
FROM python:3.9-slim

# Set direktori kerja di dalam kontainer
WORKDIR /app

# Salin file requirements.txt dan instal dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin kode aplikasi ke dalam kontainer
COPY model ./model
COPY main.py . 

# Expose port yang akan digunakan
EXPOSE 8000

# Jalankan aplikasi FastAPI menggunakan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]