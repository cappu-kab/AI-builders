from datasets import Dataset

# ระบุ Path ไปที่ "ไฟล์" .arrow ที่ต้องการเลย (ในที่นี้คือตัว train)
file_path = r"C:\Users\rocha\.cache\huggingface\datasets\fsicoli___common_voice_16_0\th\15.0.0\e10daa6c775fc97754936ed692129c926b79f612d32698dc6793403ff5b359e0\common_voice_16_0-train.arrow"

# ใช้ Dataset.from_file แทน load_from_disk
dataset = Dataset.from_file(file_path)

# เช็ครายชื่อคอลัมน์ (เฉลยจะอยู่ในนี้)
print("คอลัมน์ทั้งหมด:", dataset.column_names)

# ลองดูข้อมูลแถวแรก
print("ข้อมูลแถวแรก:", dataset[0])
for i in range(5):
    print(f"ไฟล์ที่ {i}: {dataset[i]['sentence']}")