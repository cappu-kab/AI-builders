'''

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler

# 1. โหลดข้อมูลลงเครื่อง (เอา streaming=True ออกเพื่อให้โหลดไฟล์ดิบลง Hard Drive)
print("กำลังโหลดข้อมูล... อาจใช้เวลานานขึ้นอยู่กับความเร็วเน็ต")

train_data = load_dataset("fsicoli/common_voice_16_0", "th", split="train", trust_remote_code=True)
test_data  = load_dataset("fsicoli/common_voice_16_0", "th", split="test",  trust_remote_code=True)
dev_data   = load_dataset("fsicoli/common_voice_16_0", "th", split="validation", trust_remote_code=True)

print(f"โหลดเสร็จแล้ว! \nTrain: {len(train_data)} \nTest: {len(test_data)} \nDev: {len(dev_data)}")

# 2. สร้าง DataLoader สำหรับแต่ละชุด (ยกตัวอย่างของ Train)
batch_size = 32

train_sampler = BatchSampler(RandomSampler(train_data), batch_size=batch_size, drop_last=False)
train_loader = DataLoader(train_data, batch_sampler=train_sampler)

# ตอนนี้คุณมี train_loader, test_data, และ dev_data พร้อมใช้งานแล้วครับ
'''
'''
import os
import pandas as pd
from datasets import load_dataset
import soundfile as sf

def export_isan_dialect_to_local(splits=["train"]):
    for split in splits:
        print(f"กำลังจัดการชุดข้อมูล Thai Dialect Isan: {split}...")
        
        # 1. โหลดข้อมูล (หมายเหตุ: ใช้ชื่อ scb10x/... ตามที่หน้าเว็บบ่งชี้)
        try:
            ds = load_dataset("scb10x/thai-dialect-isan-dataset", split=split, trust_remote_code=True)
        except Exception as e:
            print(f"ไม่พบ Split '{split}': {e}")
            continue
        
        # 2. สร้างโฟลเดอร์สำหรับเก็บไฟล์
        output_dir = f"thai_isan_dialect_local/{split}"
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        metadata = []

        # 3. วนลูปจัดการข้อมูลเสียงและข้อความ
        for i, entry in enumerate(ds):
            file_name = f"{split}_{i}.wav"
            file_path = os.path.join(audio_dir, file_name)
            
            # ดึงข้อมูลเสียง
            audio_array = entry["audio"]["array"]
            sampling_rate = entry["audio"]["sampling_rate"]
            
            # เซฟเป็น .wav
            sf.write(file_path, audio_array, sampling_rate)
            
            # เก็บข้อมูลลง Metadata (เพิ่มทั้งภาษาอีสานและไทยกลาง)
            metadata.append({
                "file_name": f"audio/{file_name}",
                "transcription_isan": entry["isan_spelling"], # ภาษาอีสาน
                "transcription_thai": entry["thai_spelling"], # ภาษาไทยกลาง
                "transcription": entry["thai_spelling"],      # ทำคอลัมน์มาตรฐานไว้ให้เหมือนอันอื่น
                "sampling_rate": sampling_rate
            })
            
            if i % 500 == 0:
                print(f"บันทึกไปแล้ว {i} ไฟล์...")

        # 4. เซฟเป็น CSV
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False, encoding='utf-8-sig')
        print(f"เสร็จสิ้น! ข้อมูลอยู่ที่: {output_dir}")

# เริ่มรัน (ปกติชุดข้อมูลนี้จะมีแค่ train)
export_isan_dialect_to_local()

'''
import os
import pandas as pd
from datasets import load_dataset
import soundfile as sf

def export_thai_elderly_all():
    dataset_name = "SEACrowd/thai_elderly_speech"
    # รายชื่อ Config ที่ระบบบอกว่ามี (เลือกตัวที่เป็น seacrowd_sptext)
    configs = [
        'thai_elderly_speech_healthcare_seacrowd_sptext',
        'thai_elderly_speech_smarthome_seacrowd_sptext'
    ]
    
    output_dir = "thai_elderly_local/train"
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    metadata = []
    global_count = 0 # ตัวนับรวมสำหรับตั้งชื่อไฟล์ไม่ให้ซ้ำกัน

    for config in configs:
        print(f"กำลังโหลดข้อมูลหมวด: {config}...")
        try:
            # โหลดข้อมูล (ใส่ trust_remote_code=True)
            ds = load_dataset(dataset_name, config, split="train", trust_remote_code=True)
            
            for entry in ds:
                file_name = f"elderly_{global_count}.wav"
                file_path = os.path.join(audio_dir, file_name)
                
                # ดึงเสียงและเซฟ
                audio_array = entry["audio"]["array"]
                sampling_rate = entry["audio"]["sampling_rate"]
                sf.write(file_path, audio_array, sampling_rate)
                
                # เก็บ Metadata (ใช้คอลัมน์ 'text' เพราะเป็น seacrowd schema)
                metadata.append({
                    "file_name": f"audio/{file_name}",
                    "transcription": entry["text"],
                    "category": config.split('_')[3] # เก็บไว้ดูว่าเป็น healthcare หรือ smarthome
                })
                
                global_count += 1
                if global_count % 100 == 0:
                    print(f"บันทึกรวมไปแล้ว {global_count} ไฟล์...")
                    
        except Exception as e:
            print(f"เกิดข้อผิดพลาดที่ {config}: {e}")

    # เซฟ CSV รวมทั้งหมด
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False, encoding='utf-8-sig')
    print(f"เสร็จเรียบร้อย! โหลดมาได้ทั้งหมด {global_count} ไฟล์")
    print(f"ข้อมูลอยู่ที่: {output_dir}")

export_thai_elderly_all()