'''
import os
import pandas as pd
from datasets import load_dataset
import soundfile as sf

def export_fleurs_to_local(lang_code="th_th", splits=["train", "test", "validation"]):
    for split in splits:
        print(f"กำลังจัดการชุดข้อมูล: {split}...")
        
        # 1. โหลดข้อมูลจาก Cache ที่คุณมีอยู่แล้ว
        # FLEURS ใช้รหัสภาษาแบบ th_th
        ds = load_dataset("google/fleurs", lang_code, split=split, trust_remote_code=True)
        
        # 2. สร้างโฟลเดอร์สำหรับเก็บไฟล์
        output_dir = f"fleurs_local/{split}"
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        metadata = []

        # 3. วนลูปเพื่อเซฟไฟล์เสียงและเก็บ Label
        for i, entry in enumerate(ds):
            # ตั้งชื่อไฟล์เสียง
            file_name = f"{split}_{i}.wav"
            file_path = os.path.join(audio_dir, file_name)
            
            # เซฟไฟล์เสียงเป็น .wav (FLEURS มักเป็น 16kHz หรือ 44.1kHz)
            audio_array = entry["audio"]["array"]
            sampling_rate = entry["audio"]["sampling_rate"]
            sf.write(file_path, audio_array, sampling_rate)
            
            # เก็บข้อมูล Label และ Metadata อื่นๆ
            metadata.append({
                "file_name": f"audio/{file_name}",
                "transcription": entry["transcription"],
                "raw_transcription": entry["raw_transcription"],
                "gender": entry["gender"],
                "duration": len(audio_array) / sampling_rate
            })
            
            if i % 100 == 0:
                print(f"บันทึกไปแล้ว {i} ไฟล์...")

        # 4. เซฟ Label เป็นไฟล์ CSV
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False, encoding='utf-8-sig')
        print(f"เสร็จสิ้นชุด {split}! ไฟล์ทั้งหมดอยู่ที่: {output_dir}")

# สั่งรัน (เลือกภาษา th_th สำหรับไทย)
export_fleurs_to_local(lang_code="th_th")

'''

import os
import pandas as pd
from datasets import load_dataset
import soundfile as sf

def export_commonvoice_to_local(lang_code="th", splits=["train", "test", "validation"]):
    for split in splits:
        print(f"กำลังจัดการชุดข้อมูล Common Voice: {split}...")
        
        # 1. โหลดข้อมูลจากระบบ (ใช้ไฟล์ .arrow ที่คุณมีอยู่แล้ว)
        ds = load_dataset("fsicoli/common_voice_16_0", lang_code, split=split, trust_remote_code=True)
        
        # 2. สร้างโฟลเดอร์สำหรับเก็บไฟล์
        output_dir = f"common_voice_local/{split}"
        audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        metadata = []

        # 3. วนลูปดึงข้อมูลเสียงและข้อความออกมา
        for i, entry in enumerate(ds):
            # ตั้งชื่อไฟล์เสียง (ใช้ .mp3 ตามต้นฉบับ หรือจะเปลี่ยนเป็น .wav ก็ได้)
            # ในที่นี้ขอเซฟเป็น .wav เพื่อให้เหมือน FLEURS นะครับ
            file_name = f"{split}_{i}.wav"
            file_path = os.path.join(audio_dir, file_name)
            
            # ดึงข้อมูลเสียง
            audio_array = entry["audio"]["array"]
            sampling_rate = entry["audio"]["sampling_rate"]
            
            # เซฟไฟล์เสียง
            sf.write(file_path, audio_array, sampling_rate)
            
            # เก็บข้อมูล Label (ใช้คอลัมน์ 'sentence' สำหรับ Common Voice)
            metadata.append({
                "file_name": f"audio/{file_name}",
                "transcription": entry["sentence"],
                "up_votes": entry.get("up_votes", 0),
                "gender": entry.get("gender", ""),
                "age": entry.get("age", "")
            })
            
            if i % 500 == 0:
                print(f"จัดการไปแล้ว {i} ไฟล์...")

        # 4. เซฟเป็น CSV
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False, encoding='utf-8-sig')
        print(f"เสร็จสิ้น! ข้อมูลอยู่ที่: {output_dir}")

# เริ่มรันสำหรับภาษาไทย ("th")
export_commonvoice_to_local(lang_code="th")