# Model Narrative — Thai Speech Denoiser

ตัวเลขทั้งหมดในส่วนนี้มาจาก `Run/benchmark_outputs/wer/summary_table.txt`
ซึ่งรัน Whisper large-v3 บน test set ขนาดใหญ่กว่าตัวอย่าง 16 คลิปที่อยู่ใน article เดิม
SI-SDRi แยก SNR มาจาก `Run/benchmark_outputs/fast_v2/`
per-dataset มาจาก `Run/listening_test_outputs/comparison_table.txt` (SNR −5 dB)

---

## CRN BiLSTM — โมเดลหลักของเรา

เมื่อดูตัวเลข SI-SDRi รวม CRN ได้ +9.34 dB ซึ่งไม่ใช่อันดับหนึ่งในกลุ่มค่ะ แต่ตัวเลขนั้นไม่ใช่สิ่งที่โมเดลถูกออกแบบมาให้แข่ง use case ของเราคือ LF noise ต่ำกว่า 200 Hz เมื่อวัดเฉพาะ LF noise ผล CRN อยู่ที่ +12.10 dB เพิ่มขึ้นจาก All Noise ราว 3 dB เพราะ dataset ฝึกมี LF noise 70% และ loss function ให้น้ำหนักย่านนั้นเป็นพิเศษ ช่องว่างระหว่าง All Noise กับ LF Noise นี้ไม่ใช่ความบังเอิญ แต่เป็นผลโดยตรงของ data curation และ loss design

สิ่งที่ยืนยันการออกแบบนี้ชัดที่สุดคือผลบน Thai Elderly ซึ่งได้ +15.24 dB ที่ SNR −5 dB ค่านี้สูงกว่า Resemble pretrained ที่ได้เพียง +2.64 dB ในชุดเดียวกัน และสูงกว่า U-Net ที่ +6.06 dB เสียงผู้สูงอายุมีพลังงานต่ำกว่าและรูปแบบสระต่างจาก Standard Thai ชัดเจน การเทรนด้วยชุดข้อมูล Thai Elderly โดยตรงจึงให้ผลที่ pretrained weight ภาษาอังกฤษไม่สามารถให้ได้ ในแง่ WER เมื่อส่งเสียงที่ผ่าน CRN เข้า Whisper large-v3 WER ลดจาก 5.396 เป็น 4.988 บน All Noise และจาก 2.693 เป็น 2.543 บน LF Noise เทียบกับ U-Net ที่ WER เพิ่มขึ้นเป็น 8.064

ข้อจำกัดหลักคือ CRN ไม่ได้ optimize สำหรับ General Noise บน General Noise SI-SDRi อยู่ที่ +8.42 dB ซึ่งต่ำกว่า Resemble ที่ +9.15 dB และห่างจาก MossFormerGAN ที่ +13.94 dB ชัดเจน นี่คือ trade-off ที่ตั้งใจไว้ตั้งแต่ต้น โมเดลที่ดีในทุก noise type จะต้องแลกกับ performance บน noise type เฉพาะทาง และเราเลือกเส้นทางนั้น

---

## U-Net — architecture แรกที่ลอง

U-Net เป็นตัวเลือกแรกของเราด้วยเหตุผลที่ชัดเจนค่ะ โมเดลมีเพียง 2.14M parameters เทียบกับ CRN ที่ 15.9M คือเล็กกว่าประมาณ 7 เท่า ฝึกบน Colab ได้ภายใน 8 GB VRAM และได้ SI-SDRi +6.73 dB All Noise และ +9.46 dB บน LF Noise ถ้า use case คือ edge device ที่ต้องจำกัดขนาดโมเดล หรือต้องการเทรนบนฮาร์ดแวร์ขั้นต่ำ U-Net ยังเป็นจุดเริ่มต้นที่สมเหตุสมผล เพราะขนาดเล็กกว่า 7 เท่าแต่ได้ LF noise +9.46 dB ซึ่งใกล้ CRN กว่าที่ parameter count จะบอก

ปัญหาที่เห็นชัดมาจาก WER ค่ะ เมื่อส่งเสียงผ่าน U-Net แล้วถอดด้วย Whisper large-v3 WER เพิ่มจาก 5.396 เป็น 8.064 บน All Noise และจาก 6.297 เป็น 9.721 บน General Noise ตัวเลขนี้บอกว่า U-Net กด noise ลงได้ แต่พาเสียงพูดบางส่วนออกไปด้วย ผล SI-SDRi จึงไม่สะท้อนความจริงทั้งหมด บน Thai Elderly สิ่งนี้ยิ่งชัด โดยได้เพียง +6.06 dB เทียบกับ CRN ที่ +15.24 dB เพราะ U-Net เป็น encoder-decoder บน STFT ที่ไม่มี temporal modeling ข้ามเฟรม จึงแยกเสียงพูดออกจาก LF noise ที่ทับกันในย่านเดียวกันได้ยาก

เราเรียนรู้จาก U-Net ว่า SI-SDRi กับ WER ไม่เสมอกันเสมอค่ะ โมเดลที่ลด energy ของ noise ได้ดีในเชิงตัวเลข อาจทำร้าย downstream ASR ถ้า suppress เสียงพูดมากเกินไป นี่คือสาเหตุที่เราเปลี่ยนมาใช้ recurrent bottleneck และฝึก CRN BiLSTM แทน ซึ่งให้ WER ดีขึ้นและ Thai Elderly สูงขึ้นกว่า 2 เท่า

---

## Resemble-Enhance — pretrained และ fine-tuned

Resemble pretrained น่าสนใจในแง่ที่หลายคนไม่คาดค่ะ WER วัดได้ดีที่สุดในกลุ่ม บน All Noise WER ลดจาก 5.396 เป็น 2.944 ต่ำกว่า CRN (4.988) และ MossFormerGAN (4.240) แสดงว่า architecture ออกแบบมาให้เสียงออกมาฟังดีและ ASR ตามได้ง่าย ไม่ใช่แค่ maximize energy ratio แต่ SI-SDRi บน Thai Elderly อยู่ที่ +2.64 dB เท่านั้น ต่ำที่สุดในกลุ่ม เพราะ pretrained weight ฝึกกับเสียงภาษาอังกฤษและไม่เคยเห็นรูปแบบเสียงผู้สูงอายุไทย

เราจึงลอง fine-tune บนชุด Thai LF noise ผลบน Thai Elderly ดีขึ้นจาก +2.64 เป็น +5.86 dB ยืนยันว่า FT ช่วยให้โมเดลรู้จักผู้พูดในกลุ่มที่โมเดลเดิมไม่เห็น แต่ผลที่ตามมาคือ SI-SDRi All Noise ที่ SNR +5 dB ลดจาก +9.87 เป็น +4.96 และ General Noise ที่ SNR เดิมลดจาก +9.44 เป็น +4.39 คือโมเดลลืม capability เดิมไปเกือบครึ่งหลัง FT บน domain เล็ก นอกจากนี้ Thai Isan ที่ Resemble pretrained ทำได้ +18.28 dB ก็ลดลงเป็น +12.71 dB หลัง FT แสดงว่า FT ช่วยบาง speaker แต่ทำร้าย speaker อื่นในเวลาเดียวกัน

นี่คือตัวอย่างของ catastrophic forgetting ที่เห็นได้ชัดค่ะ เมื่อ fine-tune โมเดลใหญ่ด้วยข้อมูลน้อยในโดเมนแคบ gradient update จากข้อมูลใหม่จะเขียนทับ weight ที่ encode general knowledge ไว้ วิธีแก้ที่น่าลองคือ freeze encoder บางชั้น หรือใช้ regularization อย่าง Elastic Weight Consolidation เพื่อรักษา capability เดิมไว้ระหว่างการ FT สำหรับงานนี้บทเรียนสำคัญกว่าตัวเลข และมันชัดกว่าการทำ FT แล้วได้ผลดีตั้งแต่ต้น

---

## MossFormerGAN — industry baseline

MossFormerGAN ไม่ใช่โมเดลที่เราเทรน แต่นำมาเป็น reference เพื่อให้เห็นว่า production model ระดับอุตสาหกรรมอยู่ที่ระดับไหนค่ะ โมเดลนี้เทรนบน dataset ขนาดใหญ่กว่าของเรามาก ใช้สถาปัตยกรรมที่ซับซ้อนกว่า และต้องการทรัพยากรเกิน free-tier CPU จึงนำขึ้น Hugging Face Space ไม่ได้ ผลที่ได้คือ +14.56 dB All Noise และ +16.42 dB LF Noise สูงสุดในทุกกลุ่ม รวมถึง Thai Elderly ที่ +21.04 dB ห่างจาก CRN ราว 6 dB

อย่างไรก็ตาม WER ของ MossFormerGAN อยู่ที่ 4.240 บน All Noise ซึ่งดีกว่า CRN (4.988) แต่แย่กว่า Resemble pretrained (2.944) แสดงว่า SI-SDRi สูงสุดไม่ได้การันตี WER ดีที่สุดเสมอ ขึ้นอยู่กับว่าโมเดล suppress เสียงพูดไปด้วยมากแค่ไหนในกระบวนการลด noise สิ่งนี้สะท้อนว่าการเลือกโมเดลควรดู metric หลายตัวพร้อมกัน ไม่ใช่แค่ energy ratio เพียงอย่างเดียว

ตัวเลขที่น่าสนใจที่สุดคือ CRN ของเรา เทรนจากศูนย์ด้วย dataset ไทย 88,110 คลิป ได้ LF Noise +12.10 dB เทียบกับ MossFormerGAN ที่ +16.42 dB ห่างกัน 4.3 dB สำหรับโมเดลที่สร้างด้วยทรัพยากรต่างกันมาก ช่องว่างนี้น้อยกว่าที่คาด และบอกว่า domain-specific training ด้วยข้อมูลที่ตรงกับ use case จริงให้ผลได้ใกล้เคียง production baseline โดยไม่ต้องใช้ทรัพยากรระดับอุตสาหกรรม

---

## ตัวเลขอ้างอิงทั้งหมด

| Model | SI-SDRi All | SI-SDRi LF | SI-SDRi Gen | WER All | WER LF | Thai Elderly |
|---|---|---|---|---|---|---|
| Raw | — | — | — | 5.396 | 2.693 | — |
| CRN | +9.34 | **+12.10** | +8.42 | 4.988 | 2.543 | **+15.24** |
| U-Net | +6.73 | +9.46 | +5.82 | 8.064 | 3.093 | +6.06 |
| Resemble | +8.90 | +8.14 | +9.15 | **2.944** | 2.475 | +2.64 |
| Resemble-FT | +7.83 | +9.43 | +7.30 | 3.644 | **2.267** | +5.86 |
| MossFormerGAN | **+14.56** | **+16.42** | **+13.94** | 4.240 | 2.475 | +21.04 |

*Thai Elderly: SNR −5 dB, source: `Run/listening_test_outputs/comparison_table.txt`*
*WER/SI-SDRi: Whisper large-v3, prefix-tolerant, source: `Run/benchmark_outputs/wer/summary_table.txt`*
*Resemble vs Resemble-FT ที่ SNR +5 dB: source `Run/benchmark_outputs/fast_v2/snr_p5/summary_table.txt`*
