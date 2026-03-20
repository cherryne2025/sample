import os
import gc
import csv
import json
import torch
import pandas as pd
from datetime import timedelta

import whisperx
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import get_writer

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & ĐẦU VÀO (CONFIG)
# ==========================================
# Cấu hình tối ưu cho phần cứng 
DEVICE = "cpu"               
COMPUTE_TYPE = "int8"     
BATCH_SIZE = 4

# Thông tin File & Model
AUDIO_FILE = "audiojp.mp3"
HF_TOKEN = "" 
outdir ="outputs"
raw_outputs = "raw_outputs"
os.makedirs(outdir, exist_ok=True)
os.makedirs(raw_outputs, exist_ok=True)
# MODEL_DIR = os.path.expanduser("~/Documents/AI_models")
# os.environ["HF_HOME"] = MODEL_DIR

# Bộ lọc chống tạp âm & bắt từ ngắn (Aizuchi)
VAD_OPTIONS = {
    "vad_onset": 0.400,            
    "vad_offset": 0.363,           
    "min_speech_duration_ms": 50
}

# ==========================================
# 2. CÁC HÀM XỬ LÝ DỮ LIỆU (UTILITIES)
# ==========================================
def format_time(seconds):
    """Chuyển đổi giây thành định dạng HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def group_segments_by_speaker(segments):
    """Gộp các câu nói liên tiếp của cùng một người thành một khối liền mạch"""
    structured_output = []
    current_speaker = None
    current_start = None
    current_end = None
    current_text = []

    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
        start = segment.get("start", "unknown")
        end = segment.get("end", "unknown")
        text = segment.get("text", "")

        # Nếu đổi người nói -> Lưu dữ liệu của người cũ lại
        if speaker != current_speaker:
            if current_speaker is not None:
                structured_output.append([current_speaker, format_time(current_start), format_time(current_end), " ".join(current_text)])
            
            # Khởi tạo dữ liệu cho người mới
            current_speaker = speaker
            current_start = start
            current_text = [text]
        else:
            # Nếu cùng một người đang nói -> Ghép nối tiếp chữ vào
            current_text.append(text)
        
        current_end = end

    # Lưu đoạn hội thoại cuối cùng tránh bị sót
    if current_speaker is not None:
        structured_output.append([current_speaker, format_time(current_start), format_time(current_end), " ".join(current_text)])
    
    return structured_output

def save_to_csv(data, filename):
    """Xuất mảng dữ liệu ra file CSV chuẩn mực"""
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Speaker", "Start", "End", "Speech"])
        writer.writerows(data)

def save_grouped_txt(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("--- KỊCH BẢN HỘI THOẠI CHUẨN (ĐÃ GỘP CÂU) ---\n\n")
        for row in data:
            speaker = row[0]
            start_time = row[1]
            end_time = row[2]
            text = row[3]
            f.write(f"[{start_time} - {end_time}] {speaker}:\n{text}\n\n")

# ==========================================
# 3. LUỒNG THỰC THI CHÍNH (MAIN WORKFLOW)
# ==========================================
def main():
    print(f"🚀 BẮT ĐẦU XỬ LÝ FILE: {AUDIO_FILE}")
    basename = os.path.splitext(os.path.basename(AUDIO_FILE))[0]

    # --- BƯỚC 1: BÓC BĂNG (TRANSCRIBE) ---
    print("\n⏳ 1. Đang tải model Whisper và bóc băng...")
    model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE, language="ja", vad_options=VAD_OPTIONS)
    audio = whisperx.load_audio(AUDIO_FILE)
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language="ja", chunk_size=10)
    
    del model
    gc.collect()
    torch.mps.empty_cache()

    # --- BƯỚC 2: ĐỒNG BỘ THỜI GIAN (ALIGN) ---
    print("\n⏱️ 2. Đang căn chỉnh thời gian...")
    model_a, metadata = whisperx.load_align_model(language_code="ja", device=DEVICE)
    result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
    result["language"] = "ja"  # Gắn nhãn ngôn ngữ vào kết quả
    
    del model_a
    gc.collect()
    torch.mps.empty_cache()

    # --- BƯỚC 3: PHÂN TÁCH GIỌNG NÓI (DIARIZE) ---
    print("\n🗣️ 3. Đang phân tách người nói...")
    diarize_model = DiarizationPipeline(model_name="pyannote/speaker-diarization-community-1", token=HF_TOKEN, device=DEVICE)
    diarize_segments = diarize_model(audio, min_speakers=3, max_speakers=5)

    # Lưu file log Diarize thô
    file_diarize = os.path.join(raw_outputs,  f"diarize_{basename}.txt")
    with open(file_diarize, "w", encoding="utf-8") as f:
        f.write(f"--- BẢNG PHÂN TÁCH GIỌNG NÓI GỐC: {AUDIO_FILE} ---\n\n")
        for index, row in diarize_segments.iterrows():
            f.write(f"[{row['start']:.2f}s -> {row['end']:.2f}s] : {row['speaker']}\n")
    print(f"✅ Đã xuất log phân tách: {file_diarize}")

    # Gắn nhãn người nói vào kết quả văn bản
    result = whisperx.assign_word_speakers(diarize_segments, result)
# lưu data sau khi đã gán nhãn người nói với định dạng segments có thêm trường "speaker" tất cả định dạng file json, csv, txt đều có thể lưu được
    
    # Khởi tạo công cụ ghi file và trỏ thẳng vào thư mục rawoutput
    writer = get_writer("all", raw_outputs) 
    
    # Các thông số định dạng bắt buộc
    writer_args = {
        "highlight_words": False,
        "max_line_count": None,
        "max_line_width": None
    }
    
    # Thực thi xuất 5 file (srt, vtt, txt, tsv, json) với tên mặc định của file gốc
    writer(result, AUDIO_FILE, writer_args)


    # --- BƯỚC 4: GỘP SPEAKER & XUẤT CSV ---
    print("\n📊 4. Đang gộp câu thoại và xuất dữ liệu CSV...")
    structured_data = group_segments_by_speaker(result["segments"])
    
    ten_file_csv = os.path.join(outdir, f"{basename}_kich_ban.csv")
    save_to_csv(structured_data, ten_file_csv)
    print(f"  ✅ Đã xuất Thành phẩm CSV vào: {ten_file_csv}")
    
    ten_file_txt = os.path.join(outdir, f"{basename}_kich_ban.txt")
    save_grouped_txt(structured_data, ten_file_txt)
    print(f"  ✅ Đã xuất Thành phẩm TXT vào: {ten_file_txt}")
    
    # file_csv = f"output_{basename}.csv"
    # save_to_csv(structured_data, file_csv)
    # print(f"✅ Đã xuất file Kịch bản: {file_csv}")

    print("\n" + "="*60)
    print("🚀 HOÀN TẤT XUẤT SẮC! HÃY KIỂM TRA THƯ MỤC CỦA BẠN!")
    print("="*60)

# Khởi chạy ứng dụng
if __name__ == "__main__":
    main()