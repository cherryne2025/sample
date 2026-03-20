import os
import gc
import csv
import torch
from datetime import timedelta

import whisperx
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import get_writer

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & ĐẦU VÀO
# ==========================================
DEVICE = "mps"               
COMPUTE_TYPE = "float16"     
BATCH_SIZE = 4

AUDIO_FILE = "test01.mp4"         # Tên file của bạn
HF_TOKEN = "MÃ_TOKEN_CỦA_BẠN"     # Token Hugging Face
MODEL_DIR = os.path.expanduser("~/Documents/AI_models")
os.environ["HF_HOME"] = MODEL_DIR

VAD_OPTIONS = {
    "vad_onset": 0.400,            
    "vad_offset": 0.363,           
    "min_speech_duration_ms": 50,  
    "min_silence_duration_ms": 200 
}

# ==========================================
# 2. CÁC HÀM XỬ LÝ (CUSTOM FUNCTIONS)
# ==========================================
def format_time_csv(seconds):
    return str(timedelta(seconds=int(seconds)))

def group_segments_by_speaker(segments):
    structured_output = []
    current_speaker = None
    current_start = None
    current_end = None
    current_text = []

    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "").strip()

        if speaker != current_speaker:
            if current_speaker is not None:
                structured_output.append([current_speaker, format_time_csv(current_start), format_time_csv(current_end), " ".join(current_text)])
            current_speaker = speaker
            current_start = start
            current_text = [text]
        else:
            current_text.append(text)
        current_end = end

    if current_speaker is not None:
        structured_output.append([current_speaker, format_time_csv(current_start), format_time_csv(current_end), " ".join(current_text)])
    return structured_output

def save_to_csv(data, filename):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Speaker", "Start", "End", "Speech"])
        writer.writerows(data)

def save_grouped_txt(data, filename):
    """Xuất file TXT kịch bản siêu sạch, có dán mốc thời gian"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("--- KỊCH BẢN HỘI THOẠI CHUẨN (ĐÃ GỘP CÂU) ---\n\n")
        for row in data:
            speaker = row[0]
            start_time = row[1]
            end_time = row[2]
            text = row[3]
            # Ghi theo format: [0:00:00 - 0:00:15] SPEAKER_00: 
            f.write(f"[{start_time} - {end_time}] {speaker}:\n{text}\n\n")

# ==========================================
# 3. LUỒNG THỰC THI CHÍNH
# ==========================================
def main():
    print(f"🚀 BẮT ĐẦU XỬ LÝ: {AUDIO_FILE}")
    ten_goc = os.path.splitext(os.path.basename(AUDIO_FILE))[0]

    # --- BÓC BĂNG ---
    print("\n⏳ 1. Bóc băng...")
    model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE, language="ja", vad_options=VAD_OPTIONS, download_root=MODEL_DIR)
    audio = whisperx.load_audio(AUDIO_FILE)
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language="ja", chunk_size=10)
    del model; gc.collect(); torch.mps.empty_cache()

    # --- ĐỒNG BỘ THỜI GIAN ---
    print("\n⏱️ 2. Căn chỉnh thời gian...")
    model_a, metadata = whisperx.load_align_model(language_code="ja", device=DEVICE, model_dir=MODEL_DIR)
    result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
    del model_a; gc.collect(); torch.mps.empty_cache()

    # --- PHÂN TÁCH GIỌNG NÓI ---
    print("\n🗣️ 3. Phân tách người nói...")
    diarize_model = DiarizationPipeline(model_name="pyannote/speaker-diarization-community-1", use_auth_token=HF_TOKEN, device=DEVICE)
    diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=5)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # ==========================================
    # 4. XUẤT HÀNG LOẠT FILE VÀO RAWOUTPUT
    # ==========================================
    thu_muc_xuat = "rawoutput"
    os.makedirs(thu_muc_xuat, exist_ok=True)
    print(f"\n🎉 4. ĐANG XUẤT 7 FILE VÀO THƯ MỤC '{thu_muc_xuat}'...")
    
    # 4.1. Xuất 5 file thô mặc định
    writer = get_writer("all", thu_muc_xuat) 
    writer_args = {"highlight_words": False, "max_line_count": None, "max_line_width": None}
    writer(result, AUDIO_FILE, writer_args)
    
    # 4.2. Xuất CSV và TXT Gộp câu (Custom)
    structured_data = group_segments_by_speaker(result["segments"])
    
    ten_file_csv = os.path.join(thu_muc_xuat, f"{ten_goc}_gop_cau.csv")
    save_to_csv(structured_data, ten_file_csv)
    
    ten_file_txt_gop = os.path.join(thu_muc_xuat, f"{ten_goc}_gop_cau.txt")
    save_grouped_txt(structured_data, ten_file_txt_gop)

    print(f"\n🚀 HOÀN TẤT XUẤT SẮC! Mời bạn mở thư mục '{thu_muc_xuat}' kiểm tra nhé!")

if __name__ == "__main__":
    main()