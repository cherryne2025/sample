import whisperx
import gc
import os
import pandas as pd # Thêm để xử lý dataframe diarize_segments
from whisperx.diarize import DiarizationPipeline

device = "cpu"
audio_file = "audio.mp3"
batch_size = 1
compute_type = "int8" 
hf_token = "" 

# 1. Transcribe
vad_options = {
    "vad_onset": 0.400,            
    "vad_offset": 0.363,           
    "min_speech_duration_ms": 50,  
    "min_silence_duration_ms": 200 
}

model = whisperx.load_model("large-v3", device, compute_type=compute_type, language="ja", vad_options=vad_options)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size, language="ja", chunk_size=10)

del model
gc.collect()

# 2. Align
model_a, metadata = whisperx.load_align_model(language_code="ja", device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# 3. Diarization
diarize_model = DiarizationPipeline(token=hf_token, device=device)
diarize_segments = diarize_model(audio, min_speakers=3, max_speakers=5)

# ==========================================
# LOGIC MỚI: CHUYỂN SPEAKER_01 -> SPEAKER_00
# ==========================================
def fix_speaker_id(speaker_str):
    if not speaker_str or not isinstance(speaker_str, str):
        return speaker_str
    try:
        # Tách lấy số cuối (ví dụ '01' từ 'SPEAKER_01')
        parts = speaker_str.split("_")
        num = int(parts[-1])
        return f"SPEAKER_{max(0, num - 1):02d}"
    except:
        return speaker_str

# Áp dụng cho bảng diarize_segments (Dành cho file TXT thô)
if 'speaker' in diarize_segments.columns:
    diarize_segments['speaker'] = diarize_segments['speaker'].apply(fix_speaker_id)

# Áp dụng cho kết quả cuối (Dành cho file Kịch bản và SRT)
result = whisperx.assign_word_speakers(diarize_segments, result)

for segment in result["segments"]:
    if "speaker" in segment:
        segment["speaker"] = fix_speaker_id(segment["speaker"])
# ==========================================

# --- XUẤT FILE 1: LƯU ĐOẠN PHÂN TÁCH GIỌNG NÓI THÔ ---
ten_goc = os.path.splitext(os.path.basename(audio_file))[0]
ten_file_diarize = f"diarize_{ten_goc}.txt"
with open(ten_file_diarize, "w", encoding="utf-8") as f:
    f.write(f"--- BẢNG PHÂN TÁCH GIỌNG NÓI GỐC (Đã fix 00): {audio_file} ---\n\n")
    for index, row in diarize_segments.iterrows():
        f.write(f"[{row['start']:.2f}s -> {row['end']:.2f}s] : {row['speaker']}\n")

# --- XUẤT FILE 2: FILE KỊCH BẢN .TXT ---
ten_file_txt = f"kich_ban_{ten_goc}.txt"
with open(ten_file_txt, "w", encoding="utf-8") as f:
    f.write(f"--- KỊCH BẢN CHI TIẾT (Đã fix 00): {audio_file} ---\n\n")
    for segment in result["segments"]:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {speaker}: {text}\n")

# --- XUẤT FILE 3: FILE PHỤ ĐỀ (.SRT) ---
ten_file_srt = f"phu_de_{ten_goc}.srt"
def format_time_srt(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

with open(ten_file_srt, "w", encoding="utf-8") as f:
    for i, segment in enumerate(result["segments"], start=1):
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        f.write(f"{i}\n{format_time_srt(segment['start'])} --> {format_time_srt(segment['end'])}\n[{speaker}] {text}\n\n")

print(f"\n✅ Đã xử lý xong với Speaker bắt đầu từ 00!")