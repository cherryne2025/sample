import csv
from datetime import timedelta

import whisperx
import gc
import os
from whisperx.diarize import DiarizationPipeline

device = "cpu"
audio_file = "audio.mp3"
batch_size = 1 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
hf_token = "" # HuggingFace token for diarization model (optional, will use a default one if not provided)
# 1. Transcribe with original whisper (batched)
vad_options = {
    "vad_onset": 0.400,            # Mặc định 0.5. Giảm xuống để nhạy hơn với tiếng nói nhỏ.
    "vad_offset": 0.363,           
    "min_speech_duration_ms": 50,  # Mặc định 250ms. Hạ xuống 50ms để bắt được tiếng "Hai", "Un" siêu ngắn.
    "min_silence_duration_ms": 200 # Mặc định 2000ms. Hạ xuống để ngắt câu ngay lập tức khi bị chen ngang.
}

model = whisperx.load_model("large-v3", device, compute_type=compute_type, language="ja", vad_options=vad_options)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)


audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size, language="ja", chunk_size=10)
print(result["segments"]) # before alignment

del model
gc.collect()
# delete model if low on GPU resources
# import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code="ja", device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = DiarizationPipeline(token=hf_token, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio, min_speakers=3, max_speakers=5)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

# --- XUẤT FILE 1: LƯU ĐOẠN PHÂN TÁCH GIỌNG NÓI THÔ ---
# Tự động trích xuất tên gốc của file (ví dụ "test01.mp4" -> "test01")
ten_goc = os.path.splitext(os.path.basename(audio_file))[0]
ten_file_diarize = f"diarize_{ten_goc}.txt"
with open(ten_file_diarize, "w", encoding="utf-8") as f:
    f.write(f"--- BẢNG PHÂN TÁCH GIỌNG NÓI GỐC: {audio_file} ---\n\n")
    # Lặp qua từng hàng trong bảng dữ liệu để lấy thời gian và tên người nói
    for index, row in diarize_segments.iterrows():
        f.write(f"[{row['start']:.2f}s -> {row['end']:.2f}s] : {row['speaker']}\n")
print(f"✅ Đã xuất file Diarize: {ten_file_diarize}")

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs

structured_output = []
current_speaker = None
current_start = None
current_end = None
current_text = []

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

for segment in result["segments"]:

    speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
    start = segment.get("start", "unknown")
    end = segment.get("end", "unknown")
    text = segment.get("text", "")

    if speaker != current_speaker:
        if current_speaker is not None:
            structured_output.append([current_speaker, format_time(current_start), format_time(current_end), " ".join(current_text)])

        current_speaker = speaker
        current_start = start
        current_text = [text]
    else:
        current_text.append(text)

    current_end = end

if current_speaker is not None:
    structured_output.append([current_speaker, format_time(current_start), format_time(current_end), " ".join(current_text)])

    # print(f"{elapsed()} 7. Save to CSV")

with open("final_output.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Speaker", "Start", "End", "Speech"])
    writer.writerows(structured_output)


# ==========================================
# PHẦN 5: XUẤT KẾT QUẢ RA FILE TXT VÀ SRT
# ==========================================
# print("\n🎉 ĐANG XUẤT CÁC FILE KẾT QUẢ CUỐI CÙNG...")

# # --- XUẤT FILE 2: FILE KỊCH BẢN .TXT ĐỌC TRỰC QUAN ---
# ten_file_txt = f"kich_ban_{ten_goc}.txt"
# with open(ten_file_txt, "w", encoding="utf-8") as f:
#     f.write(f"--- KỊCH BẢN CHI TIẾT: {audio_file} ---\n\n")
#     for segment in result["segments"]:
#         speaker = segment.get("speaker", "UNKNOWN")
#         text = segment.get("text", "").strip()
#         start = segment['start']
#         end = segment['end']
        
#         # Ghi theo format: [0.00s - 5.00s] SPEAKER_00: Nội dung...
#         f.write(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}\n")
# print(f"✅ Đã xuất file Kịch bản (TXT): {ten_file_txt}")

# # --- XUẤT FILE 3: FILE PHỤ ĐỀ VIDEO (.SRT) ---
# ten_file_srt = f"phu_de_{ten_goc}.srt"
# def format_time_srt(seconds):
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     secs = int(seconds % 60)
#     msecs = int((seconds - int(seconds)) * 1000)
#     return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"

# with open(ten_file_srt, "w", encoding="utf-8") as f:
#     for i, segment in enumerate(result["segments"], start=1):
#         speaker = segment.get("speaker", "UNKNOWN")
#         text = segment.get("text", "").strip()
#         start_time = format_time_srt(segment['start'])
#         end_time = format_time_srt(segment['end'])
        
#         f.write(f"{i}\n")
#         f.write(f"{start_time} --> {end_time}\n")
#         f.write(f"[{speaker}] {text}\n\n")
# print(f"✅ Đã xuất file Phụ đề (SRT): {ten_file_srt}")

print("\n" + "="*60)
print(f"🚀 HOÀN TẤT XUẤT SẮC! HÃY KIỂM TRA THƯ MỤC CỦA BẠN!")
print("="*60)


