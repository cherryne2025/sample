import pandas as pd

def custom_assign_speakers(transcript_result, diarize_df):
    """
    Hàm tự định nghĩa để gán nhãn người nói dựa trên thời gian giao thoa.
    Khắc phục lỗi gán sai khi có người nói chồng lấn.
    
    - transcript_result: Dictionary kết quả từ mô hình Whisper/Align.
    - diarize_df: Pandas DataFrame trả về từ DiarizationPipeline.
    """
    for segment in transcript_result["segments"]:
        seg_start = segment["start"]
        seg_end = segment["end"]
        
        # Dictionary lưu tổng thời gian xuất hiện của từng speaker trong câu này
        speaker_overlaps = {}
        
        # 1. TÌM SPEAKER CHO TOÀN BỘ CÂU (SEGMENT)
        # Duyệt qua các kết quả diarization (thường là pandas DataFrame)
        for _, row in diarize_df.iterrows():
            spk_start = row['start']
            spk_end = row['end']
            speaker_label = row['speaker']
            
            # Tính toán khoảng thời gian giao nhau thực tế
            overlap_start = max(seg_start, spk_start)
            overlap_end = min(seg_end, spk_end)
            overlap_duration = overlap_end - overlap_start
            
            # Nếu có giao thoa (thời gian kết thúc > thời gian bắt đầu)
            if overlap_duration > 0:
                if speaker_label not in speaker_overlaps:
                    speaker_overlaps[speaker_label] = 0
                speaker_overlaps[speaker_label] += overlap_duration
        
        # Gán nhãn cho người có thời gian giao thoa dài nhất
        if speaker_overlaps:
            best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
            segment["speaker"] = best_speaker
            
            # Tính năng nâng cao: Đánh dấu nếu có người nói chen vào (chiếm > 20% thời gian câu)
            total_overlap = sum(speaker_overlaps.values())
            for spk, duration in speaker_overlaps.items():
                if spk != best_speaker and (duration / total_overlap) > 0.2:
                    segment["overlap_warning"] = True # Cảnh báo: Có người nói chồng
        else:
            segment["speaker"] = "UNKNOWN"

        # 2. TÌM SPEAKER CHÍNH XÁC CHO TỪNG TỪ (WORD-LEVEL)
        # Chỉ chạy nếu bạn đã dùng hàm whisperx.align() trước đó
        if "words" in segment:
            for word in segment["words"]:
                # Có những từ Whisper không bắt được timestamp chính xác
                if "start" in word and "end" in word:
                    w_start = word["start"]
                    w_end = word["end"]
                    w_overlaps = {}
                    
                    for _, row in diarize_df.iterrows():
                        overlap_start = max(w_start, row['start'])
                        overlap_end = min(w_end, row['end'])
                        duration = overlap_end - overlap_start
                        
                        if duration > 0:
                            if row['speaker'] not in w_overlaps:
                                w_overlaps[row['speaker']] = 0
                            w_overlaps[row['speaker']] += duration
                    
                    # Gán người nói cho từ
                    if w_overlaps:
                        word["speaker"] = max(w_overlaps, key=w_overlaps.get)
                    else:
                        word["speaker"] = segment["speaker"] # Nếu không tìm thấy, lấy speaker của cả câu

    return transcript_result



# Sử dụng hàm custom để gộp kết quả
    result = custom_assign_speakers(result, diarize_segments)

    # --- In kết quả xem thử có bắt được đoạn Overlap không ---
    print("\n--- KẾT QUẢ ---")
    for segment in result["segments"]:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment['text']
        
        # Nếu có người nói chồng, in thêm cảnh báo
        if segment.get("overlap_warning"):
            print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {speaker} (⚠️ Có tiếng chèn): {text}")
        else:
            print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {speaker}: {text}")