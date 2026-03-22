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
        import pandas as pd


        def custom_assign_speakers(transcript_result, diarize_df, overlap_warn_threshold=0.2):
            """
            Gán speaker cho segment và trên cấp từ dựa trên thời gian giao thoa.

            - transcript_result: dict với key "segments" (mỗi segment có 'start','end','text' và optional 'words')
            - diarize_df: pandas.DataFrame với cột 'start','end','speaker'
            - overlap_warn_threshold: ngưỡng (0-1) để đánh dấu cảnh báo overlap
            """
            required_cols = {"start", "end", "speaker"}
            if not required_cols.issubset(set(diarize_df.columns)):
                raise ValueError(f"diarize_df phải có cột: {required_cols}")

            for segment in transcript_result.get("segments", []):
                seg_start = segment.get("start", 0.0)
                seg_end = segment.get("end", 0.0)

                # Lọc nhanh các interval diarization có khả năng giao thoa với segment
                mask = (diarize_df['end'] > seg_start) & (diarize_df['start'] < seg_end)
                candidates = diarize_df.loc[mask]

                speaker_overlaps = {}
                for _, row in candidates.iterrows():
                    overlap_start = max(seg_start, row['start'])
                    overlap_end = min(seg_end, row['end'])
                    overlap_duration = overlap_end - overlap_start
                    if overlap_duration > 0:
                        speaker_overlaps[row['speaker']] = speaker_overlaps.get(row['speaker'], 0.0) + overlap_duration

                if speaker_overlaps:
                    best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
                    segment["speaker"] = best_speaker

                    total_overlap = sum(speaker_overlaps.values())
                    for spk, dur in speaker_overlaps.items():
                        if spk != best_speaker and (dur / total_overlap) > overlap_warn_threshold:
                            segment["overlap_warning"] = True
                            break
                else:
                    segment["speaker"] = "UNKNOWN"

                # Word-level assignment (nếu có)
                if "words" in segment and isinstance(segment["words"], list):
                    for word in segment["words"]:
                        if "start" in word and "end" in word:
                            w_start = word["start"]
                            w_end = word["end"]

                            w_mask = (diarize_df['end'] > w_start) & (diarize_df['start'] < w_end)
                            w_cands = diarize_df.loc[w_mask]
                            w_overlaps = {}
                            for _, row in w_cands.iterrows():
                                o_start = max(w_start, row['start'])
                                o_end = min(w_end, row['end'])
                                dur = o_end - o_start
                                if dur > 0:
                                    w_overlaps[row['speaker']] = w_overlaps.get(row['speaker'], 0.0) + dur

                            if w_overlaps:
                                word["speaker"] = max(w_overlaps, key=w_overlaps.get)
                            else:
                                word["speaker"] = segment.get("speaker", "UNKNOWN")

            return transcript_result


        if __name__ == "__main__":
            # Ví dụ mẫu để kiểm thử nhanh
            sample_result = {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.5,
                        "text": "Hello there",
                        "words": [
                            {"start": 0.0, "end": 0.5, "text": "Hello"},
                            {"start": 0.6, "end": 1.0, "text": "there"}
                        ]
                    },
                    {
                        "start": 2.5,
                        "end": 5.0,
                        "text": "How are you",
                        "words": [
                            {"start": 2.6, "end": 3.0, "text": "How"},
                            {"start": 3.1, "end": 4.5, "text": "are you"}
                        ]
                    }
                ]
            }

            diarize_segments = pd.DataFrame([
                {"start": 0.0, "end": 1.0, "speaker": "spk0"},
                {"start": 0.9, "end": 3.0, "speaker": "spk1"},
                {"start": 3.0, "end": 5.0, "speaker": "spk0"},
            ])

            merged = custom_assign_speakers(sample_result, diarize_segments)

            print("\n--- KẾT QUẢ ---")
            for seg in merged["segments"]:
                sp = seg.get("speaker", "UNKNOWN")
                txt = seg.get("text", "")
                if seg.get("overlap_warning"):
                    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {sp} (⚠️ overlap): {txt}")
                else:
                    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {sp}: {txt}")
                for w in seg.get("words", []):
                    print(f"  - {w.get('text','')} ({w.get('start')}-{w.get('end')}) -> {w.get('speaker')}")