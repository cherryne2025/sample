import argparse
import csv
import gc
import os
from datetime import timedelta

import whisperx
from whisperx.diarize import DiarizationPipeline
import pandas as pd
import json
from copy import deepcopy


def format_time_human(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def format_time_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"


def export_diarize_txt(diarize_df, audio_file, out_prefix):
    ten_file_diarize = f"{out_prefix}_diarize.txt"
    with open(ten_file_diarize, "w", encoding="utf-8") as f:
        f.write(f"--- BẢNG PHÂN TÁCH GIỌNG NÓI GỐC: {audio_file} ---\n\n")
        for index, row in diarize_df.iterrows():
            start_s = format_time_srt(row['start'])
            end_s = format_time_srt(row['end'])
            f.write(f"[{start_s} --> {end_s}] : {row['speaker']}\n")
    print(f"✅ Đã xuất file Diarize: {ten_file_diarize}")


def export_diarize_csv(diarize_df, out_prefix):
    ten_csv = f"{out_prefix}_diarize.csv"
    diarize_df.to_csv(ten_csv, index=False, encoding="utf-8")
    print(f"✅ Đã xuất Diarize CSV: {ten_csv}")


def export_diarize_json(diarize_df, out_prefix):
    ten_json = f"{out_prefix}_diarize.json"
    diarize_df.to_json(ten_json, orient="records", force_ascii=False)
    print(f"✅ Đã xuất Diarize JSON: {ten_json}")


def export_script_txt(segments, out_prefix, audio_file):
    ten_file_txt = f"{out_prefix}_script.txt"
    with open(ten_file_txt, "w", encoding="utf-8") as f:
        f.write(f"--- KỊCH BẢN CHI TIẾT: {audio_file} ---\n\n")
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            start = segment.get('start', 0.0)
            end = segment.get('end', 0.0)
            start_s = format_time_srt(start)
            end_s = format_time_srt(end)
            f.write(f"[{start_s} --> {end_s}] {speaker}: {text}\n")
    print(f"✅ Đã xuất file Kịch bản (TXT): {ten_file_txt}")


def export_srt(segments, out_prefix):
    ten_file_srt = f"{out_prefix}.srt"
    with open(ten_file_srt, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            start_time = format_time_srt(segment.get('start', 0.0))
            end_time = format_time_srt(segment.get('end', 0.0))
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"[{speaker}] {text}\n\n")
    print(f"✅ Đã xuất file Phụ đề (SRT): {ten_file_srt}")


def export_csv_grouped(segments, out_prefix):
    ten_csv = f"{out_prefix}_final_output.csv"
    structured_output = []
    current_speaker = None
    current_start = None
    current_end = None
    current_text = []

    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
        text = segment.get("text", "")

        if speaker != current_speaker:
            if current_speaker is not None:
                structured_output.append([
                    current_speaker,
                    format_time_human(current_start),
                    format_time_human(current_end),
                    " ".join(current_text),
                ])
            current_speaker = speaker
            current_start = start
            current_text = [text]
        else:
            current_text.append(text)

        current_end = end

    if current_speaker is not None:
        # Use SRT-like timestamps for CSV Start/End
        structured_output.append([
            current_speaker,
            format_time_srt(current_start),
            format_time_srt(current_end),
            " ".join(current_text),
        ])

    with open(ten_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Speaker", "Start", "End", "Speech"])
        writer.writerows(structured_output)

    print(f"✅ Đã xuất CSV tổng hợp: {ten_csv}")


def custom_assign_speakers(transcript_result, diarize_df, overlap_warn_threshold=0.2):
    required_cols = {"start", "end", "speaker"}
    if not required_cols.issubset(set(diarize_df.columns)):
        raise ValueError(f"diarize_df phải có cột: {required_cols}")

    for segment in transcript_result.get("segments", []):
        seg_start = segment.get("start", 0.0)
        seg_end = segment.get("end", 0.0)

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


def split_segments_by_diarize(transcript_result, diarize_df):
    """Split transcript segments at diarization boundaries using word timestamps.

    Returns a new list of segments where each piece is assigned to the speaker
    overlapping that time range. Requires word-level timestamps in transcript_result.
    """
    pieces = []
    for seg in transcript_result.get("segments", []):
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        words = seg.get("words", [])
        if not words:
            # fallback: assign whole segment to best diarize speaker
            mask = (diarize_df['end'] > seg_start) & (diarize_df['start'] < seg_end)
            cands = diarize_df.loc[mask]
            speaker = cands.iloc[0]['speaker'] if len(cands) > 0 else seg.get('speaker', 'UNKNOWN')
            pieces.append({"start": seg_start, "end": seg_end, "speaker": speaker, "text": seg.get('text','').strip()})
            continue

        # group words by diarize speaker
        current = None
        for w in words:
            w_start = w.get('start')
            w_end = w.get('end')
            mask = (diarize_df['end'] > w_start) & (diarize_df['start'] < w_end)
            cands = diarize_df.loc[mask]
            speaker = cands.iloc[0]['speaker'] if len(cands) > 0 else seg.get('speaker','UNKNOWN')
            if current is None or current['speaker'] != speaker:
                if current is not None:
                    pieces.append(current)
                current = {"start": w_start, "end": w_end, "speaker": speaker, "text": w.get('text','')}
            else:
                current['end'] = w_end
                current['text'] = current['text'] + ' ' + w.get('text','')

        if current is not None:
            pieces.append(current)

    return pieces


def fill_split_text(pieces, original_result):
    """Fill text for split pieces by collecting overlapping words from original_result."""
    # build list of words with times
    words = []
    for seg in original_result.get('segments', []):
        for w in seg.get('words', []) if isinstance(seg.get('words', []), list) else []:
            if 'start' in w and 'end' in w and 'text' in w:
                words.append(w)

    for p in pieces:
        p_text = []
        p_start = p.get('start', 0.0)
        p_end = p.get('end', 0.0)
        for w in words:
            # include word if it overlaps piece
            if w['end'] > p_start and w['start'] < p_end:
                p_text.append(w['text'])
        p['text'] = ' '.join(p_text).strip()
    return pieces


def assign_transcript_to_diarize(diarize_df, transcript_result):
    """For each diarize interval, collect overlapping words from transcript_result and
    produce a list of segments with start,end,speaker,text matching diarize intervals.
    """
    diarize_segments = []
    # gather all words from transcript_result
    words = []
    for seg in transcript_result.get('segments', []):
        for w in seg.get('words', []) if isinstance(seg.get('words', []), list) else []:
            if 'start' in w and 'end' in w and 'text' in w:
                words.append(w)

    for _, row in diarize_df.iterrows():
        s = row['start']
        e = row['end']
        spk = row['speaker']
        texts = []
        for w in words:
            if w['end'] > s and w['start'] < e:
                texts.append(w['text'])
        text = ' '.join(texts).strip()
        # fallback: if no words found, try using overlapping segments' text
        if not text:
            for seg in transcript_result.get('segments', []):
                if seg.get('end', 0) > s and seg.get('start', 0) < e:
                    text = (text + ' ' + seg.get('text','')).strip()
        diarize_segments.append({'start': s, 'end': e, 'speaker': spk, 'text': text})

    return diarize_segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", "-a", default="audio.mp3", help="Path to audio file")
    parser.add_argument("--language", "-l", default="ja", help="Language code (default: ja)")
    parser.add_argument("--model", default="large-v3", help="WhisperX model (default: large-v3)")
    parser.add_argument("--compute_type", default="int8", help="Compute type (keep int8 per request)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for transcription")
    parser.add_argument("--hf_token", default="", help="HuggingFace token for diarization (optional)")
    parser.add_argument("--min_speakers", type=int, default=3, help="Minimum number of speakers for diarization")
    parser.add_argument("--max_speakers", type=int, default=3, help="Maximum number of speakers for diarization")
    parser.add_argument("--apply_custom_assign", action="store_true", help="Use custom_assign_speakers post-processing")
    parser.add_argument("--compare", action="store_true", help="Run baseline and improved flows and compare results")
    parser.add_argument("--word_timestamps", action="store_true", help="Request word-level timestamps from transcribe (if supported)")
    args = parser.parse_args()

    audio_file = args.audio
    device = "cpu"
    model_name = args.model
    compute_type = args.compute_type
    batch_size = args.batch_size
    language = args.language
    hf_token = args.hf_token

    vad_options = {
        "vad_onset": 0.400,
        "vad_offset": 0.363,
        "min_speech_duration_ms": 50,
        "min_silence_duration_ms": 100,
    }

    print(f"Loading model {model_name} on {device} (compute_type={compute_type})...")
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language, vad_options=vad_options)

    audio = whisperx.load_audio(audio_file)
    def transcribe_with_fallback(model, audio, batch_size, language, chunk_size, word_ts=False):
        # Try common word timestamp args; fallback to no-word timestamps
        try:
            if word_ts:
                try:
                    return model.transcribe(audio, batch_size=batch_size, language=language, chunk_size=chunk_size, word_timestamps=True)
                except TypeError:
                    return model.transcribe(audio, batch_size=batch_size, language=language, chunk_size=chunk_size, return_timestamps="word")
            else:
                return model.transcribe(audio, batch_size=batch_size, language=language, chunk_size=chunk_size)
        except Exception as e:
            print("WARNING: transcribe failed with word timestamps option; retrying without. error:", e)
            return model.transcribe(audio, batch_size=batch_size, language=language, chunk_size=chunk_size)

    print("Transcribing audio (requesting word-level timestamps)...")
    result = transcribe_with_fallback(model, audio, batch_size, language, chunk_size=10, word_ts=True)
    print("Transcription done. Segments before alignment:", len(result.get("segments", [])))

    del model
    gc.collect()

    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    del model_a
    gc.collect()

    print("Running diarization (baseline args)...")
    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    if args.min_speakers is not None or args.max_speakers is not None:
        diarize_segments = diarize_model(audio, min_speakers=args.min_speakers, max_speakers=args.max_speakers)
    else:
        diarize_segments = diarize_model(audio)

    if not isinstance(diarize_segments, pd.DataFrame):
        diarize_segments = pd.DataFrame(diarize_segments)

    out_prefix = os.path.splitext(os.path.basename(audio_file))[0]
    export_diarize_txt(diarize_segments, audio_file, out_prefix)
    export_diarize_csv(diarize_segments, out_prefix)
    export_diarize_json(diarize_segments, out_prefix)

    print("Assigning speakers with WhisperX assign_word_speakers (baseline)...")
    baseline_result = whisperx.assign_word_speakers(diarize_segments, result)
    baseline_segments = baseline_result.get("segments", [])

    # export baseline (WhisperX assignment) outputs
    export_script_txt(baseline_segments, f"{out_prefix}_whisper_assign", audio_file)
    export_srt(baseline_segments, f"{out_prefix}_whisper_assign_subtitles")
    export_csv_grouped(baseline_segments, f"{out_prefix}_whisper_assign")

    # also export split-by-diarize (word-level) to preserve speakers from diarize
    try:
        split_base = split_segments_by_diarize(baseline_result, diarize_segments)
        if split_base:
            split_base = fill_split_text(split_base, baseline_result)
            export_script_txt(split_base, f"{out_prefix}_whisper_assign_split", audio_file)
            export_srt(split_base, f"{out_prefix}_whisper_assign_split_subtitles")
            export_csv_grouped(split_base, f"{out_prefix}_whisper_assign_split")
    except Exception as e:
        print('Warning: split_segments_by_diarize failed:', e)

    # Now run custom assign on a copy and export results
    print("Running custom_assign_speakers (overlap-based)...")
    custom_result = custom_assign_speakers(deepcopy(baseline_result), diarize_segments)
    custom_segments = custom_result.get("segments", [])

    export_script_txt(custom_segments, f"{out_prefix}_custom_assign", audio_file)
    export_srt(custom_segments, f"{out_prefix}_custom_assign_subtitles")
    export_csv_grouped(custom_segments, f"{out_prefix}_custom_assign")

    # also export split-by-diarize for custom assign
    try:
        split_custom = split_segments_by_diarize(custom_result, diarize_segments)
        if split_custom:
            split_custom = fill_split_text(split_custom, custom_result)
            export_script_txt(split_custom, f"{out_prefix}_custom_assign_split", audio_file)
            export_srt(split_custom, f"{out_prefix}_custom_assign_split_subtitles")
            export_csv_grouped(split_custom, f"{out_prefix}_custom_assign_split")
    except Exception as e:
        print('Warning: split_segments_by_diarize failed for custom result:', e)

    # Create final output based on split_custom so final timestamps and speaker count match split
    try:
        if split_custom:
            final_csv = f"{out_prefix}_final_output.csv"
            with open(final_csv, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Speaker", "Start", "End", "Speech"])
                for seg in split_custom:
                    writer.writerow([seg.get('speaker','UNKNOWN'), format_time_srt(seg.get('start',0.0)), format_time_srt(seg.get('end',0.0)), seg.get('text','')])
            speakers = sorted(set([s.get('speaker') for s in split_custom]))
            summary = f"Total speakers: {len(speakers)}\nSpeakers: {', '.join(speakers)}\n"
            with open(f"{out_prefix}_final_summary.txt", 'w', encoding='utf-8') as sf:
                sf.write(summary)
            print(f"✅ Đã ghi final CSV based on split: {final_csv}")
            print(f"✅ Đã ghi summary: {out_prefix}_final_summary.txt")
    except Exception as e:
        print('Warning: could not write final output from split_custom:', e)

    # Additionally, create a diarize-based assigned script for custom result (match split_script)
    try:
        assigned_by_diarize = assign_transcript_to_diarize(diarize_segments, custom_result)
        if assigned_by_diarize:
            # write to the same filename pattern audio_custom_assign_script.txt
            export_script_txt(assigned_by_diarize, f"{out_prefix}_custom_assign", audio_file)
            export_srt(assigned_by_diarize, f"{out_prefix}_custom_assign_subtitles")
            export_csv_grouped(assigned_by_diarize, f"{out_prefix}_custom_assign")
            print(f"✅ Đã ghi diarize-based custom assign script: {out_prefix}_custom_assign_script.txt")
    except Exception as e:
        print('Warning: could not write diarize-based assigned script:', e)

    print("Both WhisperX-assigned and custom-assigned outputs exported.")

    print("\n" + "=" * 60)
    print("🚀 HOÀN TẤT: Kiểm tra thư mục hiện tại để xem các file kết quả.")
    print("=" * 60)


if __name__ == "__main__":
    main()
