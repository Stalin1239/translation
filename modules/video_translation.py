import os
import subprocess
import time
from modules.audio_translation import get_whisper_model
from modules.text_translation import translate_text
from config import BASE_DIR, get_aws_client

def get_ffmpeg_path():
    """
    Dynamically searches and returns the absolute path of Gyan.FFmpeg.
    Falls back to 'ffmpeg' if globally available.
    """
    local_appdata = os.getenv('LOCALAPPDATA', '')
    if local_appdata:
        packages_dir = os.path.join(local_appdata, 'Microsoft', 'WinGet', 'Packages')
        if os.path.exists(packages_dir):
            for root, dirs, files in os.walk(packages_dir):
                if 'ffmpeg.exe' in files:
                    return os.path.join(root, 'ffmpeg.exe')
    return 'ffmpeg'

def format_srt_time(seconds):
    """
    Converts decimal seconds into standard SRT timestamp format: HH:MM:SS,mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def extract_audio_from_video(video_path, output_audio_path):
    """
    Uses direct FFmpeg command line to extract audio from video as MP3.
    """
    ffmpeg = get_ffmpeg_path()
    try:
        if not os.path.exists(os.path.dirname(output_audio_path)):
            os.makedirs(os.path.dirname(output_audio_path))
            
        cmd = [
            ffmpeg, "-y",
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            output_audio_path
        ]
        # Hide console window in Windows
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo, check=True)
        return os.path.exists(output_audio_path)
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False

def generate_subtitles(video_path, target_lang='hi'):
    """
    1. Extracts audio from video.
    2. Transcribes with OpenAI Whisper to obtain timestamps.
    3. Translates segments.
    4. Writes to an SRT file.
    5. Burns SRT file into a copy of the video.
    Returns: Dict containing 'srt_path', 'video_path', and list of segments.
    """
    ffmpeg = get_ffmpeg_path()
    temp_dir = os.path.join(BASE_DIR, 'temp')
    outputs_dir = os.path.join(BASE_DIR, 'outputs')
    
    for d in [temp_dir, outputs_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    timestamp = int(time.time())
    audio_path = os.path.join(temp_dir, f"extracted_{timestamp}.mp3")
    srt_path = os.path.join(outputs_dir, f"subtitles_{timestamp}.srt")
    output_video_path = os.path.join(outputs_dir, f"translated_{timestamp}.mp4")

    # Step 1: Extract Audio
    print("Extracting audio from video...")
    success = extract_audio_from_video(video_path, audio_path)
    if not success:
        return {"error": "Failed to extract audio from the video."}

    # Step 2: Transcribe using Whisper
    print("Transcribing audio with Whisper...")
    model = get_whisper_model()
    if not model:
        return {"error": "Failed to load local Whisper transcription engine."}

    try:
        result = model.transcribe(audio_path)
        segments = result.get('segments', [])
    except Exception as e:
        return {"error": f"Transcription error: {str(e)}"}

    # Clean up temp audio
    try:
        os.remove(audio_path)
    except Exception:
        pass

    if not segments:
        return {"error": "No speech detected in the video audio track."}

    # Step 3 & 4: Translate and generate SRT text
    print(f"Translating {len(segments)} segments to {target_lang}...")
    srt_lines = []
    translated_segments = []

    for idx, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        text = seg['text'].strip()
        
        # Translate the segment text
        translated_text, _, _ = translate_text(text, 'auto', target_lang)
        
        # Format times
        start_srt = format_srt_time(start)
        end_srt = format_srt_time(end)
        
        # Compile SRT block
        srt_lines.append(f"{idx + 1}")
        srt_lines.append(f"{start_srt} --> {end_srt}")
        srt_lines.append(f"{translated_text}\n")
        
        translated_segments.append({
            'index': idx + 1,
            'start': start_srt,
            'end': end_srt,
            'original': text,
            'translated': translated_text
        })

    # Save SRT file
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(srt_lines))

    # Step 5: Burn subtitles into video
    print("Burning subtitles into video...")
    try:
        # Proper Windows path escaping for the subtitles filter
        # subtitles='C\:/path/to/sub.srt' is needed by FFmpeg on Windows.
        sub_path_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
        
        cmd = [
            ffmpeg, "-y",
            "-i", video_path,
            "-vf", f"subtitles='{sub_path_escaped}'",
            "-c:a", "copy",  # Keep the same audio track
            output_video_path
        ]
        
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo, check=True)
        
        return {
            'srt_path': srt_path,
            'output_video_path': output_video_path,
            'segments': translated_segments
        }
    except Exception as e:
        print(f"FFmpeg subtitle burning failed: {e}")
        # Return at least the SRT and original video if burning fails
        return {
            'srt_path': srt_path,
            'output_video_path': video_path,  # Fallback to original video
            'segments': translated_segments,
            'burning_warning': "Subtitle burning failed, but translation SRT file was successfully generated."
        }
