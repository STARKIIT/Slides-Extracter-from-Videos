# vid2slides_m1.py
import argparse
import collections
import cv2
import glob
import ffmpeg
import json
from matplotlib import image
import numpy as np
import os
import pytesseract
from pytesseract import Output
import subprocess
import sklearn
import sklearn.cluster
import tempfile
import math
import shutil

# --- keep any helper functions from original vid2slides (viterbi, heuristics, etc) ---
# For brevity I'm assuming you keep all helper functions like
# log_viterbi, heuristic_frames, max_likelihood_sequence, get_delta_images, detect_faces, get_slide_title
# — copy them unchanged from original vid2slides.py
# (They are unchanged; only frame extraction functions below are replaced.)

def get_video_info(path):
    """Return ffprobe video stream info (same as original)."""
    probe = ffmpeg.probe(path)
    for stream in probe["streams"]:
        if stream["codec_type"] == "video":
            return stream
    raise RuntimeError("No video stream found")

def to_timestamp(ts):
    h, m, s = ts // 3600, (ts // 60) % 60, int(ts % 60)
    return f'{int(h):02}:{int(m):02}:{int(s):02}'

# ---------- Replaced functions: use ffmpeg CLI to extract frames ----------

def extract_thumbnails(video, lo_dir, lo_size=(360,202), thumb_interval=2):
    """
    Use ffmpeg to dump low-res thumbnails at interval (seconds).
    Writes files lo_dir/thumb-0001.jpg etc.
    """
    if not os.path.exists(lo_dir):
        os.makedirs(lo_dir, exist_ok=True)

    # fps = 1 / thumb_interval
    fps_float = 1.0 / float(thumb_interval)
    # -vf scale keeps aspect
    # Use -qscale:v 3 for good jpg quality
    cmd = [
        'ffmpeg', '-y', '-i', video,
        '-vf', f"fps={fps_float},scale=w={lo_size[0]}:h=-1:force_original_aspect_ratio=decrease",
        '-qscale:v', '3',
        os.path.join(lo_dir, 'thumb-%04d.jpg')
    ]
    subprocess.run(cmd, check=True)

def extract_single_frame_by_number(video, out_path, frame_number, total_frames, width, height):
    """
    Extract exactly one frame by index using ffmpeg
    frame_number: 0-based index
    total_frames, width, height: from get_video_info()
    We'll compute timestamp = frame_number / framerate
    """
    # Determine framerate from video using ffmpeg.probe duration/nb_frames if present
    info = get_video_info(video)
    duration = float(info.get('duration', 0.0))
    nb_frames = int(info.get('nb_frames', total_frames)) if info.get('nb_frames') else total_frames
    if nb_frames <= 0 and duration > 0:
        # assume 30 fps fallback
        fps = 30.0
    elif duration > 0:
        fps = nb_frames / duration
    else:
        fps = 30.0
    ts = frame_number / fps
    # Use -ss then -frames:v 1 to seek and grab one frame
    cmd = [
        'ffmpeg', '-y', '-ss', str(ts), '-i', video,
        '-frames:v', '1',
        '-s', f'{width}x{height}',
        out_path
    ]
    subprocess.run(cmd, check=True)

def extract_frames(video, hi_dir, hi_size, times):
    """
    For times list (which are offsets computed by the pipeline in units consistent with thumbnails),
    extract high-res frames into hi_dir/thumb-%04d.png (1-based)
    We compute frame numbers using framerate inferred from probe
    """
    if not os.path.exists(hi_dir):
        os.makedirs(hi_dir, exist_ok=True)

    info = get_video_info(video)
    duration = float(info.get('duration', 0.0))
    # Prefer nb_frames field if present
    nb_frames = int(info.get('nb_frames')) if info.get('nb_frames') else None
    # Get coded width/height if available
    w = int(info.get('coded_width', hi_size[0]))
    h = int(info.get('coded_height', hi_size[1]))

    # Estimate framerate
    if nb_frames:
        framerate = float(nb_frames) / duration if duration > 0 else 30.0
    else:
        # try r_frame_rate or avg_frame_rate
        r = info.get('r_frame_rate') or info.get('avg_frame_rate') or '30/1'
        num, den = r.split('/')
        framerate = float(num) / float(den) if float(den) != 0 else 30.0

    # times are indices in the original algorithm measured as "thumb index" (not seconds).
    # The original multiplies time by 2*(index+1) when calling; to be safe, treat 'times' as seconds.
    # If original produced integers representing thumbnail indices, convert them to seconds based on thumbnail spacing.
    # Here we assume times is in seconds; if not, user can adjust 'thumb_interval' earlier.
    for i, t in enumerate(times):
        out_path = os.path.join(hi_dir, f'thumb-{i+1:04}.png')
        # if t looks like an integer less than, say, 10000, treat as seconds — original uses small numbers.
        # Use ffmpeg to extract at timestamp t seconds
        cmd = [
            'ffmpeg', '-y', '-ss', str(t), '-i', video,
            '-frames:v', '1',
            '-vf', f"scale={hi_size[0]}:-1:force_original_aspect_ratio=decrease",
            out_path
        ]
        subprocess.run(cmd, check=True)

# ---------- End frame extraction replacements ----------

# The rest of the code: call the same pipeline as original:
# implement extract_keyframes_from_video similarly to original but rely on new extract_thumbnails/extract_frames

def extract_keyframes_from_video(target, output_json, thumb_dir=None):
    lo_size = (360, 202)
    hi_size = (1280, 720)  # you can set 1920x1080 but keep 1280 for speed on some videos

    if thumb_dir is None:
        thumb_dir = tempfile.mkdtemp()
    if not os.path.exists(thumb_dir):
        os.makedirs(thumb_dir, exist_ok=True)

    lo_dir = os.path.join(thumb_dir, 'lo')
    hi_dir = os.path.join(thumb_dir, 'hi')
    os.makedirs(lo_dir, exist_ok=True)
    os.makedirs(hi_dir, exist_ok=True)

    # 1) Low-res thumbnails (every 2 seconds)
    extract_thumbnails(target, lo_dir, lo_size, thumb_interval=2)

    # 2) face detection & candidate selection (reuse original detect_faces/get_delta_images)
    face_information = detect_faces(lo_dir)   # implement or reuse existing function
    _, candidates, sse = get_delta_images(lo_dir, face_information['has_full_face'])

    to_select = np.where(~face_information['has_full_face'])[0]
    sequence = max_likelihood_sequence(sse[to_select, :])
    full_sequence = -np.ones(sse.shape[0])
    full_sequence[to_select] = sequence

    # Build slide JSON (same logic as original - copy/paste this section from original vid2slides.py)
    last_num = -2
    latest_slide = {'start_index': 0}
    slides = []
    for i, num in enumerate(full_sequence):
        if num != last_num:
            latest_slide['end_time'] = to_timestamp(2 * (i + 1))
            latest_slide['end_index'] = i
            offset = candidates[int(last_num)] if int(last_num) >= 0 and int(last_num) < len(candidates) else 0
            latest_slide['offset'] = int(offset)
            latest_slide['source'] = os.path.join(hi_dir, f'thumb-{offset+1:04}.png')
            if num == -1:
                latest_slide = {
                    'type': 'speaker',
                    'start_time': to_timestamp(2 * (i + 1)),
                    'start_index': i,
                }
            else:
                latest_slide = {
                    'type': 'slide',
                    'start_time': to_timestamp(2 * (i + 1)),
                    'start_index': i
                }
            last_num = num

    latest_slide['end_time'] = to_timestamp(2 * (i + 1))
    latest_slide['end_index'] = i
    offset = candidates[int(last_num)] if int(last_num) >= 0 and int(last_num) < len(candidates) else 0
    latest_slide['offset'] = int(offset)
    latest_slide['source'] = os.path.join(hi_dir, f'thumb-{offset+1:04}.png')
    slides.append(latest_slide)
    slides = slides[1:]

    # collect offsets
    offsets = [slide['offset'] for slide in slides if slide['type'] == 'slide']

    # 3) Extract hi-res frames for the offsets (convert offsets to times if needed)
    # Here offsets are indices into candidate list (i.e., thumbnail counts). Thumbnail interval is 2s above,
    # so thumbnail index i corresponds to time = i * thumb_interval seconds. We'll generate times accordingly:
    thumb_interval = 2
    times_in_seconds = [int(o) * thumb_interval for o in offsets]
    extract_frames(target, hi_dir, hi_size, times_in_seconds)

    # 4) build info and OCR titles
    info = {'pip_location': face_information.get('pip_location', None),
            'sequence': slides}
    info['crop'] = extract_crop(info)   # reuse original

    print(f"Found {len(slides)} canonical slides")
    for el in info['sequence']:
        if el['type'] == 'slide':
            # read generated hi-res image path (we created them as thumb-<n>.png)
            src = el.get('source')
            # if source points to a file not created, adjust to hi_dir/thumb-<index>.png
            if not os.path.exists(src):
                # attempt to remap using offset index
                idx = el.get('offset', 0)
                candidate_path = os.path.join(hi_dir, f'thumb-{idx+1:04}.png')
                if os.path.exists(candidate_path):
                    src = candidate_path
                    el['source'] = src
            im = cv2.imread(el['source'])
            if im is None:
                el['title'] = ""
                continue
            d = pytesseract.image_to_data(im, output_type=Output.DICT)
            el['title'] = get_slide_title(d)

    with open(output_json, 'w') as f:
        json.dump(info, f, indent=2)

if __name__ == "__main__":
    desc = "Extract key slides from video (arm-native version)."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("in_path", help='Input video path')
    parser.add_argument("out_path", help='Output json path')
    parser.add_argument("--tmp_path", help='Temporary path for thumbnails', default=None)
    args = parser.parse_args()
    extract_keyframes_from_video(args.in_path, args.out_path, args.tmp_path)
