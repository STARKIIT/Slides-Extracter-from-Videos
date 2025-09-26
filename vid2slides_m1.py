#!/usr/bin/env python3
"""
vid2slides_m1.py
ARM-native slide extraction pipeline using ffmpeg + OpenCV + pytesseract.

Usage:
    python vid2slides_m1.py input_video.mp4 output.json [--tmp tmpdir] [--thumb_interval 2]

Output:
    JSON file with `sequence` describing slides/speakers and `crop`/pip info.
"""

import argparse
import os
import sys
import glob
import subprocess
import tempfile
import math
import json
import shutil
from datetime import timedelta

from tqdm import tqdm

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# ---- CONFIG ----
DEFAULT_LO_SIZE = (360, 202)
DEFAULT_HI_SIZE = (1280, 720)
DEFAULT_THUMB_INTERVAL = 2  # seconds between thumbnails
HAAR_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"  # keep in repo or will fallback
# ----------------

# ---------- Utilities ----------

def get_video_info(path):
    """Return ffprobe video stream info dictionary for first video stream."""
    # Use ffmpeg.probe
    try:
        import ffmpeg as _ffmpeg
        probe = _ffmpeg.probe(path)
    except Exception:
        # fallback to ffprobe binary
        proc = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", path],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
        probe = json.loads(proc.stdout)
    streams = probe.get("streams", [])
    for s in streams:
        if s.get("codec_type") == "video":
            return s
    raise RuntimeError("No video stream found")

def to_timestamp(seconds):
    # seconds may be float
    td = timedelta(seconds=float(seconds))
    # produce H:MM:SS
    total_seconds = int(td.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# ---------- Frame extraction (ffmpeg-backed) ----------

def extract_thumbnails(video, lo_dir, lo_size=DEFAULT_LO_SIZE, thumb_interval=DEFAULT_THUMB_INTERVAL):
    """
    Uses ffmpeg to extract low-res thumbnails into lo_dir/thumb-0001.jpg etc.
    Wraps ffmpeg and uses tqdm to show progress (by polling output dir).
    """
    os.makedirs(lo_dir, exist_ok=True)

    info = get_video_info(video)
    duration = float(info.get("duration", 0.0))
    total_thumbs = max(1, math.ceil(duration / thumb_interval))

    # ffmpeg fps filter: fps=1/<thumb_interval>
    out_pattern = os.path.join(lo_dir, "thumb-%04d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y", "-i", video,
        "-vf", f"fps=1/{thumb_interval},scale=w={lo_size[0]}:h=-1:force_original_aspect_ratio=decrease",
        "-qscale:v", "3",
        out_pattern
    ]

    # Start ffmpeg
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Poll output dir
    with tqdm(total=total_thumbs, desc="Extracting thumbnails") as pbar:
        prev_count = 0
        try:
            while proc.poll() is None:
                files = len(glob.glob(os.path.join(lo_dir, "thumb-*.jpg")))
                if files > prev_count:
                    pbar.update(files - prev_count)
                    prev_count = files
        except KeyboardInterrupt:
            proc.kill()
            raise
        # final catch-up
        files = len(glob.glob(os.path.join(lo_dir, "thumb-*.jpg")))
        if files > prev_count:
            pbar.update(files - prev_count)

    if proc.returncode not in (0, None):
        out, err = proc.communicate()
        raise RuntimeError(f"ffmpeg returned code {proc.returncode}. stderr:\n{err.decode('utf8', 'ignore')}")

    # ensure thumbnails sorted list exists
    thumbs = sorted(glob.glob(os.path.join(lo_dir, "thumb-*.jpg")))
    return thumbs

def extract_frames_at_times(video, hi_dir, hi_size, times):
    """
    Extract single high-res frames at specified `times` (seconds) into hi_dir/thumb-0001.png etc.
    times: list of floats (seconds)
    """
    os.makedirs(hi_dir, exist_ok=True)
    out_paths = []
    for i, t in enumerate(times):
        out_path = os.path.join(hi_dir, f"thumb-{i+1:04d}.png")
        # ffmpeg best practice: use -ss before -i for fast seek to keyframe; but for exact frame we use -ss then -frames:v 1 -accurate_seek
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-ss", str(t), "-i", video,
            "-frames:v", "1",
            "-vf", f"scale={hi_size[0]}:-1:force_original_aspect_ratio=decrease",
            out_path
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            # try fallback with seeking after input (slower but sometimes more accurate)
            cmd2 = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-i", video, "-ss", str(t),
                "-frames:v", "1",
                "-vf", f"scale={hi_size[0]}:-1:force_original_aspect_ratio=decrease",
                out_path
            ]
            proc2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc2.returncode != 0:
                raise RuntimeError(f"ffmpeg failed extracting frame at {t}s: {proc2.stderr.decode('utf8', 'ignore')}")
        out_paths.append(out_path)
    return out_paths

# ---------- Image / detection helpers ----------
def load_haar_cascade():
    # Prefer local models/ directory
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(repo_dir, "models", "haarcascade_frontalface_default.xml")
    if os.path.exists(local_path):
        cascade_path = local_path
    elif os.path.exists(HAAR_CASCADE_FILENAME):
        cascade_path = HAAR_CASCADE_FILENAME
    else:
        raise FileNotFoundError(
            "Cannot find haarcascade_frontalface_default.xml. "
            "Place it in the 'models/' folder or repo root."
        )
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return cascade

# def load_haar_cascade():
#     # try repo-local first, then cv2 default
#     if os.path.exists(HAAR_CASCADE_FILENAME):
#         cascade_path = HAAR_CASCADE_FILENAME
#     else:
#         # try cv2 default
#         cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
#     if not os.path.exists(cascade_path):
#         raise FileNotFoundError("Haar cascade for face detection not found. Place 'haarcascade_frontalface_default.xml' in repo or install OpenCV data.")
#     cascade = cv2.CascadeClassifier(cascade_path)
#     return cascade

def detect_faces(lo_dir, min_size=(30,30)):
    """
    Runs Haar cascade face detection on each thumbnail in lo_dir.
    Returns dict:
      - has_full_face: boolean array per thumb (True if a reasonably large face present)
      - face_boxes: list of box lists per image
      - pip_location: None (placeholder) or coordinates if PiP detected
    """
    cascade = load_haar_cascade()
    files = sorted(glob.glob(os.path.join(lo_dir, "thumb-*.jpg")))
    has_full_face = []
    face_boxes = []
    pip_location = None
    for fname in tqdm(files, desc="Detecting faces"):
        im = cv2.imread(fname)
        if im is None:
            has_full_face.append(False)
            face_boxes.append([])
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # scale factor and minNeighbors tuned for thumbnails
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=min_size)
        face_boxes.append([tuple(map(int, f)) for f in faces])
        # Decide "full face" if a face covers >6% of area (heuristic)
        h, w = gray.shape[:2]
        found = False
        for (x,y,fw,fh) in faces:
            area = (fw * fh) / float(w * h)
            if area > 0.06:
                found = True
                break
        has_full_face.append(found)
    return {"has_full_face": np.array(has_full_face, dtype=bool), "face_boxes": face_boxes, "pip_location": pip_location, "files": files}

def compute_delta_sse(lo_dir, blur=3):
    """
    Compute per-thumbnail difference energy (SSE) between consecutive thumbnails.
    Returns sse: 1D array length N-1 (difference between i and i+1)
    and gray images list length N for future use.
    """
    files = sorted(glob.glob(os.path.join(lo_dir, "thumb-*.jpg")))
    grays = []
    for f in files:
        im = cv2.imread(f)
        if im is None:
            grays.append(None)
        else:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if blur and blur > 0:
                g = cv2.GaussianBlur(g, (blur, blur), 0)
            grays.append(g)
    sse = []
    for i in range(len(grays)-1):
        a = grays[i]
        b = grays[i+1]
        if a is None or b is None:
            sse.append(0.0)
            continue
        # normalize sizes if different
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a2 = cv2.resize(a, (w,h))
            b2 = cv2.resize(b, (w,h))
        else:
            a2, b2 = a, b
        diff = (a2.astype(np.float32) - b2.astype(np.float32)) ** 2
        sse_value = float(np.mean(diff))
        sse.append(sse_value)
    sse = np.array(sse, dtype=float)
    return sse, files, grays

def simple_change_points_from_sse(sse, multiplier=3.0, min_spacing=3):
    """
    Determine change points given sse array.
    Adaptive threshold = median(sse) * multiplier (but at least small epsilon).
    min_spacing = min number of thumbnails between changes to avoid noisy toggles
    Returns change_indices = list of indexes where a change occurs in terms of thumbnail index
    (index refers to the later frame, i.e., change at i means boundary between i-1 and i)
    """
    if len(sse) == 0:
        return []
    med = np.median(sse)
    thr = max(med * multiplier, 1e-3)
    peaks = np.where(sse > thr)[0] + 1  # change index refers to second frame in pair
    if peaks.size == 0:
        return []
    # enforce min spacing
    cps = []
    last = -999
    for p in peaks:
        if p - last >= min_spacing:
            cps.append(int(p))
            last = p
    return cps

# ---------- Slide segmentation & selection ----------

def build_segments_from_changes(num_thumbs, change_points):
    """
    Given number of thumbnails and change_points (indexes where changes occur),
    produce segments as list of (start_index, end_index) inclusive indices in thumbnail space.
    """
    if num_thumbs == 0:
        return []
    starts = [0] + change_points
    ends = change_points + [num_thumbs]
    segments = []
    for s, e in zip(starts, ends):
        # s .. e-1 are the thumbnails in this segment
        segments.append((int(s), int(e-1)))
    return segments

def choose_representatives(segments):
    """
    For each segment (start,end) choose representative thumbnail index (midpoint).
    Return list of representative indices (0-based).
    """
    reps = []
    for s,e in segments:
        reps.append((s + e) // 2)
    return reps

# ---------- OCR title extraction ----------

def get_slide_title(ocr_data, min_words=1):
    """
    Given pytesseract image_to_data output (DICT), extract a short title.
    Heuristic: choose the top-most text block (smallest top y) that has >= min_words and not just digits.
    Returns a short string (maybe empty).
    """
    if ocr_data is None or 'text' not in ocr_data:
        return ""
    texts = ocr_data['text']
    lefts = ocr_data['left']
    tops = ocr_data['top']
    heights = ocr_data['height']
    confs = ocr_data.get('conf', [None] * len(texts))

    candidates = []
    N = len(texts)
    for i in range(N):
        t = texts[i].strip()
        if len(t) < 2:
            continue
        # ignore mostly punctuation
        if all(ch in ".,:;()-–—/\\'\"[]{}" for ch in t):
            continue
        # prefer words with reasonable confidence if available
        try:
            conf = float(confs[i])
        except Exception:
            conf = None
        candidates.append((tops[i], i, t, conf))
    if not candidates:
        return ""
    # sort by top coordinate (y); choose the top-most cluster of words within small vertical window
    candidates.sort(key=lambda x: x[0])
    # take the top-most one and attempt to collect subsequent words close vertically
    top_y = candidates[0][0]
    collected = [candidates[0][2]]
    for (y,i,t,conf) in candidates[1:]:
        if abs(y - top_y) < max(20, int(0.05 * heights[i] if i < len(heights) else 20)):
            collected.append(t)
        else:
            break
    title = " ".join(collected).strip()
    # basic cleanup: remove odd newlines
    title = " ".join(title.split())
    return title

# ---------- Crop extraction (simple heuristic) ----------

def extract_crop_from_images(example_img_path):
    """
    Heuristic to compute crop (slide bounding box) from one representative image.
    This tries to find the largest contiguous bright rectangle area (slides are often bright)
    Fallback returns the full frame rectangle.
    """
    im = cv2.imread(example_img_path)
    if im is None:
        return None
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # adaptive threshold to find content
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 5)
    # morphological closing to join regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    # find contours
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h,w = gray.shape
        return {"x":0, "y":0, "w":w, "h":h}
    # pick largest contour by area
    areas = [cv2.contourArea(c) for c in cnts]
    max_idx = int(np.argmax(areas))
    x,y,wc,hc = cv2.boundingRect(cnts[max_idx])
    return {"x": int(x), "y": int(y), "w": int(wc), "h": int(hc)}

# ---------- Main pipeline ----------

def extract_keyframes_from_video(video_path, output_json, tmp_path=None,
                                 lo_size=DEFAULT_LO_SIZE, hi_size=DEFAULT_HI_SIZE,
                                 thumb_interval=DEFAULT_THUMB_INTERVAL):
    # Prepare temp dirs
    if tmp_path is None:
        tmp_root = tempfile.mkdtemp(prefix="vid2slides_")
    else:
        tmp_root = tmp_path
        os.makedirs(tmp_root, exist_ok=True)

    lo_dir = os.path.join(tmp_root, "lo")
    hi_dir = os.path.join(tmp_root, "hi")
    os.makedirs(lo_dir, exist_ok=True)
    os.makedirs(hi_dir, exist_ok=True)

    print("Step 1: extracting low-res thumbnails (ffmpeg)...")
    thumbs = extract_thumbnails(video_path, lo_dir, lo_size=lo_size, thumb_interval=thumb_interval)
    num_thumbs = len(thumbs)
    print(f" -> extracted {num_thumbs} thumbnails into {lo_dir}")

    print("Step 2: face detection on thumbnails...")
    face_info = detect_faces(lo_dir)
    has_face = face_info["has_full_face"]

    print("Step 3: compute delta energy between thumbnails...")
    sse, files, _ = compute_delta_sse(lo_dir)
    # sse length = num_thumbs - 1
    # produce single per-thumb SSE aligned to second frame (we will mark change at index i meaning between i-1 and i)
    if len(sse) == 0:
        print("No thumbnails or too few frames; aborting.")
        info_out = {"sequence": [], "crop": None, "pip_location": None}
        with open(output_json, "w") as f:
            json.dump(info_out, f, indent=2)
        return

    print("Step 4: detect change points (slide boundaries)...")
    change_points = simple_change_points_from_sse(sse, multiplier=3.0, min_spacing=3)
    segments = build_segments_from_changes(num_thumbs, change_points)
    reps = choose_representatives(segments)
    print(f" -> {len(segments)} segments detected, {len(reps)} representative frames chosen")

    # For each representative, compute time in seconds (thumb index => time = index * thumb_interval)
    rep_times = [r * thumb_interval for r in reps]

    print("Step 5: extracting high-res representative frames (ffmpeg one-by-one)...")
    # This will create hi_dir/thumb-0001.png ... in same order as reps
    hi_paths = []
    # chunk the times if too many (but we'll iterate)
    for t in tqdm(rep_times, desc="Extracting hi-res frames"):
        idx = rep_times.index(t)
        # use direct extraction
        paths = extract_frames_at_times(video_path, hi_dir, hi_size, [t])
        hi_paths.append(paths[0])

    # Build sequence JSON objects
    sequence = []
    for (seg_idx, (s,e)), rep_idx, hi_path in zip(enumerate(segments), reps, hi_paths):
        # determine start_time and end_time (seconds)
        start_time = to_timestamp(s * thumb_interval)
        end_time = to_timestamp((e+1) * thumb_interval)  # end is inclusive; set end_time to next-second
        # choose offset = rep thumbnail index relative to segment start
        offset = int(rep_idx - s)
        # build slide dict
        slide = {
            "type": "slide",
            "start_time": start_time,
            "end_time": end_time,
            "start_index": int(s),
            "end_index": int(e),
            "offset": offset,
            "source": hi_path,
            "title": ""
        }
        sequence.append(slide)

    # Optionally insert speaker segments between slides if gaps are small or large (simple heuristic)
    # For simplicity, keep sequence as slides only.

    # Compute crop based on first hi image if present
    crop = None
    if hi_paths:
        crop = extract_crop_from_images(hi_paths[0])

    # Step 6: OCR titles (only on slides)
    print("Step 6: OCR on representative frames (pytesseract)...")
    for sl in tqdm(sequence, desc="Running OCR"):
        src = sl.get("source")
        if not src or not os.path.exists(src):
            sl["title"] = ""
            continue
        im = cv2.imread(src)
        if im is None:
            sl["title"] = ""
            continue
        # optionally crop to crop box before OCR
        if crop:
            x,y,wc,hc = crop["x"], crop["y"], crop["w"], crop["h"]
            h_im, w_im = im.shape[:2]
            # clamp
            x = max(0, min(x, w_im-1))
            y = max(0, min(y, h_im-1))
            wc = max(1, min(wc, w_im-x))
            hc = max(1, min(hc, h_im-y))
            roi = im[y:y+hc, x:x+wc]
        else:
            roi = im
        try:
            d = pytesseract.image_to_data(roi, output_type=Output.DICT)
            title = get_slide_title(d)
        except Exception as e:
            # if tesseract fails, set empty
            title = ""
        sl["title"] = title

    # Build final info
    info = {
        "pip_location": face_info.get("pip_location", None),
        "sequence": sequence,
        "crop": crop,
        "thumb_interval": thumb_interval,
        "tmp_root": tmp_root
    }

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(info, f, indent=2)

    print(f"Saved JSON to {output_json}. Temporary files are in {tmp_root}")
    print("You can now run slides2pdf.py / slides2gif.py / slides2chapters.py on the JSON output.")

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Extract slides from a lecture video (ARM-native).")
    p.add_argument("in_path", help="Input video path")
    p.add_argument("out_path", help="Output JSON path")
    p.add_argument("--tmp_path", help="Temporary folder (optional)", default=None)
    p.add_argument("--thumb_interval", type=float, default=DEFAULT_THUMB_INTERVAL,
                   help="Seconds between low-res thumbnails (default 2)")
    p.add_argument("--lo_w", type=int, default=DEFAULT_LO_SIZE[0], help="Low-res thumbnail width")
    p.add_argument("--hi_w", type=int, default=DEFAULT_HI_SIZE[0], help="High-res extraction width")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    lo_size = (args.lo_w, int(round(args.lo_w * DEFAULT_LO_SIZE[1] / DEFAULT_LO_SIZE[0])))
    hi_size = (args.hi_w, int(round(args.hi_w * DEFAULT_HI_SIZE[1] / DEFAULT_HI_SIZE[0])))
    try:
        extract_keyframes_from_video(args.in_path, args.out_path, tmp_path=args.tmp_path,
                                     lo_size=lo_size, hi_size=hi_size, thumb_interval=args.thumb_interval)
    except Exception as e:
        print("ERROR:", str(e))
        raise
