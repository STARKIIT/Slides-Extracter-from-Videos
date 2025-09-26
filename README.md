# 📊 vid2slides (M1-Optimized)

`vid2slides` is a tool to **extract slides from lecture or meeting recordings**.  
This version is optimized for **Apple Silicon (M1/M2)** and replaces `decord` with **ffmpeg + OpenCV**.  
It also uses **pytesseract** for OCR and **tqdm** for progress tracking.

---

## 📂 Directory Structure

Your project directory should look like this:

vid2slides-main/
│── demo/ # demo outputs (optional)
│── models/
│ └── haarcascade_frontalface_default.xml # Haar cascade for face detection
│── vid2slides/
│ ├── slides2chapters.py # Convert JSON → YouTube chapters
│ ├── slides2gif.py # Convert JSON → GIF
│ ├── slides2pdf.py # Convert JSON → PDF
│ ├── vid2slides.py # Original (uses decord)
│ └── vid2slides_m1.py # ✅ Updated version for M1 (ffmpeg + OpenCV)
│── environment_m1.yml # Conda environment file for M1
│── my_video.mp4 # Example input video
│── output.json # Example extracted slides (JSON)
│── README.md / come.md # Documentation file (this one)



---

## 🛠 Technologies Used

- **ffmpeg** → Extracts thumbnails & high-resolution frames from video.  
- **OpenCV** → Handles image operations (grayscale, face detection, slide difference, cropping).  
- **Haar Cascade Classifier** → Detects faces to avoid false slide detections.  
- **pytesseract (Tesseract OCR)** → Extracts text (titles) from slides.  
- **tqdm** → Displays progress bars during long operations.  
- **Python 3.10+** → Main scripting language.  

---

## ⚙️ How the Project Works

1. **Thumbnail Extraction (ffmpeg)**  
   - Extracts low-res thumbnails every `thumb_interval` seconds.  
   - Example: 5h lecture → ~10,000 thumbnails.

2. **Face Detection (OpenCV Haar Cascade)**  
   - Detects faces in thumbnails to avoid classifying PiP (Picture-in-Picture) speaker frames as slides.

3. **Slide Change Detection**  
   - Computes frame-to-frame difference (SSE).  
   - Marks change points when the slide content changes significantly.

4. **Segment Building**  
   - Groups frames between change points into slide segments.  
   - Picks one representative frame per segment.

5. **High-Res Frame Extraction (ffmpeg)**  
   - Extracts clean high-resolution frames for the representative slides.

6. **Crop Detection (OpenCV)**  
   - Finds the bounding box of the slide region.  
   - Ensures OCR only processes the actual slide content.

7. **OCR (pytesseract)**  
   - Reads text from slides.  
   - Extracts likely **titles** (usually the top-most text).

8. **JSON Output**  
   - Stores structured data: slide start/end times, titles, image paths, crop info.  
   - Used by converters (`slides2pdf.py`, `slides2gif.py`, `slides2chapters.py`).

---

## 🚀 Installation

1. Clone the repo:
```bash
git clone https://github.com/YOURNAME/vid2slides.git
cd vid2slides-main
Create environment:

bash

conda env create -f environment_m1.yml
conda activate vid2slides-m1
Install tqdm (progress bar):

bash

pip install tqdm
Ensure ffmpeg & Tesseract are installed:

bash

ffmpeg -version
tesseract --version
🛠 Usage
1️⃣ Extract slides to JSON
bash

python vid2slides/vid2slides_m1.py my_video.mp4 output.json
Input: my_video.mp4

Output: output.json with slide data.

2️⃣ Convert JSON → PDF
bash

python vid2slides/slides2pdf.py output.json slides.pdf
Output: slides.pdf (OCR-enabled, searchable).

3️⃣ Convert JSON → GIF
bash

python vid2slides/slides2gif.py output.json slides.gif
Output: slides.gif with slide transitions.

4️⃣ Convert JSON → YouTube Chapters
bash

python vid2slides/slides2chapters.py output.json > chapters.txt
Output: chapters.txt (ready to paste into YouTube description).

📊 Example Outputs
GIF of slides


OCR-enabled PDF
Sample PDF

YouTube Chapters

makefile
Copy code
00:00 Start
00:01:12 Motivation
00:02:16 Inferring Generative Models from Data
00:04:14 Increasing Computational Capacity
...
⏱ Performance
Thumbnail extraction: ~30–40× faster than real-time on M1.

Face detection + slide change detection: a few minutes for ~10k thumbnails.

OCR: ~0.5–1.5s per slide (e.g. 300 slides → ~10–15 min).

Total for a 5h lecture: ~2–4 hours.



---






