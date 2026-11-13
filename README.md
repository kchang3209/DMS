# DMS Starter

A personal Driver Monitoring System prototype. This milestone adds MediaPipe Face Landmarker to the webcam preview and draws eye and mouth landmarks. The preview now calculates and displays EAR (eye aspect ratio) and MAR (mouth aspect ratio) in real time.

## Requirements

- Python 3.10 or 3.11
- VS Code with the Python extension
- A built-in or USB webcam

## Setup on macOS

Open this folder in VS Code, then choose **Terminal → New Terminal** and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Select `.venv/bin/python` as the VS Code interpreter if needed. macOS may ask you to allow camera access for VS Code or Terminal.

## Run the MediaPipe preview

With the virtual environment active:

```bash
python src/webcam_preview.py
```

The first run downloads Google's Face Landmarker task model into `models/face_landmarker.task`; an internet connection is needed for this download. Later runs use the saved local copy. The preview mirrors the camera image and outlines the eyes and mouth. Press `q` or `Escape` in the preview window to quit.

On this Mac, the built-in camera is index `1`. To select another camera, pass its index:

```bash
python src/webcam_preview.py --camera 0
```

## Setup on Windows (optional)

In PowerShell, run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked, use Command Prompt and run `.venv\Scripts\activate.bat`, or select `.venv\Scripts\python.exe` as the VS Code interpreter. Then run `python src/webcam_preview.py`.

## Project layout

```text
DMS/
├── models/                 # Downloaded task model (ignored by Git)
├── src/
│   └── webcam_preview.py
├── .gitignore
├── README.md
└── requirements.txt
```

This is a learning prototype for webcam experimentation, not a safety system for real driving.


## EAR and MAR indicators

EAR averages both eyes using landmarks (33, 160, 158, 133, 153, 144) and (362, 385, 387, 263, 373, 380). MAR uses inner lip points 13 and 14 divided by mouth width between 78 and 308. These are experimental geometric indicators, not a safety or medical assessment.