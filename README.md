This repository contains a small sensor-fusion demo and supporting code for perception and tracking on an autonomous surface vessel. The project is written in Python and includes synthetic data generation, a simple detector, a Kalman tracker, and a basic fusion component.

Table of contents
- Project overview
- Repository layout
- Quick start (setup & run)
- Running the synthetic demo
- Development notes

Project overview
----------------
The code here is intended as a compact demo for experimentation with sensor fusion, detection, and tracking. It provides:

- Synthetic dataset generation (in `src/datasets/synthetic.py`).
- A detector interface and a toy detector implementation (`src/perception/detector.py`).
- A Kalman filter based tracker (`src/tracking/kalman.py`).
- A fusion manager / fuser module (`src/fusion/fuser.py`).
- Utility scripts for metrics and visualization (`src/utils/metrics.py`, `src/utils/vis.py`).
- A small script to run a synthetic demo: `scripts/run_synthetic_demo.py`.

Repository layout
-----------------

Top-level files
- `README.md` - (this file)
- `requirements.txt` - Python dependencies
- `scripts/` - runnable demo scripts
- `src/` - package sources (datasets, perception, fusion, tracking, utils)

Quick start (setup & run)
-------------------------
Recommended: use a virtual environment. These commands are written for PowerShell on Windows.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the synthetic demo:

```powershell
python .\scripts\run_synthetic_demo.py
```

If you run into permission errors when activating the venv on Windows, you may need to set the execution policy for the session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Running the synthetic demo
--------------------------
The `scripts/run_synthetic_demo.py` script creates synthetic data and exercises the detector, tracker, and fusion pipeline. Open the script to configure parameters (number of timesteps, noise levels, visualization options).

Development notes
-----------------
- Code is organized for clarity rather than production performance.
- Add tests under a `tests/` folder and run via `pytest` if you want test coverage.
- Keep secrets (API keys, tokens) out of the repo. Add any credentials to local environment variables or an ignored `.env` file.


