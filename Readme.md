# LatentMatFusion

**LatentMatFusion** is a machine-learning application for materials property prediction based on **VASP CHGCAR** data.

The workflow consists of:
1. **Compression of CHGCAR charge-density data using a Variational Autoencoder (VAE)** to obtain a low-dimensional latent representation.
2. **Prediction of material properties using Fusion Model** trained on the latent features.

---

## Predicted Properties
The model predicts the following material properties:
- **Bulk modulus**
- **Young’s modulus**
- **Shear modulus**
- **Formation energy**
- **Debye temperature**

---

## Project Setup Guide

This repository requires **Python 3.11** and uses a virtual environment for dependency management.



## Installation Steps

### For Linux/Mac:

#### 1. Install Python 3.11 and required system packages
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-tk
```

#### 2. Create and activate a virtual environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Python dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### For Windows:

#### 1. Install Python 3.11
- Download Python 3.11 from [python.org](https://www.python.org/downloads/)
- During installation, **ensure you check "Add Python to PATH"**
- `tkinter` is included by default with Python on Windows

#### 2. Create and activate a virtual environment
```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
```

#### 3. Install Python dependencies
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

### Linux/Mac:
Make sure the virtual environment is activated:
```bash
source .venv/bin/activate
```

Launch the graphical interface:
```bash
python gui.py
```

### Windows:
Make sure the virtual environment is activated:
```cmd
.venv\Scripts\activate
```

Launch the graphical interface:
```cmd
python gui.py
```

---

## Notes
- Always activate the virtual environment before running the application.
- If you encounter `tkinter`-related errors on Linux, ensure `python3.11-tk` is installed.
- Tested on **Ubuntu 20.04+** and **Windows 10/11**.
- CHGCAR files should be generated using standard VASP settings.

---
