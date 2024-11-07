# Sentinel
Submission for IndiaAI CyberGuard AI Hackathon.

Sentinel
Sentinel is a text classification application designed to analyze text or images, extract key insights, and detect sensitive information. This guide provides steps to clone, set up, and run the project locally.

Prerequisites
Ensure that the following are installed on your machine:

Python 3.7+
Git
Git LFS (for downloading .safetensor files)
1. Clone the Repository
First, clone the repository:
```bash
git clone https://github.com/altf4-games/Sentinel
cd Sentinel
```

2. Pull Model Weights with Git LFS
The model weights are stored using Git LFS. Make sure Git LFS is installed, then run:

```bash
git lfs pull
```
This will download the necessary .safetensor files for model inference.

3. Install Dependencies
Navigate to the api folder and install the required Python packages:
```bash
cd api
pip install uvicorn fastapi requests torch transformers numpy pandas python-multipart
```

4. Run the FastAPI Backend
Once dependencies are installed, start the FastAPI backend using uvicorn:

```bash
python -m uvicorn main:app --reload
```

This will start the server at http://127.0.0.1:8000.
