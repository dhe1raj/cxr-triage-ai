# CXR-Triage-AI 🏥

**A production-leaning chest X-ray triage, report drafting, and human review system.**

## 🚀 Features

- ✅ **AI Triage**: Binary (Normal/Abnormal) + 6 Multi-label findings
- ✅ **Explainability**: Grad-CAM heatmaps overlay
- ✅ **Report Drafting**: Radiology-style report generation
- ✅ **Human Review**: Streamlit UI for doctors to approve/edit/reject
- ✅ **Audit Trail**: Full PostgreSQL logging of every prediction and review
- ✅ **API-first**: FastAPI backend for seamless integration

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.11, PyTorch
- **Model**: DenseNet121 (binary + multi-label)
- **Explainability**: Grad-CAM
- **Database**: PostgreSQL + SQLAlchemy
- **UI**: Streamlit
- **Deployment**: Docker, docker-compose

## 📊 Dataset

- **NIH Chest X-ray**: 112,120 frontal-view images, 14 disease labels
- **V1 Product Labels**: Normal, Cardiomegaly, Effusion, Pneumothorax, Consolidation, Edema

## 🏁 Getting Started

```bash
git clone https://github.com/YourGitHubUsername/cxr-triage-ai.git
cd cxr-triage-ai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements/train.txt
