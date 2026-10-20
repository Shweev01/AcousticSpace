# AcousticSpace
### Deepfake Audio Detection Using Room Impulse Response (RIR)

## 1. Project Overview

AcousticSpace is a project focused on exploring whether room acoustics and audio characteristics can help distinguish genuine speech from AI-generated or manipulated speech.

The first phase focuses only on setting up the project folder structure so that all four team members can work in an organized way.

## 2. Initial Objectives

- Create the main project folder.
- Create separate folders for the frontend, backend, machine learning, and audio processing.
- Add folders for datasets, documentation, and testing.
- Prepare the repository for collaborative development using Git and GitHub.

## 3. Project Folder Structure

```text
AcousticSpace/
├── README.md
├── .gitignore
├── docs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── rir/
├── ml/
├── audio_processing/
├── backend/
├── frontend/
└── tests/
```

### Folder Descriptions

| Folder | Purpose |
|---|---|
| `docs/` | Project planning, diagrams, and documentation |
| `data/raw/` | Original datasets and audio files |
| `data/processed/` | Preprocessed audio and generated features |
| `data/rir/` | Room impulse response datasets |
| `ml/` | Machine learning models and training code |
| `audio_processing/` | Audio preprocessing and acoustic feature extraction |
| `backend/` | API and application integration |
| `frontend/` | User interface and dashboard |
| `tests/` | Unit and integration tests |

**Note:** Large datasets, audio files, and trained model weights should not be committed directly to GitHub.

## 4. Team Responsibilities

- **Member 1 — Machine Learning:** Responsible for datasets, model development, training, and evaluation.
- **Member 2 — Audio Processing:** Responsible for audio preprocessing, acoustic features, and RIR-related experiments.
- **Member 3 — Backend:** Responsible for the API and integration of the audio-processing and ML modules.
- **Member 4 — Frontend:** Responsible for the upload interface, waveform visualization, and results dashboard.

During this phase, all members help establish the folder structure and agree on the project layout.

## 5. Tools

- **Git and GitHub:** Version control and team collaboration.
- **Python:** Planned language for audio processing, ML, and backend development.
- **React and TypeScript:** Planned frontend technologies.
- **VS Code:** Recommended development editor.

These tools can be installed and configured in later setup steps.

## 6. First-Phase Checklist

- [ ] Create the `AcousticSpace` project folder.
- [ ] Create the subfolders listed above.
- [ ] Add this `README.md` file.
- [ ] Add a `.gitignore` file.
- [ ] Initialize a Git repository.
- [ ] Create a GitHub repository and push the initial structure.
- [ ] Confirm that all four team members can access the repository.

## 7. Next Phase

Once the initial structure is ready, the team will define module interfaces, prepare the development environments, inspect suitable datasets, and implement the first audio-processing and API prototypes.

**Important:** This repository is currently in the initial setup phase. The folder structure does not yet indicate that the detection model or application has been implemented.

---

*Project: AcousticSpace — Deepfake Detection via Room Impulse Response (RIR)*