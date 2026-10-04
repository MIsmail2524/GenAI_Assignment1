 # Restoration Suite (Tasks 1–4)

FastAPI + React + ONNX, containerized with Docker Compose.

See [design.md](design.md) for the Google Stitch design brief and instructions
for capturing genuine design evidence for the submission.

## Quick start

### 1. Generate dummy ONNX models (first time only)

    pip install torch onnx
    python scripts/make_dummy_models.py

This creates seven `.onnx` files in `models/`. They output noise but
have the exact same I/O contract as the real models, so you can develop
the entire app before you have trained weights.

### 2. Start the stack

    docker compose up --build

- Frontend:  http://localhost:5173
- Backend:   http://localhost:8000
- API docs:  http://localhost:8000/docs

### 3. Stop

    docker compose down

## Swapping in the real models

When you have the trained `.onnx` files from Colab:

1. Place the real model files in the top-level `models/` directory,
    using the filenames listed below. The backend adapts to the input
    and restored-image output names exposed by these models.
2. Restart:

       docker compose restart backend

## Model contract

| Task | File                             | Inputs                                    | Outputs                             |
|------|----------------------------------|-------------------------------------------|-------------------------------------|
| 1    | `task1.onnx`                    | `corrupted_image` (N,3,128,128)           | `restored_image` (N,3,128,128)      |
| 2    | `classifier.onnx`               | `corrupted_image` (N,3,128,128)           | `logits` (N,4)                      |
| 2    | `specialist_salt_pepper.onnx`   | `corrupted_image` (N,3,128,128)           | `restored_image` (N,3,128,128)      |
| 2    | `specialist_gaussian_blur.onnx` | `corrupted_image` (N,3,128,128)           | `restored_image` (N,3,128,128)      |
| 2    | `specialist_occlusion.onnx`     | `corrupted_image` (N,3,128,128)           | `restored_image` (N,3,128,128)      |
| 3    | `task3.onnx`                    | `corrupted_image` (N,3,128,128)           | `restored_image`, `routing_weights` |
| 4    | `task4.onnx`                    | `photo` (N,3,128,128), `style` (N,) int64 | `sketch` (N,1,128,128)              |

Tasks 1 and 3 each use one model. Task 2 uses its classifier and three
specialist models. Task 4 uses the face-to-sketch model.

## Demonstration and run records

The workspaces support image upload, runtime corruption and severity controls,
restoration/routing results, and PNG downloads. The face-to-sketch workspace
also supports webcam capture and style selection. Successful inference runs
are kept in the browser's local storage (up to 100) and can be exported as
`restoration-inference-runs.json` from the Inference Run Log.

These are application inference records, not training histories or Optuna
trial results. Add the genuine training metrics, study summaries, and trial
records separately if they are part of the assessment submission.

## Endpoints

- `GET  /health`
- `POST /universal-restoration`  (multipart: `file`) → PNG
- `POST /hard-routing`           (multipart: `file`, `force_corruption`, `severity`) → JSON
- `POST /soft-moe`               (multipart: `file`, `force_corruption`, `severity`) → JSON
- `POST /face-to-sketch`         (multipart: `file`, `style`) → PNG