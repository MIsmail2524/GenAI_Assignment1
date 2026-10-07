# Generative AI Assignment 1 — Restoration Suite (Tasks 1–4)

Four image-generation systems, trained in PyTorch, tuned with **Optuna**, tracked with **MLflow**,
exported to **ONNX**, and served in a single **React + Tailwind CSS** app with a **FastAPI** backend,
started with **Docker Compose**.

| Task | Model | App workspace |
|---|---|---|
| 1 | Universal multi-corruption denoising autoencoder (spatial bottleneck + contrastive degradation encoder, FiLM) | Universal Restoration |
| 2 | 4-class corruption classifier + 3 specialist autoencoders, hard routing with identity bypass | Hard-Routed Restoration |
| 3 | Soft mixture-of-experts (gate + identity + 3 experts), jointly fine-tuned | Soft Mixture-of-Experts Restoration |
| 4 | Style-conditioned conditional GAN (U-Net + FiLM style embedding, PatchGAN) on FS2K | Face-to-Sketch Generator |

The technical report (IEEE format, LaTeX) is in [`report/`](report/); the compiled PDF is [`report/GenAI_Assignment1_Report.pdf`](report/GenAI_Assignment1_Report.pdf).

---

## 1. Run the application (one command)

Requirements: Git, [Git LFS](https://git-lfs.com), Docker with Docker Compose.

```bash
git clone https://github.com/MIsmail2524/GenAI_Assignment1.git
cd GenAI_Assignment1
git lfs pull                 # downloads the trained ONNX models into models/
docker compose up --build
```

- App: http://localhost:5173
- API docs (Swagger): http://localhost:8000/docs
- Health check: http://localhost:8000/health (lists which models are loaded)

Stop with `docker compose down`. You do not need VS Code or any manual Python step.

Optional model check, without Docker: `pip install onnxruntime numpy && python scripts/check_models.py`

### Trained models (`models/`, stored with Git LFS)

| File | Task | Inputs | Outputs |
|---|---|---|---|
| `task1.onnx` | 1 | `corrupted_image` (N,3,128,128) in [0,1] | `restored_image` (N,3,128,128) |
| `classifier.onnx` | 2 | `corrupted_image` | `logits` (N,4): clean, salt_pepper, gaussian_blur, occlusion |
| `specialist_{salt_pepper,gaussian_blur,occlusion}.onnx` | 2 | `corrupted_image` | `restored_image` |
| `task3.onnx` | 3 | `corrupted_image` | `restored_image`, `routing_weights` (N,4) |
| `task4.onnx` | 4 | `photo` (N,3,128,128) in [-1,1], `style_idx` (N,) int64 | `sketch` (N,3,128,128) in [-1,1] |

## 2. API (FastAPI, `backend/`)

| Method | Endpoint | Form fields | Returns |
|---|---|---|---|
| GET | `/health` | – | status and per-model availability |
| POST | `/universal-restoration` | `file`, `force_corruption` (none/salt_pepper/gaussian_blur/occlusion), `severity` (low/medium/high) | corrupted + restored image, settings, inference ms |
| POST | `/hard-routing` | same | 4 class probabilities, predicted class, selected expert, restored image, ms |
| POST | `/soft-moe` | same | 4 routing weights, dominant expert, restored image, ms |
| POST | `/face-to-sketch` | `file`, `style` (0/1/2) | sketch PNG (`X-Inference-Ms` header) |

Uploads are validated: PNG, JPEG, WEBP or BMP only, at most 10 MB, and the file must be a decodable image. Otherwise the API returns HTTP 400 or 413.
Server-side corruptions use the fixed test severities (salt-and-pepper 0.03/0.08/0.15; blur (3,0.7)/(5,1.5)/(7,2.5); occlusion 1/2/3 rectangles ≈10/20/35%).

## 3. Training, evaluation and experiment tracking

All training was done on Google Colab (T4 GPU) with Google Drive for checkpoints.

| Path | Contents |
|---|---|
| `Colab_Files/Task{1..4}_Final.ipynb` | Executed notebooks with all outputs. Each covers data preparation, the runtime corruption pipeline and manifests, model, Optuna study, training, evaluation, figures, ONNX export and verification, and MLflow logging. |
| `training/*.py` | The same code as plain Python scripts, one per task. |
| `configs/*.json` | Search spaces, trial counts and selected hyperparameters for every task. |
| `requirements-training.txt` | Training dependencies. |

To reproduce a task: open the notebook in Colab and run all cells. The notebooks:
- download the datasets (torchvision Oxford-IIIT Pet; FS2K via `gdown`);
- build the seed-42 splits and deterministic manifests;
- resume from Drive checkpoints if they exist, or train otherwise.

The datasets are **not** stored in this repository.

- **Optuna:** Task 1's study is stored as SQLite (`optuna_study.db`). Tasks 2–4 store `*_optuna_all_trials.json` and `*_optuna_best.json`. These sit in each task's Drive `results/` folder.
- **MLflow:** all four tasks log to `/content/drive/MyDrive/mlflow_genai/mlflow.db` (artifacts alongside). The experiments are `Task1-Universal-AE`, `Task2-Hard-Routing`, `Task3-Soft-MoE` and `Task4-Face2Sketch-cGAN`, and they hold parameters, per-epoch losses, Optuna trials (nested runs), test metrics, checkpoints, ONNX files and figures. The last cells of the Task 4 notebook open the MLflow UI from Colab. Locally: `mlflow ui --backend-store-uri sqlite:///<path>/mlflow_genai/mlflow.db`.

### Key results (official test sets)

| | Result |
|---|---|
| Task 2 classifier | accuracy 0.983, macro-F1 0.973 |
| Task 2 hard routing (predicted) | SSIM 0.70 / 0.70–0.76 / 0.61–0.70 on salt-and-pepper / blur / occlusion |
| Task 3 soft MoE | SSIM 0.70–0.74 / 0.72–0.83 / 0.66–0.80 on the same conditions, better than hard routing on every corrupted condition |
| Task 4 cGAN | SSIM 0.464 / 0.391 / 0.620 for Styles 1 / 2 / 3, style match 83% |
| ONNX vs PyTorch | max absolute difference ≤ 2.6e-5 for every model |

See the report for the full tables, figures and the Task 1 checkpoint discussion.

## 4. Repository layout

```
backend/            FastAPI app (routers per task, ONNX Runtime inference, corruptions), Dockerfile
frontend/           React + Vite + Tailwind CSS app (4 workspaces), Dockerfile
models/             trained ONNX models (Git LFS)
Colab_Files/        executed training notebooks (Tasks 1–4)
training/           notebook code as Python scripts
configs/            hyperparameter search spaces and selected configurations
scripts/            check_models.py (ONNX smoke test), make_dummy_models.py (I/O-compatible dummy models)
report/             IEEE LaTeX report (main.tex + figures/)
design.md           Google Stitch design brief
docker-compose.yml  starts backend + frontend
```

## 5. Building the report

Upload the `report/` folder to Overleaf, or compile locally with `pdflatex main.tex` (run it twice). It uses the `IEEEtran` class.
