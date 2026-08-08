import json
import os
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from torch import nn
from transformers import AutoModel, AutoTokenizer


MODEL_DIR = Path(os.getenv("MODEL_DIR", "/model/intent_classifier"))
DEVICE_SETTING = os.getenv("INTENT_DEVICE", "auto").lower()


def mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return (hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(embeddings)


def resolve_device() -> torch.device:
    if DEVICE_SETTING == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE_SETTING not in {"cpu", "cuda"}:
        raise RuntimeError("INTENT_DEVICE must be auto, cpu, or cuda")
    if DEVICE_SETTING == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("INTENT_DEVICE=cuda, but CUDA is unavailable")
    return torch.device(DEVICE_SETTING)


class IntentRuntime:
    def __init__(self, model_dir: Path):
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(f"Missing model metadata: {metadata_path}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.device = resolve_device()
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir / "tokenizer", local_files_only=True
        )
        self.encoder = AutoModel.from_pretrained(
            model_dir / "encoder", local_files_only=True
        ).to(self.device)
        self.head = ClassificationHead(
            input_dim=int(self.metadata["embedding_dim"]),
            output_dim=len(self.metadata["labels"]),
            hidden_dim=int(self.metadata["hidden_dim"]),
            dropout=float(self.metadata["dropout"]),
        ).to(self.device)
        self.head.load_state_dict(
            torch.load(model_dir / "head.pt", map_location=self.device, weights_only=True)
        )
        self.encoder.eval()
        self.head.eval()

    def predict(self, text: str) -> dict:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=int(self.metadata["max_length"]),
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.lock, torch.inference_mode():
            output = self.encoder(**encoded)
            embedding = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            probabilities = torch.softmax(self.head(embedding), dim=-1).squeeze(0).cpu()

        scores = [
            {"intent": label, "score": float(probabilities[index])}
            for index, label in enumerate(self.metadata["labels"])
        ]
        scores.sort(key=lambda item: item["score"], reverse=True)
        return {
            "intent": scores[0]["intent"],
            "score": scores[0]["score"],
            "predictions": scores,
            "model_version": self.metadata["version"],
        }


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


runtime = IntentRuntime(MODEL_DIR)
app = FastAPI(title="NMA Query Intent Router", version=runtime.metadata["version"])


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_version": runtime.metadata["version"],
        "model_stage": runtime.metadata["stage"],
        "device": str(runtime.device),
    }


@app.get("/metadata")
def metadata() -> dict:
    return {
        key: runtime.metadata[key]
        for key in (
            "package_name",
            "version",
            "stage",
            "task_type",
            "base_model",
            "labels",
            "max_length",
            "evaluation",
        )
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be blank")
    return runtime.predict(text)
