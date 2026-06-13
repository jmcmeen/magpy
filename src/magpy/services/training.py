"""
Training service -- the seam over bioamla's AST model ops (``models ast``).

Wraps three things into MagPy-owned DTOs:

* :func:`run_training` -> ``bioamla.ml.train_ast`` (fine-tune an AST classifier).
  It is **long-running, heavyweight, and has no progress hook** -- it downloads a
  base model, needs torch (and ideally a GPU), and runs in-process. There is no
  way to cancel it cooperatively (no ``on_progress``), so the UI must run it on a
  busy/indeterminate bar with no Cancel button. TensorBoard (under
  ``{training_dir}/logs``) is its real progress channel.
* :func:`evaluate_model` -> ``bioamla.ml.evaluate_directory`` (accuracy/precision/
  recall/F1 over a labelled directory vs. a ground-truth CSV).
* :func:`predict_audio` -> ``bioamla.ml.predict_file`` (top-k classification of one
  file).

``train_ast`` takes ~28 keyword-only args; MagPy exposes a curated, commonly-tuned
subset via :data:`TRAIN_PARAMS` (reusing the generic :class:`BatchParam` form
descriptor) and lets bioamla default the rest. Param names match ``train_ast``'s
keyword arguments exactly, and :func:`run_training` forwards only the known ones
(defensive against the screen passing extras).

These paths require model downloads / torch, so they are construct- and
wiring-verified, not run, in the headless harness -- the same honesty boundary as
the catalog network paths.
"""

from __future__ import annotations

from dataclasses import dataclass

# bioamla import is confined to the services layer.
from bioamla.ml import (
    evaluate_directory,
    list_models,
    predict_file,
    train_ast,
)

from .batch import BatchParam


@dataclass(frozen=True)
class TrainOutcome:
    model_path: str
    epochs: int
    final_accuracy: float | None
    final_loss: float | None
    message: str = ""


@dataclass(frozen=True)
class EvalOutcome:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_samples: int
    message: str = ""


@dataclass(frozen=True)
class PredictOutcome:
    predicted_label: str
    confidence: float
    top_k: list[tuple[str, float]]
    message: str = ""


# Curated subset of train_ast's keyword args (names match exactly). The rest keep
# bioamla's defaults. finetune_mode 'feature-extraction' freezes the encoder.
TRAIN_PARAMS: tuple[BatchParam, ...] = (
    BatchParam("base_model", "Base model", "str", "MIT/ast-finetuned-audioset-10-10-0.4593",
               help="Pretrained AST checkpoint to fine-tune (HF id)."),
    BatchParam("num_train_epochs", "Epochs", "int", 1, 1, 1000, 1, 0),
    BatchParam("learning_rate", "Learning rate", "float", 5e-05, 0.0, 1.0, 1e-05, 6),
    BatchParam("per_device_train_batch_size", "Batch size", "int", 8, 1, 512, 1, 0),
    BatchParam("category_label_column", "Label column", "str", "category",
               help="CSV/HF column whose unique values are the classes (audiofolders use 'label')."),
    BatchParam("finetune_mode", "Fine-tune mode", "choice", "full",
               choices=("full", "feature-extraction")),
    BatchParam("eval_strategy", "Eval strategy", "choice", "epoch", choices=("epoch", "steps", "no")),
    BatchParam("save_strategy", "Save strategy", "choice", "epoch", choices=("epoch", "steps", "no")),
    BatchParam("fp16", "FP16 (NVIDIA)", "bool", False),
    BatchParam("bf16", "BF16 (Ampere+)", "bool", False),
)

_TRAIN_KEYS = tuple(p.name for p in TRAIN_PARAMS)


def list_available_models() -> list[str]:
    """List the AST models bioamla knows about (for the model picker)."""
    try:
        return list(list_models())
    except Exception:  # noqa: BLE001 - never let a model-registry hiccup break the UI
        return []


def run_training(train_dataset: str, training_dir: str, params: dict) -> TrainOutcome:
    """Fine-tune an AST classifier. Long-running, no progress -- run in a Worker.

    ``train_dataset`` is a HuggingFace dataset id, a metadata CSV, or a directory
    of class-named subdirectories. Only the known :data:`TRAIN_PARAMS` keys are
    forwarded to ``train_ast``.
    """
    kwargs = {k: params[k] for k in _TRAIN_KEYS if k in params}
    # transformers requires eval_strategy == save_strategy (and not "no") when
    # load_best_model_at_end is True (train_ast's default). The form exposes the
    # two strategies independently, so an incompatible combo would raise inside
    # TrainingArguments *after* the heavyweight model download. Disable
    # load-best-at-end for incompatible combos so the run proceeds instead.
    eval_strategy = kwargs.get("eval_strategy", "epoch")
    save_strategy = kwargs.get("save_strategy", "epoch")
    if eval_strategy != save_strategy or eval_strategy == "no":
        kwargs["load_best_model_at_end"] = False
    result = train_ast(train_dataset=train_dataset, training_dir=training_dir, **kwargs)
    return TrainOutcome(
        model_path=getattr(result, "model_path", training_dir),
        epochs=int(getattr(result, "epochs", 0) or 0),
        final_accuracy=getattr(result, "final_accuracy", None),
        final_loss=getattr(result, "final_loss", None),
        message="Training complete.",
    )


def evaluate_model(
    audio_dir: str,
    model_path: str,
    ground_truth_csv: str,
    *,
    file_column: str = "file_name",
    label_column: str = "label",
    use_fp16: bool = False,
) -> EvalOutcome:
    """Evaluate ``model_path`` over ``audio_dir`` against a ground-truth CSV."""
    r = evaluate_directory(
        audio_dir, model_path, ground_truth_csv,
        file_column=file_column, label_column=label_column, use_fp16=use_fp16,
    )
    return EvalOutcome(
        accuracy=float(getattr(r, "accuracy", 0.0) or 0.0),
        precision=float(getattr(r, "precision", 0.0) or 0.0),
        recall=float(getattr(r, "recall", 0.0) or 0.0),
        f1_score=float(getattr(r, "f1_score", 0.0) or 0.0),
        total_samples=int(getattr(r, "total_samples", 0) or 0),
        message="Evaluation complete.",
    )


def predict_audio(filepath: str, model_path: str = "bioamla/scp-frogs") -> PredictOutcome:
    """Top-k classify one audio file with an AST model."""
    r = predict_file(filepath, model_path=model_path)
    labels = list(getattr(r, "top_k_labels", None) or [])
    scores = list(getattr(r, "top_k_scores", None) or [])
    return PredictOutcome(
        predicted_label=str(getattr(r, "predicted_label", "")),
        confidence=float(getattr(r, "confidence", 0.0) or 0.0),
        top_k=list(zip(labels, scores)),
        message="Prediction complete.",
    )
