# CLAUDE.md — Dialogue Summarization LLM Fine-Tuning

## What this project is

A fine-tuned small language model for dialogue summarization, trained on [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) — a public dataset of messenger-style conversations paired with human-written summaries. The task: given a short multi-turn conversation, generate a concise summary of what was discussed and agreed upon.

The model is fine-tuned with LoRA (via PEFT) on top of Phi-3-mini, tracked with MLflow, and published to HuggingFace Hub. A Gradio demo is deployed on HuggingFace Spaces.

## Tech stack

| Layer | Tool |
|-------|------|
| Base model | `microsoft/Phi-3-mini-4k-instruct` (3.8B) |
| Fine-tuning | LoRA via `peft` |
| Training | HuggingFace `transformers` + `Trainer` API |
| Dataset | `knkarthick/dialogsum` on HF Hub |
| Evaluation | ROUGE-1 / ROUGE-2 / ROUGE-L |
| Experiment tracking | MLflow |
| Model hosting | HuggingFace Hub (`rotemso23/dialogsum-phi3-lora`) |
| Demo | Gradio on HuggingFace Spaces |

## Project structure

```
dialogue-summarizer/
├── src/
│   ├── data.py        — dataset loading, formatting, tokenization
│   ├── model.py       — base model + LoRA config setup
│   ├── train.py       — Trainer setup, MLflow logging, training loop
│   ├── evaluate.py    — ROUGE scores on test split, baseline vs. fine-tuned
│   └── infer.py       — single-conversation summarization (used by app.py)
├── app.py             — Gradio demo UI (entry point for HF Spaces)
├── requirements.txt
├── .env.example       — HF_TOKEN placeholder
└── notebooks/
    ├── train_colab.ipynb
    └── evaluate_colab.ipynb
```

## Dataset: DialogSum

- Load via: `datasets.load_dataset("knkarthick/dialogsum")`
- Splits: train (12,460 → subsampled to 4,000, seed=42) / validation (500) / test (1,500)
- Fields: `dialogue` (multi-turn conversation) and `summary` (one short paragraph)

## Prompt format

```
<|user|>
Summarize the following conversation in a few sentences.

Conversation:
{dialogue}
<|end|>
<|assistant|>
{summary}
<|end|>
```

Use `tokenizer.apply_chat_template(messages, tokenize=False)` — do not build the string by hand.
Loss is computed only on the assistant turn; prompt tokens are masked with `-100`.

## Key conventions

- All secrets via environment variables (`.env` locally, Space secrets on HF). Never hardcoded.
- `mlflow_runs/` in `.gitignore`
- Type hints on all functions
- Each module has an `if __name__ == "__main__"` block for standalone testing
- Use `transformers>=4.40,<5.0` — Phi-3 support added in 4.40; transformers 5.x has breaking API changes

## Compute notes

- Training requires a GPU — use Google Colab (free T4)
- ~1–2 hours for 3 epochs on 4k examples on T4
- If T4 OOMs: reduce `per_device_train_batch_size` to 2, increase `gradient_accumulation_steps` to 8

## HuggingFace Spaces deployment notes

- Push to Space with: `git push space master:main` (Space default branch is `main`)
- Do NOT include `bitsandbytes` in requirements.txt — it imports `triton` (GPU-only) and crashes on the CPU free tier
- Load the tokenizer from `MODEL_ID` (base model), not `HUB_REPO` — the pushed tokenizer config references `TokenizersBackend` which fails to resolve on the Space
- Use `peft>=0.14.0` — adapter was saved with a version that added `alora_invocation_tokens`; older versions throw `TypeError`
- `mlflow` conflicts with `datasets>=4.x` (pyarrow clash) — keep it commented out in requirements.txt
- `gradio>=5.9.1` required for Python 3.13; use `flagging_mode="never"` (not `allow_flagging`)
- Free tier is CPU-only — inference takes ~60s per request
