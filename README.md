# Medical Dialogue Summary

[![CI](https://github.com/jonleed/Medical-Dialogue-Summary/actions/workflows/ci.yml/badge.svg)](https://github.com/jonleed/Medical-Dialogue-Summary/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/model-BioBART--large--LoRA-yellow)](https://huggingface.co/jonleed/clinical-note-biobart-large-lora)

A local Flask application that helps clinicians turn a post-visit doctor-patient transcript into copy-ready EHR sections. The clinician pastes the full transcript, selects only the sections needed for that encounter, reviews each generated block, and copies it into the EHR.

This is a continuation of a Deep Learning course project for the University of Surrey. The original BART fine-tuning approach is preserved; the default backbone is now `GanjinZero/biobart-large`, a biomedical BART model released in 2022.

<p align="center">
  <img src="assets/screenshot-note-app.png" alt="Clinical Note Generator Flask application" width="900">
</p>

## Current Application

Run the doctor-facing application:

```bash
python app.py
```

Open <http://127.0.0.1:5002>.

The workflow is deliberately section-based because the MTS-Dialog training data pairs each dialogue with a documented EHR section. It does not force irrelevant sections into every note.

- Paste the complete conversation, not an isolated snippet.
- Select the sections relevant to the encounter.
- Generate each section using the model's original dialogue-to-section task.
- Review the output before copying it into the EHR.
- Routine outputs such as `None.` and `Noncontributory.` are visibly flagged.

The older dataset-table explorer remains available:

```bash
python eval.py
```

Open <http://127.0.0.1:5001>.

## Model

The committed adapter is `biobart-large-lora/`. It contains only the LoRA adapter and tokenizer; the 407M-parameter BioBART-large base downloads from Hugging Face on first use.

- Base model: [`GanjinZero/biobart-large`](https://huggingface.co/GanjinZero/biobart-large)
- Adapter: [`jonleed/clinical-note-biobart-large-lora`](https://huggingface.co/jonleed/clinical-note-biobart-large-lora)
- Fine-tuning method: PEFT LoRA
- LoRA rank: 8
- LoRA alpha: 64
- LoRA dropout: 0.01
- Target modules: BART `q_proj` and `v_proj`
- Input context: 512 tokens
- Output budget: 512 tokens in the Flask app
- Decoding: deterministic beam search, 4 beams, repetition penalty 2.0, no-repeat 4-gram constraint

## Results

Evaluation uses the same prompt and section-target construction as the original notebook. Scores use ROUGE with stemming on 100 validation dialogues, seed 42.

| BioBART-large LoRA evaluation     |   ROUGE-1 |   ROUGE-2 |   ROUGE-L | ROUGE-Lsum |
| --------------------------------- | --------: | --------: | --------: | ---------: |
| 100 validation dialogues, seed 42 | **0.553** | **0.325** | **0.472** |  **0.474** |

The evaluation output is checked in at [`results/eval-biobart-large-2026-08.json`](results/eval-biobart-large-2026-08.json).

### Controlled Model Study

Using the same validation split, seed, prompt construction, token limits, deterministic decoding, and ROUGE scorer:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---:|---:|---:|---:|
| BART-base, no adapter | 0.2195 | 0.0746 | 0.1673 | 0.1671 |
| BART-base with LoRA | 0.4987 | 0.2858 | 0.4342 | 0.4336 |
| BioBART-large with LoRA | **0.5589** | **0.3334** | **0.4803** | **0.4803** |

The complete study record is [`results/model-study-2026-08.json`](results/model-study-2026-08.json). The results show the value of the original LoRA fine-tuning approach and support BioBART-large as the project's selected backbone.

Run the original notebook-style evaluation:

```bash
python eval_notebook_protocol.py --split val
```

Run the Flask integration evaluation through the actual test client:

```bash
python eval_app_integration.py --app note --n 100
python eval_app_integration.py --app section --n 100
```

An earlier BioBART-large Flask integration result reported ROUGE-L 0.4719. The cached integration predictions lack enough model and input provenance for factuality scoring. Flask and notebook output limits also differ; see the [factuality protocol audit](docs/factuality_report.md#coverage-and-protocol-audit).

## Targeted Factuality Evaluation

Beyond ROUGE, this project includes a targeted factuality evaluation for **age and gender contradictions, unsupported claims, and omissions** in clinical summaries.

### Approach

The evaluation combines two layers:

1. **Local demographic screening** — rule-based extraction of patient age, sex, gender identity and pronoun references from both the source dialogue and the generated summary, with automatic comparison to flag contradictions, supported claims and unclear/review-required cases. Runs offline with no API calls.
2. **Free-model semantic review** — a [targeted development pilot](docs/openrouter_factuality_pilot.md) using Nemotron 3 Ultra Free (via OpenRouter at \$0 actual cost) to assess unsupported claims and omissions through blinded, exact-evidence validation on 24 controlled cases.

Design choices were informed by published evaluation methods from OpenAI, Anthropic and Google, adapted for a free-inference budget. See the [research sources](docs/frontier_eval_research.md) and [evaluation redesign](docs/frontier_eval_report.md).

### 300-Summary Demographic Screening

All 300 saved validation summaries (100 per model) were screened with the local demographic checker (`patient_demographics_v2`):

| Model | Cases | Suggestions | Contradictions | Supported | Unclear | Flagged cases |
|---|---:|---:|---:|---:|---:|---:|
| BioBART-large with LoRA | 100 | 58 | 3 | 20 | 35 | 27 |
| BART-base with LoRA | 100 | 75 | 4 | 17 | 54 | 38 |
| BART-base, no adapter | 100 | 51 | 2 | 49 | 0 | 1 |

BioBART-large produces the fewest contradiction flags (3) while maintaining the most supported demographic claims (20). The base model without fine-tuning shows fewer unclear items because its shorter outputs contain fewer demographic references overall; its 2 contradictions are both age errors. The full screening record is [`results/factuality/saved-summary-screening-2026-10-18/screening_results.json`](results/factuality/saved-summary-screening-2026-10-18/screening_results.json).

These are **automatic rule-based suggestions**, not annotated error rates. Checker precision and recall are unestablished on this sample. Unclear suggestions are review candidates, not confirmed errors.

### Development Pilot Results

The [targeted pilot](docs/openrouter_factuality_pilot.md) ran 19 requests through Nemotron 3 Ultra Free at \$0 actual cost:

| Capability | Planned | Valid reviews | Correct target decisions |
|---|---:|---:|---:|
| Patient age / relative-age controls | 4 | 3 | 2 |
| Sex, gender identity and pronouns | 6 | 6 | 5 |
| Unsupported claims | 3 | 3 | 2 |
| Omissions / section exclusions | 6 | 5 | 5 |
| **Total** | **19** | **17** | **14** |

Five observed failure modes (invented plan IDs, misclassified relative ages, missing pronoun targets, unsupported/contradicted confusion, source-only evidence) are documented in the [pilot report](docs/openrouter_factuality_pilot.md). These are development results on implementer-authored fixtures, not independent clinical validation.

### Reproduce

```bash
# Offline tests (no inference)
.venv/bin/python -m unittest discover -s tests -v

# Offline report from saved pilot reviews (no inference)
.venv/bin/python -m factuality.openrouter_pilot report

# Demographic screening of saved summaries (no inference)
.venv/bin/python -m factuality.evaluate_saved_summaries
```

### Limitations

- The demographic checker's precision and recall are not calibrated on an independent annotated sample
- Unsupported claims and omissions require a model reviewer; only the 24-case pilot covers these
- The development pilot uses implementer-authored fixtures, not clinician-annotated holdout data
- ROUGE scores above measure surface overlap, not clinical accuracy; they remain separate from factuality
- This project is educational and is not a medical device

## Setup

```bash
git clone --recurse-submodules https://github.com/jonleed/Medical-Dialogue-Summary.git
cd Medical-Dialogue-Summary
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Configuration

| Environment variable | Default                    | Purpose                              |
| -------------------- | -------------------------- | ------------------------------------ |
| `BASE_MODEL_ID`      | `GanjinZero/biobart-large` | Base seq2seq model                   |
| `ADAPTER_MODEL_ID`   | `./biobart-large-lora`     | Local or Hugging Face LoRA adapter   |
| `HF_TOKEN`           | unset                      | Optional Hugging Face authentication |

## Reproducing Training

`train_lora.py` mirrors the original notebook while using BioBART-large by default:

```bash
python train_lora.py \
  --base_model GanjinZero/biobart-large \
  --output_dir ./biobart-large-lora \
  --per_device_batch 2 \
  --grad_accum 4 \
  --epochs 20
```

The original notebook recipe is retained in [`Medical_Dialogue_Generation.ipynb`](Medical_Dialogue_Generation.ipynb). Training uses the MTS-Dialog training split, the original prompt and target construction, 512-token inputs, 256-token targets, learning rate `2e-4`, weight decay `0.01`, and per-epoch validation ROUGE. The effective batch size is 8 in the recommended command.

## Architecture

```mermaid
graph LR
    A["Full doctor-patient transcript"] --> B["Clinician selects EHR sections"]
    B --> C["Original section prompt"]
    C --> D["BioBART-large"]
    D --> E["LoRA adapter"]
    E --> F["Deterministic beam search"]
    F --> G["Reviewable EHR section"]
```

## Project History

The original project used BART-base with LoRA. The restoration preserves that approach and improves the deployable artifact by:

- Loading the base model and adapter separately through PEFT
- Using the biomedical BioBART-large backbone
- Increasing serving input context from 128 to the 512-token training context
- Providing clinician-selected section generation in demo

This project is educational and is not a medical device.

## Credits

- Dataset: Ben Abacha et al., [MTS-Dialog](https://github.com/abachaa/MTS-Dialog), EACL 2023
- Biomedical base: [GanjinZero/biobart-large](https://huggingface.co/GanjinZero/biobart-large)
- Frameworks: [PEFT](https://github.com/huggingface/peft), [Transformers](https://github.com/huggingface.co/docs/transformers), Flask

## License

[MIT](LICENSE)
