<div align="center">

# Measuring Real-World Prompt Injection Attacks in LLM-based Resume Screening

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Artifact](https://img.shields.io/badge/Artifact-Code%20%2B%20Examples-2E8B57.svg)](examples/)

</div>

<table align="center">
  <tr>
    <td align="center"><img src="assets/fig_injection_example_1.png" width="390" alt="Instruction injection example"></td>
    <td align="center"><img src="assets/fig_injection_example_2.png" width="390" alt="Data injection example"></td>
  </tr>
  <tr>
    <td align="center"><b>Instruction injection</b></td>
    <td align="center"><b>Data injection</b></td>
  </tr>
</table>

## Overview

This repository provides the detection and analysis code used in the paper. It includes two complementary detectors:

- **HCD**: a hybrid cascade detector that first performs rule-based PDF visual analysis and then verifies candidate excerpts with an LLM.
- **VDA**: a visual discrepancy analyzer that compares rendered PDF pages against machine-extracted text using a vision-language model.

It also includes the LLM-based scripts used to classify injection type, instruction-injection subtype, and candidate profile attributes.

## Methods

### Hybrid Cascade Detector (HCD)

HCD first applies lightweight rule-based visual analyses to extract candidate hidden excerpts from a resume PDF, then asks an LLM to distinguish intentional manipulation from benign PDF artifacts.

<p align="center">
  <img src="assets/fig_hcd_pipeline.png" width="780" alt="Hybrid Cascade Detector pipeline">
</p>

### Visual Discrepancy Analyzer (VDA)

VDA compares two views of the same resume: rendered page images, which approximate what a human reader sees, and machine-extracted text, which may contain hidden content.

<p align="center">
  <img src="assets/fig_vda_pipeline.png" width="780" alt="Visual Discrepancy Analyzer pipeline">
</p>

## Installation

```bash
git clone https://github.com/UNITES-Lab/resume-injection-measurement.git
cd resume-injection-measurement
pip install .
```

The package requires Python 3.9+. VDA uses `pdf2image`, which requires Poppler:

```bash
sudo apt-get install poppler-utils
```

LLM-based components require an OpenAI API key:

```bash
export OPENAI_API_KEY=...
```

## Quick Start

Run the full HCD pipeline on a sanitized example PDF:

```bash
mkdir -p outputs
rim-hcd \
  --pdf examples/pdfs/instruction_override_case.pdf \
  --output outputs
```

Run VDA on the same example:

```bash
rim-vda \
  --pdf examples/pdfs/instruction_override_case.pdf \
  --output outputs/instruction_override_case_vda.json
```

Both commands should report a hidden-text injection on the provided examples when the OpenAI API is configured.

## Examples

The repository includes two synthetic PDFs under `examples/pdfs/`:

| Example | Pattern | HCD result | VDA result |
| --- | --- | --- | --- |
| `instruction_override_case.pdf` | hidden instruction injection | `label=1` | `label=1` |
| `data_skills_keywords_case.pdf` | hidden skill-keyword injection | `label=1` | `label=1` |

The generated outputs from running the released code are included under:

```text
examples/results/hcd/
examples/results/vda/
```

## Command Line Tools

| Command | Purpose |
| --- | --- |
| `rim-hcd` | Full HCD pipeline: rule-based scan followed by LLM verification when `OPENAI_API_KEY` is configured. |
| `rim-hcd-rule` | HCD Stage 1: rule-based detection of visually hidden PDF text. |
| `rim-hcd-verify` | HCD Stage 2: LLM verification of Stage-1 excerpts. |
| `rim-vda` | VDA pipeline for rendered-page vs extracted-text comparison. |
| `rim-classify-type` | Classify malicious excerpts as instruction injection or data injection. |
| `rim-classify-subtype` | Classify instruction-injection subtype. |
| `rim-classify-profile` | Classify candidate industry and job function from structured profile data. |

## Repository Layout

```text
src/resume_injection_measurement/
  hcd_pipeline.py               end-to-end HCD pipeline
  hcd_rule_analysis.py          HCD Stage 1
  hcd_llm_verification.py       HCD Stage 2
  vda_detector.py               VDA detector
  classify_injection_type.py    injection type classification
  classify_injection_subtype.py instruction-injection subtype classification
  classify_profile.py           profile classification

examples/
  pdfs/                         sanitized synthetic PDF examples
  results/hcd/                  full HCD outputs generated from examples
  results/vda/                  VDA outputs generated from examples
```

## Data Availability

This repository releases shareable code and sanitized synthetic examples. The original resume datasets, raw PDFs, and production detection outputs cannot be released because of applicant privacy and contractual restrictions.
