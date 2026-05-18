"""End-to-end HCD pipeline: rule-based scan followed by optional LLM verification."""

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from .hcd_llm_verification import format_prompt, parse_json_response
from .hcd_rule_analysis import process_pdf


def verify_stage1(stage1_result, model):
    detections = stage1_result.get("detections", [])
    texts = [d.get("excerpt") or d.get("text") for d in detections]
    texts = [t for t in texts if t]

    if stage1_result.get("label", 0) == 0 or not texts:
        return {"resume_label": 0, "excerpts": [], "status": "skipped_no_stage1_detections"}

    prompt = format_prompt(texts)
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    parsed = parse_json_response(response.choices[0].message.content)

    labels = parsed.get("labels", [])
    reasoning = parsed.get("reasoning", [])
    excerpts = []
    for i, text in enumerate(texts):
        excerpts.append({
            "text": text,
            "label": labels[i] if i < len(labels) else 0,
            "reasoning": reasoning[i] if i < len(reasoning) else "",
        })

    return {
        "resume_label": int(any(label == 1 for label in labels)),
        "excerpts": excerpts,
        "status": "completed",
    }


def run_hcd(pdf_path, output_dir=None, model="gpt-5"):
    stage1_result = process_pdf(pdf_path, output_dir)
    if stage1_result is None:
        return None

    pdf_file = Path(pdf_path)
    out_path = Path(output_dir) if output_dir else pdf_file.parent
    out_path.mkdir(parents=True, exist_ok=True)

    if os.environ.get("OPENAI_API_KEY"):
        stage2_result = verify_stage1(stage1_result, model)
    else:
        stage2_result = {
            "resume_label": None,
            "excerpts": [],
            "status": "skipped_no_openai_api_key",
        }

    result = {
        "pdf": str(pdf_file),
        "model": model,
        "stage1": stage1_result,
        "stage2": stage2_result,
        "resume_label": stage2_result.get("resume_label"),
    }

    output_path = out_path / f"{pdf_file.stem}_hcd_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"HCD pipeline result: resume_label={result['resume_label']} ({stage2_result['status']})")
    print(f"Wrote {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the full HCD pipeline: Stage 1 rule scan plus Stage 2 LLM verification when OPENAI_API_KEY is available."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Directory for output files. Defaults to the same directory as the input PDF.",
    )
    parser.add_argument("--model", default="gpt-5", help="LLM model for Stage 2 verification (default: gpt-5)")
    args = parser.parse_args()
    run_hcd(args.pdf, args.output, model=args.model)


if __name__ == "__main__":
    main()
