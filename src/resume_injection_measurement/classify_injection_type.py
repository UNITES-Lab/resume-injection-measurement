"""
Classify Injection Type: prompt_injection vs content_injection

Takes malicious excerpts extracted from a resume and classifies the overall
attack type (prompt injection vs content injection) and, for content
injections, the specific content subtype.

Input format (JSON):
{
  "excerpts": ["malicious hidden text 1", "malicious hidden text 2"]
}

Output format (JSON):
{
  "attack_type": "content_injection",
  "content_type": "skills_keywords",
  "reasoning": "Hidden list of technical skills"
}

Usage:
  python classify_injection_type.py --input excerpts.json --output attack_type.json
  python classify_injection_type.py --input excerpts.json --output attack_type.json --model gpt-5
"""

import json
import argparse
from openai import OpenAI

PROMPT_TEMPLATE = """You are an expert at classifying hidden content injected into resumes. Your task is to analyze all hidden text detected in a single resume and classify the overall attack.

## Hidden Content Detected
{excerpts_formatted}

## Classification Task

### 1. Attack Type (choose ONE)
- **prompt_injection**: Direct instructions to manipulate AI systems (e.g., "ignore other content", "say hire me", "this candidate is perfect", instructions to AI/LLM)
- **content_injection**: Hidden text to game keyword matching or ATS systems (skills, experience, qualifications stuffed into resume)

### 2. Content Type (choose ONE, only if attack_type is "content_injection")
- **skills_keywords**: Technical skills, programming languages, tools, certifications, soft skills
- **experience_description**: Work experience, job responsibilities, achievements
- **job_requirements**: Job description content, qualification requirements copied from JD
- **education_credentials**: Degrees, certifications, academic achievements
- **mixed**: Multiple content types combined together

## Output Format

Return ONLY a JSON object with no additional text:
{"attack_type": "prompt_injection" or "content_injection", "content_type": "skills_keywords" or "experience_description" or "job_requirements" or "education_credentials" or "mixed" or null, "reasoning": "Brief explanation in 1-2 sentences"}"""

def parse_json_response(text):
    """Extract a JSON object from LLM response text, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    if "{" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    return json.loads(text)


def format_prompt(excerpts):
    """Format excerpt list into numbered segments and fill the prompt template."""
    lines = []
    for i, excerpt in enumerate(excerpts, 1):
        if len(excerpt) > 2000:
            excerpt = excerpt[:2000] + "..."
        lines.append(f"[Segment {i}]\n{excerpt}")
    excerpts_formatted = "\n\n".join(lines)
    return PROMPT_TEMPLATE.replace("{excerpts_formatted}", excerpts_formatted)


def main():
    parser = argparse.ArgumentParser(
        description="Classify hidden content attack type (prompt_injection vs content_injection)"
    )
    parser.add_argument("--input", required=True, help="Input JSON file with excerpts")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--model", default="gpt-5", help="LLM model (default: gpt-5)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    excerpts = data.get("excerpts", [])
    if not excerpts:
        result = {"attack_type": None, "content_type": None, "reasoning": "No excerpts provided"}
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"No excerpts to classify. Wrote {args.output}")
        return

    # Call LLM once
    client = OpenAI()
    prompt = format_prompt(excerpts)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content
    parsed = parse_json_response(raw)

    result = {
        "attack_type": parsed.get("attack_type"),
        "content_type": parsed.get("content_type"),
        "reasoning": parsed.get("reasoning", ""),
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Done. attack_type={result['attack_type']}. Wrote {args.output}")


if __name__ == "__main__":
    main()