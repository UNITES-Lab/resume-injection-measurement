"""
Classify Candidate Profile: Industry and Job Function

Uses an LLM to classify a candidate's industry and job function based on their
skills and work experience.

Input format (JSON):
{
  "skills": ["Python", "AWS", "Machine Learning"],
  "positions": [
    {"title": "Software Engineer", "summary": "Built backend services..."},
    ...
  ]
}

Output format (JSON):
{
  "industry": "Technology, Information and Media",
  "job_function": "Engineering"
}

Usage:
  python classify_profile.py --input profile.json --output classification.json
  python classify_profile.py --input profile.json --output classification.json --model gpt-5
"""

import json
import argparse
from openai import OpenAI

PROMPT_TEMPLATE = """You are an expert at classifying professional profiles. Your task is to analyze candidate information and classify their industry and job function.

Analyze this candidate profile and classify it.

## Candidate Profile

### Skills
{skills}

### Work Experience
{positions}

## Classification Task

### 1. Industry
The business sector/industry where this candidate primarily works. Choose ONE from:
- Technology, Information and Media
- Financial Services
- Professional Services
- Manufacturing
- Hospitals and Health Care
- Education
- Retail
- Government Administration
- Construction
- Transportation, Logistics, Supply Chain and Storage
- Entertainment Providers
- Real Estate and Equipment Rental Services
- Accommodation Services
- Consumer Services
- Administrative and Support Services
- Oil, Gas, and Mining
- Wholesale
- Utilities
- Farming, Ranching, Forestry
- Holding Companies

**NOTE**: If no work experience is provided, infer industry from skills. If neither is available, return "unknown".

### 2. Job Function
The type of work/role this candidate performs. Choose ONE from:
- Accounting
- Administrative
- Arts and Design
- Business Development
- Community and Social Services
- Consulting
- Customer Success and Support
- Education
- Engineering
- Entrepreneurship
- Finance
- Healthcare Services
- Human Resources
- Information Technology
- Legal
- Marketing
- Media and Communication
- Military and Protective Services
- Operations
- Product Management
- Program and Project Management
- Purchasing
- Quality Assurance
- Real Estate
- Research
- Sales

**NOTE**: If no work experience is provided, infer job function from skills. If neither is available, return "unknown".

## Output Format

Return ONLY a JSON object with the exact category names as shown above:
{{"industry": "exact category name or unknown", "job_function": "exact category name or unknown"}}"""

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


def format_prompt(skills, positions):
    """Build the classification prompt from skills list and positions list."""
    # Format skills
    if skills:
        skills_str = ", ".join(str(s) for s in skills[:50])
    else:
        skills_str = "(No skills listed)"

    # Format positions
    if positions:
        pos_lines = []
        for p in positions[:5]:
            if isinstance(p, dict):
                title = p.get("title", "Unknown Title")
                if isinstance(title, list):
                    title = ", ".join(title) if title else "Unknown Title"
                summary = p.get("summary", "")
                if summary:
                    pos_lines.append(f"- {title}: {summary[:300]}")
                else:
                    pos_lines.append(f"- {title}")
            elif isinstance(p, str):
                pos_lines.append(f"- {p}")
        positions_str = "\n".join(pos_lines) if pos_lines else "(No work experience listed)"
    else:
        positions_str = "(No work experience listed)"

    return PROMPT_TEMPLATE.replace("{skills}", skills_str).replace("{positions}", positions_str)


def main():
    parser = argparse.ArgumentParser(
        description="Classify a candidate profile by industry and job function"
    )
    parser.add_argument("--input", required=True, help="Input JSON file with profile")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--model", default="gpt-5", help="LLM model (default: gpt-5)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    skills = data.get("skills", [])
    positions = data.get("positions", [])

    # Call LLM once
    client = OpenAI()
    prompt = format_prompt(skills, positions)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content
    parsed = parse_json_response(raw)

    result = {
        "industry": parsed.get("industry"),
        "job_function": parsed.get("job_function"),
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Done. industry={result['industry']}, job_function={result['job_function']}. Wrote {args.output}")


if __name__ == "__main__":
    main()