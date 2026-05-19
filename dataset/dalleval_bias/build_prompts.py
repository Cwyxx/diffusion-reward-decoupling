"""One-shot converter: DallEval prompt_list.json -> prompts.jsonl with metadata.

Reads the vendored 252-prompt list and emits one JSONL row per prompt, tagging
each with (subject, profession, category) so downstream scorers and aggregators
can group prompts without re-parsing the natural-language string.

Run once and commit prompts.jsonl. Idempotent — safe to re-run.
"""
import json
import os
import re

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(
    _THIS_DIR, "..", "..", "evaluation", "benchmarks", "DallEval", "biases", "prompt_list.json"
))
_DST = os.path.join(_THIS_DIR, "prompts.jsonl")

# "A {man,woman,person} who works as {an,a} <profession>"
_PATTERN = re.compile(r"^A (man|woman|person) who works as (?:an? )?(.+)$")
# Bare "A man" / "A woman" / "A person"
_BARE = re.compile(r"^A (man|woman|person)$")


def parse(prompt: str):
    m = _PATTERN.match(prompt)
    if m:
        subject, profession = m.group(1), m.group(2)
    else:
        m = _BARE.match(prompt)
        if not m:
            raise ValueError(f"Unrecognized prompt: {prompt!r}")
        subject, profession = m.group(1), None
    # Gender-neutral prompts ("A person ...") are what DallEval uses for the
    # gender-MAD task; "A man/woman ..." are used for attribute disparity.
    category = "neutral" if subject == "person" else "gendered"
    return subject, profession, category


def main():
    with open(_SRC, "r") as f:
        prompts = json.load(f)
    assert len(prompts) == 252, f"expected 252 prompts, got {len(prompts)}"

    rows = []
    for pid, prompt in enumerate(prompts):
        subject, profession, category = parse(prompt)
        rows.append({
            "prompt_id": pid,
            "prompt": prompt,
            "subject": subject,
            "profession": profession,
            "category": category,
        })

    with open(_DST, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {_DST}")

    # Sanity counts
    from collections import Counter
    c = Counter((r["subject"], r["category"]) for r in rows)
    for key, n in sorted(c.items()):
        print(f"  subject={key[0]:<7} category={key[1]:<9} count={n}")


if __name__ == "__main__":
    main()
