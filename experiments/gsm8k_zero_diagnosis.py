"""Diagnose the initial GSM8K replay's zero score from saved generations."""

import argparse
import json
from pathlib import Path
from statistics import median

from experiments.audit_gsm8k_regroup import ANSWER, NUMBER, check_score, read_split
from experiments.audit_kv_online import read_json, require


def diagnose(run, test_path):
    protocol = read_json(run / "protocol.json")
    summary = read_json(run / "summary.json")
    test = read_split(test_path, protocol["test_sha256"])
    initial = summary["initial_eval"]
    prompt_ids = protocol["heldout_prompt_ids"]
    require([row["prompt_id"] for row in initial] == prompt_ids, "initial prompts differ from the protocol")
    for arm in summary["arms"].values():
        require([row["prompt_id"] for row in arm["evaluated"]] == prompt_ids,
                "arm prompts differ from the protocol")

    questions = []
    for index, row in enumerate(initial):
        prompt_id = row["prompt_id"]
        gold = test[prompt_id][1]
        tagged = ANSWER.findall(row["completion"])
        candidate = tagged[-1] if tagged else row["completion"]
        numbers = NUMBER.findall(candidate.replace(",", ""))
        final_number = float(numbers[-1]) if numbers else None
        format_relaxed = final_number is not None and abs(final_number - gold) < 1e-4
        unchanged = all(arm["evaluated"][index]["token_ids"] == row["token_ids"]
                        for arm in summary["arms"].values())
        questions.append({"prompt_id": prompt_id, "gold": gold, "final_number": final_number,
                          "tagged": bool(tagged), "truncated": row["truncated"],
                          "tokens": len(row["token_ids"]), "unchanged_across_arms": unchanged,
                          "strict_correct": check_score(row, gold, f"initial/{prompt_id}"),
                          "format_relaxed_correct": int(format_relaxed)})

    return {"questions": questions, "heldout_count": len(questions),
            "tagged": sum(row["tagged"] for row in questions),
            "truncated": sum(row["truncated"] for row in questions),
            "nontruncated_median_tokens": median(row["tokens"] for row in questions if not row["truncated"])
            if any(not row["truncated"] for row in questions) else None,
            "unchanged_across_arms": sum(row["unchanged_across_arms"] for row in questions),
            "strict_correct": sum(row["strict_correct"] for row in questions),
            "format_relaxed_correct": sum(row["format_relaxed_correct"] for row in questions)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.run, args.test), indent=2))
