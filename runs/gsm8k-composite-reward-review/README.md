# GSM8K composite reward training-trace review

Two blinded AI reviews from the same model family agreed with the composite reward on 12 of 15 previously unseen-by-design question IDs. They judged two ties and preferred the other answer once. Both reviews made the same choice on every pair. These are descriptive counts on selected groups where the reward broke a tie. They do not estimate agreement on random answers, human judgment, or the effect of training.

## Source and selection

The historical feasibility output was written on 2026-11-24 at 11:34:53.381801 UTC. The index fixes the 53 training archive files available to the original scan, their order, timestamps and SHA-256 hashes. The online V2 parsed-outcome seeds with complete training by the cutoff were 0, 1, 3 and 4. Seed 2 finished later. Duplicate groups use the question ID and the unordered pair of generated token sequences. Copies across sources have the precedence recorded in the index.

A group was eligible if both V2 outcome scores were zero and the old simple-step rewards tied. The new reward ranks a group only when one answer's matched reference steps strictly contain the other's. The original candidate list is represented by the question ID, source, two completion digests and winner index. Replaying the current group scorer reproduces every one of the original 156 directions.

| Frozen group population | Groups / IDs | Eligible groups / IDs | Newly ranked groups / IDs |
| --- | ---: | ---: | ---: |
| Original design traces | 427 / 414 | 212 / 208 | 30 / 30 |
| All archives on original design IDs, including the row above | 1,830 / 414 | 1,017 / 309 | 138 / 79 |
| All archives on other training IDs | 304 / 137 | 200 / 123 | 18 / 15 |

For each of the 15 question IDs in the last row, the reviewed pair came from the earliest archived file modification time among its newly ranked groups. Modification time is the recorded tie-breaker, not a verified individual generation time. Question 6695 initially used a mixed-run pair. Its stale-run pair had an earlier timestamp and was reviewed separately before reconciliation. The lock retains both versions and both reviews. The reviewed sources were the five-seed online-lag archive (12 pairs), the five-seed budget archive (ID 4517), the stable recipe sweep (ID 4756) and the online-low GRPO diagnosis (ID 5669).

Each reviewer saw the question, reference rationale and anonymized completions. Labels and reasons were saved before joining the automatic winner. The two original blind payloads can be reconstructed byte-for-byte. Review times and hashes are in the lock. The audit reads the training archives, selected training dataset rows and a plan containing reserved IDs. Full replay verifies the checksum of each original trace it uses. For eight local smoke archives, a checksum-bound supplement retains their ten unique two-response groups, with original line numbers. Four of those files contributed zero unique groups. The supplement contains only public GSM8K training IDs, generated answer text, truncation flags and generated token IDs. It contains no private prompts or user messages.

## Where the automatic order diverged

- On question 71 both answers compute 36 and 43 before making different one-step mistakes. Equation formatting gave A a unique machine match for the multiplication. Both reviewers called a tie.
- On question 2742 A computes 120 sour candies then abandons the result. B divides among three people but starts from the wrong good-candy count. Both reviewers called a tie.
- On question 3058 both answers derive 50 bones. A earned the unique machine match, then invented new monthly rates. Both reviewers preferred B because it retained the distinction between the total and the remaining stock.
- On question 1199 both reviewers and the reward preferred B. Both answers actually obtain 240 available square feet. The reviewers' reason was B's setup for the next division, even though the machine's distinguishing match was the subtraction.

## Final-answer parsing

The simple and composite comparisons both use V2 outcome parsing. All 15 reviewed pairs still have two zero outcomes under the separate V3 parser. As a sensitivity check over the full inventory, the broader V3 parser changes four group outcomes involving two design-pool questions. Three groups involve ID 855 and one involves ID 5771. Two formerly double-zero groups become mixed, and two mixed groups become double-correct. None of the four receives a new composite-step order. The audit output records both outcome vectors and the source lines.

## Reproduction

From the Narl repository root, obtain the official GSM8K training JSONL using the pinned source in `docs/gpu-validation.md`. Verify the published repository checksum before running the audit.

```sh
mkdir -p runs/gsm8k-data
curl -fsSL https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl -o runs/gsm8k-data/train.jsonl
echo '17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465  runs/gsm8k-data/train.jsonl' | shasum -a 256 -c -
```

Use Python 3.11 or newer, including a plain standard-library environment. The 15 pairs use tracked training traces. The full replay prefers each original smoke archive when available and otherwise uses the SHA-256-verified supplement in this directory.

```sh
python3 -m experiments.audit_gsm8k_composite_review --inventory \
  --output /tmp/narl-composite-audit.json --export-blind /tmp/narl-composite-blind.json
```

Omit `--inventory` to replay just the reviewed pairs. The blind export contains the same original A/B order and reference text, with the re-reviewed pair substituted for 6695. `review_lock.json` stores both reviewers' reasons, original-packet digests, one-based trace line numbers and zero-based A/B sample indices. `archive_index.json` stores the historical source inventory and candidate list. `audit.json` records both reward scores and reason strings for each reviewed completion, along with scorer hashes, reconciliation and parser sensitivity.
