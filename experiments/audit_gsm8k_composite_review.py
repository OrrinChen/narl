"""Reconstruct the frozen training-only review. Run with python -m experiments.audit_gsm8k_composite_review."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import mmap
from pathlib import Path
import re

from experiments.gsm8k_composite_reward import score_group_training_reward
from experiments.gsm8k_step_reward import score_training_reward as simple_reward
from experiments.gsm8k_training_reward_v2 import score_training_reward as outcome_v2
from experiments.gsm8k_training_reward_v3 import score_training_reward as outcome_v3


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'runs/gsm8k-composite-reward-review'
BASE = 'runs/gsm8k-stable-v1/models/lr-0.0001-groups-16/'
TRAIN = ROOT / 'runs/gsm8k-data/train.jsonl'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def training_rows(ids):
    """Use newline offsets to deserialize only training rows used by this audit."""
    result = {}
    with TRAIN.open('rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        offset = 0
        for i in range(max(ids) + 1):
            end = data.find(b'\n', offset)
            if end < 0:
                end = len(data)
            if i in ids:
                row = json.loads(data[offset:end])
                rationale, value = row['answer'].rsplit('####', 1)
                result[i] = {'question': row['question'], 'rationale': rationale, 'value': value.strip().replace(',', '')}
            offset = end + 1
    return result


def order(results):
    a, b = (r[0] for r in results)
    return 'A' if a > b else 'B' if b > a else 'tie'


def read_pair(row, example):
    i = row['id']
    with (ROOT / row['source']).open('rb') as trace:
        raw = next(raw for n, raw in enumerate(trace, 1) if n == row['line'])
    record = json.loads(raw)
    require(record['prompt_id'] == i, f'prompt locator changed for {i}')
    samples = [record['samples'][n] for n in row['ab_sample_indices']]
    text = [s['completion'] for s in samples]
    require([digest(t.encode()) for t in text] == row['completion_sha256_ab'], f'completion changed for {i}')
    return samples, {'id': i, 'question': example['question'], 'reference': example['rationale'], 'A': text[0], 'B': text[1]}


def review_rows(lock, reserved):
    selected = lock['pairs']
    ids = {row['id'] for row in selected}
    require(len(selected) == len(ids) == 15 and not (ids & reserved), 'review split or ID count changed')
    examples = training_rows(ids)
    report, packet = [], []
    for row in selected:
        i = row['id']
        ex = examples[i]
        samples, blind = read_pair(row, ex)
        packet.append(blind)
        simple = [simple_reward(s, ex) for s in samples]
        full = score_group_training_reward(samples, ex)
        old_end = [outcome_v2(s, ex['value']) for s in samples]
        relaxed_end = [outcome_v3(s, ex['value']) for s in samples]
        require(order(simple) == 'tie' and not any(r[0] for r in old_end), f'ineligible review group {i}')
        require(order(full) == row['automatic'], f'reviewed automatic direction changed for {i}')
        judges = [row[k]['choice'] for k in ('primary', 'secondary')]
        relation = 'uncertain' if judges[0] != judges[1] or 'tie' in judges or order(full) == 'tie' else (
            'supports' if judges[0] == order(full) else 'opposes')
        report.append({'id': i, 'simple_ab': simple, 'composite_ab': full, 'winner': order(full),
                       'historical_winner': row['automatic'], 'judges_ab': judges, 'match': relation,
                       'outcome_v2_ab': old_end, 'outcome_v3_ab': relaxed_end})
    # Rebuild the two original pre-unblinding payloads, including the superseded pair.
    old_row = lock['superseded_6695']
    _, first_6695 = read_pair(old_row, examples[old_row['id']])
    first_packet = [first_6695 if r['id'] == old_row['id'] else r for r in packet]
    second_packet = next(r for r in packet if r['id'] == old_row['id'])
    original = lock['original_files']
    for name, payload in [('blinded', json.dumps(first_packet, ensure_ascii=False, indent=2)),
                          ('earliest-recheck', json.dumps(second_packet))]:
        require(digest(payload.encode()) == original[name]['sha256'], f'blinded packet changed: {name}')
    return packet, report


def training_groups(path, expected_sha, reserved):
    sha = hashlib.sha256()
    with path.open('rb') as file:
        for line, raw in enumerate(file, 1):
            sha.update(raw)
            prefix = re.search(rb'"prompt_id":\s*(\d+)', raw[:512])
            require(prefix is not None, f'missing training prompt ID at {path}:{line}')
            i = int(prefix[1])
            if i in reserved:
                continue
            row = json.loads(raw)
            require(i == row['prompt_id'], 'training ID mismatch')
            for samples in ([row['samples']] if 'samples' in row else row['cohorts'].values()):
                yield line, i, samples
    require(sha.hexdigest() == expected_sha, f'training archive changed: {path}')


def scan_index(index, reserved):
    """Scan explicit training files with historical first-copy precedence."""
    design_ids, groups = set(), {}
    raw = (ARCHIVE / 'smoke_group_supplement.json').read_bytes()
    require(digest(raw) == index['smoke_group_supplement_sha256'], 'smoke supplement changed')
    supplement = json.loads(raw)
    for name, meta in index['training_archives'].items():
        path = ROOT / name
        require(path.name in {'trace.jsonl', 'samples.jsonl'}, 'expected training groups')
        require(meta['mtime_ns'] <= index['feasibility_cutoff_ns'], f'archive postdates freeze: {name}')
        if path.is_file():
            records = training_groups(path, meta['sha256'], reserved)
        else:
            require(name in supplement, f'missing training archive: {path}')
            records = ((r['line'], r['prompt_id'], r['samples']) for r in supplement[name])
        for line, i, samples in records:
            require(i not in reserved, f'reserved prompt ID at {name}:{line}')
            if name.startswith(BASE):
                design_ids.add(i)
            if len(samples) == 2:
                tokens = tuple(sorted(tuple(s['token_ids']) for s in samples))
                groups.setdefault((i, tokens), (name, line, samples))
    return design_ids, groups


def inventory_report(index, reserved):
    design_ids, groups = scan_index(index, reserved)
    examples = training_rows({i for i, _ in groups})
    historical = {(g['source'], g['id'], tuple(g['hashes'])): g['winner_index'] for g in index['historical_discordant_groups']}
    seen = set()
    totals, distinct = defaultdict(Counter), defaultdict(lambda: defaultdict(set))
    parser_cases, historical_differences = [], []
    for (i, _), (name, line, samples) in groups.items():
        categories = ['design_union'] if i in design_ids else ['independent']
        if name.startswith(BASE):
            categories.append('original_design')
        ex = examples[i]
        old = [simple_reward(s, ex) for s in samples]
        ends = [outcome_v2(s, ex['value'])[0] for s in samples]
        eligible = not any(ends) and order(old) == 'tie'
        new = score_group_training_reward(samples, ex)
        added = eligible and order(new) != 'tie'
        identity = (name, i, tuple(digest(s['completion'].encode()) for s in samples))
        original = identity in historical
        if original:
            seen.add(identity)
            require(eligible, f'ineligible historical candidate {i}')
        if added != original or (original and order(new) != 'AB'[historical[identity]]):
            historical_differences.append({'id': i, 'source': name, 'line': line,
                                           'historical': 'AB'[historical[identity]] if original else 'tie',
                                           'composite': order(new)})
        relaxed = [outcome_v3(s, ex['value'])[0] for s in samples]
        if relaxed != ends:
            parser_cases.append({'id': i, 'source': name, 'line': line, 'eligible': eligible,
                                 'new_step_order': added, 'v2_outcomes': ends, 'v3_outcomes': relaxed})
        for cat in categories:
            for key, include in [('all', True), ('eligible', eligible), ('historical_new', original), ('current_new', added)]:
                if include:
                    totals[cat][key] += 1
                    distinct[cat][key].add(i)
    require(not historical_differences, f'current scores differ from frozen directions: {historical_differences}')
    require(seen == historical.keys(), 'historical candidate could not be located')
    counts = {c: {k: {'groups': n, 'questions': len(distinct[c][k])} for k, n in counter.items()} for c, counter in totals.items()}
    for cat, all_groups, novel_groups in [('original_design', 427, 30), ('independent', 304, 18), ('design_union', 1830, 138)]:
        require(counts[cat]['all']['groups'] == all_groups and counts[cat]['historical_new']['groups'] == novel_groups,
                f'historical archive counts differ for {cat}')
    require(counts['independent']['all']['questions'] == 137 and counts['independent']['eligible']['groups'] == 200
            and counts['independent']['historical_new']['questions'] == 15, 'independent cohort mismatch')
    return {'counts': counts, 'historical_vs_current_differences': historical_differences,
            'separate_v3_parser_changes': parser_cases}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export-blind', type=Path, help='write question, reference and anonymous A/B texts')
    p.add_argument('--inventory', action='store_true', help='also re-score all training groups in the frozen source list')
    p.add_argument('--output', type=Path, help='save per-pair scores and any inventory counts as JSON')
    args = p.parse_args()
    plan = json.loads((ROOT / 'runs/gsm8k-stable-v1/plan.json').read_bytes())
    reserved = set().union(*plan['qualification_prompt_ids'][:2])
    lock = json.loads((ARCHIVE / 'review_lock.json').read_bytes())
    packet, rows = review_rows(lock, reserved)
    if args.export_blind:
        write(args.export_blind, packet)
    result = {'summary': dict(Counter(row['match'] for row in rows)), 'pairs': rows}
    result['scorer_sha256'] = {name: digest((ROOT / 'experiments' / name).read_bytes()) for name in (
        'gsm8k_composite_reward.py', 'gsm8k_step_reward.py', 'gsm8k_training_reward.py',
        'gsm8k_training_reward_v2.py', 'gsm8k_training_reward_v3.py')}
    require(result['summary'] == {'supports': 12, 'uncertain': 2, 'opposes': 1}, 'locked review summary changed')
    if args.inventory:
        index = json.loads((ARCHIVE / 'archive_index.json').read_bytes())
        cutoff = index['feasibility_cutoff_ns']
        require(all(t <= cutoff for t in index['completed_v2_seeds'].values()), 'seed completion postdates freeze')
        for row in lock['pairs']:
            candidates = [g for g in index['historical_discordant_groups'] if g['id'] == row['id']]
            earliest = min(candidates, key=lambda g: index['training_archives'][g['source']]['mtime_ns'])
            require(row['source'] == earliest['source'] and sorted(row['completion_sha256_ab']) == sorted(earliest['hashes']),
                    f"earliest training pair differs for {row['id']}")
        result.update(inventory_report(index, reserved))
    if args.output:
        write(args.output, result)
    print(json.dumps(result if not args.output else {k: result[k] for k in ('summary', 'counts') if k in result}, indent=2))


if __name__ == '__main__':
    main()
