"""SkillNet skill hierarchy tokenization utilities.

This script exposes the public, sanitized version of the SkillNet hierarchy
tokenizer. It has three layers:

1. A 6-digit motion code describing physical interaction semantics.
2. A motion-code cluster id.
3. VerbNet and verb ids from the released tokenization strategy.

No API key or private endpoint is stored in this file. Optional LLM-based motion
code annotation reads configuration from environment variables.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_STRATEGY_PATH = Path(__file__).with_name("tokenization_strategy.json")

MOTION_CODE_CENTERS = {
    "200200": "cluster_0",
    "200100": "cluster_1",
    "100200": "cluster_2",
    "200010": "cluster_3",
    "100100": "cluster_4",
    "201010": "cluster_5",
    "200001": "cluster_6",
    "201201": "cluster_7",
    "000001": "cluster_8",
    "220001": "cluster_9",
    "100001": "cluster_10",
    "200201": "cluster_11",
}

MOTION_CODE_WEIGHTS = (3.056, 1.539, 1.350, 2.844, 1.039, 2.632)

MOTION_CODE_PROMPT = """\
Role: You are a Robotic Action Semantic Encoder. Analyze {n_items} natural
language descriptions of robotic manipulation subtasks in "Verb + Object"
format and convert each one into a precise 6-digit motion code.

Output Format: A single line containing {n_items} 6-digit codes separated by
spaces. Do not output explanations.

Code Definitions:
Digit 1: Contact Time
  0 = No Contact, 1 = Short Contact, 2 = Long Contact
Digit 2: Deformation Type
  0 = Non-Permanent, 1 = Plastic, 2 = Rigid separation/cutting/fracture
Digit 3: Arm Fixed-Axis Rotation
  0 = No, 1 = Yes
Digit 4: Object Constrained Translation
  0 = No Translation, 1 = 1D Constrained, 2 = Unconstrained 3D motion
Digit 5: Object Fixed-Axis Rotation
  0 = No, 1 = Yes
Digit 6: Tool Usage
  0 = No, 1 = Yes

Examples:
Input: Pick the red block
Output: 200200

Input: Cut the apple
Output: 220001

Input: Tap the button
Output: 100100
"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_phrase(phrase: str) -> str:
    return " ".join(phrase.strip().lower().split())


def first_verb(phrase: str) -> str:
    phrase = normalize_phrase(phrase)
    if not phrase:
        return "unknown"
    return phrase.split()[0]


def validate_motion_code(code: str) -> str:
    code = code.strip()
    if not re.fullmatch(r"[0-2][0-2][0-1][0-2][0-1][0-1]", code):
        raise ValueError(
            f"Invalid motion code {code!r}; expected six digits with ranges "
            "[0-2][0-2][0-1][0-2][0-1][0-1]."
        )
    return code


def motion_code_distance(code_a: str, code_b: str, weights: tuple[float, ...] = MOTION_CODE_WEIGHTS) -> float:
    code_a = validate_motion_code(code_a)
    code_b = validate_motion_code(code_b)
    distance = 0.0

    for idx, (a_digit, b_digit, weight) in enumerate(zip(code_a, code_b, weights)):
        if idx in (0, 1, 3):
            if (a_digit == "0") != (b_digit == "0"):
                distance += weight
            elif a_digit != b_digit:
                distance += weight / 2.0
        elif a_digit != b_digit:
            distance += weight

    return distance


def assign_motion_cluster(motion_code: str) -> str:
    motion_code = validate_motion_code(motion_code)
    best_center = min(
        MOTION_CODE_CENTERS,
        key=lambda center: motion_code_distance(motion_code, center),
    )
    return MOTION_CODE_CENTERS[best_center]


def motion_cluster_distances(motion_code: str) -> list[dict[str, Any]]:
    motion_code = validate_motion_code(motion_code)
    records = [
        {
            "center": center,
            "cluster": cluster,
            "distance": motion_code_distance(motion_code, center),
        }
        for center, cluster in MOTION_CODE_CENTERS.items()
    ]
    return sorted(records, key=lambda record: (record["distance"], record["cluster"]))


def load_strategy(path: Path = DEFAULT_STRATEGY_PATH) -> dict[str, dict[str, int]]:
    strategy = load_json(path)
    required = {"category_map", "verbnet_map", "verb_map"}
    missing = required.difference(strategy)
    if missing:
        raise ValueError(f"Tokenization strategy is missing keys: {sorted(missing)}")
    return strategy


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def record_phrase(record: dict[str, Any]) -> str:
    for key in ("subtask", "phrase", "plan_step"):
        value = record.get(key)
        if value:
            return str(value)
    raise KeyError("Batch records must contain one of: subtask, phrase, plan_step.")


def record_motion_code(record: dict[str, Any]) -> str:
    for key in ("motion_code", "code"):
        value = record.get(key)
        if value:
            return validate_motion_code(str(value))
    raise KeyError("Cluster records must contain a motion_code field.")


def choose_medoid(codes: list[str], counts: Counter[str]) -> str:
    return min(
        codes,
        key=lambda candidate: (
            sum(motion_code_distance(candidate, code) * counts[code] for code in codes),
            -counts[candidate],
            candidate,
        ),
    )


def build_motion_code_clusters(records: list[dict[str, Any]], num_clusters: int) -> dict[str, Any]:
    """Build deterministic weighted k-medoids centers from annotated motion codes."""
    codes = [record_motion_code(record) for record in records]
    counts: Counter[str] = Counter(codes)
    unique_codes = sorted(counts)
    if not unique_codes:
        raise ValueError("No motion_code values found.")
    if num_clusters < 1:
        raise ValueError("--num-motion-clusters must be at least 1.")
    if num_clusters > len(unique_codes):
        raise ValueError(
            f"Requested {num_clusters} clusters, but only {len(unique_codes)} unique motion codes are present."
        )

    centers = [min(unique_codes, key=lambda code: (-counts[code], code))]
    while len(centers) < num_clusters:
        remaining = [code for code in unique_codes if code not in centers]
        next_center = max(
            remaining,
            key=lambda code: (min(motion_code_distance(code, center) for center in centers), counts[code], code),
        )
        centers.append(next_center)

    assignments: dict[str, list[str]] = {}
    for _ in range(100):
        assignments = {center: [] for center in centers}
        for code in unique_codes:
            best_center = min(centers, key=lambda center: (motion_code_distance(code, center), center))
            assignments[best_center].append(code)

        new_centers = sorted(choose_medoid(cluster_codes, counts) for cluster_codes in assignments.values())
        if new_centers == sorted(centers):
            centers = new_centers
            break
        centers = new_centers

    assignments = {center: [] for center in centers}
    for code in unique_codes:
        best_center = min(centers, key=lambda center: (motion_code_distance(code, center), center))
        assignments[best_center].append(code)

    ordered_centers = sorted(centers)
    return {
        "weights": list(MOTION_CODE_WEIGHTS),
        "distance_rule": (
            "Digits 1, 2, and 4 use full weight for zero/nonzero changes and half weight "
            "for nonzero/nonzero changes; digits 3, 5, and 6 use full weight for any change."
        ),
        "centers": [
            {
                "center": center,
                "cluster": f"cluster_{idx}",
                "count": counts[center],
                "members": [
                    {
                        "motion_code": code,
                        "count": counts[code],
                        "distance": motion_code_distance(code, center),
                    }
                    for code in sorted(assignments[center])
                ],
            }
            for idx, center in enumerate(ordered_centers)
        ],
    }


def tokenize_jsonl_records(
    records: list[dict[str, Any]],
    *,
    strategy: dict[str, dict[str, int]],
    skill_graph: dict[str, Any] | None,
    use_llm: bool,
    include_cluster_distances: bool,
) -> list[dict[str, Any]]:
    phrases = [record_phrase(record) for record in records]
    motion_codes: list[str | None] = [record.get("motion_code") for record in records]

    missing_indices = [idx for idx, code in enumerate(motion_codes) if not code]
    if missing_indices:
        if not use_llm:
            raise ValueError("Every batch record must contain motion_code unless --use-llm is set.")
        annotated_codes = annotate_motion_codes_with_llm([phrases[idx] for idx in missing_indices])
        for idx, code in zip(missing_indices, annotated_codes, strict=True):
            motion_codes[idx] = code

    outputs = []
    for record, phrase, motion_code in zip(records, phrases, motion_codes, strict=True):
        tokenized = tokenize_phrase(
            phrase,
            validate_motion_code(str(motion_code)),
            strategy,
            skill_graph,
            include_cluster_distances,
        )
        outputs.append({**record, **tokenized})
    return outputs


def build_strategy_from_graph(graph_path: Path) -> dict[str, dict[str, int]]:
    graph = load_json(graph_path)
    category_map: dict[str, int] = {}
    verbnet_map: dict[str, int] = {}
    verb_map: dict[str, int] = {}

    for category, category_content in graph.items():
        if category not in category_map:
            category_map[category] = int(category.split("_")[1])
        for verbnet_class, verbnet_content in category_content.items():
            if verbnet_class not in verbnet_map:
                verbnet_map[verbnet_class] = len(verbnet_map)
            for verb in verbnet_content:
                if verb not in verb_map:
                    verb_map[verb] = len(verb_map)

    return {
        "category_map": category_map,
        "verbnet_map": verbnet_map,
        "verb_map": verb_map,
    }


def choose_verbnet_from_graph(verb: str, graph: dict[str, Any]) -> str | None:
    counts: dict[str, int] = {}
    for category_content in graph.values():
        for verbnet_class, verbnet_content in category_content.items():
            if verb in verbnet_content:
                counts[verbnet_class] = counts.get(verbnet_class, 0) + len(verbnet_content[verb])
    if not counts:
        return None
    best_class, _ = max(counts.items(), key=lambda item: item[1])
    return best_class if best_class != "unknown" else None


def choose_verbnet_from_nltk(verb: str, strategy: dict[str, dict[str, int]]) -> str:
    try:
        from nltk.corpus import verbnet
    except Exception:
        return "unknown"

    try:
        for class_id in verbnet.classids(verb):
            if class_id in strategy["verbnet_map"]:
                return class_id
    except Exception:
        return "unknown"
    return "unknown"


def choose_verbnet_class(
    phrase: str,
    strategy: dict[str, dict[str, int]],
    skill_graph: dict[str, Any] | None = None,
) -> str:
    verb = first_verb(phrase)
    if skill_graph is not None:
        graph_class = choose_verbnet_from_graph(verb, skill_graph)
        if graph_class is not None:
            return graph_class
    return choose_verbnet_from_nltk(verb, strategy)


def tokenize_phrase(
    phrase: str,
    motion_code: str,
    strategy: dict[str, dict[str, int]],
    skill_graph: dict[str, Any] | None = None,
    include_cluster_distances: bool = False,
) -> dict[str, Any]:
    phrase = normalize_phrase(phrase)
    verb = first_verb(phrase)
    cluster = assign_motion_cluster(motion_code)
    verbnet_class = choose_verbnet_class(phrase, strategy, skill_graph)

    category_id = strategy["category_map"][cluster]
    verbnet_id = strategy["verbnet_map"].get(verbnet_class, strategy["verbnet_map"].get("unknown"))
    verb_id = strategy["verb_map"].get(verb)

    if verbnet_id is None:
        raise KeyError("The strategy does not contain an 'unknown' VerbNet class.")
    if verb_id is None:
        raise KeyError(f"Verb {verb!r} is not in the released verb map.")

    record = {
        "phrase": phrase,
        "motion_code": motion_code,
        "motion_cluster": cluster,
        "verbnet_class": verbnet_class,
        "verb": verb,
        "tokens": [category_id, verbnet_id, verb_id],
    }
    if include_cluster_distances:
        record["motion_cluster_distances"] = motion_cluster_distances(motion_code)
    return record


def annotate_motion_codes_with_llm(phrases: list[str]) -> list[str]:
    endpoint = os.environ.get("SKILLNET_LLM_BASE_URL")
    api_key = os.environ.get("SKILLNET_LLM_API_KEY")
    model = os.environ.get("SKILLNET_LLM_MODEL", "gpt-5-mini")
    if not endpoint or not api_key:
        raise RuntimeError(
            "Set SKILLNET_LLM_BASE_URL and SKILLNET_LLM_API_KEY, or provide "
            "--motion-code explicitly."
        )

    prompt = MOTION_CODE_PROMPT.format(n_items=len(phrases))
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt + "\n\nInput:\n" + " ## ".join(phrases),
            }
        ],
        "max_tokens": 256,
        "temperature": 0,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    content = result["choices"][0]["message"]["content"].strip()
    codes = [validate_motion_code(code) for code in content.split()]
    if len(codes) != len(phrases):
        raise RuntimeError(f"Expected {len(phrases)} motion codes, got {len(codes)}: {content!r}")
    return codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY_PATH)
    parser.add_argument("--skill-graph", type=Path, default=None, help="Optional skill_graph.json for VerbNet disambiguation.")
    parser.add_argument("--phrase", action="append", default=[], help="Subtask phrase, e.g. 'pick red block'. Repeatable.")
    parser.add_argument("--motion-code", action="append", default=[], help="6-digit motion code. Repeatable, aligned with --phrase.")
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Batch input JSONL with subtask/phrase and motion_code fields.")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Write batch tokenization records as JSONL.")
    parser.add_argument("--use-llm", action="store_true", help="Annotate motion codes through an OpenAI-compatible endpoint.")
    parser.add_argument(
        "--show-cluster-distances",
        action="store_true",
        help="Include weighted distances from the motion code to all released cluster centers.",
    )
    parser.add_argument("--build-strategy-from-graph", type=Path, default=None)
    parser.add_argument("--output-strategy", type=Path, default=None)
    parser.add_argument(
        "--build-motion-clusters",
        type=Path,
        default=None,
        help="Build deterministic weighted k-medoids centers from a JSONL file containing motion_code fields.",
    )
    parser.add_argument("--num-motion-clusters", type=int, default=12)
    parser.add_argument("--output-motion-clusters", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_strategy_from_graph is not None:
        strategy = build_strategy_from_graph(args.build_strategy_from_graph)
        output = args.output_strategy or args.strategy
        save_json(output, strategy)
        print(f"Wrote tokenization strategy to {output}")
        return

    if args.build_motion_clusters is not None:
        records = load_jsonl(args.build_motion_clusters)
        clusters = build_motion_code_clusters(records, args.num_motion_clusters)
        if args.output_motion_clusters is not None:
            save_json(args.output_motion_clusters, clusters)
            print(f"Wrote motion-code clusters to {args.output_motion_clusters}")
        else:
            print(json.dumps(clusters, indent=2, ensure_ascii=False))
        return

    strategy = load_strategy(args.strategy)
    skill_graph = load_json(args.skill_graph) if args.skill_graph else None

    if args.input_jsonl is not None:
        records = load_jsonl(args.input_jsonl)
        outputs = tokenize_jsonl_records(
            records,
            strategy=strategy,
            skill_graph=skill_graph,
            use_llm=args.use_llm,
            include_cluster_distances=args.show_cluster_distances,
        )
        if args.output_jsonl is not None:
            write_jsonl(args.output_jsonl, outputs)
            print(f"Wrote tokenized records to {args.output_jsonl}")
        else:
            print(json.dumps(outputs, indent=2, ensure_ascii=False))
        return

    if not args.phrase:
        raise SystemExit("Provide at least one --phrase or --input-jsonl.")

    if args.use_llm:
        motion_codes = annotate_motion_codes_with_llm(args.phrase)
    else:
        motion_codes = [validate_motion_code(code) for code in args.motion_code]

    if len(motion_codes) != len(args.phrase):
        raise SystemExit("The number of --motion-code values must match --phrase values.")

    records = [
        tokenize_phrase(phrase, motion_code, strategy, skill_graph, args.show_cluster_distances)
        for phrase, motion_code in zip(args.phrase, motion_codes)
    ]
    print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
