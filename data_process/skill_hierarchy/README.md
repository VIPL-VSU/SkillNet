# Skill Hierarchy Tokenization

This directory contains the public SkillNet skill hierarchy tokenizer. It turns
short manipulation subtasks such as `pick red block` into hierarchical skill
tokens used by SkillNet.

## Files

- `skill_hierarchy_tokenizer.py`: sanitized tokenizer and strategy-regeneration
  CLI. It contains no hard-coded API keys or private endpoints.
- `tokenization_strategy.json`: released mapping from hierarchy labels to
  integer ids.
- `skill_graph_example.json`: small public graph showing the expected
  `cluster -> VerbNet class -> verb -> examples` schema.
- `motion_code_clusters.json`: released motion-code centers and distance
  weights used for nearest-center assignment.
- `motion_code_annotation_examples.jsonl`: small public annotation examples for
  calibrating and smoke-testing the 6-digit motion-code schema.

## Hierarchy

SkillNet uses three semantic levels:

1. Motion-code cluster id.
2. VerbNet class id.
3. Verb id.

For a subtask phrase, the tokenizer outputs:

```text
[motion_cluster_id, verbnet_class_id, verb_id]
```

The motion-code layer starts from a 6-digit action code:

| Digit | Meaning | Values |
| --- | --- | --- |
| 1 | Contact time | `0` no contact, `1` short contact, `2` long contact |
| 2 | Deformation type | `0` non-permanent, `1` plastic, `2` rigid separation/cutting/fracture |
| 3 | Arm fixed-axis rotation | `0` no, `1` yes |
| 4 | Object constrained translation | `0` static, `1` 1D constrained, `2` unconstrained 3D motion |
| 5 | Object fixed-axis rotation | `0` no, `1` yes |
| 6 | Tool usage | `0` no, `1` yes |

The released tokenizer assigns the motion code to the nearest cluster center
with the weighted distance used in the SkillNet experiments.

## Tokenize a Phrase

If the motion code is already known:

```bash
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --phrase "pick red block" \
  --motion-code 200200 \
  --skill-graph data_process/skill_hierarchy/skill_graph_example.json
```

The output is a JSON record containing the motion cluster, VerbNet class, verb,
and the final integer token triplet.

To inspect the weighted clustering decision, add
`--show-cluster-distances`:

```bash
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --phrase "open door" \
  --motion-code 200010 \
  --show-cluster-distances
```

If a skill graph is available, pass it to select VerbNet classes using the graph
statistics before falling back to NLTK:

```bash
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --phrase "open door" \
  --motion-code 200010 \
  --skill-graph data_process/skill_hierarchy/skill_graph_example.json
```

## Batch Tokenization

Batch input is JSONL. Each row should contain `subtask` or `phrase`, plus
`motion_code`; optional metadata fields such as `task_id` are copied to the
output.

```bash
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --input-jsonl data_process/skill_hierarchy/motion_code_annotation_examples.jsonl \
  --skill-graph data_process/skill_hierarchy/skill_graph_example.json \
  --output-jsonl ./skillnet_tokenized_examples.jsonl
```

The output JSONL adds `motion_cluster`, `verbnet_class`, `verb`, and `tokens`.
The released strategy is a frozen vocabulary: if a phrase uses a verb absent
from `verb_map`, the tokenizer raises an error instead of inventing a new id.
Extend or rebuild the strategy before tokenizing new corpora with new verbs.

## Optional LLM Annotation

To reproduce automatic motion-code annotation, provide your own
OpenAI-compatible endpoint through environment variables:

```bash
export SKILLNET_LLM_BASE_URL=https://your-endpoint/v1/chat/completions
export SKILLNET_LLM_API_KEY=your_key
export SKILLNET_LLM_MODEL=your_model

python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --phrase "cut apple" \
  --use-llm
```

The repository intentionally does not contain any private endpoint, API key, or
machine-specific path.

## Rebuild or Inspect a Strategy

The released `tokenization_strategy.json` is the canonical public artifact. It
contains 13 motion category ids, 127 VerbNet ids, and 267 verb ids. Its SHA256
checksum is:

```text
13d025527a240738e20be3a5e2a208157d623250667db6fc8949830a1ffd8081
```

The released strategy was generated from the full skill graph used in the
SkillNet experiments. That full graph is not required for tokenizing new
subtasks with the released strategy. To inspect the graph schema or test the
strategy builder, use the included example graph:

```bash
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --build-strategy-from-graph data_process/skill_hierarchy/skill_graph_example.json \
  --output-strategy ./skillnet_tokenization_strategy_example.json
```

To regenerate a full strategy for a new corpus, provide your own graph with the
same schema. The top-level keys are motion clusters, the next level is VerbNet
classes, the third level is verbs, and each verb stores example records. The
example graph is intentionally small and will not reproduce the released
checksum.
