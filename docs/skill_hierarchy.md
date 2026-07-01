# Skill Hierarchy

SkillNet represents each high-level manipulation subtask with a compact
hierarchical token. This document describes the public reproduction path for
the hierarchy construction used by the release.

## Pipeline

The hierarchy is built in three stages:

1. Decompose an instruction into short manipulation subtasks, typically
   `verb object` phrases such as `pick block`, `open drawer`, or `place bowl`.
2. Annotate each subtask with a 6-digit motion code that captures physical
   interaction semantics.
3. Map the motion code, VerbNet class, and verb into integer tokens with the
   released tokenization strategy.

The public tokenizer lives in:

```bash
data_process/skill_hierarchy/skill_hierarchy_tokenizer.py
```

The released mapping file is:

```bash
data_process/skill_hierarchy/tokenization_strategy.json
```

`tokenization_strategy.json` is the canonical released artifact for public
reproduction. It contains 13 motion category ids, 127 VerbNet ids, and 267 verb
ids. Its SHA256 checksum is:

```text
13d025527a240738e20be3a5e2a208157d623250667db6fc8949830a1ffd8081
```

The full experiment skill graph used to produce that mapping contains task
examples from the paper data mixture and is not required for applying the
released tokenizer. A small public graph with the same schema is included at
`data_process/skill_hierarchy/skill_graph_example.json` for format checks and
for testing the strategy builder; it is not intended to regenerate the full
released strategy exactly.

The released LIBERO and RoboTwin training configs consume flat integer skill-id
sequences plus masks, because that is the format stored in the public LeRobot
datasets and expected by the released Skill-MoE checkpoints. The hierarchical
`[motion_cluster_id, verbnet_class_id, verb_id]` tokens in this document are
provided as the public hierarchy-construction path and as metadata for analysis
and hierarchy-token experiments; do not replace the flat `skills` field in the
released configs with this three-token tuple unless you also update the model
input contract.

## Motion Code

The 6-digit code describes:

| Digit | Meaning |
| --- | --- |
| 1 | Contact time |
| 2 | Deformation type |
| 3 | Arm fixed-axis rotation |
| 4 | Object constrained translation |
| 5 | Object fixed-axis rotation |
| 6 | Tool usage |

Allowed values:

| Digit | Values |
| --- | --- |
| 1 | `0` no contact, `1` short contact, `2` long contact |
| 2 | `0` non-permanent, `1` plastic deformation, `2` rigid separation/cutting/fracture |
| 3 | `0` no arm fixed-axis rotation, `1` arm fixed-axis rotation |
| 4 | `0` static object, `1` 1D constrained translation, `2` unconstrained 3D motion |
| 5 | `0` no object fixed-axis rotation, `1` object fixed-axis rotation |
| 6 | `0` no tool, `1` tool used |

Manual annotation protocol:

1. Split a task instruction into short manipulation subtasks before assigning
   motion codes. Each subtask should contain one dominant verb phrase such as
   `pick up the bowl`, `open the drawer`, or `turn off the stove`.
2. Assign the 6-digit motion code from the physical effect of that subtask, not
   from object names alone. For example, `pick up the bowl` is long contact with
   unconstrained 3D object motion, while `open the drawer` is long contact with
   1D constrained translation.
3. Validate that every digit is in the allowed value set and keep the raw
   subtask text next to the code. `motion_code_annotation_examples.jsonl`
   provides small calibration examples for this schema.
4. If using the optional LLM endpoint, treat the model output as a proposal.
   The released tokenizer validates the digit format, but human or scripted
   review should still check the physical interpretation before using the code
   for training data.

The released centers and weights are embedded in
`skill_hierarchy_tokenizer.py` and recorded in
`data_process/skill_hierarchy/motion_code_clusters.json`. A subtask's motion
code is assigned to the nearest motion cluster by the weighted distance
function.

The 12 released motion-code centers are:

| Center | Cluster |
| --- | --- |
| `200200` | `cluster_0` |
| `200100` | `cluster_1` |
| `100200` | `cluster_2` |
| `200010` | `cluster_3` |
| `100100` | `cluster_4` |
| `201010` | `cluster_5` |
| `200001` | `cluster_6` |
| `201201` | `cluster_7` |
| `000001` | `cluster_8` |
| `220001` | `cluster_9` |
| `100001` | `cluster_10` |
| `200201` | `cluster_11` |

The weighted distance uses weights
`[3.056, 1.539, 1.350, 2.844, 1.039, 2.632]`. For digits 1, 2, and 4, changing
between zero and nonzero receives the full weight, while changing between two
nonzero values receives half weight. For digits 3, 5, and 6, any change
receives the full weight.

The public release freezes these centers from the SkillNet paper data. If you
want to rebuild centers for a new annotation corpus, use the same 6-digit
schema and report the resulting center set together with the released distance
rule.

## Tokenization Strategy

The final token for a subtask is:

```text
[motion_cluster_id, verbnet_class_id, verb_id]
```

`tokenization_strategy.json` provides the released id maps:

- `category_map`: motion cluster to integer id.
- `verbnet_map`: VerbNet class to integer id.
- `verb_map`: surface verb to integer id.

The released strategy has 13 motion category ids, 127 VerbNet ids, and 267 verb
ids. The default motion-code centers cover the 12 clusters used by the public
motion-code assignment; the extra category id is kept for compatibility with
the released strategy file.

The released strategy is a frozen vocabulary. If a phrase uses a surface verb
that is absent from `verb_map`, the tokenizer raises an error instead of
inventing a new id. Extend or rebuild the strategy before tokenizing new
corpora with verbs outside the released vocabulary.

## Commands

Tokenize a phrase with a known motion code:

```bash
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --phrase "pick red block" \
  --motion-code 200200 \
  --skill-graph data_process/skill_hierarchy/skill_graph_example.json
```

Inspect the weighted clustering decision:

```bash
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --phrase "open door" \
  --motion-code 200010 \
  --show-cluster-distances
```

Use a graph to disambiguate VerbNet classes from examples:

```bash
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --phrase "open door" \
  --motion-code 200010 \
  --skill-graph data_process/skill_hierarchy/skill_graph_example.json
```

Batch-tokenize annotated subtasks:

```bash
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --input-jsonl data_process/skill_hierarchy/motion_code_annotation_examples.jsonl \
  --skill-graph data_process/skill_hierarchy/skill_graph_example.json \
  --output-jsonl ./skillnet_tokenized_examples.jsonl
```

Each input JSONL row should contain `subtask` or `phrase`, plus `motion_code`.
Optional metadata fields such as `task_id` are copied to the output.

Use an optional OpenAI-compatible endpoint to annotate the motion code:

```bash
export SKILLNET_LLM_BASE_URL=https://your-endpoint/v1/chat/completions
export SKILLNET_LLM_API_KEY=your_key
export SKILLNET_LLM_MODEL=your_model

python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --phrase "open drawer" \
  --use-llm
```

Inspect the public graph schema and regenerate a toy strategy:

```bash
python data_process/skill_hierarchy/skill_hierarchy_tokenizer.py \
  --build-strategy-from-graph data_process/skill_hierarchy/skill_graph_example.json \
  --output-strategy ./skillnet_tokenization_strategy_example.json
```

To regenerate a full strategy for a new corpus, provide a full graph with the
same schema as `skill_graph_example.json`. Its top-level keys are motion
clusters, the next level is VerbNet classes, the third level is verbs, and each
verb stores example records. The small example graph is only a schema and smoke
test artifact, so it will not reproduce the released checksum above.

The public release does not include private API keys, private endpoints, or
machine-specific paths. Users who want automatic motion-code annotation should
provide their own endpoint through environment variables.
