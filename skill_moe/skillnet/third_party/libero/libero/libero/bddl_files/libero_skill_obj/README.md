# LIBERO-Skill Assets

This directory contains bddl assets used while constructing the SkillNet
LIBERO-Skill benchmark. The public reported benchmark is not every bddl file in
this directory.

Authoritative public benchmark:

- Benchmark key: `libero_skill_obj`
- Task count: 9
- Machine-readable task order: `public_task_manifest.json`
- Runtime registration: `../../benchmark/libero_suite_task_map.py`
- Skill/object annotations:
  `../../../../../../examples/libero/annotations/libero_skill_obj_annotations.json`

The registered task ids are compact aliases (`skill_obj_01` through
`skill_obj_09`) so the public checkout remains portable on path-length-limited
platforms. `public_task_manifest.json` records the original generated
`source_task` names and language strings, and each bddl file keeps the language
inside its `:language` field.

`tasks_info.txt` is an asset inventory for bundled bddl files and includes
auxiliary candidate tasks. It should not be used as the reported LIBERO-Skill
evaluation list. The evaluator validates that the installed benchmark exactly
matches the 9 tasks in `public_task_manifest.json` before running rollouts.
