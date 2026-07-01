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

`tasks_info.txt` is an asset inventory for bundled bddl files and includes
auxiliary candidate tasks. It should not be used as the reported LIBERO-Skill
evaluation list. The evaluator validates that the installed benchmark exactly
matches the 9 tasks in `public_task_manifest.json` before running rollouts.
