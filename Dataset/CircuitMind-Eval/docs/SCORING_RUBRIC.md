# CircuitMind Eval v0.2.0 — Scoring Rubric

Each task is scored 0–100. Grade numeric fields deterministically against `ground_truth.json`, using the tolerances included for each task. For reasoning fields, use a fixed rubric and human review; if an LLM judge is used, publish its exact model ID, prompt, settings, and calibration agreement.

| ID | Task | Scoring dimensions |
|---|---|---|
| CM001 | Series | Req 20; current 20; voltage drops 40; KVL 20 |
| CM002 | Parallel | Req 20; total/branch currents 40; branch voltage 20; KCL 20 |
| CM003 | User-supplied multi-source circuit | polarity/reference interpretation 20; supernode relation 15; node voltages 20; R3 current/direction 15; power 20; derivation 10 |
| CM004 | Nodal analysis | reference/node identification 15; KCL equation 20; Va 25; currents 30; units 10 |
| CM005 | Thevenin | open-circuit Vth 20; source deactivation 20; Rth 20; load current 20; load voltage 20 |
| CM006 | Superposition | V1 contribution 25; V2 contribution 25; total Va 20; load current 15; linear superposition check 15 |
| CM007 | Norton | IN 25; RN 20; load current 20; load voltage 15; current-division verification 20 |
| CM008 | Mesh | mesh setup 20; equations 20; node voltage 15; branch currents and directions 30; KCL verification 15 |
| CM009 | Maximum power transfer | condition RL=Rth 25; optimal load 20; maximum power 25; load voltage 15; load current 15 |
| CM010 | GATE-level bridge | topology/reference 10; nodal equations 25; VL and VR 25; bridge current/direction 15; source current 10; KCL checks 15 |

## Numerical grading rules
- Compare numeric answers to the exact reference values with the per-task absolute tolerances.
- Accept equivalent units and reasonable rounding after unit normalization.
- Current direction must be correct. A signed answer is acceptable if the reference direction is clearly stated.
- Do not award full reasoning points for a lucky numeric guess.
- Missing fields earn zero for that field; never infer an answer the model did not provide.

## Critical-error flags
Record separately from the numeric score:
1. CM003: ignoring or reversing a source polarity shown in the diagram.
2. CM003: reporting negative power absorbed by a passive resistor without explaining the sign convention.
3. CM006: failing to deactivate the other ideal voltage source correctly during superposition.
4. CM010: assuming a balanced bridge or omitting bridge current despite the nonzero bridge branch.

## Aggregate results
Report macro-average score, easy/medium/hard averages, per-task score, numeric within-tolerance rate, critical-error rate, and raw model responses. Freeze dataset, prompts, model versions, and settings before benchmark runs.
