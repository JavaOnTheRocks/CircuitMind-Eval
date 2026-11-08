
---
license: mit
task_categories:
  - visual-question-answering
language:
  - en
pretty_name: CircuitMind-Eval
size_categories:
  - 10K<n<100K
---

# CircuitMind-Eval

A benchmark dataset for evaluating AI models on
image-based electrical circuit reasoning.

## Dataset Structure

- dataset/questions.json: Benchmark questions.
- dataset/metadata.json: Task metadata.
- dataset/images/: Circuit diagrams.

## Tasks

The benchmark contains 10 electrical circuit reasoning
problems covering:

- Series and parallel circuits
- Nodal analysis
- Mesh analysis
- Thevenin and Norton theorems
- Superposition
- Maximum power transfer
- Bridge circuits

## Evaluation

Models are evaluated on their ability to interpret
circuit diagrams, apply electrical engineering
principles, and produce correct numerical answers.
