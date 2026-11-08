# CircuitMind-Eval
### A Benchmark for Evaluating Frontier Models on Electrical and Electronic Circuit Understanding and Network Theory Reasoning


**CircuitMind-Eval** is an evaluation benchmark designed to investigate how reliably frontier models (VLMs) understand electrical circuit diagrams and solve problems involving DC circuit analysis and network theory and AC Circuit Analysis,Analog Electronics, Digital Electronics.

The benchmark evaluates more than whether a model produces a plausible final answer. It examines whether the model can interpret a circuit, identify its electrical relationships, formulate appropriate equations, perform calculations correctly, and maintain physical consistency in its explanation.

The project compares two frontier models using a common dataset, standardized prompts, deterministic numerical scoring, and an independent AI-based evaluation process.

---

## 1. The Research Gap

### What can an electrical and electronics engineering expert do reliably?

An electrical & electronic  engineering student, engineer, or domain expert can approach a circuit problem through a structured reasoning process:

1. Interpret the circuit diagram and identify its components.
2. Recognize circuit topology, nodes, branches, loops, and source polarities.
3. Select the appropriate analytical method.
4. Formulate the required network equations.
5. Solve the equations while preserving current directions and voltage polarities.
6. Verify the result using electrical laws and physical relationships.

For example, when solving a circuit involving nodal analysis, an expert identifies the essential nodes, selects a reference node, applies Kirchhoff's Current Law (KCL), constructs the node equations, and solves for the unknown voltages.

A frontier model may recognize the components and produce a convincing explanation, but still:

- Misinterpret the circuit topology.
- Reverse a source polarity.
- Assign an incorrect current direction.
- Formulate an incorrect equation.
- Make mistakes in simultaneous equations.
- Produce incorrect numerical values despite apparently correct reasoning.
- Violate physical consistency while presenting a confident answer.
- Return malformed or incomplete structured output.

This creates an important evaluation gap.

### The gap we investigate

Traditional answer-based evaluation can overlook the difference between a correct-looking explanation and a technically correct electrical engineering solution.

CircuitMind-Eval investigates whether models can reliably connect:

**Visual circuit interpretation → Electrical relationships → Mathematical formulation → Numerical solution → Physical verification.**

Our objective is to measure these capabilities independently rather than relying exclusively on a final-answer score.

The benchmark is an initial evaluation study, not a claim that all models fail at circuit reasoning or that human experts never make mistakes.

---

## 2. What Is CircuitMind-Eval?

CircuitMind-Eval is an evaluation suite for models electrical circuit reasoning.

It contains circuit images, problem statements, expected output fields, and reference answers. Models receive the same circuit image and question and are required to return a structured answer.

The evaluation pipeline consists of:

1. Dataset loading and validation.
2. Image-based model inference.
3. Structured response validation.
4. Deterministic numerical evaluation.
5. Independent AI-judge evaluation.
6. Metric calculation.
7. Model-level and question-level performance reporting.

The architecture is designed to support additional questions without hardcoding the original dataset size.

### Evaluation workflow

<img src="mermaid-diagram (1).png" alt="Evaluation workflow" style="max-width: 50%;">

Both models are evaluated using the same dataset and evaluation criteria.

---

## 3. Domain and Technical Expertise

CircuitMind-Eval draws on electrical and Electronic engineering, particularlyDC circuit analysis and network theory and AC Circuit Analysis,Analog Electronics, Digital Electronics.

The benchmark currently covers the following problem categories:

for this **micro1 Eval Hackathon** the Dataset we constructed Just included the **Basic Network Theroy circuit Problem**.


| Circuit analysis topic | What it evaluates |
|---|---|
| Series circuits | Equivalent resistance, loop current and voltage drops |
| Parallel circuits | Equivalent resistance, branch currents and KCL |
| Multi-source circuits | Source polarity, node voltages and power |
| Nodal analysis | Node-voltage equations and current relationships |
| Thevenin's theorem | Open-circuit voltage and equivalent resistance |
| Superposition theorem | Individual source contributions and combined response |
| Norton's theorem | Norton current, equivalent resistance and load current |
| Mesh analysis | Loop equations and shared-branch currents |
| Maximum power transfer | Optimal load and maximum power |
| Bridge circuits | Unbalanced bridge analysis and node relationships |


### Electrical principles used

The benchmark draws on the following principles:

**Ohm's Law**

V = IR

**Kirchhoff's Current Law (KCL)**

The algebraic sum of currents at a node is zero.

**Kirchhoff's Voltage Law (KVL)**

The algebraic sum of voltages around a closed loop is zero.

**Series resistance**

R_eq = R_1 + R_2 + ... + R_n

**Parallel resistance**

1/R_eq = 1/R_1 + 1/R_2 + ... + 1/R_n

**Electrical power**

P = VI = I²R = V²/R

These principles provide the technical foundation for constructing reference solutions and evaluating model outputs.

---

## 4. Dataset Preparation

### Dataset design

The dataset is organized around electrical circuit problems rather than general-purpose question answering.

Each example contains:

- A circuit diagram image.
- A corresponding problem statement.
- A unique task identifier.
- Required output fields.
- Reference numerical answers.
- Expected units and sign conventions.
- Numerical tolerances where applicable.
- Additional derivation or explanation requirements when needed.

### Current dataset structure

```text
Dataset/
└── CircuitMind-Eval/
    └── dataset/
        ├── images/
        ├── questions.json
        ├── ground_truth.json
        ├── metadata.json
        └── data.jsonl

```


| File | Purpose |
|---|---|
| `images/` | Circuit diagram images |
| `questions.json` | Questions and required output fields |
| `ground_truth.json` | Reference solutions |
| `metadata.json` | Dataset information and evaluation metadata |
| `data.jsonl` | Line-oriented dataset representation |


### How the reference dataset is prepared

The benchmark uses a structured preparation process:

1. Select problems representing different network theory concepts.
2. Prepare or curate the corresponding circuit diagrams.
3. Associate each diagram with a clearly defined engineering question.
4. Identify the expected output fields for that question.
5. Establish reference numerical answers using the relevant circuit equations.
6. Specify tolerances for numerical comparisons.
7. Validate that the image, question, and ground truth refer to the same task.

A key design decision is that not every circuit question has the same answer structure.

For example, a Thevenin problem requires the Thevenin voltage and equivalent resistance, whereas a mesh-analysis problem may require mesh equations, branch currents, and a KCL verification.

Therefore, the evaluator uses task-specific required fields rather than assuming that every question has an identical output schema.

### Important dataset considerations

The benchmark preserves signed numerical quantities and current-direction conventions. For example, a negative current can represent a current flowing opposite to the assumed reference direction; it must not automatically be treated as an incorrect magnitude.

Numerical tolerances are used to accommodate small floating-point differences without accepting materially incorrect electrical results.

**Current evaluated dataset size:** 10 questions (CM001–CM010).

The pipeline is designed for expansion to larger datasets. Any expanded dataset must be validated before running the benchmark.

### Dataset access

**Hugging Face Dataset:** [https://huggingface.co/datasets/rakesh-bishnoi29/circuitmind-eval/]

The public dataset should include the circuit images, question definitions, metadata, and ground-truth annotations needed to reproduce the evaluation, subject to the dataset's licensing and redistribution permissions.

---

## Future Dataset Expansion: Toward a Comprehensive Electrical & Electronics Engineering Benchmark

The current CircuitMind-Eval dataset focuses on DC circuit analysis and electrical DC network theory, AC circuit analysis. However, evaluating frontier models on electrical engineering requires a broader range of circuit configurations, electronic components across like Analog electronics and Digital electronics , and problem-solving methods.

We therefore plan to expand CircuitMind-Eval into a multi-domain electrical and electronics engineering benchmark.

### Planned domains

**1. DC Electrical Network Theory**

The existing dataset covers fundamental network analysis, including KCL, KVL, nodal analysis, mesh analysis, Thevenin's theorem, Norton's theorem, superposition, bridge circuits, and maximum power transfer.

**2. AC Circuit Analysis**

The expanded dataset will include alternating-current circuit problems involving impedance, reactance, phasor analysis, series and parallel RLC circuits, resonance, power factor, frequency response, and AC power calculations.

**3. Analog Electronics**

This domain will evaluate whether models can interpret semiconductor and analog circuit diagrams and solve problems involving diode circuits, rectifiers, clipping and clamping circuits, BJT and MOSFET configurations, transistor biasing, operational amplifiers, amplifiers, and filters.

**4. Digital Electronics**

This domain will investigate visual and logical reasoning through logic gates, Boolean algebra, truth tables, Karnaugh maps, combinational circuits, multiplexers, adders, flip-flops, counters, and sequential logic.

### Why this expansion matters

Electrical engineering problems require different forms of reasoning.

DC network analysis requires electrical equation formulation and numerical solving. AC circuit analysis introduces complex impedance and phase relationships. Analog electronics requires understanding semiconductor behavior and circuit operating conditions. Digital electronics requires logical interpretation, Boolean reasoning, and verification of output states.

A comprehensive benchmark should evaluate these capabilities separately instead of assuming that success in one type of circuit implies competence in every other domain.

### Domain-specific evaluation

The expanded benchmark will retain the existing deterministic scoring and independent AI-judge framework while introducing domain-specific evaluation criteria.

Examples include:

- Numerical and equation accuracy for DC and AC circuit problems.
- Component and operating-condition understanding for analog electronics.
- Boolean expression, truth-table, and logical-output accuracy for digital electronics.
- Visual circuit interpretation across all four domains.

Results will be reported both at the overall benchmark level and separately for each domain.

This expansion aims to create a more comprehensive evaluation framework for investigating the electrical and electronics engineering reasoning capabilities and limitations of frontier  models.


## 5. What Does the Benchmark Measure?

A central design decision is to evaluate models across multiple dimensions.

A model may obtain a reasonable semantic score while making numerical errors. Conversely, it may calculate a correct value without providing a complete or convincing derivation.

For this reason, CircuitMind-Eval uses deterministic numerical evaluation and AI-based reasoning evaluation as complementary components.


### 5.1 Primary metric: Numerical Accuracy Among Scored Fields

This is the primary metric used for comparing numerical problem-solving performance.

It measures how many numerical fields evaluated by the deterministic scorer match their corresponding reference values within the specified tolerances.

**Formula:**

Numerical Accuracy (%) =

(Number of Correct Numerical Fields / Number of Scored Numerical Fields) × 100

A numerical field is counted as correct when the prediction satisfies the applicable numerical tolerance.

For a field with reference value G and predicted value P, the evaluator checks the numerical difference against the configured tolerance.

The benchmark supports field-specific tolerances and numerical comparison settings. The exact scoring implementation and precedence of these tolerances should be taken from the published scorer code.

**Why it matters:**

A model might explain nodal analysis correctly but calculate the node voltage incorrectly. Numerical accuracy exposes this failure even when the response sounds convincing .

The benchmark also reports dataset-wide numerical accuracy:

Dataset-wide Numerical Accuracy (%) =

(Correct Numerical Fields / Total Expected Numerical Fields Across the Dataset) × 100

Unlike accuracy among scored fields, the dataset-wide denominator retains expected fields from questions that were not successfully scored. This makes missing and invalid responses visible in the overall result.

Both values are reported because they answer different questions:

- Accuracy among scored fields: How accurate were the numerical fields the scorer could evaluate?
- Dataset-wide accuracy: How many of all expected numerical answers were correct?

### 5.2 Global Accuracy

Global Accuracy measures the proportion of benchmark questions that receive a final gated AI-judge score of 5/5.

Formula:

Global Accuracy (%) =

(Questions with Final Score 5/5 / Total Dataset Questions) × 100

Questions that are missing, invalid, or not evaluated are not counted as correct.

This is a strict metric: a question receiving 4/5 is not counted as globally correct.

### 5.3 Circuit Understanding

This dimension evaluates whether the model demonstrates an appropriate understanding of the circuit problem.

It considers whether the response identifies the relevant circuit relationships and addresses the problem's electrical structure.

### 5.4 Method Correctness

This dimension evaluates whether the model selects and applies an appropriate circuit-analysis method.

Examples include:

- Nodal analysis.
- Mesh analysis.
- Thevenin's theorem.
- Norton's theorem.
- Superposition.
- Series and parallel reduction.

### 5.5 Reasoning Correctness

This dimension evaluates the logical and mathematical consistency of the explanation.

It considers whether the presented equations and reasoning support the claimed result.

### 5.6 Physical Consistency

This dimension evaluates whether the response is consistent with the relevant electrical laws and physical relationships.

Examples include:

- KCL consistency.
- KVL consistency.
- Ohm's Law.
- Power relationships.
- Current and voltage direction conventions.

### 5.7 Answer Completeness

Answer Completeness measures how many required output fields are present in the model response.

Formula:

Completeness (%) =

(Present Required Fields / Total Required Fields) × 100

A response may contain several correct values but still be incomplete if it omits a required equation, current, voltage, or verification field.

### 5.8 Response Coverage

Response Coverage measures how many dataset questions have a saved model response.

Formula:

Response Coverage (%) =

(Questions with Available Responses / Total Dataset Questions) × 100

### 5.9 Valid JSON Coverage

Valid JSON Coverage measures the percentage of model responses that can be parsed as valid JSON and satisfy the evaluator's response-validation requirements.

Formula:

Valid JSON Coverage (%) =

(Valid JSON Responses / Total Dataset Questions) × 100

This metric matters because a model's answer must be machine-readable for reliable automated evaluation.

### 5.10 Deterministic Scoring Coverage

Deterministic Scoring Coverage measures the percentage of dataset questions that successfully pass through the deterministic scoring stage.

Formula:

Scoring Coverage (%) =

(Successfully Scored Questions / Total Dataset Questions) × 100

### 5.11 AI Evaluation Coverage

AI Evaluation Coverage measures the proportion of dataset questions that successfully receive an AI-judge evaluation.

Formula:

AI Evaluation Coverage (%) =

(Successfully AI-Evaluated Questions / Total Dataset Questions) × 100

### 5.12 Average AI-Judge Score

The AI judge assigns a score from 0 to 5 for each evaluated question.

The benchmark reports:

- Average score among successfully evaluated questions.
- End-to-end average, where missing or unevaluated questions contribute zero.

The second measure prevents missing evaluations from being hidden by reporting only the average of successfully evaluated answers.

PDF

---

## 6. Evaluation Methodology

CircuitMind-Eval uses two complementary evaluation approaches.

### Stage 1: Model inference

Each model receives:

- The circuit image.
- The corresponding problem statement.
- Required output fields.
- The standardized response instructions.

Both models receive the same task content and prompt structure.

The model is expected to return structured JSON containing the required numerical answers and explanations.

### Stage 2: Response validation

The inference output is checked for:

- Whether a response exists.
- Whether the response is valid JSON.
- Whether the task identifier matches.
- Whether the required response structure is available.

Invalid responses are recorded rather than silently corrected.

### Stage 3: Deterministic scoring

The deterministic scorer compares model outputs with the reference ground truth.

It evaluates:

- Required-field completeness.
- Numerical correctness.
- Numerical tolerance.
- Missing fields.
- Incorrect fields.
- Invalid or missing responses.

The scorer does not use an AI judge to decide whether a numerical value is correct.

This separation makes the numerical evaluation more reproducible and less dependent on subjective interpretation.

### Stage 4: Independent AI-judge evaluation

An independent model, GPT-4.1 mini, evaluates the model's response using the question, reference answer, and available deterministic scoring information.

The judge evaluates four dimensions:

1. Circuit understanding.
2. Method correctness.
3. Reasoning correctness.
4. Physical consistency.

It also produces a question-level score from 0 to 5 with supporting evidence.

The deterministic scoring result acts as a safeguard: a high semantic score cannot automatically override missing required fields or numerical errors.

**Important limitation:** The current judge implementation does not receive the original circuit image. Consequently, it evaluates the response against the question and reference information, but it cannot independently verify whether the model interpreted every visual feature of the original circuit correctly.

This is an important distinction when interpreting the results.

### Stage 5: Metric calculation

The metric calculator aggregates:

- Numerical accuracy.
- Global accuracy.
- AI-judge scores.
- Semantic dimension scores.
- Completeness.
- Response coverage.
- Valid JSON coverage.
- Scoring coverage.
- AI evaluation coverage.
- Task-level failures.

Metrics are calculated separately for each model and combined into a benchmark report.

### Stage 6: Failure tracking

A failed task does not automatically terminate the complete benchmark.

The pipeline records failures at the task and model level and continues processing other available tasks.

Missing or invalid outputs are not fabricated or manually assigned scores.

This is important because a benchmark should report both model performance and the reliability of the evaluation process.

---

## 7. Models Tested and Experimental Configuration

The initial experiment compares two vision-language models.

| Configuration | OpenAI | NVIDIA |
|---|---|---|
| Model ID | `gpt-6.1-terra` | `nvidia/Nemotron-3.5-Super-VL` |
| Provider | OpenAI | NVIDIA model through Nebius Token Factory |
| Input | Circuit image + question | Circuit image + question |
| Output | Structured JSON | Structured JSON |
| Temperature | Not explicitly set in the Responses API request | 0, subject to the actual request configuration |
| Dataset | Same benchmark dataset | Same benchmark dataset |
| Prompt | Standardized | Standardized |

**Independent AI judge:** `gpt-4.1-mini`

**NVIDIA API base URL:**

`https://api.tokenfactory.nebius.com/v1`

**Run date:** [8-11-2026]

**Dataset version:** [0.1.0]

**Inference settings:** [RECORD ACTUAL MAX OUTPUT TOKENS - 8192, RETRIES - 1, TIMEOUT - 180, TEMPERATURE - 0]

The model identifiers above represent the identifiers used in this experiment. They should not be interpreted as claims that the providers expose identical model versions or inference implementations.

---

## 8. Experimental Results

The following results are from the initial 10-question benchmark run.

### 8.1 Primary metric: Numerical Accuracy


| Metric | OpenAI | NVIDIA |
|---|---:|---:|
| Correct numerical fields | 14 | 28 |
| Scored numerical fields | 43 | 39 |
| Accuracy among scored fields | 32.56% | 71.79% |
| Expected numerical fields in dataset | 41 | 41 |
| Dataset-wide numerical accuracy | 34.15% | 68.29% |


The primary metric is numerical accuracy among scored fields.

The dataset-wide value is also reported to expose missing or invalid evaluations.

**Data-quality note:** The current scorer output reports 43 scored numerical fields for OpenAI, although the dataset-derived expected count is 41. NVIDIA reports 39 scored fields. This discrepancy must be investigated against the per-task scoring records before these numerical accuracy figures are treated as final publication-grade results. The figures above reproduce the current pipeline output; they are not presented as a fully reconciled result.

### 8.2 Global Accuracy and AI-Judge Results



The NVIDIA evaluated-only average is calculated over nine successfully evaluated questions, whereas the end-to-end average includes the missing evaluation as zero.

### 8.3 Response and Evaluation Coverage

| Metric | OpenAI | NVIDIA |
|---|---:|---:|
| Dataset questions | 10 | 10 |
| Questions with final score 5/5 | 1 | 3 |
| Global accuracy | 10% | 30% |
| Average judge score, evaluated questions | 4.10/5 | 4.33/5 |
| End-to-end average judge score | 4.10/5 | 3.90/5 |

### 8.4 Answer Completeness

| Model | Present fields | Expected fields | Completeness |
|---|---:|---:|---:|
| OpenAI | 53 | 53 | 100% |
| NVIDIA | 48 | 53 | 90.57% |

### 8.5 Semantic Evaluation

The AI judge evaluates four dimensions independently.

The following figures represent the percentage of successfully evaluated questions receiving a fully correct dimension score, as reported by the current evaluation output.

| Dimension | OpenAI | NVIDIA |
|---|---:|---:|
| Circuit understanding | 100% | 88.89% |
| Method correctness | 100% | 77.78% |
| Reasoning correctness | 90% | 66.67% |
| Physical consistency | 100% | 100% |

These are dimension-level full-correct percentages, not average scores out of five.

### 8.6 Important Evaluation Failure

One NVIDIA response, CM003, returned invalid JSON.

As a result:

- The response was not included in deterministic scoring.
- The task was not evaluated by the AI judge.
- The task remains part of the original dataset denominator.
- No missing numerical answers or judge scores were fabricated.

This demonstrates why a benchmark must measure response reliability and evaluation coverage in addition to answer accuracy.

### 8.7 Interpreting the Results

In this initial experiment, NVIDIA produced higher numerical accuracy among the fields successfully scored and a higher global accuracy.

OpenAI achieved complete response, scoring, and AI-evaluation coverage in this run, while NVIDIA had one invalid JSON response.

The results show that answer accuracy and evaluation reliability are separate dimensions.

The sample contains only ten questions, so the findings should be treated as preliminary observations rather than broad conclusions about the overall capabilities of either model.

---

## 9. Failure Analysis: Where Can Models Go Wrong?

CircuitMind-Eval is designed to make technical failures observable instead of treating every incorrect answer as a single undifferentiated error.

The benchmark tracks several failure categories.

| Failure category | Example | Why it matters |
|---|---|---|
| Visual interpretation | Incorrectly identifying a connection | Changes the circuit being solved |
| Topology recognition | Confusing series and parallel connections | Produces incorrect equivalent resistance |
| Source polarity | Reversing a voltage source | Changes node and loop equations |
| Equation formulation | Incorrect KCL or KVL expression | Produces an incorrect mathematical model |
| Numerical calculation | Incorrect simultaneous equation solution | Leads to wrong electrical quantities |
| Sign convention | Reversing a current direction | Misrepresents the physical result |
| Physical inconsistency | Violating KCL or KVL | Indicates an invalid electrical solution |
| Output formatting | Invalid JSON | Prevents reliable automated evaluation |
| Missing fields | Omitting a required current or voltage | Makes the response incomplete |

### Observed failure in the initial run

The most clearly identifiable pipeline-level model failure was NVIDIA CM003, which returned invalid JSON.

The deterministic scorer and AI judge correctly excluded this response from their respective successful evaluation counts.

Further technical failure analysis should be based on the saved per-question responses and scoring evidence rather than inferring the cause of an incorrect answer without examining it.

---

## 10. Reproducibility: Run the Complete Evaluation Suite

One of the main goals of CircuitMind-Eval is to make model evaluation repeatable.

The project includes a complete pipeline that runs the evaluation stages in sequence.

### Repository structure

                                    ```text
                                    CircuitMind-Eval/
                                    │
                                    ├── Dataset/
                                    │   └── CircuitMind-Eval/
                                    │       └── dataset/
                                    │           ├── images/
                                    │           ├── questions.json
                                    │           ├── ground_truth.json
                                    │           ├── metadata.json
                                    │           └── data.jsonl
                                    │
                                    ├── src/
                                    │   ├── config.py
                                    │   └── prompt_builder.py
                                    │
                                    ├── evals/
                                    │   ├── dataset_loader.py
                                    │   ├── run_inference.py
                                    │   ├── scorer.py
                                    │   ├── model_evaluator.py
                                    │   ├── metric_calculator.py
                                    │   ├── comparison.py
                                    │   └── run_benchmark.py
                                    │
                                    ├── results/
                                    │   ├── openai/
                                    │   ├── nvidia/
                                    │   ├── scoring/
                                    │   ├── model_evaluation/
                                    │   ├── metrics/
                                    │   └── comparison/
                                    │
                                    ├── .env.example
                                    ├── requirements.txt
                                    └── README.md
                                    ```

### Requirements

- Python 3.11 or a version supported by the project's dependencies.
- OpenAI API access for the configured model and AI judge.
- NVIDIA/Nebius Token Factory API access.
- Internet connectivity for API inference and judge evaluation.

### Step 1: Clone the repository

```bash
git clone https://github.com/JavaOnTheRocks/CircuitMind-Eval.git
cd CircuitMind-Eval
```

### Step 2: Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API keys

Create a `.env` file in the project root.

```env
OPENAI_API_KEY=your_openai_api_key
NVIDIA_API_KEY=your_nebius_token_factory_api_key
```

The `NVIDIA_API_KEY` variable is used for the NVIDIA model endpoint hosted by Nebius Token Factory.

Never commit your `.env` file or expose API keys in public logs.

### Step 5: Run the complete benchmark

```bash
python -m evals.run_benchmark
```

This is the intended single-command entry point.

The pipeline executes:

1. Dataset loading and validation.
2. OpenAI and NVIDIA inference.
3. Deterministic scoring.
4. AI-judge evaluation.
5. Metric calculation.
6. Final benchmark report generation.

The pipeline is designed to record individual task failures and continue processing the remaining tasks.

To force fresh inference rather than reuse valid existing responses, use:

```bash
python -m evals.run_benchmark --force-inference
```

The exact CLI options should be checked against the current repository version.

## Running Individual Evaluation Modules

CircuitMind-Eval allows you to execute each evaluation stage independently or run the complete evaluation pipeline with a single command.

Run all commands from the project root directory.

### 1. Dataset Validation

Validate the dataset, ground-truth annotations, metadata, required fields, and circuit images.

```bash
python -m evals.dataset_loader
```

### 2. Model Inference

Run inference using both models:

```bash
python -m evals.run_inference --model both
```

Run inference using OpenAI only:

```bash
python -m evals.run_inference --model openai
```

Run inference using NVIDIA only:

```bash
python -m evals.run_inference --model nvidia
```

Force fresh inference instead of reusing existing valid responses:

```bash
python -m evals.run_inference --model both --force
```

### 3. Deterministic Scoring

Evaluate model responses against the reference ground truth.

Run scoring for both models:

```bash
python -m evals.scorer --model both
```

Run scoring individually:

```bash
python -m evals.scorer --model openai
python -m evals.scorer --model nvidia
```

### 4. AI-Judge Evaluation

Evaluate circuit understanding, method correctness, reasoning correctness, and physical consistency using the independent AI judge (GPT-4.1 mini).

Run evaluation for both models:

```bash
python -m evals.model_evaluator --model both
```

Run evaluation individually:

```bash
python -m evals.model_evaluator --model openai
python -m evals.model_evaluator --model nvidia
```

### 5. Metric Calculation

Calculate numerical accuracy, global accuracy, answer completeness, semantic evaluation scores, response coverage, and other benchmark metrics.

Run for both models:

```bash
python -m evals.metric_calculator --model both
```

Run individually:

```bash
python -m evals.metric_calculator --model openai
python -m evals.metric_calculator --model nvidia
```

### 6. Model Comparison

Generate the model comparison report using the calculated benchmark metrics.

```bash
python -m evals.comparison
```

### 7. Run the Complete Evaluation Pipeline

Alternatively, execute the entire evaluation suite automatically using a single command:

```bash
python -m evals.run_benchmark
```

This executes inference, deterministic scoring, AI-judge evaluation, metric calculation, and final report generation in sequence.

### Recommended Execution Order

When running individual modules, follow this order:

                                    ```text
                                    Dataset Validation
                                        |
                                        v
                                    Model Inference
                                        |
                                        v
                                    Deterministic Scoring
                                        |
                                        v
                                    AI-Judge Evaluation
                                        |
                                        v
                                    Metric Calculation
                                        |
                                        v
                                    Model Comparison
                                        |
                                        v
                                    Final Benchmark Reports
                                    ```

**Note:** Each evaluation stage depends on the outputs of the preceding stages. If you run inference separately before executing the complete pipeline, the inference module is designed to reuse valid existing responses and skip completed tasks unless fresh inference is explicitly requested.

### Expected output files

                                    ```text
                                    results/
                                    │
                                    ├── openai/
                                    │   └── CM001.json
                                    │
                                    ├── nvidia/
                                    │   └── CM001.json
                                    │
                                    ├── scoring/
                                    │   ├── openai/
                                    │   └── nvidia/
                                    │
                                    ├── model_evaluation/
                                    │   ├── openai/
                                    │   ├── nvidia/
                                    │   ├── evaluation_summary.json
                                    │   └── combined_summary.json
                                    │
                                    ├── metrics/
                                    │   ├── openai_metrics.json
                                    │   ├── nvidia_metrics.json
                                    │   └── benchmark_metrics.json
                                    │
                                    ├── comparison/
                                    │   ├── model_performance_report.json
                                    │   └── model_performance_report.md
                                    │
                                    └── benchmark_run.json
                                    ```

These are the intended output locations; the actual generated files depend on the current run and repository version.


## 11. Reproducibility and Evaluation Quality

CircuitMind-Eval is designed around several reproducibility principles.

### Identical evaluation inputs

Both models receive the same benchmark questions and circuit images.

### Standardized prompts

A common prompt-building process is used to reduce differences caused by inconsistent instructions.

### Ground-truth-based numerical scoring

Numerical answers are checked against reference values rather than relying exclusively on an AI judge.

### Independent semantic evaluation

A separate AI judge evaluates reasoning-related dimensions using a common rubric.

### Explicit failure tracking

Invalid JSON, missing answers, failed inference requests, and unsuccessful evaluations are recorded.

### Dataset-size-independent processing

The pipeline loads tasks dynamically instead of assuming that the dataset always contains ten questions.

### Persistent evaluation artifacts

Model responses, scoring results, judge evaluations, and metric reports are saved so that the experiment can be inspected and audited.

### Important reproducibility limitation

Running the same code and dataset does not guarantee byte-for-byte identical results from hosted models.

Model providers may update model versions, inference infrastructure, or underlying behavior. API availability, model access, and numerical nondeterminism can also affect results.

For stronger reproducibility, record:

- Dataset version or commit hash.
- Repository commit hash.
- Exact model IDs.
- Evaluation date.
- Prompt version.
- Scorer version.
- Judge model ID.
- Inference settings.
- API errors and retries.
- Python and dependency versions.

Temperature zero helps reduce sampling variability where supported, but it does not guarantee identical outputs across provider updates.

---

## 12. Evaluation Report and Logs

For transparency, the project maintains machine-readable outputs and human-readable reports.

### Evaluation report

**Attach the generated evaluation report PDF here:**

[https://docs.google.com/document/d/1_nzB-VXYp69DveILrbWeGMPv9e8wQXxoyy1eoXTUYaU/edit?usp=sharing]

The PDF should include:

- Benchmark objective and methodology.
- Dataset overview.
- Model configurations.
- Metric definitions and formulas.
- Overall model results.
- Evaluation limitations.



## 13. Video Walkthrough

A video walkthrough demonstrates the benchmark and its execution.

**Video:** [https://docs.google.com/document/d/1f0lC3WyE5-KjWMJBv6R68B9NwwHEOGglJfVt2Xp1fK8/edit?usp=sharing]

The walkthrough video covers:

1. The electrical circuit reasoning problem.
2. The gap between plausible model explanations and technically correct solutions.
3. Dataset structure and preparation.
4. The two models being evaluated.
5. The complete evaluation pipeline.
6. Numerical scoring and AI-judge evaluation.
7. Metric definitions and results.
8. Model failure analysis.
9. Running the benchmark using one command.
10. Generated reports and reproducibility artifacts.


---

## 14. Prior Work and Existing Benchmarks

CircuitMind-Eval is a domain-specific evaluation project focused on  electrical and electronics circuit reasoning.

Its dataset and evaluation approach should be distinguished from general-purpose circuit question answering and general mathematical reasoning benchmarks.

The current experiment uses a project-specific collection of electrical  network theory problems and reference answers.

**Dataset provenance and prior-work disclosure:**

[I prepared the Dataset form my prior Enginneering Experience and GATE papers]

---

## 15. Current Limitations and Future Work

The current benchmark is an initial study with ten evaluated questions.

Several improvements are planned:

- Expand the dataset to 50–100 or more carefully validated circuit questions.
- Increase diversity in circuit topology and electrical and elctronics concepts.
- Include more complex multi-loop and multi-source circuits.
- Improve analysis of visual interpretation errors.
- Investigate separate evaluation of equation formulation and numerical solving.
- Reconcile numerical field-count discrepancies in the initial scoring results.
- Add additional model families.
- Improve dataset versioning and experiment tracking.
- Provide stronger validation of ground-truth annotations.
- Explore independent expert review of difficult examples.

---

## 16. Conclusion

CircuitMind-Eval investigates the Elcetrical and Electronics capabilities of frontier Models:

**Can frontier models reliably reason about electrical circuits, or can they only produce explanations that appear technically convincing?**

By combining circuit images, electrical and electronics questions, structured outputs, deterministic numerical scoring, and independent AI-based reasoning evaluation, the benchmark provides a more detailed view of model performance than final-answer evaluation alone.

The initial ten-question experiment revealed differences in numerical performance, response validity, completeness, and evaluation coverage.

The project is designed to grow into a larger and more comprehensive benchmark, with transparent scoring, explicit failure tracking, and reproducible execution.

The goal is not simply to rank models. It is to understand where their electrical engineering reasoning succeeds, where it fails, and how reliably those capabilities can be measured.

---

## 17. Project Links

| Resource | Link |
|---|---|
| Source code | [https://github.com/JavaOnTheRocks/CircuitMind-Eval.git] |
| Hugging Face dataset | [https://huggingface.co/datasets/rakesh-bishnoi29/circuitmind-eval/] |
| Evaluation report PDF | [https://docs.google.com/document/d/1_nzB-VXYp69DveILrbWeGMPv9e8wQXxoyy1eoXTUYaU/edit?usp=sharing] |
| Video walkthrough | [https://docs.google.com/document/d/1f0lC3WyE5-KjWMJBv6R68B9NwwHEOGglJfVt2Xp1fK8/edit?usp=sharing] |


