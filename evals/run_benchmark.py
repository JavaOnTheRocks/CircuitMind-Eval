
"""
CircuitMind-Eval: End-to-End Benchmark Runner

Runs:
1. Inference for OpenAI and NVIDIA
2. Deterministic numerical scoring
3. gpt-6.1-terra semantic evaluation
4. Benchmark metric calculation
5. Neutral performance report

"""

import argparse
import json
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

METRICS_DIR = RESULTS_DIR / "metrics"
COMPARISON_DIR = RESULTS_DIR / "comparison"

BENCHMARK_METRICS_FILE = (
    METRICS_DIR / "benchmark_metrics.json"
)

COMPARISON_JSON_FILE = (
    COMPARISON_DIR / "model_performance_report.json"
)

COMPARISON_MD_FILE = (
    COMPARISON_DIR / "model_performance_report.md"
)

PIPELINE_RUN_FILE = (
    RESULTS_DIR / "benchmark_run.json"
)

MODELS = ("openai", "nvidia")


# ============================================================
# GENERAL HELPERS
# ============================================================

def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    temporary_path.replace(path)


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def get_nested(data, *keys, default=None):
    """Safely retrieve a nested dictionary value."""

    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def display_percent(value):
    if value is None:
        return "N/A"

    return f"{value:.2f}%"


def display_score(value):
    if value is None:
        return "N/A"

    return f"{value:.2f}/5"


def display_count(numerator, denominator):
    if numerator is None or denominator is None:
        return "N/A"

    return f"{numerator}/{denominator}"


# ============================================================
# RUN ONE PIPELINE STAGE
# ============================================================

def run_stage(stage_name, command, stage_results):
    """
    Execute one pipeline stage.
    
    """

    print("\n" + "=" * 75)
    print(f"PIPELINE STAGE: {stage_name.upper()}")
    print("=" * 75)

    print("Command:")
    print(" ".join(command))
    print("-" * 75)

    started_at = timestamp()

    try:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False
        )

        success = process.returncode == 0

        result = {
            "stage": stage_name,
            "status": (
                "completed" if success else "failed"
            ),
            "return_code": process.returncode,
            "started_at": started_at,
            "completed_at": timestamp(),
        }

        if not success:
            result["error"] = (
                f"Process exited with code "
                f"{process.returncode}"
            )

    except Exception as error:
        result = {
            "stage": stage_name,
            "status": "failed",
            "return_code": None,
            "started_at": started_at,
            "completed_at": timestamp(),
            "error": str(error),
        }

    stage_results.append(result)

    print("\nStage status:", result["status"].upper())

    if result.get("error"):
        print("Stage error:", result["error"])

    return result["status"] == "completed"


# ============================================================
# BUILD PIPELINE COMMANDS
# ============================================================

def build_pipeline_commands(force_inference=False):

    inference_command = [
        sys.executable,
        "-m",
        "evals.run_inference",
        "--model",
        "both",
    ]

    if force_inference:
        inference_command.append("--force")

    return [
        (
            "inference",
            inference_command,
        ),
        (
            "deterministic_scoring",
            [
                sys.executable,
                "-m",
                "evals.scorer",
                "--model",
                "both",
            ],
        ),
        (
            "ai_judge_evaluation",
            [
                sys.executable,
                "-m",
                "evals.model_evaluator",
                "--model",
                "both",
            ],
        ),
        (
            "metric_calculation",
            [
                sys.executable,
                "-m",
                "evals.metric_calculator",
                "--model",
                "both",
            ],
        ),
    ]


# ============================================================
# EXTRACT METRICS FOR ONE MODEL
# ============================================================

def extract_model_metrics(model_metrics):

    numerical = model_metrics.get(
        "numerical_accuracy",
        {}
    )

    global_accuracy = model_metrics.get(
        "global_accuracy",
        {}
    )

    average_score = model_metrics.get(
        "average_question_score",
        {}
    )

    completeness = model_metrics.get(
        "answer_completeness",
        {}
    )

    coverage = model_metrics.get(
        "coverage",
        {}
    )

    semantic = model_metrics.get(
        "semantic_metrics",
        {}
    )

    score_distribution = model_metrics.get(
        "question_score_distribution",
        {}
    )

    deterministic_accuracy = model_metrics.get(
        "deterministic_task_accuracy",
        {}
    )

    return {
        "numerical_accuracy": {
            "correct_numeric_fields": numerical.get(
                "correct_numeric_fields"
            ),
            "scored_numeric_fields": numerical.get(
                "scored_numeric_fields"
            ),
            "expected_numeric_fields_in_dataset": numerical.get(
                "expected_numeric_fields_in_dataset"
            ),
            "accuracy_among_scored_numeric_fields_percent": (
                numerical.get(
                    "accuracy_among_scored_numeric_fields_percent"
                )
            ),
            "dataset_wide_accuracy_percent": numerical.get(
                "accuracy_percent"
            ),
        },

        "global_accuracy": {
            "correct_questions": global_accuracy.get(
                "correct_questions"
            ),
            "dataset_questions": global_accuracy.get(
                "dataset_questions"
            ),
            "accuracy_percent": global_accuracy.get(
                "accuracy_percent"
            ),
            "accuracy_among_evaluated_percent": (
                global_accuracy.get(
                    "accuracy_among_evaluated_percent"
                )
            ),
        },

        "average_question_score": {
            "average_among_evaluated_answers": (
                average_score.get(
                    "average_among_evaluated_answers"
                )
            ),
            "end_to_end_average_including_missing_as_zero": (
                average_score.get(
                    "end_to_end_average_including_missing_as_zero"
                )
            ),
        },

        "answer_completeness": {
            "present_fields": completeness.get(
                "present_fields"
            ),
            "expected_required_fields_in_dataset": (
                completeness.get(
                    "expected_required_fields_in_dataset"
                )
            ),
            "completeness_percent": completeness.get(
                "completeness_percent"
            ),
            "completeness_among_scored_responses_percent": (
                completeness.get(
                    "completeness_among_scored_responses_percent"
                )
            ),
        },

        "semantic_metrics": semantic,

        "coverage": {
            "dataset_questions": coverage.get(
                "dataset_questions"
            ),
            "available_responses": coverage.get(
                "available_responses"
            ),
            "missing_responses": coverage.get(
                "missing_responses"
            ),
            "response_coverage_percent": coverage.get(
                "response_coverage_percent"
            ),
            "valid_json_responses": coverage.get(
                "valid_json_responses"
            ),
            "invalid_or_unscored_responses": coverage.get(
                "invalid_or_unscored_responses"
            ),
            "valid_json_percent_of_dataset": coverage.get(
                "valid_json_percent_of_dataset"
            ),
            "completed_ai_evaluations": coverage.get(
                "completed_ai_evaluations"
            ),
            "ai_evaluation_coverage_percent": coverage.get(
                "ai_evaluation_coverage_percent"
            ),
        },

        "deterministic_task_accuracy": (
            deterministic_accuracy
        ),

        "question_score_distribution": (
            score_distribution
        ),

        "benchmark_status": model_metrics.get(
            "benchmark_status",
            {}
        ),
    }


# ============================================================
# TERMINAL PERFORMANCE REPORT
# ============================================================

def print_performance_report(report):

    models = report.get("models", {})

    print("\n\n")
    print("=" * 100)
    print("                    CIRCUITMIND-EVAL")
    print("              FINAL BENCHMARK PERFORMANCE")
    print("=" * 100)

    print(
        f"\nDataset size: {report.get('dataset_size', 'N/A')}"
    )

    print(
        f"Generated at: {report.get('generated_at', 'N/A')}"
    )

    print("\n" + "-" * 100)
    print("1. PRIMARY METRIC: NUMERICAL ACCURACY AMONG SCORED FIELDS")
    print("-" * 100)

    print(
        f"{'Metric':<48}"
        f"{'OpenAI':>22}"
        f"{'NVIDIA':>22}"
    )

    print("-" * 100)

    for label, key in [
        (
            "Numerical accuracy among scored fields",
            "accuracy_among_scored_numeric_fields_percent",
        ),
        (
            "Correct numeric fields",
            "correct_numeric_fields",
        ),
        (
            "Scored numeric fields",
            "scored_numeric_fields",
        ),
        (
            "Dataset-wide numerical accuracy",
            "dataset_wide_accuracy_percent",
        ),
    ]:

        values = {}

        for model in MODELS:
            values[model] = get_nested(
                models,
                model,
                "numerical_accuracy",
                key,
            )

        if key.endswith("_percent"):
            openai_value = display_percent(
                values["openai"]
            )
            nvidia_value = display_percent(
                values["nvidia"]
            )
        else:
            openai_value = (
                str(values["openai"])
                if values["openai"] is not None
                else "N/A"
            )
            nvidia_value = (
                str(values["nvidia"])
                if values["nvidia"] is not None
                else "N/A"
            )

        print(
            f"{label:<48}"
            f"{openai_value:>22}"
            f"{nvidia_value:>22}"
        )

    # --------------------------------------------------------
    # GLOBAL ACCURACY AND JUDGE SCORES
    # --------------------------------------------------------

    print("\n" + "-" * 100)
    print("2. GLOBAL ACCURACY AND AI JUDGE SCORES")
    print("-" * 100)

    print(
        f"{'Metric':<48}"
        f"{'OpenAI':>22}"
        f"{'NVIDIA':>22}"
    )

    print("-" * 100)

    rows = [
        (
            "Global accuracy",
            "global_accuracy",
            "accuracy_percent",
            "percent",
        ),
        (
            "Accuracy among evaluated questions",
            "global_accuracy",
            "accuracy_among_evaluated_percent",
            "percent",
        ),
        (
            "Average judge score (evaluated answers)",
            "average_question_score",
            "average_among_evaluated_answers",
            "score",
        ),
        (
            "End-to-end average judge score",
            "average_question_score",
            "end_to_end_average_including_missing_as_zero",
            "score",
        ),
    ]

    for label, section, key, value_type in rows:

        openai_value = get_nested(
            models, "openai", section, key
        )

        nvidia_value = get_nested(
            models, "nvidia", section, key
        )

        if value_type == "percent":
            openai_text = display_percent(openai_value)
            nvidia_text = display_percent(nvidia_value)
        else:
            openai_text = display_score(openai_value)
            nvidia_text = display_score(nvidia_value)

        print(
            f"{label:<48}"
            f"{openai_text:>22}"
            f"{nvidia_text:>22}"
        )

    # --------------------------------------------------------
    # SEMANTIC DIMENSIONS
    # --------------------------------------------------------

    print("\n" + "-" * 100)
    print("3. SEMANTIC EVALUATION DIMENSIONS")
    print("-" * 100)

    print(
        f"{'Dimension':<48}"
        f"{'OpenAI':>22}"
        f"{'NVIDIA':>22}"
    )

    print("-" * 100)

    dimensions = [
        "circuit_understanding",
        "method_correctness",
        "reasoning_correctness",
        "physical_consistency",
    ]

    for dimension in dimensions:

        openai_value = get_nested(
            models,
            "openai",
            "semantic_metrics",
            dimension,
            "average_score_out_of_5",
        )

        nvidia_value = get_nested(
            models,
            "nvidia",
            "semantic_metrics",
            dimension,
            "average_score_out_of_5",
        )

        print(
            f"{dimension.replace('_', ' ').title():<48}"
            f"{display_score(openai_value):>22}"
            f"{display_score(nvidia_value):>22}"
        )

    # --------------------------------------------------------
    # COMPLETENESS AND COVERAGE
    # --------------------------------------------------------

    print("\n" + "-" * 100)
    print("4. ANSWER COMPLETENESS AND RELIABILITY")
    print("-" * 100)

    rows = [
        (
            "Answer completeness",
            "answer_completeness",
            "completeness_percent",
        ),
        (
            "Completeness among scored responses",
            "answer_completeness",
            "completeness_among_scored_responses_percent",
        ),
        (
            "Response coverage",
            "coverage",
            "response_coverage_percent",
        ),
        (
            "Valid JSON coverage (dataset)",
            "coverage",
            "valid_json_percent_of_dataset",
        ),
        (
            "AI evaluation coverage",
            "coverage",
            "ai_evaluation_coverage_percent",
        ),
    ]

    print(
        f"{'Metric':<48}"
        f"{'OpenAI':>22}"
        f"{'NVIDIA':>22}"
    )

    print("-" * 100)

    for label, section, key in rows:

        openai_value = get_nested(
            models, "openai", section, key
        )

        nvidia_value = get_nested(
            models, "nvidia", section, key
        )

        print(
            f"{label:<48}"
            f"{display_percent(openai_value):>22}"
            f"{display_percent(nvidia_value):>22}"
        )

    # --------------------------------------------------------
    # FAILURE AND COVERAGE COUNTS
    # --------------------------------------------------------

    print("\n" + "-" * 100)
    print("5. RESPONSE AND EVALUATION COUNTS")
    print("-" * 100)

    count_rows = [
        ("Available responses", "available_responses"),
        ("Missing responses", "missing_responses"),
        ("Valid JSON responses", "valid_json_responses"),
        (
            "Invalid or unscored responses",
            "invalid_or_unscored_responses",
        ),
        (
            "Completed AI evaluations",
            "completed_ai_evaluations",
        ),
    ]

    print(
        f"{'Metric':<48}"
        f"{'OpenAI':>22}"
        f"{'NVIDIA':>22}"
    )

    print("-" * 100)

    for label, key in count_rows:

        openai_value = get_nested(
            models, "openai", "coverage", key
        )

        nvidia_value = get_nested(
            models, "nvidia", "coverage", key
        )

        print(
            f"{label:<48}"
            f"{str(openai_value if openai_value is not None else 'N/A'):>22}"
            f"{str(nvidia_value if nvidia_value is not None else 'N/A'):>22}"
        )

    # --------------------------------------------------------
    # PER QUESTION PERFORMANCE
    # --------------------------------------------------------

    per_question = report.get(
        "per_question_results",
        {}
    )

    if isinstance(per_question, dict) and per_question:

        print("\n" + "-" * 100)
        print("6. PER-QUESTION PERFORMANCE")
        print("-" * 100)

        print(
            f"{'Task ID':<25}"
            f"{'OpenAI Score':>25}"
            f"{'NVIDIA Score':>25}"
            f"{'OpenAI Numeric':>20}"
            f"{'NVIDIA Numeric':>20}"
        )

        print("-" * 115)

        for task_id, task_result in per_question.items():

            openai_task = task_result.get("openai", {})
            nvidia_task = task_result.get("nvidia", {})

            openai_score = openai_task.get(
                "final_question_score"
            )

            nvidia_score = nvidia_task.get(
                "final_question_score"
            )

            openai_numeric = display_count(
                openai_task.get("correct_numeric_fields"),
                openai_task.get("numeric_fields"),
            )

            nvidia_numeric = display_count(
                nvidia_task.get("correct_numeric_fields"),
                nvidia_task.get("numeric_fields"),
            )

            print(
                f"{task_id:<25}"
                f"{display_score(openai_score):>25}"
                f"{display_score(nvidia_score):>25}"
                f"{openai_numeric:>20}"
                f"{nvidia_numeric:>20}"
            )

    # --------------------------------------------------------
    # FINAL NOTE
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("BENCHMARK PERFORMANCE REPORT GENERATED")
    print("=" * 100)

    print(
        "\nThis report presents measured performance for both models."
    )

    print(
        "No winner or loser has been assigned."
    )

    print(
        "\nPrimary metric: Numerical accuracy among scored numeric fields."
    )

    print(
        "Secondary metrics: Global accuracy, judge scores, "
        "semantic dimensions, completeness and coverage."
    )

    print("\nSaved reports:")
    print(COMPARISON_JSON_FILE)
    print(COMPARISON_MD_FILE)

    print("=" * 100)


# ============================================================
# SAVE NEUTRAL COMPARISON REPORT
# ============================================================

def build_comparison_report(benchmark):

    raw_models = benchmark.get("models", {})

    models = {}

    for model_name in MODELS:

        model_metrics = raw_models.get(model_name)

        if not isinstance(model_metrics, dict):
            models[model_name] = None
            continue

        models[model_name] = extract_model_metrics(
            model_metrics
        )

    report = {
        "benchmark": "CircuitMind-Eval",
        "generated_at": timestamp(),
        "dataset_size": benchmark.get("dataset_size"),
        "comparison_policy": {
            "primary_metric": (
                "Numerical accuracy among scored numeric fields"
            ),
            "secondary_metrics": [
                "Global accuracy",
                "Average judge score",
                "Circuit understanding",
                "Method correctness",
                "Reasoning correctness",
                "Physical consistency",
                "Answer completeness",
                "Response coverage",
                "Valid JSON coverage",
                "AI evaluation coverage",
            ],
            "winner_declared": False,
            "loser_declared": False,
        },
        "models": models,
        "per_question_results": benchmark.get(
            "per_question_results",
            benchmark.get(
                "per_question_comparison",
                {}
            )
        ),
    }

    return report


# ============================================================
# MARKDOWN REPORT
# ============================================================

def generate_markdown_report(report):

    lines = [
        "# CircuitMind-Eval: Benchmark Performance Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Dataset size: {report.get('dataset_size', 'N/A')}",
        "",
        "## Primary metric: Numerical accuracy among scored fields",
        "",
        "| Metric | OpenAI | NVIDIA |",
        "|---|---:|---:|",
    ]

    models = report.get("models", {})

    def metric(model, section, key):
        return get_nested(
            models,
            model,
            section,
            key,
        )

    primary_rows = [
        (
            "Numerical accuracy among scored fields",
            "numerical_accuracy",
            "accuracy_among_scored_numeric_fields_percent",
        ),
        (
            "Correct numeric fields",
            "numerical_accuracy",
            "correct_numeric_fields",
        ),
        (
            "Scored numeric fields",
            "numerical_accuracy",
            "scored_numeric_fields",
        ),
        (
            "Dataset-wide numerical accuracy",
            "numerical_accuracy",
            "dataset_wide_accuracy_percent",
        ),
    ]

    for label, section, key in primary_rows:

        values = []

        for model in MODELS:
            value = metric(model, section, key)

            if value is None:
                values.append("N/A")
            elif key.endswith("_percent"):
                values.append(f"{value:.2f}%")
            else:
                values.append(str(value))

        lines.append(
            f"| {label} | {values[0]} | {values[1]} |"
        )

    lines.extend([
        "",
        "## Secondary metrics",
        "",
        "| Metric | OpenAI | NVIDIA |",
        "|---|---:|---:|",
    ])

    secondary_rows = [
        (
            "Global accuracy",
            "global_accuracy",
            "accuracy_percent",
            "percent",
        ),
        (
            "Average judge score",
            "average_question_score",
            "average_among_evaluated_answers",
            "score",
        ),
        (
            "Circuit understanding",
            "semantic_metrics",
            "circuit_understanding",
            "dimension",
        ),
        (
            "Method correctness",
            "semantic_metrics",
            "method_correctness",
            "dimension",
        ),
        (
            "Reasoning correctness",
            "semantic_metrics",
            "reasoning_correctness",
            "dimension",
        ),
        (
            "Physical consistency",
            "semantic_metrics",
            "physical_consistency",
            "dimension",
        ),
        (
            "Answer completeness",
            "answer_completeness",
            "completeness_percent",
            "percent",
        ),
        (
            "Response coverage",
            "coverage",
            "response_coverage_percent",
            "percent",
        ),
        (
            "Valid JSON coverage",
            "coverage",
            "valid_json_percent_of_dataset",
            "percent",
        ),
        (
            "AI evaluation coverage",
            "coverage",
            "ai_evaluation_coverage_percent",
            "percent",
        ),
    ]

    for row in secondary_rows:

        label, section, key, value_type = row

        values = []

        for model in MODELS:

            if value_type == "dimension":
                value = get_nested(
                    models,
                    model,
                    section,
                    key,
                    "average_score_out_of_5",
                )
            else:
                value = metric(model, section, key)

            if value is None:
                text_value = "N/A"
            elif value_type == "percent":
                text_value = f"{value:.2f}%"
            elif value_type in ("score", "dimension"):
                text_value = f"{value:.2f}/5"
            else:
                text_value = str(value)

            values.append(text_value)

        lines.append(
            f"| {label} | {values[0]} | {values[1]} |"
        )

    lines.extend([
        "",
        "## Comparison policy",
        "",
        "- Numerical accuracy among scored fields is the primary metric.",
        "- Other metrics provide supporting performance and reliability context.",
        "- Missing and invalid responses remain visible in coverage metrics.",
        "- No winner or loser is declared.",
        "",
    ])

    return "\n".join(lines)


# ============================================================
# MAIN BENCHMARK PIPELINE
# ============================================================

def run_benchmark(force_inference=False):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_started = datetime.now(timezone.utc)
    pipeline_start_timestamp = pipeline_started.timestamp()

    stage_results = []

    print("\n")
    print("=" * 75)
    print("                 CIRCUITMIND-EVAL")
    print("               COMPLETE BENCHMARK RUN")
    print("=" * 75)

    print("Project:", PROJECT_ROOT)
    print("Results:", RESULTS_DIR)
    print("Started:", pipeline_started.isoformat())

    commands = build_pipeline_commands(
        force_inference=force_inference
    )

    # Run all stages in sequence.
    # A stage failure is recorded; later stages still run.

    for stage_name, command in commands:
        run_stage(
            stage_name,
            command,
            stage_results
        )

    # --------------------------------------------------------
    # LOAD FRESH METRICS
    # --------------------------------------------------------

    report = None
    report_error = None

    if not BENCHMARK_METRICS_FILE.exists():

        report_error = (
            "Metric calculator did not create benchmark_metrics.json."
        )

    elif (
        BENCHMARK_METRICS_FILE.stat().st_mtime
        < pipeline_start_timestamp
    ):

        report_error = (
            "The metrics file was not regenerated during this run. "
            "Refusing to report potentially stale metrics."
        )

    else:

        try:
            benchmark = load_json(
                BENCHMARK_METRICS_FILE
            )

            if not isinstance(benchmark, dict):
                raise ValueError(
                    "Benchmark metrics file must contain a JSON object."
                )

            report = build_comparison_report(
                benchmark
            )

            save_json(
                COMPARISON_JSON_FILE,
                report
            )

            markdown = generate_markdown_report(
                report
            )

            COMPARISON_MD_FILE.write_text(
                markdown,
                encoding="utf-8"
            )

        except Exception as error:
            report_error = str(error)

    # --------------------------------------------------------
    # PIPELINE SUMMARY
    # --------------------------------------------------------

    failed_stages = [
        item
        for item in stage_results
        if item["status"] == "failed"
    ]

    pipeline_completed = (
        len(failed_stages) == 0
        and report is not None
    )

    pipeline_summary = {
        "benchmark": "CircuitMind-Eval",
        "started_at": pipeline_started.isoformat(),
        "completed_at": timestamp(),
        "pipeline_status": (
            "COMPLETED"
            if pipeline_completed
            else "COMPLETED_WITH_ERRORS"
        ),
        "report_generated": report is not None,
        "stage_results": stage_results,
        "failed_stage_count": len(failed_stages),
        "report_error": report_error,
        "winner_declared": False,
        "loser_declared": False,
        "report_files": {
            "benchmark_metrics": str(
                BENCHMARK_METRICS_FILE
            ),
            "comparison_json": str(
                COMPARISON_JSON_FILE
            ),
            "comparison_markdown": str(
                COMPARISON_MD_FILE
            ),
        },
    }

    save_json(
        PIPELINE_RUN_FILE,
        pipeline_summary
    )

    # --------------------------------------------------------
    # PRINT FINAL PERFORMANCE
    # --------------------------------------------------------

    if report is not None:

        print_performance_report(report)

    else:

        print("\n" + "=" * 75)
        print("FINAL PERFORMANCE REPORT NOT AVAILABLE")
        print("=" * 75)

        print(report_error)

        print(
            "\nCheck the stage errors and existing files "
            "under the results directory."
        )

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 75)

    for stage in stage_results:

        print(
            f"{stage['stage']:<28}"
            f"{stage['status'].upper()}"
        )

    print("-" * 75)

    print(
        "Pipeline status:",
        pipeline_summary["pipeline_status"]
    )

    print(
        "Failed stages:",
        pipeline_summary["failed_stage_count"]
    )

    print(
        "Final report generated:",
        pipeline_summary["report_generated"]
    )

    print(
        "Winner declared: NO"
    )

    print(
        "Loser declared: NO"
    )

    print(
        "\nPipeline log:",
        PIPELINE_RUN_FILE
    )

    print("=" * 75)

    return pipeline_summary


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete CircuitMind-Eval benchmark."
        )
    )

    parser.add_argument(
        "--force-inference",
        action="store_true",
        help=(
            "Regenerate inference responses even when "
            "existing valid responses are present."
        ),
    )

    args = parser.parse_args()

    run_benchmark(
        force_inference=args.force_inference
    )


if __name__ == "__main__":
    main()
