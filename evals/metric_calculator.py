import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# CIRCUITMIND-EVAL METRIC CALCULATOR v3.1.0
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

INFERENCE_DIR = RESULTS_DIR
SCORING_DIR = RESULTS_DIR / "scoring"

# IMPORTANT: The evaluator writes to model_evaluation,
# not evaluation.
EVALUATION_DIR = RESULTS_DIR / "model_evaluation"

OUTPUT_DIR = RESULTS_DIR / "metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ("openai", "nvidia")

SEMANTIC_DIMENSIONS = (
    "circuit_understanding",
    "method_correctness",
    "reasoning_correctness",
    "physical_consistency",
)

MAX_QUESTION_SCORE = 5


# ============================================================
# GENERAL HELPERS
# ============================================================

def timestamp():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    temporary_path.replace(path)


def safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def percentage(numerator, denominator):
    result = safe_divide(numerator, denominator)

    if result is None:
        return None

    return round(result * 100, 2)


def get_task_id(task):
    return task.get("task_id") or task.get("test_id")


def get_required_fields(task):
    fields = task.get("required_output", [])

    if isinstance(fields, dict):
        return list(fields.keys())

    if isinstance(fields, list):
        return fields

    return []


def is_numeric_ground_truth(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def get_error_message(result):
    """Extract a useful failure message from a result record."""

    if not isinstance(result, dict):
        return None

    for key in (
        "error",
        "error_message",
        "failure_reason",
        "scoring_error",
        "evaluation_error",
        "message",
    ):
        value = result.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


# ============================================================
# LOAD INDIVIDUAL TASK RESULT FILES
# ============================================================

def load_task_results(directory):
    """
    Load individual JSON result files.

    Aggregate summary files are excluded.
    Task IDs are obtained from the JSON record or filename.
    """

    results = {}

    if not directory.exists():
        return results

    excluded_names = {
        "summary.json",
        "scoring_summary.json",
        "evaluation_summary.json",
        "combined_summary.json",
        "benchmark_metrics.json",
        "openai_evaluation_summary.json",
        "nvidia_evaluation_summary.json",
    }

    for path in directory.glob("*.json"):

        if path.name.lower() in excluded_names:
            continue

        try:
            data = load_json(path)

            if not isinstance(data, dict):
                continue

            task_id = data.get("task_id", path.stem)
            results[str(task_id)] = data

        except (OSError, json.JSONDecodeError) as error:
            print(f"Warning: Could not load {path}: {error}")

    return results


# ============================================================
# LOAD DATASET AND GROUND TRUTH
# ============================================================

def load_benchmark_data():

    from evals.dataset_loader import (
        load_dataset,
        load_ground_truth_map,
    )

    dataset = load_dataset()
    ground_truth_map = load_ground_truth_map()

    if not dataset:
        raise ValueError("The dataset contains no tasks.")

    task_map = {}

    for task in dataset:

        task_id = get_task_id(task)

        if not task_id:
            raise ValueError(
                "A dataset task is missing task_id/test_id."
            )

        task_id = str(task_id)

        if task_id in task_map:
            raise ValueError(
                f"Duplicate task ID found: {task_id}"
            )

        task_map[task_id] = task

    return dataset, task_map, ground_truth_map


# ============================================================
# LOAD MODEL RESULTS AND ERROR LOGS
# ============================================================

def load_model_data(model_name):

    inference_results = load_task_results(
        INFERENCE_DIR / model_name
    )

    scoring_results = load_task_results(
        SCORING_DIR / model_name
    )

    semantic_results = load_task_results(
        EVALUATION_DIR / model_name
    )

    return (
        inference_results,
        scoring_results,
        semantic_results,
    )


def load_error_records(model_name):
    """
    Load available error logs.

    These logs are supplementary. Metrics are calculated
    from the individual task result files.
    """

    errors = {
        "inference": [],
        "scoring": [],
        "evaluation": [],
    }

    # Inference errors are stored in one shared file.
    inference_error_path = (
        RESULTS_DIR / "inference_errors.json"
    )

    if inference_error_path.exists():

        try:
            data = load_json(inference_error_path)

            if isinstance(data, list):
                errors["inference"] = [
                    item for item in data
                    if isinstance(item, dict)
                    and item.get("model") == model_name
                ]

            elif isinstance(data, dict):
                records = data.get("errors", [])

                if isinstance(records, list):
                    errors["inference"] = [
                        item for item in records
                        if isinstance(item, dict)
                        and item.get("model") == model_name
                    ]

        except (OSError, json.JSONDecodeError):
            pass

    # Scoring and evaluation errors are model-specific.
    error_paths = {
        "scoring": SCORING_DIR / model_name / "errors.json",
        "evaluation": (
            EVALUATION_DIR / model_name / "errors.json"
        ),
    }

    for stage, path in error_paths.items():

        if not path.exists():
            continue

        try:
            data = load_json(path)

            if isinstance(data, list):
                errors[stage] = data

            elif isinstance(data, dict):
                records = data.get("errors", [])

                if isinstance(records, list):
                    errors[stage] = records

        except (OSError, json.JSONDecodeError):
            pass

    return errors


# ============================================================
# VALIDATE SCORING RESULTS
# ============================================================

def is_successfully_scored(result):
    """
    A task is successfully scored when:
    - A scoring result exists.
    - The model response was valid JSON.
    - The deterministic scorer produced field_results.
    - No explicit scoring error is recorded.
    """

    if not isinstance(result, dict):
        return False

    if result.get("response_valid_json") is not True:
        return False

    field_results = result.get("field_results")

    if not isinstance(field_results, dict):
        return False

    if result.get("scoring_error"):
        return False

    if result.get("status") in ("failed", "error"):
        return False

    return True


def is_valid_ai_evaluation(result):

    if not isinstance(result, dict):
        return False

    if result.get("review_status") != "evaluated":
        return False

    score = result.get("question_score")

    if not isinstance(score, int):
        return False

    if isinstance(score, bool):
        return False

    return 0 <= score <= MAX_QUESTION_SCORE


# ============================================================
# SEMANTIC DIMENSION VALIDATION
# ============================================================

def get_dimension_score(result, dimension):

    if not isinstance(result, dict):
        return None

    value = result.get(dimension)

    if not isinstance(value, dict):
        return None

    score = value.get("score")
    max_score = value.get("max_score", 5)

    if not is_number(score):
        return None

    if not is_number(max_score) or max_score <= 0:
        return None

    if not 0 <= score <= max_score:
        return None

    return {
        "score": score,
        "max_score": max_score,
        "normalized_score": round(
            score / max_score * 5,
            2,
        ),
    }


# ============================================================
# FINAL QUESTION SCORE
# ============================================================

def get_final_question_score(
    scoring_result,
    semantic_result,
):
    """
    Use the AI judge's holistic score.

    A judge score of 5 is capped at 4 if deterministic
    scoring detects incorrect numerical fields or missing
    required fields.
    """

    if not is_valid_ai_evaluation(semantic_result):
        return None

    score = semantic_result["question_score"]

    if score == 5 and is_successfully_scored(scoring_result):

        field_results = scoring_result.get(
            "field_results",
            {},
        )

        numeric_fields = field_results.get(
            "numeric_fields",
            0,
        )

        correct_numeric_fields = field_results.get(
            "correct_numeric_fields",
            0,
        )

        incorrect_numeric_fields = field_results.get(
            "incorrect_numeric_fields",
            max(
                0,
                numeric_fields - correct_numeric_fields,
            ),
        )

        total_required = field_results.get(
            "total_required_fields",
            0,
        )

        present_fields = field_results.get(
            "present_fields",
            0,
        )

        has_numeric_errors = (
            incorrect_numeric_fields > 0
        )

        has_missing_fields = (
            total_required > present_fields
        )

        if has_numeric_errors or has_missing_fields:
            score = 4

    return score


# ============================================================
# EXPECTED DATASET FIELD COUNTS
# ============================================================

def calculate_expected_field_counts(
    task_map,
    ground_truth_map,
):

    total_required_fields = 0
    total_numeric_fields = 0
    per_task = {}

    for task_id, task in task_map.items():

        required_fields = get_required_fields(task)

        ground_truth = ground_truth_map.get(
            task_id,
            {},
        )

        numeric_fields = []

        for field in required_fields:

            if field in ground_truth:

                if is_numeric_ground_truth(
                    ground_truth[field]
                ):
                    numeric_fields.append(field)

        total_required_fields += len(required_fields)
        total_numeric_fields += len(numeric_fields)

        per_task[task_id] = {
            "required_fields": required_fields,
            "total_required_fields": len(required_fields),
            "numeric_fields": numeric_fields,
            "total_numeric_fields": len(numeric_fields),
        }

    return {
        "total_required_fields": total_required_fields,
        "total_numeric_fields": total_numeric_fields,
        "per_task": per_task,
    }


# ============================================================
# METRIC 1: GLOBAL ACCURACY
# ============================================================

def calculate_global_accuracy(
    final_scores,
    expected_count,
):

    evaluated_count = len(final_scores)

    correct = sum(
        1
        for score in final_scores.values()
        if score == 5
    )

    return {
        "correct_questions": correct,
        "dataset_questions": expected_count,
        "evaluated_questions": evaluated_count,
        "pending_questions": expected_count - evaluated_count,

        "accuracy_percent": percentage(
            correct,
            expected_count,
        ),

        "accuracy_among_evaluated_percent": percentage(
            correct,
            evaluated_count,
        ),

        "is_complete": evaluated_count == expected_count,

        "definition": (
            "Percentage of all dataset questions receiving "
            "a final score of 5/5. Missing or unevaluated "
            "questions are not counted as correct."
        ),
    }


# ============================================================
# METRIC 2: SCORE DISTRIBUTION
# ============================================================

def calculate_score_distribution(
    final_scores,
    expected_task_ids,
):

    evaluated_distribution = {
        str(score): 0
        for score in range(6)
    }

    end_to_end_distribution = {
        str(score): 0
        for score in range(6)
    }

    for score in final_scores.values():

        if score in range(6):
            evaluated_distribution[str(score)] += 1

    for task_id in expected_task_ids:

        score = final_scores.get(task_id)

        if score is None:
            score = 0

        end_to_end_distribution[str(score)] += 1

    evaluated_count = len(final_scores)
    expected_count = len(expected_task_ids)

    return {
        "evaluated_answer_distribution": (
            evaluated_distribution
        ),

        "evaluated_answer_percentages": {
            key: percentage(value, evaluated_count)
            for key, value in evaluated_distribution.items()
        },

        "end_to_end_distribution": (
            end_to_end_distribution
        ),

        "end_to_end_percentages": {
            key: percentage(value, expected_count)
            for key, value in end_to_end_distribution.items()
        },

        "evaluated_questions": evaluated_count,
        "dataset_questions": expected_count,

        "note": (
            "End-to-end distribution represents missing or "
            "unevaluated answers as zero. The evaluated "
            "distribution excludes them."
        ),
    }


# ============================================================
# METRIC 3: AVERAGE QUESTION SCORE
# ============================================================

def calculate_average_score(
    final_scores,
    expected_task_ids,
):

    evaluated_scores = list(final_scores.values())

    evaluated_average = (
        round(
            sum(evaluated_scores) / len(evaluated_scores),
            2,
        )
        if evaluated_scores
        else None
    )

    end_to_end_scores = [
        (
            final_scores[task_id]
            if final_scores.get(task_id) is not None
            else 0
        )
        for task_id in expected_task_ids
    ]

    end_to_end_average = (
        round(
            sum(end_to_end_scores) / len(end_to_end_scores),
            2,
        )
        if end_to_end_scores
        else None
    )

    return {
        "average_among_evaluated_answers": evaluated_average,

        "end_to_end_average_including_missing_as_zero": (
            end_to_end_average
        ),

        "maximum_score": MAX_QUESTION_SCORE,
        "evaluated_questions": len(evaluated_scores),
        "dataset_questions": len(expected_task_ids),
    }


# ============================================================
# METRIC 4: NUMERICAL ACCURACY
# ============================================================

def calculate_numerical_accuracy(
    scoring_results,
    expected_task_ids,
    expected_field_counts,
):

    correct_fields = 0
    scored_numeric_fields = 0
    successfully_scored_tasks = 0

    for task_id in expected_task_ids:

        result = scoring_results.get(task_id)

        if not is_successfully_scored(result):
            continue

        field_results = result["field_results"]

        numeric_count = field_results.get(
            "numeric_fields",
            0,
        )

        correct_count = field_results.get(
            "correct_numeric_fields",
            0,
        )

        scored_numeric_fields += numeric_count
        correct_fields += correct_count
        successfully_scored_tasks += 1

    dataset_numeric_fields = expected_field_counts[
        "total_numeric_fields"
    ]

    return {
        "correct_numeric_fields": correct_fields,

        "scored_numeric_fields": scored_numeric_fields,

        "expected_numeric_fields_in_dataset": (
            dataset_numeric_fields
        ),

        "accuracy_percent": percentage(
            correct_fields,
            dataset_numeric_fields,
        ),

        "accuracy_among_scored_numeric_fields_percent": (
            percentage(
                correct_fields,
                scored_numeric_fields,
            )
        ),

        "successfully_scored_tasks": successfully_scored_tasks,

        "note": (
            "Dataset-wide numerical accuracy uses all "
            "numerical ground-truth fields as the denominator. "
            "Missing or invalid predictions do not disappear "
            "from the calculation."
        ),
    }


# ============================================================
# METRIC 5: ANSWER COMPLETENESS
# ============================================================

def calculate_answer_completeness(
    scoring_results,
    expected_task_ids,
    expected_field_counts,
):

    present_fields = 0
    scored_required_fields = 0

    for task_id in expected_task_ids:

        result = scoring_results.get(task_id)

        if not is_successfully_scored(result):
            continue

        field_results = result["field_results"]

        present_fields += field_results.get(
            "present_fields",
            0,
        )

        scored_required_fields += field_results.get(
            "total_required_fields",
            0,
        )

    dataset_required_fields = expected_field_counts[
        "total_required_fields"
    ]

    return {
        "present_fields": present_fields,

        "scored_required_fields": scored_required_fields,

        "expected_required_fields_in_dataset": (
            dataset_required_fields
        ),

        "completeness_percent": percentage(
            present_fields,
            dataset_required_fields,
        ),

        "completeness_among_scored_responses_percent": (
            percentage(
                present_fields,
                scored_required_fields,
            )
        ),

        "note": (
            "Measures whether required output fields were "
            "provided. It does not measure whether their "
            "values are correct."
        ),
    }


# ============================================================
# METRIC 6: SEMANTIC DIMENSION METRICS
# ============================================================

def calculate_semantic_metrics(
    semantic_results,
    expected_task_ids,
):

    output = {}

    for dimension in SEMANTIC_DIMENSIONS:

        reviewed = 0
        fully_correct = 0
        normalized_scores = []

        for task_id in expected_task_ids:

            result = semantic_results.get(task_id)

            if not is_valid_ai_evaluation(result):
                continue

            dimension_result = get_dimension_score(
                result,
                dimension,
            )

            if dimension_result is None:
                continue

            reviewed += 1

            normalized = dimension_result[
                "normalized_score"
            ]

            normalized_scores.append(normalized)

            if normalized == 5:
                fully_correct += 1

        output[dimension] = {
            "fully_correct_questions": fully_correct,

            "reviewed_questions": reviewed,

            "fully_correct_percent": percentage(
                fully_correct,
                reviewed,
            ),

            "average_score_out_of_5": (
                round(
                    sum(normalized_scores)
                    / len(normalized_scores),
                    2,
                )
                if normalized_scores
                else None
            ),
        }

    return output


# ============================================================
# METRIC 7: RESPONSE, SCORING AND EVALUATION COVERAGE
# ============================================================

def calculate_coverage(
    inference_results,
    scoring_results,
    semantic_results,
    expected_task_ids,
):

    expected_count = len(expected_task_ids)

    available_responses = sum(
        1
        for task_id in expected_task_ids
        if task_id in inference_results
    )

    valid_json_responses = sum(
        1
        for task_id in expected_task_ids
        if (
            task_id in scoring_results
            and scoring_results[task_id].get(
                "response_valid_json"
            ) is True
        )
    )

    successfully_scored = sum(
        1
        for task_id in expected_task_ids
        if is_successfully_scored(
            scoring_results.get(task_id)
        )
    )

    evaluated_responses = sum(
        1
        for task_id in expected_task_ids
        if is_valid_ai_evaluation(
            semantic_results.get(task_id)
        )
    )

    return {
        "dataset_questions": expected_count,

        "available_responses": available_responses,

        "missing_responses": (
            expected_count - available_responses
        ),

        "response_coverage_percent": percentage(
            available_responses,
            expected_count,
        ),

        "valid_json_responses": valid_json_responses,

        "invalid_json_responses": (
            available_responses - valid_json_responses
        ),

        "valid_json_percent_of_dataset": percentage(
            valid_json_responses,
            expected_count,
        ),

        "valid_json_percent_of_available_responses": (
            percentage(
                valid_json_responses,
                available_responses,
            )
        ),

        "successfully_scored_tasks": successfully_scored,

        "deterministic_scoring_failures_or_missing": (
            expected_count - successfully_scored
        ),

        "deterministic_scoring_coverage_percent": percentage(
            successfully_scored,
            expected_count,
        ),

        "completed_ai_evaluations": evaluated_responses,

        "pending_or_failed_ai_evaluations": (
            expected_count - evaluated_responses
        ),

        "ai_evaluation_coverage_percent": percentage(
            evaluated_responses,
            expected_count,
        ),

        "is_response_coverage_complete": (
            available_responses == expected_count
        ),

        "is_valid_json_coverage_complete": (
            valid_json_responses == expected_count
        ),

        "is_deterministic_scoring_complete": (
            successfully_scored == expected_count
        ),

        "is_ai_evaluation_complete": (
            evaluated_responses == expected_count
        ),
    }


# ============================================================
# METRIC 8: DETERMINISTIC TASK ACCURACY
# ============================================================

def calculate_deterministic_task_accuracy(
    scoring_results,
    expected_task_ids,
    expected_field_counts,
):

    numeric_tasks = 0
    numerically_correct_tasks = 0

    complete_tasks = 0
    numerically_correct_and_complete_tasks = 0

    for task_id in expected_task_ids:

        task_counts = expected_field_counts[
            "per_task"
        ][task_id]

        expected_numeric = task_counts[
            "total_numeric_fields"
        ]

        expected_required = task_counts[
            "total_required_fields"
        ]

        if expected_numeric > 0:
            numeric_tasks += 1

        result = scoring_results.get(task_id)

        if not is_successfully_scored(result):
            continue

        field_results = result["field_results"]

        numeric_count = field_results.get(
            "numeric_fields",
            0,
        )

        correct_numeric = field_results.get(
            "correct_numeric_fields",
            0,
        )

        present_count = field_results.get(
            "present_fields",
            0,
        )

        required_count = field_results.get(
            "total_required_fields",
            0,
        )

        numeric_pass = (
            expected_numeric > 0
            and numeric_count == expected_numeric
            and correct_numeric == expected_numeric
        )

        completeness_pass = (
            required_count == expected_required
            and present_count == expected_required
        )

        if numeric_pass:
            numerically_correct_tasks += 1

        if completeness_pass:
            complete_tasks += 1

        if numeric_pass and completeness_pass:
            numerically_correct_and_complete_tasks += 1

    return {
        "tasks_with_numeric_targets": numeric_tasks,

        "tasks_with_all_numeric_fields_correct": (
            numerically_correct_tasks
        ),

        "numeric_task_accuracy_percent": percentage(
            numerically_correct_tasks,
            numeric_tasks,
        ),

        "tasks_with_all_required_fields_present": (
            complete_tasks
        ),

        "tasks_with_all_numeric_fields_and_required_fields_correct": (
            numerically_correct_and_complete_tasks
        ),

        "numeric_and_complete_task_accuracy_percent": (
            percentage(
                numerically_correct_and_complete_tasks,
                len(expected_task_ids),
            )
        ),

        "dataset_questions": len(expected_task_ids),

        "note": (
            "These are deterministic task-level diagnostics. "
            "They do not replace the holistic 5/5 AI-judge score."
        ),
    }


# ============================================================
# TASK-LEVEL FAILURE REPORT
# ============================================================

def build_failure_report(
    expected_task_ids,
    inference_results,
    scoring_results,
    semantic_results,
):

    failures = {
        "missing_inference_responses": [],
        "invalid_json_responses": [],
        "missing_scoring_records": [],
        "deterministic_scoring_failures": [],
        "missing_ai_evaluations": [],
        "ai_evaluation_failures": [],
    }

    for task_id in expected_task_ids:

        inference = inference_results.get(task_id)
        scoring = scoring_results.get(task_id)
        semantic = semantic_results.get(task_id)

        if not inference:
            failures["missing_inference_responses"].append(
                task_id
            )

        if inference and scoring:
            if scoring.get("response_valid_json") is not True:
                failures["invalid_json_responses"].append(
                    {
                        "task_id": task_id,
                        "reason": (
                            get_error_message(scoring)
                            or "Response was not valid JSON."
                        ),
                    }
                )

        if inference and not scoring:
            failures["missing_scoring_records"].append(
                task_id
            )

        if scoring and not is_successfully_scored(scoring):
            failures["deterministic_scoring_failures"].append(
                {
                    "task_id": task_id,
                    "reason": (
                        get_error_message(scoring)
                        or "Scoring record is incomplete or unsuccessful."
                    ),
                }
            )

        if not semantic:
            failures["missing_ai_evaluations"].append(
                task_id
            )

        elif not is_valid_ai_evaluation(semantic):

            failures["ai_evaluation_failures"].append(
                {
                    "task_id": task_id,
                    "reason": (
                        get_error_message(semantic)
                        or semantic.get(
                            "review_status",
                            "AI evaluation was not completed.",
                        )
                    ),
                }
            )

    failures["counts"] = {
        key: len(value)
        for key, value in failures.items()
        if isinstance(value, list)
    }

    return failures


# ============================================================
# PER-QUESTION RESULTS
# ============================================================

def build_per_question_results(
    expected_task_ids,
    inference_results,
    scoring_results,
    semantic_results,
    expected_field_counts,
):

    output = {}

    for task_id in expected_task_ids:

        inference = inference_results.get(task_id, {})
        scoring = scoring_results.get(task_id, {})
        semantic = semantic_results.get(task_id, {})

        field_results = scoring.get(
            "field_results",
            {},
        )

        final_score = get_final_question_score(
            scoring,
            semantic,
        )

        dimensions = {}

        for dimension in SEMANTIC_DIMENSIONS:

            dimensions[dimension] = get_dimension_score(
                semantic,
                dimension,
            )

        task_counts = expected_field_counts[
            "per_task"
        ][task_id]

        numeric_count = field_results.get(
            "numeric_fields",
            0,
        )

        correct_numeric = field_results.get(
            "correct_numeric_fields",
            0,
        )

        present_fields = field_results.get(
            "present_fields",
            0,
        )

        total_required = field_results.get(
            "total_required_fields",
            0,
        )

        numeric_pass = (
            is_successfully_scored(scoring)
            and task_counts["total_numeric_fields"] > 0
            and numeric_count == task_counts["total_numeric_fields"]
            and correct_numeric == task_counts["total_numeric_fields"]
        )

        completeness_pass = (
            is_successfully_scored(scoring)
            and total_required == task_counts["total_required_fields"]
            and present_fields == task_counts["total_required_fields"]
        )

        output[task_id] = {
            "has_inference_response": bool(inference),

            "response_valid_json": scoring.get(
                "response_valid_json",
                False,
            ),

            "deterministic_scoring_status": (
                "scored"
                if is_successfully_scored(scoring)
                else "failed_or_missing"
            ),

            "ai_evaluation_status": semantic.get(
                "review_status",
                "not_evaluated",
            ),

            "judge_question_score": semantic.get(
                "judge_question_score"
            ),

            "final_question_score": final_score,

            "deterministic_gate_applied": semantic.get(
                "deterministic_gate_applied",
                False,
            ),

            "deterministic_gate_reasons": semantic.get(
                "deterministic_gate_reasons",
                [],
            ),

            "numeric_fields": numeric_count,

            "correct_numeric_fields": correct_numeric,

            "expected_numeric_fields": (
                task_counts["total_numeric_fields"]
            ),

            "all_numeric_fields_correct": numeric_pass,

            "present_fields": present_fields,

            "total_required_fields": (
                task_counts["total_required_fields"]
            ),

            "all_required_fields_present": completeness_pass,

            "scoring_error": get_error_message(scoring),

            "evaluation_error": get_error_message(semantic),

            "dimensions": dimensions,
        }

    return output


# ============================================================
# CALCULATE ALL METRICS FOR ONE MODEL
# ============================================================

def calculate_model_metrics(
    model_name,
    expected_task_ids,
    task_map,
    ground_truth_map,
):

    (
        inference_results,
        scoring_results,
        semantic_results,
    ) = load_model_data(model_name)

    expected_field_counts = calculate_expected_field_counts(
        task_map,
        ground_truth_map,
    )

    final_scores = {}

    for task_id in expected_task_ids:

        semantic = semantic_results.get(task_id)

        if not is_valid_ai_evaluation(semantic):
            continue

        score = get_final_question_score(
            scoring_results.get(task_id, {}),
            semantic,
        )

        if score is not None:
            final_scores[task_id] = score

    coverage = calculate_coverage(
        inference_results,
        scoring_results,
        semantic_results,
        expected_task_ids,
    )

    failures = build_failure_report(
        expected_task_ids,
        inference_results,
        scoring_results,
        semantic_results,
    )

    metrics = {
        "metric_calculator_version": "3.1.0",

        "model": model_name,

        "generated_at": timestamp(),

        "dataset_size": len(expected_task_ids),

        "global_accuracy": calculate_global_accuracy(
            final_scores,
            len(expected_task_ids),
        ),

        "average_question_score": calculate_average_score(
            final_scores,
            expected_task_ids,
        ),

        "question_score_distribution": (
            calculate_score_distribution(
                final_scores,
                expected_task_ids,
            )
        ),

        "semantic_metrics": calculate_semantic_metrics(
            semantic_results,
            expected_task_ids,
        ),

        "numerical_accuracy": calculate_numerical_accuracy(
            scoring_results,
            expected_task_ids,
            expected_field_counts,
        ),

        "answer_completeness": calculate_answer_completeness(
            scoring_results,
            expected_task_ids,
            expected_field_counts,
        ),

        "deterministic_task_accuracy": (
            calculate_deterministic_task_accuracy(
                scoring_results,
                expected_task_ids,
                expected_field_counts,
            )
        ),

        "coverage": coverage,

        "failures": failures,

        "per_question_results": build_per_question_results(
            expected_task_ids,
            inference_results,
            scoring_results,
            semantic_results,
            expected_field_counts,
        ),

        "benchmark_status": {
            "all_responses_available": (
                coverage["is_response_coverage_complete"]
            ),

            "all_responses_valid_json": (
                coverage["is_valid_json_coverage_complete"]
            ),

            "all_tasks_deterministically_scored": (
                coverage["is_deterministic_scoring_complete"]
            ),

            "all_ai_evaluations_complete": (
                coverage["is_ai_evaluation_complete"]
            ),

            "is_complete": (
                coverage["is_response_coverage_complete"]
                and coverage["is_valid_json_coverage_complete"]
                and coverage["is_deterministic_scoring_complete"]
                and coverage["is_ai_evaluation_complete"]
            ),

            "status": (
                "COMPLETED"
                if (
                    coverage["is_response_coverage_complete"]
                    and coverage["is_valid_json_coverage_complete"]
                    and coverage["is_deterministic_scoring_complete"]
                    and coverage["is_ai_evaluation_complete"]
                )
                else "INCOMPLETE"
            ),
        },

        "scoring_policy": {
            "final_score_source": "GPT-4.1 mini AI judge",

            "semantic_dimensions": list(
                SEMANTIC_DIMENSIONS
            ),

            "five_out_of_five_gate": (
                "A judge score of 5 is capped at 4 if "
                "deterministic scoring identifies an incorrect "
                "numerical field or missing required field."
            ),

            "numerical_accuracy": (
                "Deterministic comparison against ground truth "
                "using the scorer's configured tolerances."
            ),

            "answer_completeness": (
                "Deterministic comparison of supplied required "
                "fields against the dataset's required fields."
            ),

            "missing_response_policy": (
                "Missing responses are not counted as evaluated "
                "answers. They remain in dataset-wide denominators."
            ),

            "missing_question_score_policy": (
                "Missing or unevaluated answers are not counted "
                "as correct. They are represented as zero only "
                "in explicitly labelled end-to-end distributions "
                "and averages."
            ),
        },
    }

    return metrics


# ============================================================
# CROSS-MODEL PER-QUESTION COMPARISON
# ============================================================

def build_question_comparison(
    all_metrics,
    expected_task_ids,
):

    comparison = {}

    for task_id in expected_task_ids:

        row = {}

        for model_name in MODEL_NAMES:

            model_metrics = all_metrics.get(
                model_name,
                {},
            )

            task_result = model_metrics.get(
                "per_question_results",
                {},
            ).get(task_id, {})

            row[model_name] = {
                "score": task_result.get(
                    "final_question_score"
                ),

                "judge_score": task_result.get(
                    "judge_question_score"
                ),

                "valid_json": task_result.get(
                    "response_valid_json"
                ),

                "deterministic_scoring_status": (
                    task_result.get(
                        "deterministic_scoring_status"
                    )
                ),

                "ai_evaluation_status": task_result.get(
                    "ai_evaluation_status"
                ),

                "all_numeric_fields_correct": (
                    task_result.get(
                        "all_numeric_fields_correct"
                    )
                ),

                "all_required_fields_present": (
                    task_result.get(
                        "all_required_fields_present"
                    )
                ),
            }

        openai_score = row.get(
            "openai",
            {},
        ).get("score")

        nvidia_score = row.get(
            "nvidia",
            {},
        ).get("score")

        if openai_score is None or nvidia_score is None:
            winner = "incomplete_comparison"

        elif openai_score > nvidia_score:
            winner = "openai"

        elif nvidia_score > openai_score:
            winner = "nvidia"

        else:
            winner = "tie"

        comparison[task_id] = {
            **row,
            "question_winner": winner,
        }

    return comparison


# ============================================================
# CROSS-MODEL AGGREGATE COMPARISON
# ============================================================

def build_aggregate_comparison(all_metrics):

    comparison = {}

    for model_name in MODEL_NAMES:

        metrics = all_metrics.get(model_name)

        if not metrics:
            continue

        comparison[model_name] = {
            "global_accuracy_percent": (
                metrics["global_accuracy"]["accuracy_percent"]
            ),

            "accuracy_among_evaluated_percent": (
                metrics["global_accuracy"][
                    "accuracy_among_evaluated_percent"
                ]
            ),

            "average_question_score": (
                metrics["average_question_score"][
                    "average_among_evaluated_answers"
                ]
            ),

            "end_to_end_average_score": (
                metrics["average_question_score"][
                    "end_to_end_average_including_missing_as_zero"
                ]
            ),

            "numerical_accuracy_percent": (
                metrics["numerical_accuracy"][
                    "accuracy_percent"
                ]
            ),

            "answer_completeness_percent": (
                metrics["answer_completeness"][
                    "completeness_percent"
                ]
            ),

            "response_coverage_percent": (
                metrics["coverage"][
                    "response_coverage_percent"
                ]
            ),

            "valid_json_percent_of_dataset": (
                metrics["coverage"][
                    "valid_json_percent_of_dataset"
                ]
            ),

            "deterministic_scoring_coverage_percent": (
                metrics["coverage"][
                    "deterministic_scoring_coverage_percent"
                ]
            ),

            "ai_evaluation_coverage_percent": (
                metrics["coverage"][
                    "ai_evaluation_coverage_percent"
                ]
            ),

            "semantic_metrics": metrics["semantic_metrics"],

            "benchmark_complete": (
                metrics["benchmark_status"]["is_complete"]
            ),
        }

    return comparison


# ============================================================
# WINNER ELIGIBILITY
# ============================================================

def calculate_comparison_status(all_metrics):

    both_models_available = all(
        model in all_metrics
        for model in MODEL_NAMES
    )

    if not both_models_available:
        return {
            "both_models_available": False,
            "both_models_complete": False,
            "winner_can_be_declared": False,
            "winner": None,
            "status": "SINGLE_MODEL_RUN",
            "note": (
                "Both models must be evaluated on the same "
                "dataset before a benchmark winner can be declared."
            ),
        }

    both_complete = all(
        all_metrics[model]["benchmark_status"]["is_complete"]
        for model in MODEL_NAMES
    )

    if not both_complete:
        return {
            "both_models_available": True,
            "both_models_complete": False,
            "winner_can_be_declared": False,
            "winner": None,
            "status": "INCOMPLETE",
            "note": (
                "At least one model has missing, invalid, "
                "unscored, or unevaluated tasks. No final winner "
                "can be declared."
            ),
        }

    openai_accuracy = all_metrics["openai"][
        "global_accuracy"
    ]["accuracy_percent"]

    nvidia_accuracy = all_metrics["nvidia"][
        "global_accuracy"
    ]["accuracy_percent"]

    if openai_accuracy > nvidia_accuracy:
        winner = "openai"

    elif nvidia_accuracy > openai_accuracy:
        winner = "nvidia"

    else:
        openai_average = all_metrics["openai"][
            "average_question_score"
        ]["average_among_evaluated_answers"]

        nvidia_average = all_metrics["nvidia"][
            "average_question_score"
        ]["average_among_evaluated_answers"]

        if openai_average > nvidia_average:
            winner = "openai"

        elif nvidia_average > openai_average:
            winner = "nvidia"

        else:
            winner = "tie"

    return {
        "both_models_available": True,
        "both_models_complete": True,
        "winner_can_be_declared": True,
        "winner": winner,
        "status": "COMPLETED",
        "ranking_policy": (
            "Primary metric: global accuracy. "
            "Tie-breaker: average AI judge question score."
        ),
        "note": (
            "Both models completed all benchmark tasks. "
            "The winner is determined using the stated ranking policy."
        ),
    }


# ============================================================
# MAIN METRIC CALCULATION PIPELINE
# ============================================================

def run_metric_calculation(selected_model="both"):

    (
        dataset,
        task_map,
        ground_truth_map,
    ) = load_benchmark_data()

    expected_task_ids = list(task_map.keys())

    models = (
        list(MODEL_NAMES)
        if selected_model == "both"
        else [selected_model]
    )

    print("\n" + "=" * 75)
    print("CIRCUITMIND-EVAL: METRIC CALCULATOR v3.1.0")
    print("=" * 75)

    print(f"Dataset size: {len(expected_task_ids)}")
    print(f"Models: {', '.join(models)}")
    print(f"Evaluation directory: {EVALUATION_DIR}")

    all_metrics = {}

    for model_name in models:

        print(
            f"\nCalculating metrics for "
            f"{model_name.upper()}..."
        )

        metrics = calculate_model_metrics(
            model_name=model_name,
            expected_task_ids=expected_task_ids,
            task_map=task_map,
            ground_truth_map=ground_truth_map,
        )

        all_metrics[model_name] = metrics

        output_path = (
            OUTPUT_DIR / f"{model_name}_metrics.json"
        )

        save_json(output_path, metrics)

        print("\nCoverage:")
        print(
            json.dumps(
                metrics["coverage"],
                indent=4,
                ensure_ascii=False,
            )
        )

        print("\nPerformance:")
        print(
            json.dumps(
                {
                    "global_accuracy": (
                        metrics["global_accuracy"]
                    ),
                    "average_question_score": (
                        metrics["average_question_score"]
                    ),
                    "numerical_accuracy": (
                        metrics["numerical_accuracy"]
                    ),
                    "answer_completeness": (
                        metrics["answer_completeness"]
                    ),
                },
                indent=4,
                ensure_ascii=False,
            )
        )

        print("\nBenchmark status:")
        print(
            json.dumps(
                metrics["benchmark_status"],
                indent=4,
                ensure_ascii=False,
            )
        )

        print("\nFailure counts:")
        print(
            json.dumps(
                metrics["failures"]["counts"],
                indent=4,
                ensure_ascii=False,
            )
        )

        print(f"\nSaved: {output_path}")

    # --------------------------------------------------------
    # COMBINED BENCHMARK RESULTS
    # --------------------------------------------------------

    combined = {
        "benchmark": "CircuitMind-Eval",

        "metric_calculator_version": "3.1.0",

        "generated_at": timestamp(),

        "dataset_size": len(expected_task_ids),

        "models": all_metrics,
    }

    if selected_model == "both":

        combined["per_question_comparison"] = (
            build_question_comparison(
                all_metrics,
                expected_task_ids,
            )
        )

        combined["aggregate_comparison"] = (
            build_aggregate_comparison(
                all_metrics,
            )
        )

        combined["comparison_status"] = (
            calculate_comparison_status(all_metrics)
        )

    combined_path = (
        OUTPUT_DIR / "benchmark_metrics.json"
    )

    save_json(combined_path, combined)

    print("\n" + "=" * 75)
    print("METRIC CALCULATION COMPLETED")
    print("=" * 75)

    print(f"Dataset questions: {len(expected_task_ids)}")
    print(f"Models processed: {', '.join(models)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Combined metrics: {combined_path}")

    if selected_model == "both":

        status = combined["comparison_status"]

        print("\nComparison status:")
        print(json.dumps(status, indent=4))

        if not status["winner_can_be_declared"]:
            print(
                "\nNo final winner declared. "
                "At least one model has incomplete benchmark coverage."
            )
        else:
            print(f"\nBenchmark winner: {status['winner']}")

    return combined


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate CircuitMind-Eval benchmark metrics."
        )
    )

    parser.add_argument(
        "--model",
        choices=["openai", "nvidia", "both"],
        default="both",
        help="Select which model's metrics to calculate.",
    )

    args = parser.parse_args()

    run_metric_calculation(args.model)


if __name__ == "__main__":
    main()