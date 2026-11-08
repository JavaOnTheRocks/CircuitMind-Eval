
import argparse
import json
import math
import re

from datetime import datetime, timezone
from pathlib import Path

from evals.dataset_loader import (
    load_dataset,
    load_ground_truth_map,
)

from src.config import (
    ABSOLUTE_TOLERANCE,
    RELATIVE_TOLERANCE,
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIRECTORIES = {
    "openai": (
        RESULTS_DIR / "openai",
        RESULTS_DIR / "scoring" / "openai",
    ),
    "nvidia": (
        RESULTS_DIR / "nvidia",
        RESULTS_DIR / "scoring" / "nvidia",
    ),
}

SCORING_VERSION = "3.0.0"

for _, output_dir in MODEL_DIRECTORIES.values():
    output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# GENERAL UTILITIES
# ============================================================

def get_task_id(task):
    task_id = task.get("task_id") or task.get("test_id")

    if not task_id:
        raise ValueError("Task is missing task_id/test_id.")

    return str(task_id)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    temporary_path.replace(path)


# ============================================================
# UNIT DEFINITIONS
# ============================================================

UNIT_FACTORS = {
    "voltage": {
        "v": 1.0,
        "mv": 1e-3,
        "uv": 1e-6,
    },
    "current": {
        "a": 1.0,
        "ma": 1e-3,
        "ua": 1e-6,
    },
    "resistance": {
        "ohm": 1.0,
        "ω": 1.0,
        "kohm": 1e3,
        "kω": 1e3,
        "mohm": 1e-3,
        "megohm": 1e6,
        "mω": 1e6,
    },
    "power": {
        "w": 1.0,
        "mw": 1e-3,
        "kw": 1e3,
    },
}


# ============================================================
# FIELD CLASSIFICATION
# ============================================================

def get_field_dimension_and_unit(field_name):
    name = field_name.lower()

    if "voltage" in name or "vth" in name:
        if "_mv" in name or name.endswith("mv"):
            return "voltage", "mv"
        return "voltage", "v"

    if "current" in name:
        if "_ua" in name or name.endswith("ua"):
            return "current", "ua"
        if "_ma" in name or name.endswith("ma"):
            return "current", "ma"
        return "current", "a"

    if "resistance" in name or "ohm" in name:
        if "kohm" in name or "_kω" in name:
            return "resistance", "kohm"
        return "resistance", "ohm"

    if "power" in name:
        if "_mw" in name or name.endswith("mw"):
            return "power", "mw"
        if "_kw" in name or name.endswith("kw"):
            return "power", "kw"
        return "power", "w"

    return None, None


def is_numeric_value(value):
    if isinstance(value, bool):
        return False

    if isinstance(value, (int, float)):
        return math.isfinite(float(value))

    if isinstance(value, str):
        number, _ = extract_scalar(value)
        return number is not None

    return False


def is_numeric_field(field_name, ground_truth_value):
    if isinstance(ground_truth_value, bool):
        return False

    if isinstance(ground_truth_value, (int, float)):
        return True

    dimension, _ = get_field_dimension_and_unit(field_name)

    return dimension is not None


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s.+-]", "", value)

    return value


def is_present(value):
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, (dict, list)):
        return bool(value)

    return True


# ============================================================
# NUMERICAL EXTRACTION
# ============================================================

NUMBER_PATTERN = re.compile(
    r"""
    ^\s*
    ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    \s*
    ([a-zA-ZµμΩω]+)?
    \s*$
    """,
    re.VERBOSE,
)


def normalize_unit(unit):
    if not unit:
        return None

    unit = unit.strip()
    unit = unit.replace("μ", "u")
    unit = unit.replace("µ", "u")

    return unit.lower().replace(" ", "")


def extract_scalar(value):
    if isinstance(value, bool):
        return None, None

    if isinstance(value, (int, float)):
        number = float(value)

        if math.isfinite(number):
            return number, None

        return None, None

    if not isinstance(value, str):
        return None, None

    match = NUMBER_PATTERN.fullmatch(value.strip())

    if not match:
        return None, None

    try:
        number = float(match.group(1))

        if not math.isfinite(number):
            return None, None

        unit = normalize_unit(match.group(2))

        return number, unit

    except (TypeError, ValueError):
        return None, None


def extract_number(value):
    number, _ = extract_scalar(value)
    return number


# ============================================================
# UNIT CONVERSION
# ============================================================

def convert_to_expected_unit(
    number,
    supplied_unit,
    dimension,
    expected_unit,
):
    if supplied_unit is None:
        return number, "assumed_ground_truth_unit"

    supplied_unit = normalize_unit(supplied_unit)
    expected_unit = normalize_unit(expected_unit)

    factors = UNIT_FACTORS.get(dimension, {})

    if supplied_unit not in factors:
        return None, "unsupported_predicted_unit"

    if expected_unit not in factors:
        return None, "unsupported_expected_unit"

    base_value = number * factors[supplied_unit]

    converted_value = base_value / factors[expected_unit]

    return converted_value, "converted"


# ============================================================
# TOLERANCE SELECTION
# ============================================================

def get_tolerance_key(field_name, dimension):
    name = field_name.lower()

    if dimension == "voltage":
        if "_mv" in name:
            return "voltage_abs_mv"
        return "voltage_abs_v"

    if dimension == "current":
        if "_ma" in name:
            return "current_abs_ma"
        if "_ua" in name:
            return "current_abs_ua"
        return "current_abs_a"

    if dimension == "resistance":
        return "resistance_abs_ohm"

    if dimension == "power":
        return "power_abs_w"

    return None


def get_field_tolerance(field_name, ground_truth):
    dimension, expected_unit = (
        get_field_dimension_and_unit(field_name)
    )

    tolerances = ground_truth.get("tolerances", {}) or {}

    tolerance_key = get_tolerance_key(
        field_name,
        dimension,
    )

    if tolerance_key and tolerance_key in tolerances:
        abs_tol = float(tolerances[tolerance_key])

    else:
        base_keys = {
            "voltage": "voltage_abs_v",
            "current": "current_abs_a",
            "resistance": "resistance_abs_ohm",
            "power": "power_abs_w",
        }

        base_key = base_keys.get(dimension)

        if (
            base_key
            and base_key in tolerances
            and expected_unit
        ):
            base_tolerance = float(tolerances[base_key])

            factor = UNIT_FACTORS[dimension][expected_unit]

            abs_tol = base_tolerance / factor

        else:
            abs_tol = float(ABSOLUTE_TOLERANCE)

    rel_tol = float(RELATIVE_TOLERANCE)

    return abs_tol, rel_tol


# ============================================================
# NUMERICAL COMPARISON
# ============================================================

def compare_numeric_values(
    predicted,
    expected,
    field_name,
    ground_truth,
):
    predicted_number, predicted_unit = extract_scalar(predicted)

    expected_number, expected_unit = extract_scalar(expected)

    if predicted_number is None:
        return {
            "status": "invalid_numeric_value",
            "correct": False,
            "predicted": predicted,
            "expected": expected,
            "absolute_error": None,
        }

    if expected_number is None:
        return {
            "status": "invalid_ground_truth",
            "correct": False,
            "predicted": predicted,
            "expected": expected,
            "absolute_error": None,
        }

    dimension, field_expected_unit = (
        get_field_dimension_and_unit(field_name)
    )

    expected_unit = field_expected_unit or expected_unit

    conversion_status = "not_applicable"

    if dimension and expected_unit:
        converted_number, conversion_status = (
            convert_to_expected_unit(
                predicted_number,
                predicted_unit,
                dimension,
                expected_unit,
            )
        )

        if converted_number is None:
            return {
                "status": conversion_status,
                "correct": False,
                "predicted": predicted,
                "expected": expected,
                "absolute_error": None,
            }

        predicted_number = converted_number

    abs_tol, rel_tol = get_field_tolerance(
        field_name,
        ground_truth,
    )

    absolute_error = abs(predicted_number - expected_number)

    allowed_error = (
        abs_tol + rel_tol * abs(expected_number)
    )

    correct = absolute_error <= allowed_error

    return {
        "status": "correct" if correct else "incorrect",
        "correct": correct,
        "predicted": predicted_number,
        "expected": expected_number,
        "predicted_unit": predicted_unit or expected_unit,
        "expected_unit": expected_unit,
        "unit_conversion": conversion_status,
        "absolute_error": absolute_error,
        "allowed_error": allowed_error,
        "absolute_tolerance": abs_tol,
        "relative_tolerance": rel_tol,
    }


# ============================================================
# MODEL RESPONSE PARSING
# ============================================================

def parse_model_response(response_text):
    if isinstance(response_text, dict):
        return response_text

    if not isinstance(response_text, str):
        raise ValueError(
            "Model response must be a string or dictionary."
        )

    response_text = response_text.strip()

    if not response_text:
        raise ValueError("Empty model response.")

    # Attempt 1: Standard JSON.
    try:
        parsed = json.loads(response_text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Attempt 2: JSON inside Markdown code fences.
    fenced_match = re.search(
        r"```(?:json)?\s*(.*?)```",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )

    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1).strip())

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

    # Attempt 3: Extract a JSON object from surrounding text.
    decoder = json.JSONDecoder()

    for index, character in enumerate(response_text):
        if character == "{":
            try:
                parsed, _ = decoder.raw_decode(
                    response_text[index:]
                )

                if isinstance(parsed, dict):
                    return parsed

            except json.JSONDecodeError:
                continue

    raise ValueError(
        "Could not extract valid JSON from model response."
    )


def extract_final_answer(parsed_response):
    if not isinstance(parsed_response, dict):
        raise ValueError(
            "Parsed model response is not a JSON object."
        )

    final_answer = parsed_response.get("final_answer")

    if not isinstance(final_answer, dict):
        raise ValueError(
            "Missing or invalid final_answer object."
        )

    return final_answer


# ============================================================
# SCORE REQUIRED OUTPUT FIELDS
# ============================================================

def score_required_fields(
    task,
    ground_truth,
    final_answer,
):
    required_fields = task.get("required_output", [])

    if not isinstance(required_fields, list):
        raise ValueError(
            f"Invalid required_output for {get_task_id(task)}."
        )

    field_results = {}

    total_fields = len(required_fields)
    present_fields = 0

    numeric_fields = 0
    correct_numeric_fields = 0
    missing_numeric_fields = 0

    semantic_review_fields = []
    exact_text_matches = 0

    for field in required_fields:
        predicted = final_answer.get(field)
        expected = ground_truth.get(field)

        if expected is None:
            raise ValueError(
                f"Ground truth missing required field: {field}"
            )

        numeric_field = is_numeric_field(field, expected)
        present = is_present(predicted)

        if present:
            present_fields += 1

        # Missing fields.
        if not present:
            if numeric_field:
                numeric_fields += 1
                missing_numeric_fields += 1

            field_results[field] = {
                "status": "missing",
                "present": False,
                "correct": False,
                "is_numeric": numeric_field,
                "predicted": predicted,
                "expected": expected,
            }

            continue

        # Numerical fields.
        if numeric_field:
            numeric_fields += 1

            result = compare_numeric_values(
                predicted,
                expected,
                field,
                ground_truth,
            )

            result["present"] = True
            result["is_numeric"] = True

            if result["correct"]:
                correct_numeric_fields += 1

            field_results[field] = result
            continue

        # Qualitative fields.
        predicted_text = normalize_text(predicted)
        expected_text = normalize_text(expected)

        exact_match = (
            predicted_text == expected_text
            and bool(expected_text)
        )

        if exact_match:
            exact_text_matches += 1
            status = "exact_text_match"

        else:
            status = "requires_semantic_review"
            semantic_review_fields.append(field)

        field_results[field] = {
            "status": status,
            "present": True,
            "is_numeric": False,
            "exact_match": exact_match,
            "predicted": predicted,
            "expected": expected,
            "semantic_review_required": not exact_match,
        }

    missing_fields = total_fields - present_fields

    completeness = (
        present_fields / total_fields
        if total_fields else 0.0
    )

    numeric_accuracy = (
        correct_numeric_fields / numeric_fields
        if numeric_fields else None
    )

    return {
        "total_required_fields": total_fields,
        "present_fields": present_fields,
        "missing_fields": missing_fields,
        "completeness": completeness,
        "numeric_fields": numeric_fields,
        "correct_numeric_fields": correct_numeric_fields,
        "missing_numeric_fields": missing_numeric_fields,
        "incorrect_numeric_fields": (
            numeric_fields - correct_numeric_fields
        ),
        "numeric_accuracy": numeric_accuracy,
        "text_fields_exactly_matched": exact_text_matches,
        "semantic_review_fields": semantic_review_fields,
        "field_results": field_results,
    }


# ============================================================
# AUTOMATIC DIAGNOSTIC SCORE
# ============================================================

def calculate_automatic_score(field_results):
    """
    Diagnostic score only.
    The final holistic 0-5 score belongs to model_evaluator.py.
    """

    total_fields = field_results["total_required_fields"]

    if total_fields == 0:
        return None

    if field_results["semantic_review_fields"]:
        return None

    if field_results["numeric_fields"] == 0:
        return None

    numeric_accuracy = field_results["numeric_accuracy"]
    completeness = field_results["completeness"]

    if numeric_accuracy is None:
        return None

    diagnostic_score = 5 * (
        0.8 * numeric_accuracy
        + 0.2 * completeness
    )

    return round(diagnostic_score, 2)


# ============================================================
# SCORE ONE MODEL RESPONSE
# ============================================================

def score_response(
    response_record,
    task,
    ground_truth,
):
    task_id = get_task_id(task)

    raw_response = response_record.get("response", "")

    base_result = {
        "task_id": task_id,
        "model": response_record.get("model", "unknown"),
        "scoring_version": SCORING_VERSION,
        "scored_at": timestamp(),
        "status": "failed",
        "question_score": None,
        "final_score_requires_review": True,
    }

    try:
        parsed_response = parse_model_response(raw_response)

        final_answer = extract_final_answer(parsed_response)

        field_results = score_required_fields(
            task,
            ground_truth,
            final_answer,
        )

        semantic_review_required = bool(
            field_results["semantic_review_fields"]
        )

        return {
            **base_result,
            "status": "scored",
            "response_valid_json": True,
            "automatic_diagnostic_score": (
                calculate_automatic_score(field_results)
            ),
            "semantic_review_required": semantic_review_required,
            "field_results": field_results,
            "model_reasoning": parsed_response.get("reasoning", ""),
            "model_method": parsed_response.get("method", ""),
            "identified_components": parsed_response.get(
                "identified_components", []
            ),
            "calculations": parsed_response.get("calculations", []),
            "assumptions": parsed_response.get("assumptions", []),
        }

    except Exception as error:
        return {
            **base_result,
            "status": "failed",
            "response_valid_json": False,
            "automatic_diagnostic_score": None,
            "semantic_review_required": True,
            "failure_stage": "response_parsing_or_scoring",
            "error": str(error),
        }


# ============================================================
# FAILURE RECORDS
# ============================================================

def make_failure_record(
    task_id,
    model_name,
    status,
    error,
):
    return {
        "task_id": task_id,
        "model": model_name,
        "scoring_version": SCORING_VERSION,
        "scored_at": timestamp(),
        "status": status,
        "response_valid_json": False,
        "question_score": None,
        "automatic_diagnostic_score": None,
        "final_score_requires_review": True,
        "semantic_review_required": True,
        "failure_stage": "scoring",
        "error": str(error),
    }


# ============================================================
# SCORE ALL RESPONSES FOR ONE MODEL
# ============================================================

def score_model(
    model_results_dir,
    scoring_output_dir,
    dataset,
    ground_truth_map,
    model_name="unknown",
):
    scoring_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "scoring_version": SCORING_VERSION,
        "total_tasks": len(dataset),
        "responses_found": 0,
        "scored": 0,
        "valid_json_responses": 0,
        "invalid_responses": 0,
        "missing_responses": 0,
        "numeric_fields": 0,
        "correct_numeric_fields": 0,
        "missing_numeric_fields": 0,
        "present_fields": 0,
        "required_fields": 0,
        "semantic_reviews_required": 0,
        "scoring_errors": 0,
        "failed_tasks": [],
    }

    for task in dataset:
        task_id = get_task_id(task)

        response_file = (
            model_results_dir / f"{task_id}.json"
        )

        output_file = (
            scoring_output_dir / f"{task_id}.json"
        )

        # ------------------------------------------------
        # Missing response
        # ------------------------------------------------

        if not response_file.exists():
            summary["missing_responses"] += 1
            summary["scoring_errors"] += 1
            summary["failed_tasks"].append(task_id)

            failure_record = make_failure_record(
                task_id,
                model_name,
                "failed",
                "Inference response file is missing.",
            )

            save_json_atomic(output_file, failure_record)

            print(f"MISSING RESPONSE: {task_id}")
            continue

        summary["responses_found"] += 1

        # ------------------------------------------------
        # Read and score response
        # ------------------------------------------------

        try:
            with response_file.open(
                "r",
                encoding="utf-8",
            ) as file:
                response_record = json.load(file)

            if not isinstance(response_record, dict):
                raise ValueError(
                    "Inference response file must contain a JSON object."
                )

            stored_task_id = str(
                response_record.get("task_id", "")
            )

            if stored_task_id != task_id:
                raise ValueError(
                    f"Response task ID mismatch: "
                    f"expected {task_id}, got {stored_task_id}"
                )

            if task_id not in ground_truth_map:
                raise KeyError(
                    f"No ground truth found for task {task_id}"
                )

            result = score_response(
                response_record,
                task,
                ground_truth_map[task_id],
            )

            save_json_atomic(output_file, result)

            if not result["response_valid_json"]:
                summary["invalid_responses"] += 1
                summary["scoring_errors"] += 1
                summary["failed_tasks"].append(task_id)

                print(
                    f"INVALID RESPONSE: {task_id} | "
                    f"{result.get('error')}"
                )

                continue

            summary["scored"] += 1
            summary["valid_json_responses"] += 1

            fields = result["field_results"]

            summary["numeric_fields"] += (
                fields["numeric_fields"]
            )

            summary["correct_numeric_fields"] += (
                fields["correct_numeric_fields"]
            )

            summary["missing_numeric_fields"] += (
                fields["missing_numeric_fields"]
            )

            summary["present_fields"] += (
                fields["present_fields"]
            )

            summary["required_fields"] += (
                fields["total_required_fields"]
            )

            if result["semantic_review_required"]:
                summary["semantic_reviews_required"] += 1

            print(f"SCORED: {task_id}")

        except Exception as error:
            summary["scoring_errors"] += 1
            summary["failed_tasks"].append(task_id)

            failure_record = make_failure_record(
                task_id,
                model_name,
                "failed",
                error,
            )

            save_json_atomic(output_file, failure_record)

            print(
                f"SCORING FAILED: {task_id}: {error}"
            )

            # Continue with the next task.
            continue

    # ----------------------------------------------------
    # MODEL-LEVEL METRICS
    # ----------------------------------------------------

    total_tasks = len(dataset)

    summary["response_coverage_percent"] = (
        100 * summary["responses_found"] / total_tasks
        if total_tasks else 0.0
    )

    summary["valid_json_percent_of_responses"] = (
        100 * summary["valid_json_responses"]
        / summary["responses_found"]
        if summary["responses_found"] else 0.0
    )

    summary["scoring_coverage_percent"] = (
        100 * summary["scored"] / total_tasks
        if total_tasks else 0.0
    )

    summary["numerical_accuracy_percent"] = (
        100 * summary["correct_numeric_fields"]
        / summary["numeric_fields"]
        if summary["numeric_fields"] else None
    )

    summary["answer_completeness_percent"] = (
        100 * summary["present_fields"]
        / summary["required_fields"]
        if summary["required_fields"] else None
    )

    summary["failed_task_count"] = len(
        set(summary["failed_tasks"])
    )

    return summary


# ============================================================
# MAIN SCORING PIPELINE
# ============================================================

def run_scoring(model="both"):
    if model not in {"openai", "nvidia", "both"}:
        raise ValueError(
            "Model must be openai, nvidia, or both."
        )

    print("\n" + "=" * 65)
    print("CIRCUITMIND-EVAL: DETERMINISTIC SCORING")
    print("=" * 65)

    dataset = load_dataset()
    ground_truth_map = load_ground_truth_map()

    if not dataset:
        raise ValueError(
            "Dataset is empty. Cannot run scoring."
        )

    dataset = sorted(
        dataset,
        key=get_task_id,
    )

    task_ids = [get_task_id(task) for task in dataset]

    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task IDs found in dataset.")

    selected_models = (
        ["openai", "nvidia"]
        if model == "both"
        else [model]
    )

    print(f"\nDataset examples: {len(dataset)}")
    print(f"Selected models: {selected_models}")

    all_summaries = {}

    for model_key in selected_models:
        results_dir, scoring_dir = MODEL_DIRECTORIES[model_key]

        print("\n" + "=" * 65)
        print(f"SCORING MODEL: {model_key.upper()}")
        print("=" * 65)

        summary = score_model(
            results_dir,
            scoring_dir,
            dataset,
            ground_truth_map,
            model_name=model_key,
        )

        all_summaries[model_key] = summary

        print("\nMODEL SCORING SUMMARY")
        print(json.dumps(summary, indent=4))

    summary_file = (
        RESULTS_DIR
        / "scoring"
        / "scoring_summary.json"
    )

    save_json_atomic(
        summary_file,
        {
            "generated_at": timestamp(),
            "scoring_version": SCORING_VERSION,
            "dataset_size": len(dataset),
            "task_ids": task_ids,
            "models": all_summaries,
        },
    )

    print("\n" + "=" * 65)
    print("SCORING STAGE FINISHED")
    print("=" * 65)

    print("Scoring output:", RESULTS_DIR / "scoring")
    print("Summary:", summary_file)

    return all_summaries


# ============================================================
# COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CircuitMind-Eval deterministic scoring engine"
    )

    parser.add_argument(
        "--model",
        choices=["openai", "nvidia", "both"],
        default="both",
    )

    args = parser.parse_args()

    run_scoring(args.model)
