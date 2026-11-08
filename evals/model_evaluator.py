
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from src.config import (
    OPENAI_API_KEY,
    JUDGE_MODEL,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
)

from evals.dataset_loader import (
    load_dataset,
    load_ground_truth_map,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

INFERENCE_DIR = RESULTS_DIR
SCORING_DIR = RESULTS_DIR / "scoring"

# IMPORTANT:
# This must match metric_calculator.py.
EVALUATION_DIR = RESULTS_DIR / "model_evaluation"

EVALUATOR_VERSION = "3.1.0"

DIMENSIONS = (
    "circuit_understanding",
    "method_correctness",
    "reasoning_correctness",
    "physical_consistency",
)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=REQUEST_TIMEOUT,
    max_retries=0,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    temporary_path.replace(path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def file_fingerprint(path):
    if not path or not path.exists():
        return None

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def get_task_id(task):
    return task.get("task_id") or task.get("test_id")


def get_model_directories(model):
    return {
        "inference": INFERENCE_DIR / model,
        "scoring": SCORING_DIR / model,
        "evaluation": EVALUATION_DIR / model,
    }


def make_failure_record(
    task_id,
    model,
    reason,
    error_type="evaluation_failure",
):
    return {
        "task_id": task_id,
        "model": model,
        "judge_model": JUDGE_MODEL,
        "evaluator_version": EVALUATOR_VERSION,
        "review_status": "failed",
        "error_type": error_type,
        "error": str(reason),
        "failed_at": timestamp(),
    }


# ============================================================
# MODEL RESPONSE LOADING
# ============================================================

def load_model_response(model, task_id):

    directories = get_model_directories(model)

    response_path = (
        directories["inference"] / f"{task_id}.json"
    )

    if not response_path.exists():
        raise FileNotFoundError(
            f"Inference response not found: {response_path}"
        )

    response_record = load_json(response_path)

    if not isinstance(response_record, dict):
        raise ValueError(
            "Inference response record must be a JSON object."
        )

    recorded_task_id = (
        response_record.get("task_id")
        or response_record.get("test_id")
    )

    if recorded_task_id and recorded_task_id != task_id:
        raise ValueError(
            f"Task ID mismatch. Expected {task_id}, "
            f"found {recorded_task_id}."
        )

    model_response = response_record.get("response")

    if isinstance(model_response, dict):
        model_response = json.dumps(
            model_response,
            indent=2,
            ensure_ascii=False,
        )

    if not isinstance(model_response, str):
        raise ValueError(
            f"Invalid response type for {task_id}."
        )

    if not model_response.strip():
        raise ValueError(
            f"Empty model response for {task_id}."
        )

    return model_response, response_path


# ============================================================
# DETERMINISTIC SCORING VALIDATION
# ============================================================

def load_deterministic_scoring(model, task_id):

    directories = get_model_directories(model)

    scoring_path = (
        directories["scoring"] / f"{task_id}.json"
    )

    if not scoring_path.exists():
        return None, None

    scoring_result = load_json(scoring_path)

    if not isinstance(scoring_result, dict):
        raise ValueError(
            f"Invalid deterministic scoring record: {task_id}"
        )

    return scoring_result, scoring_path


def validate_deterministic_result(
    deterministic_result,
    task_id,
):
    """
    Prevent invalid model responses from receiving
    an AI judge score.
    """

    if deterministic_result is None:
        raise ValueError(
            f"Deterministic scoring record missing for {task_id}."
        )

    status = deterministic_result.get("status")

    if status != "scored":
        raise ValueError(
            f"Deterministic scoring failed for {task_id}. "
            f"Status: {status}"
        )

    if deterministic_result.get("response_valid_json") is not True:
        raise ValueError(
            f"Model response is not valid JSON for {task_id}."
        )

    summary = deterministic_result.get("field_results")

    if not isinstance(summary, dict):
        raise ValueError(
            f"Deterministic field results missing for {task_id}."
        )

    return True


# ============================================================
# JUDGE PROMPT
# ============================================================

def build_judge_prompt(
    task,
    ground_truth,
    model_response,
    deterministic_result=None,
):

    required_fields = task.get("required_output", [])

    if isinstance(required_fields, dict):
        required_fields = list(required_fields.keys())

    reference_solution = {
        key: value
        for key, value in ground_truth.items()
        if key not in ("task_id", "test_id")
    }

    evaluation_data = {
        "task_id": get_task_id(task),
        "category": task.get("category"),
        "difficulty": task.get("difficulty"),
        "question": task.get("question"),
        "required_output_fields": required_fields,
        "ground_truth": reference_solution,
        "model_response": model_response,
        "deterministic_scoring_results": deterministic_result,
    }

    return f"""
You are an independent expert evaluator for CircuitMind-Eval,
an electrical and electronics circuit reasoning benchmark.

Your task is to evaluate a model's answer using the supplied
question, required output fields, ground truth, and model response.

IMPORTANT EVALUATION RULES:

1. Treat the supplied ground truth as the reference answer.
2. Evaluate the model response against the actual question.
3. Check numerical values, signs, units, directions, equations,
   calculations, and final conclusions.
4. Check whether the model used an appropriate electrical
   engineering method.
5. Check whether the reasoning is consistent with the reference
   solution and established electrical principles.
6. Do not reward a correct final answer if the explanation
   contains serious contradictions.
7. Do not penalize harmless differences in wording or notation.
8. Respect the reference current directions and voltage polarities.
9. Consider the task's stated numerical tolerances, if provided
   in the ground truth.
10. Missing required fields should reduce answer completeness.
11. Do not invent missing reference values or assumptions.
12. If the ground truth does not provide enough information to
    verify a claim, explain that limitation in your evidence.
13. Do not assume unsupported claims are correct.
14. Do not use external information or search for answers.
15. The deterministic scoring results are supporting evidence.
    Independently assess the explanation and method as well.
16. The original circuit image is intentionally not provided.
    Do not claim to have visually inspected the circuit.

SCORING RUBRIC:

Score each evaluation dimension from 0 to 5.

5 = Completely correct.
4 = Mostly correct; only minor noncritical issues.
3 = Partially correct; some important errors.
2 = Major conceptual or calculation errors.
1 = Very limited correct information.
0 = Incorrect, irrelevant, or unusable.

EVALUATION DIMENSIONS:

A. circuit_understanding

Evaluate whether the response correctly identifies and explains
the electrical quantities, relationships, and circuit behavior
required by the question, based on the supplied reference.

B. method_correctness

Evaluate whether the selected electrical engineering method is
appropriate and correctly applied. Examples include Ohm's law,
KCL, KVL, nodal analysis, mesh analysis, Thevenin, Norton,
superposition, and maximum power transfer.

C. reasoning_correctness

Evaluate the mathematical derivation, intermediate steps,
equations, calculations, and logical progression.

D. physical_consistency

Evaluate whether the final answer and reasoning are consistent
with electrical principles, reference directions, polarities,
units, and the supplied ground truth.

QUESTION AND EVALUATION DATA:

{json.dumps(evaluation_data, indent=2, ensure_ascii=False)}

OUTPUT REQUIREMENTS:

Return ONLY valid JSON.

Use exactly this structure:

{{
    "task_id": "{get_task_id(task)}",
    "circuit_understanding": {{
        "score": 0,
        "max_score": 5,
        "evidence": "Detailed explanation"
    }},
    "method_correctness": {{
        "score": 0,
        "max_score": 5,
        "evidence": "Detailed explanation"
    }},
    "reasoning_correctness": {{
        "score": 0,
        "max_score": 5,
        "evidence": "Detailed explanation"
    }},
    "physical_consistency": {{
        "score": 0,
        "max_score": 5,
        "evidence": "Detailed explanation"
    }},
    "question_score": 0,
    "overall_explanation": "Overall evaluation with strengths and weaknesses"
}}

The question_score must be an integer from 0 to 5.

It represents the overall quality of the complete answer.
It is NOT the average of the four dimension scores automatically.
Use your expert judgment and the scoring rubric.

Do not return Markdown fences.
"""


# ============================================================
# JUDGE RESPONSE VALIDATION
# ============================================================

def extract_json_response(response):

    text = response.output_text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "", 1)
        text = text.replace("```", "").strip()

    result = json.loads(text)

    if not isinstance(result, dict):
        raise ValueError(
            "Judge response must be a JSON object."
        )

    return result


def validate_judge_result(result, task_id):

    if result.get("task_id") != task_id:
        raise ValueError(
            f"Judge returned incorrect task_id: "
            f"{result.get('task_id')}"
        )

    for dimension in DIMENSIONS:

        if dimension not in result:
            raise ValueError(
                f"Missing evaluation dimension: {dimension}"
            )

        dimension_result = result[dimension]

        if not isinstance(dimension_result, dict):
            raise ValueError(
                f"Invalid dimension structure: {dimension}"
            )

        score = dimension_result.get("score")
        evidence = dimension_result.get("evidence")

        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= 5
        ):
            raise ValueError(
                f"Invalid score for {dimension}: {score}"
            )

        if (
            not isinstance(evidence, str)
            or not evidence.strip()
        ):
            raise ValueError(
                f"Missing evidence for {dimension}"
            )

        dimension_result["max_score"] = 5

    question_score = result.get("question_score")

    if (
        isinstance(question_score, bool)
        or not isinstance(question_score, int)
        or not 0 <= question_score <= 5
    ):
        raise ValueError(
            f"Invalid overall question score: {question_score}"
        )

    explanation = result.get("overall_explanation")

    if (
        not isinstance(explanation, str)
        or not explanation.strip()
    ):
        raise ValueError(
            "Missing overall explanation."
        )

    return result


# ============================================================
# DETERMINISTIC SCORE GATE
# ============================================================

def apply_deterministic_gate(
    judge_result,
    deterministic_result,
):
    """
    A judge cannot award 5/5 when deterministic checks show
    missing required fields or incorrect numerical answers.

    This gate is applied only when scorer results are available.
    """

    judge_score = judge_result["question_score"]

    result = {
        "judge_question_score": judge_score,
        "question_score": judge_score,
        "deterministic_gate_applied": False,
        "deterministic_gate_reasons": [],
    }

    if not deterministic_result:
        return result

    summary = deterministic_result.get("field_results", {})

    missing_fields = summary.get("missing_fields", 0)

    incorrect_numeric_fields = summary.get(
        "incorrect_numeric_fields",
        0,
    )

    reasons = []

    if missing_fields > 0:
        reasons.append(
            f"{missing_fields} required fields are missing."
        )

    if incorrect_numeric_fields > 0:
        reasons.append(
            f"{incorrect_numeric_fields} numerical fields are incorrect."
        )

    if judge_score == 5 and reasons:
        result["question_score"] = 4
        result["deterministic_gate_applied"] = True
        result["deterministic_gate_reasons"] = reasons

    return result


# ============================================================
# SINGLE TASK EVALUATION
# ============================================================

def evaluate_one(
    task,
    ground_truth,
    model,
):

    task_id = get_task_id(task)

    model_response, response_path = load_model_response(
        model,
        task_id,
    )

    deterministic_result, scoring_path = (
        load_deterministic_scoring(
            model,
            task_id,
        )
    )

    validate_deterministic_result(
        deterministic_result,
        task_id,
    )

    prompt = build_judge_prompt(
        task=task,
        ground_truth=ground_truth,
        model_response=model_response,
        deterministic_result=deterministic_result,
    )

    response = client.responses.create(
        model=JUDGE_MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_object",
            }
        },
    )

    judge_result = extract_json_response(response)

    judge_result = validate_judge_result(
        judge_result,
        task_id,
    )

    gate_result = apply_deterministic_gate(
        judge_result,
        deterministic_result,
    )

    judge_result.update(gate_result)

    judge_result.update({
        "task_id": task_id,
        "model": model,
        "judge_model": JUDGE_MODEL,
        "evaluator_version": EVALUATOR_VERSION,
        "review_status": "evaluated",
        "evaluated_at": timestamp(),
        "source_fingerprints": {
            "response": file_fingerprint(response_path),
            "scoring": file_fingerprint(scoring_path),
        },
    })

    return judge_result


# ============================================================
# RESUME SUPPORT
# ============================================================

def is_existing_result_valid(
    output_path,
    response_path,
    scoring_path,
    task_id,
    model,
):

    if not output_path.exists():
        return False

    try:

        existing = load_json(output_path)

        if not isinstance(existing, dict):
            return False

        if existing.get("review_status") != "evaluated":
            return False

        if existing.get("task_id") != task_id:
            return False

        if existing.get("model") != model:
            return False

        if existing.get("evaluator_version") != EVALUATOR_VERSION:
            return False

        if existing.get("judge_model") != JUDGE_MODEL:
            return False

        if not isinstance(existing.get("question_score"), int):
            return False

        if not 0 <= existing["question_score"] <= 5:
            return False

        for dimension in DIMENSIONS:

            dimension_result = existing.get(dimension)

            if not isinstance(dimension_result, dict):
                return False

            score = dimension_result.get("score")
            evidence = dimension_result.get("evidence")

            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or not 0 <= score <= 5
            ):
                return False

            if not isinstance(evidence, str) or not evidence.strip():
                return False

        fingerprints = existing.get(
            "source_fingerprints",
            {},
        )

        if fingerprints.get("response") != file_fingerprint(
            response_path
        ):
            return False

        if fingerprints.get("scoring") != file_fingerprint(
            scoring_path
        ):
            return False

        return True

    except Exception:
        return False


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model):

    print("\n" + "=" * 65)
    print(f"STARTING MODEL EVALUATION: {model.upper()}")
    print("=" * 65)

    tasks = load_dataset()
    ground_truth_map = load_ground_truth_map()

    directories = get_model_directories(model)

    inference_dir = directories["inference"]
    scoring_dir = directories["scoring"]
    evaluation_dir = directories["evaluation"]

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    error_path = evaluation_dir / "errors.json"
    summary_path = evaluation_dir / "evaluation_summary.json"

    # Current-run errors only.
    # Resolved failures will not remain in this summary.
    errors = []

    summary = {
        "model": model,
        "judge_model": JUDGE_MODEL,
        "evaluator_version": EVALUATOR_VERSION,
        "dataset_size": len(tasks),
        "evaluated": 0,
        "skipped": 0,
        "missing_responses": 0,
        "invalid_responses": 0,
        "scoring_failures": 0,
        "failed": 0,
        "started_at": timestamp(),
    }

    for index, task in enumerate(tasks, start=1):

        task_id = get_task_id(task)

        print(
            f"\n[{index}/{len(tasks)}] Evaluating {task_id}"
        )

        if not task_id:

            summary["failed"] += 1

            errors.append(
                make_failure_record(
                    None,
                    model,
                    "Task has no task_id or test_id.",
                    "invalid_dataset_task",
                )
            )

            continue

        response_path = (
            inference_dir / f"{task_id}.json"
        )

        scoring_path = (
            scoring_dir / f"{task_id}.json"
        )

        output_path = (
            evaluation_dir / f"{task_id}.json"
        )

        # ----------------------------------------------------
        # Ground-truth validation
        # ----------------------------------------------------

        if task_id not in ground_truth_map:

            summary["failed"] += 1

            failure = make_failure_record(
                task_id,
                model,
                "Ground truth not found.",
                "missing_ground_truth",
            )

            save_json(output_path, failure)
            errors.append(failure)

            print("  Failed: Ground truth not found.")
            continue

        # ----------------------------------------------------
        # Inference response validation
        # ----------------------------------------------------

        if not response_path.exists():

            summary["missing_responses"] += 1
            summary["failed"] += 1

            failure = make_failure_record(
                task_id,
                model,
                "Inference response file not found.",
                "missing_response",
            )

            save_json(output_path, failure)
            errors.append(failure)

            print("  Missing inference response.")
            continue

        # ----------------------------------------------------
        # Deterministic scoring validation
        # ----------------------------------------------------

        if not scoring_path.exists():

            summary["scoring_failures"] += 1
            summary["failed"] += 1

            failure = make_failure_record(
                task_id,
                model,
                "Deterministic scoring record not found.",
                "missing_scoring_record",
            )

            save_json(output_path, failure)
            errors.append(failure)

            print("  Failed: Scoring record missing.")
            continue

        try:

            scoring_record = load_json(scoring_path)

            if not isinstance(scoring_record, dict):
                raise ValueError(
                    "Scoring record must be a JSON object."
                )

            if (
                scoring_record.get("status") != "scored"
                or scoring_record.get("response_valid_json") is not True
            ):

                summary["invalid_responses"] += 1
                summary["scoring_failures"] += 1
                summary["failed"] += 1

                reason = (
                    scoring_record.get("error")
                    or "Response failed deterministic validation."
                )

                failure = make_failure_record(
                    task_id,
                    model,
                    reason,
                    "invalid_model_response",
                )

                save_json(output_path, failure)
                errors.append(failure)

                print(
                    "  Invalid response. Judge evaluation skipped."
                )

                continue

        except Exception as error:

            summary["scoring_failures"] += 1
            summary["failed"] += 1

            failure = make_failure_record(
                task_id,
                model,
                str(error),
                "scoring_record_error",
            )

            save_json(output_path, failure)
            errors.append(failure)

            print(f"  Scoring validation failed: {error}")
            continue

        # ----------------------------------------------------
        # Resume validation
        # ----------------------------------------------------

        if is_existing_result_valid(
            output_path=output_path,
            response_path=response_path,
            scoring_path=scoring_path,
            task_id=task_id,
            model=model,
        ):

            summary["skipped"] += 1

            print(
                "  Existing evaluation is valid. Skipping."
            )

            continue

        # ----------------------------------------------------
        # AI judge evaluation with retries
        # ----------------------------------------------------

        last_error = None

        for attempt in range(MAX_RETRIES + 1):

            try:

                result = evaluate_one(
                    task=task,
                    ground_truth=ground_truth_map[task_id],
                    model=model,
                )

                save_json(output_path, result)

                summary["evaluated"] += 1

                print(
                    f"  Completed. "
                    f"Score: {result['question_score']}/5"
                )

                last_error = None
                break

            except Exception as error:

                last_error = str(error)

                print(
                    f"  Attempt {attempt + 1} failed: "
                    f"{last_error}"
                )

                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)

        if last_error:

            summary["failed"] += 1

            failure = make_failure_record(
                task_id,
                model,
                last_error,
                "judge_evaluation_failure",
            )

            save_json(output_path, failure)
            errors.append(failure)

            print(
                f"  Evaluation failed: {last_error}"
            )

        # Save progress after every task.
        save_json(error_path, errors)
        save_json(summary_path, summary)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary["completed_at"] = timestamp()

    completed_count = (
        summary["evaluated"] + summary["skipped"]
    )

    summary["completed_count"] = completed_count

    summary["evaluation_coverage_percent"] = (
        completed_count / len(tasks) * 100
        if tasks
        else 0
    )

    summary["failed_task_count"] = summary["failed"]

    summary["is_complete"] = (
        completed_count == len(tasks)
        and summary["missing_responses"] == 0
        and summary["invalid_responses"] == 0
        and summary["scoring_failures"] == 0
        and summary["failed"] == 0
    )

    summary["error_count"] = len(errors)

    save_json(error_path, errors)
    save_json(summary_path, summary)

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 65)
    print(f"EVALUATION SUMMARY: {model.upper()}")
    print("=" * 65)

    print(f"Dataset size:          {summary['dataset_size']}")
    print(f"Evaluated:             {summary['evaluated']}")
    print(f"Skipped:               {summary['skipped']}")
    print(f"Missing responses:     {summary['missing_responses']}")
    print(f"Invalid responses:     {summary['invalid_responses']}")
    print(f"Scoring failures:      {summary['scoring_failures']}")
    print(f"Failed:                {summary['failed']}")

    print(
        f"Evaluation coverage:   "
        f"{summary['evaluation_coverage_percent']:.2f}%"
    )

    print(f"Complete:              {summary['is_complete']}")

    print(f"Summary: {summary_path}")
    print(f"Errors:  {error_path}")

    return summary


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation(model="both"):

    if model == "both":
        models = ["openai", "nvidia"]
    else:
        models = [model]

    all_summaries = {}

    for selected_model in models:

        try:

            all_summaries[selected_model] = evaluate_model(
                selected_model
            )

        except Exception as error:

            print(
                f"\nCRITICAL EVALUATION ERROR "
                f"FOR {selected_model.upper()}: {error}"
            )

            all_summaries[selected_model] = {
                "model": selected_model,
                "judge_model": JUDGE_MODEL,
                "evaluator_version": EVALUATOR_VERSION,
                "is_complete": False,
                "pipeline_error": str(error),
                "completed_at": timestamp(),
            }

    combined_path = (
        EVALUATION_DIR / "combined_summary.json"
    )

    save_json(
        combined_path,
        {
            "judge_model": JUDGE_MODEL,
            "evaluator_version": EVALUATOR_VERSION,
            "generated_at": timestamp(),
            "models": all_summaries,
        },
    )

    print("\n" + "=" * 65)
    print("AI EVALUATION PROCESS FINISHED")
    print("=" * 65)

    for selected_model, result in all_summaries.items():

        print(
            f"{selected_model.upper()}: "
            f"Complete={result.get('is_complete', False)}"
        )

    print(f"\nCombined summary: {combined_path}")

    return all_summaries


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CircuitMind-Eval semantic model evaluator"
    )

    parser.add_argument(
        "--model",
        choices=["openai", "nvidia", "both"],
        default="both",
        help="Select the model to evaluate.",
    )

    args = parser.parse_args()

    run_evaluation(args.model)
