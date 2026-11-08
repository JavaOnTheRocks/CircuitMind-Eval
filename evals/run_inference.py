
import argparse
import base64
import json
import mimetypes
import time

from pathlib import Path
from datetime import datetime, timezone

from openai import OpenAI

from src.config import (
    OPENAI_API_KEY,
    NVIDIA_API_KEY,
    OPENAI_MODEL,
    NVIDIA_MODEL,
    NVIDIA_BASE_URL,
    REQUEST_TIMEOUT,
    TEMPERATURE,
    MAX_RETRIES,
    MAX_OUTPUT_TOKENS,
)

from evals.dataset_loader import load_dataset
from evals.prompt_builder import build_prompt


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_CONFIG = {
    "openai": {
        "model_name": OPENAI_MODEL,
        "output_dir": RESULTS_DIR / "openai",
    },
    "nvidia": {
        "model_name": NVIDIA_MODEL,
        "output_dir": RESULTS_DIR / "nvidia",
    },
}

FAILURE_FILE = RESULTS_DIR / "inference_errors.json"

for config in MODEL_CONFIG.values():
    config["output_dir"].mkdir(parents=True, exist_ok=True)


# ==================================================
# MODEL CLIENTS
# Application-level retry is controlled below.
# ==================================================

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=REQUEST_TIMEOUT,
    max_retries=0,
)

nvidia_client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
    timeout=REQUEST_TIMEOUT,
    max_retries=0,
)


# ==================================================
# UTILITIES
# ==================================================

def get_task_id(task):
    task_id = task.get("task_id") or task.get("test_id")

    if not task_id:
        raise ValueError("Dataset task is missing task_id/test_id.")

    return str(task_id)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    temporary_file.replace(path)


def encode_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Circuit image not found: {image_path}"
        )

    mime_type, _ = mimetypes.guess_type(str(image_path))

    supported_formats = {
        "image/png",
        "image/jpeg",
        "image/webp",
    }

    if mime_type not in supported_formats:
        raise ValueError(
            f"Unsupported image format: {mime_type}"
        )

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_image}"


# ==================================================
# FAILURE MANAGEMENT
# ==================================================

def load_errors():
    if not FAILURE_FILE.exists():
        return []

    try:
        with open(FAILURE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (json.JSONDecodeError, OSError):
        print(
            "WARNING: Existing inference error log is unreadable. "
            "Starting a new error log."
        )
        return []


failed_records = load_errors()


def save_errors():
    atomic_save_json(FAILURE_FILE, failed_records)


def record_failure(model_key, task_id, stage, error):
    # Keep one current failure record per model/task.
    failed_records[:] = [
        item for item in failed_records
        if not (
            item.get("model_key") == model_key
            and item.get("task_id") == task_id
        )
    ]

    failed_records.append({
        "task_id": task_id,
        "model_key": model_key,
        "model": MODEL_CONFIG[model_key]["model_name"],
        "stage": stage,
        "status": "failed",
        "error": str(error),
        "failed_at": timestamp(),
    })

    save_errors()


def clear_failure(model_key, task_id):
    previous_count = len(failed_records)

    failed_records[:] = [
        item for item in failed_records
        if not (
            item.get("model_key") == model_key
            and item.get("task_id") == task_id
        )
    ]

    if len(failed_records) != previous_count:
        save_errors()


# ==================================================
# RESPONSE VALIDATION AND STORAGE
# ==================================================

def is_valid_existing_response(output_file, task_id):
    if not output_file.is_file():
        return False

    try:
        with open(output_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        return (
            isinstance(data, dict)
            and str(data.get("task_id")) == task_id
            and isinstance(data.get("response"), str)
            and bool(data["response"].strip())
        )

    except (json.JSONDecodeError, OSError):
        return False


def save_response(
    output_dir,
    task,
    model_name,
    prompt,
    response_text,
):
    task_id = get_task_id(task)

    output_file = output_dir / f"{task_id}.json"

    result = {
        "task_id": task_id,
        "model": model_name,
        "prompt_type": "zero-shot",
        "question": task.get("question", ""),
        "category": task.get("category", ""),
        "difficulty": task.get("difficulty", ""),
        "image_name": task.get("image_name", ""),
        "image_path": str(task.get("image_path", "")),
        "prompt": prompt,
        "response": response_text,
        "saved_at": timestamp(),
    }

    atomic_save_json(output_file, result)

    print(f"Saved: {output_file}")


# ==================================================
# OPENAI INFERENCE
# ==================================================

def run_openai(task, prompt, image_url):
    print("Running OpenAI inference...")

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                    },
                ],
            }
        ],
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    output = response.output_text

    if not isinstance(output, str) or not output.strip():
        raise ValueError("OpenAI returned an empty response.")

    return output.strip()


# ==================================================
# NVIDIA NEMOTRON THROUGH NEBIUS
# ==================================================

def run_nvidia(task, prompt, image_url):
    print("Running NVIDIA Nemotron inference...")

    response = nvidia_client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    output = response.choices[0].message.content

    # Some compatible APIs may return structured text content.
    if isinstance(output, list):
        output = "\n".join(
            item.get("text", "")
            for item in output
            if isinstance(item, dict)
        )

    if not isinstance(output, str) or not output.strip():
        raise ValueError("NVIDIA returned an empty response.")

    return output.strip()


# ==================================================
# RETRY HANDLER
# ==================================================

def execute_with_retry(
    inference_function,
    task,
    prompt,
    image_url,
    model_key,
):
    max_attempts = max(1, int(MAX_RETRIES) + 1)

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"{model_key.upper()} | "
                f"Attempt {attempt}/{max_attempts}"
            )

            return inference_function(
                task,
                prompt,
                image_url,
            )

        except Exception as error:
            last_error = error

            print(
                f"{model_key.upper()} error "
                f"on attempt {attempt}: {error}"
            )

            if attempt < max_attempts:
                wait_time = min(2 ** attempt, 30)

                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    raise RuntimeError(
        f"Failed after {max_attempts} attempts: {last_error}"
    ) from last_error


# ==================================================
# PROCESS ONE MODEL FOR ONE TASK
# ==================================================

def process_model(
    task,
    prompt,
    image_url,
    model_key,
    force=False,
):
    task_id = get_task_id(task)

    config = MODEL_CONFIG[model_key]

    model_name = config["model_name"]
    output_dir = config["output_dir"]

    output_file = output_dir / f"{task_id}.json"

    if (
        not force
        and is_valid_existing_response(output_file, task_id)
    ):
        print(
            f"{model_key.upper()}: Valid existing response. "
            "Skipping."
        )

        clear_failure(model_key, task_id)

        return "skipped"

    inference_function = (
        run_openai if model_key == "openai"
        else run_nvidia
    )

    try:
        response_text = execute_with_retry(
            inference_function,
            task,
            prompt,
            image_url,
            model_key,
        )

        save_response(
            output_dir,
            task,
            model_name,
            prompt,
            response_text,
        )

        clear_failure(model_key, task_id)

        return "completed"

    except Exception as error:
        print(
            f"{model_key.upper()} FAILED "
            f"for {task_id}: {error}"
        )

        record_failure(
            model_key,
            task_id,
            "inference",
            error,
        )

        return "failed"


# ==================================================
# VALID RESULT COUNTS
# ==================================================

def count_valid_results(model_key, dataset):
    output_dir = MODEL_CONFIG[model_key]["output_dir"]

    return sum(
        is_valid_existing_response(
            output_dir / f"{get_task_id(task)}.json",
            get_task_id(task),
        )
        for task in dataset
    )


# ==================================================
# MAIN INFERENCE PIPELINE
# ==================================================

def run_inference(model="both", force=False):

    if model not in {"both", "openai", "nvidia"}:
        raise ValueError(
            "Model must be: both, openai, or nvidia."
        )

    selected_models = (
        ["openai", "nvidia"]
        if model == "both"
        else [model]
    )

    print("\n" + "=" * 65)
    print("CIRCUITMIND-EVAL: ZERO-SHOT INFERENCE")
    print("=" * 65)

    dataset = load_dataset()

    if not dataset:
        raise ValueError("Dataset is empty.")

    dataset = sorted(
        dataset,
        key=get_task_id,
    )

    task_ids = [get_task_id(task) for task in dataset]

    if len(task_ids) != len(set(task_ids)):
        raise ValueError(
            "Duplicate task IDs found in the dataset."
        )

    total_tasks = len(dataset)

    print(f"\nDataset examples: {total_tasks}")
    print(f"Selected models: {selected_models}")
    print(f"Force rerun: {force}")

    stats = {
        key: {
            "completed": 0,
            "skipped": 0,
            "failed": 0,
        }
        for key in selected_models
    }

    for index, task in enumerate(dataset, start=1):
        task_id = get_task_id(task)

        print("\n" + "=" * 65)
        print(f"QUESTION {index}/{total_tasks}: {task_id}")
        print("=" * 65)

        # Prepare inputs once per task.
        try:
            prompt = build_prompt(task)

            image_url = encode_image(
                task["image_path"]
            )

        except Exception as error:
            print(
                f"INPUT PREPARATION FAILED "
                f"for {task_id}: {error}"
            )

            # Record failure independently for each selected model.
            for model_key in selected_models:
                record_failure(
                    model_key,
                    task_id,
                    "input_preparation",
                    error,
                )

                stats[model_key]["failed"] += 1

            # Continue to the next dataset example.
            continue

        # Run each selected model independently.
        for model_key in selected_models:
            status = process_model(
                task=task,
                prompt=prompt,
                image_url=image_url,
                model_key=model_key,
                force=force,
            )

            stats[model_key][status] += 1

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print("\n" + "=" * 65)
    print("CIRCUITMIND-EVAL INFERENCE SUMMARY")
    print("=" * 65)

    print(f"Total dataset examples: {total_tasks}")

    for model_key in selected_models:
        valid_count = count_valid_results(
            model_key,
            dataset,
        )

        failure_count = sum(
            1 for item in failed_records
            if item.get("model_key") == model_key
            and item.get("task_id") in task_ids
        )

        print(f"\n{model_key.upper()}")
        print(
            f"Newly completed: "
            f"{stats[model_key]['completed']}"
        )
        print(
            f"Skipped: {stats[model_key]['skipped']}"
        )
        print(
            f"Failed this run: "
            f"{stats[model_key]['failed']}"
        )
        print(
            f"Valid responses in dataset: "
            f"{valid_count}/{total_tasks}"
        )
        print(
            f"Remaining failures: {failure_count}"
        )

    save_errors()

    print("\nFailure log:", FAILURE_FILE)
    print("Results directory:", RESULTS_DIR)

    print("\nInference stage finished.")
    print(
        "Individual task failures do not stop the "
        "remaining inference tasks."
    )

    return stats


# ==================================================
# COMMAND LINE INTERFACE
# ==================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CircuitMind-Eval inference runner"
    )

    parser.add_argument(
        "--model",
        choices=["both", "openai", "nvidia"],
        default="both",
        help="Choose which model(s) to run.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun tasks even if valid responses already exist.",
    )

    args = parser.parse_args()

    run_inference(
        model=args.model,
        force=args.force,
    )
