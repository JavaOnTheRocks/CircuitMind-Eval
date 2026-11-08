
import json
from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "Dataset"
    / "CircuitMind-Eval"
    / "dataset"
)

QUESTIONS_FILE = DATASET_DIR / "questions.json"
METADATA_FILE = DATASET_DIR / "metadata.json"
GROUND_TRUTH_FILE = DATASET_DIR / "ground_truth.json"



# ==========================================
# JSON LOADER
# ==========================================

def load_json(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {file_path}: {error}"
        ) from error


# ==========================================
# LOAD QUESTIONS
# ==========================================

def load_questions():

    questions_data = load_json(QUESTIONS_FILE)

    questions = (
        questions_data
        if isinstance(questions_data, list)
        else questions_data.get("questions", [])
    )

    if not questions:
        raise ValueError("No questions found in questions.json")

    return questions


# ==========================================
# LOAD GROUND TRUTH
# ==========================================

def load_ground_truth():

    ground_truth_data = load_json(GROUND_TRUTH_FILE)

    records = (
        ground_truth_data
        if isinstance(ground_truth_data, list)
        else ground_truth_data.get("ground_truth", [])
    )

    if not records:
        raise ValueError("No ground-truth records found")

    return records


def load_ground_truth_map():

    records = load_ground_truth()
    ground_truth_map = {}

    for record in records:

        task_id = (
            record.get("test_id")
            or record.get("task_id")
        )

        if not task_id:
            raise ValueError(
                f"Ground-truth record has no task ID: {record}"
            )

        if task_id in ground_truth_map:
            raise ValueError(
                f"Duplicate ground-truth ID: {task_id}"
            )

        ground_truth_map[task_id] = record

    return ground_truth_map


# ==========================================
# LOAD METADATA
# ==========================================

def load_metadata():

    metadata = load_json(METADATA_FILE)

    if not isinstance(metadata, (dict, list)):
        raise ValueError("Invalid metadata.json structure")

    return metadata


# ==========================================
# DATASET VALIDATION
# ==========================================

def validate_dataset(questions, ground_truth_map):

    question_ids = []

    for question in questions:

        task_id = (
            question.get("test_id")
            or question.get("task_id")
            or question.get("id")
        )

        if not task_id:
            raise ValueError(
                f"Missing task ID: {question}"
            )

        question_ids.append(task_id)

        image_name = question.get("image")

        if not image_name:
            raise ValueError(
                f"Missing image path for {task_id}"
            )

        image_path = (DATASET_DIR / image_name).resolve()

        # Prevent paths from escaping the dataset directory.
        try:
            image_path.relative_to(DATASET_DIR.resolve())
        except ValueError:
            raise ValueError(
                f"Invalid image path for {task_id}: {image_name}"
            )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image missing for {task_id}: {image_path}"
            )

        required_output = question.get("required_output")

        if not isinstance(required_output, list) or not required_output:
            raise ValueError(
                f"Invalid required_output for {task_id}"
            )

        if task_id not in ground_truth_map:
            raise ValueError(
                f"Ground truth missing for {task_id}"
            )

        truth = ground_truth_map[task_id]

        missing_fields = [
            field
            for field in required_output
            if field not in truth
        ]

        if missing_fields:
            raise ValueError(
                f"Missing ground-truth fields for {task_id}: "
                f"{missing_fields}"
            )

    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate task IDs in questions.json")

    question_id_set = set(question_ids)
    ground_truth_id_set = set(ground_truth_map)

    extra_ids = ground_truth_id_set - question_id_set

    if extra_ids:
        raise ValueError(
            f"Unexpected ground-truth records: {sorted(extra_ids)}"
        )

    return True


# ==========================================
# MAIN DATASET LOADER
# ==========================================

def load_dataset():

    questions = load_questions()

    ground_truth_map = load_ground_truth_map()

    metadata = load_metadata()

    validate_dataset(questions, ground_truth_map)

    examples = []

    for question in questions:

        task_id = (
            question.get("test_id")
            or question.get("task_id")
            or question.get("id")
        )

        image_name = question["image"]

        image_path = (
            DATASET_DIR / image_name
        ).resolve()

        examples.append({

            "task_id": task_id,

            "title": question.get("title", ""),

            "difficulty": question.get(
                "difficulty", "unknown"
            ),

            "topic": question.get("topic", ""),

            "category": question.get("category", ""),

            "question": question.get("question", ""),

            "image_path": str(image_path),

            "image_name": image_name,

            "required_output": question["required_output"]

        })

    examples.sort(key=lambda item: item["task_id"])

    print("\n" + "=" * 55)
    print("CIRCUITMIND-EVAL DATASET VALIDATION")
    print("=" * 55)

    print(f"Questions loaded: {len(examples)}")
    print(f"Ground-truth records: {len(ground_truth_map)}")
    print("Metadata loaded: Yes")
    print("Images validated: Yes")
    print("Required output fields validated: Yes")

    print("\nDataset validation successful!")

    return examples


# ==========================================
# COMPLETE BENCHMARK LOADER
# ==========================================

def load_benchmark():

    dataset = load_dataset()

    ground_truth = load_ground_truth_map()

    metadata = load_metadata()

    return {
        "dataset": dataset,
        "ground_truth": ground_truth,
        "metadata": metadata
    }


# ==========================================
# TEST DATASET LOADER
# ==========================================

if __name__ == "__main__":

    benchmark = load_benchmark()

    dataset = benchmark["dataset"]

    print("\nDATASET PREVIEW")
    print("-" * 65)

    for example in dataset:

        print(
            example["task_id"],
            "|",
            example["category"],
            "|",
            example["difficulty"],
            "|",
            example["image_name"]
        )

    print("\nTotal validated tasks:", len(dataset))

    print("\nGROUND TRUTH PREVIEW")
    print("-" * 65)

    for task_id, truth in benchmark["ground_truth"].items():

        print(
            task_id,
            "|",
            len(truth),
            "ground-truth fields"
        )

    print("\nAll dataset checks passed!")
