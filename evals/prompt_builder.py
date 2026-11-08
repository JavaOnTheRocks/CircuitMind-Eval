
import json


def build_prompt(task):
    required_fields = task["required_output"]

    output_template = {
        field: "Provide the calculated value with units where applicable"
        for field in required_fields
    }

    return f"""
You are an expert electrical and electronics engineering assistant specializing
in circuit analysis.

Analyze the attached circuit diagram and solve the problem.

Task ID: {task["task_id"]}

Question:
{task["question"]}

Required output fields:
{json.dumps(required_fields, indent=2)}

Instructions:
1. Inspect the circuit image carefully.
2. Identify all components, connections, source polarities,
and reference directions.
3. Apply the appropriate electrical engineering principles.
4. Show equations and important calculation steps.
5. Include units and indicate current/voltage directions.
6. Do not invent values if the image is unclear.
7. Answer every required output field.

Return valid JSON only, with this structure:

{{
  "task_id": "{task["task_id"]}",
  "identified_components": [],
  "method": "",
  "reasoning": "",
  "calculations": [],
  "final_answer": {json.dumps(output_template, indent=4)},
  "assumptions": []
}}

Do not include Markdown code fences or text outside the JSON.
"""
