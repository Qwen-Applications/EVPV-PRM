# -*- coding: utf-8 -*-
"""
evpv_prm/prompts.py
-------------------
Central repository for all prompt templates used across the EVPV-PRM pipeline.

Sections
--------
1. Constraint Extraction  — extract structured visual constraints from an image
2. Visual Checklist Audit — score policy checklists against extracted constraints
3. Step Reward Judgment   — judge individual reasoning steps as 1 / -1
4. Policy Inference       — elicit step-by-step solutions with visual-dependency annotations
5. Perception Intervention — oracle description, noise injection, text-only solve prompts
6. Step Splitting         — parse free-form reasoning into a JSON array of steps
"""

import json
from typing import Optional


# =============================================================================
# 1. Constraint Extraction
# =============================================================================

STRUCTURED_VISION_SYSTEM = (
    "You are a visual understanding module. Your only task is to analyze the "
    "image content and produce a concise structured description. Do NOT solve "
    "the problem. Output ONLY a single JSON object with no extra text."
)


def structured_vision_user(question: str) -> str:
    """User-turn prompt: generate a structured JSON description of the image."""
    return (
        "Based on the question text, produce a concise structured description "
        "of the image (2-5 sentences total).\n"
        "Output JSON only, following this schema:\n"
        "{\n"
        '  "caption": "2-5 sentences summarizing the image and its '
        'solution-relevant relationships/labels/quantities/geometric constraints",\n'
        '  "key_facts": ["3-8 short key facts"],\n'
        '  "uncertain": ["anything unclear or uncertain; can be an empty array"]\n'
        "}\n"
        "Requirement: be faithful to the image; do not fabricate information.\n"
        "Question text:\n"
        "<<<" + question + ">>>"
    )


def structured_vision_user_short(question: str) -> str:
    """Abbreviated version of structured_vision_user for ablation studies."""
    return (
        "Analyze the image only; do not solve the problem. Output JSON:\n"
        '{ "caption": "...", "key_facts": ["..."], "uncertain": [] }\n'
        "Question: <<<" + question + ">>>"
    )


# Long-form image description prompt (used in EVPV-PRM inference pipeline).
PROMPT_GENERATE_IMAGE_DESCRIPTION = """
You are a top-tier image analyst and mathematics education expert. Your task
is to create a clear, accurate, and solution-critical description of the image
provided alongside a math question.

Focus on analyzing the image and generating a structured natural-language
description as if explaining the diagram to a student. Your description must
cover the following points in one coherent paragraph:

1. **[What is it?]** One sentence summarizing the image type and topic
   (e.g., "This is a geometric figure showing a combined cone and cylinder.").
2. **[Key elements and data?]** Identify the main mathematical objects (points,
   lines, shapes, graphs) and list all directly visible numbers, labels, and
   symbols.
3. **[Important relationships?]** Describe spatial layouts and geometric
   relationships that are critical for solving the problem.

Output format: a strict JSON code block. The root object must contain a single
key `image_description` whose value is the complete description string.
Do NOT add any explanatory text outside the JSON block.

---
Question text:
```
{question_text}
```
Image:
"""


# =============================================================================
# 2. Visual Checklist Audit
# =============================================================================

PROMPT_VISUAL_CHECKLIST_EVALUATION = """
### Fair Visual Checklist Audit: Penalize Direct Contradictions

**[Role and Task]**
You are a fair, objective AI auditor. Your sole task is to compare two
description lists about the same image: a "Ground-Truth Checklist" (provably
correct) and a "Candidate Checklist" (model-generated visual understanding).
Evaluate the factual accuracy of the Candidate Checklist.

**[Core Evaluation Principle]**
"Incomplete" is NOT "incorrect." Only a **direct contradiction** counts as an error.
Strictly ignore all omissions and missing details.

**[Inputs]**
1. **`[Ground-Truth Checklist]`**: Verified-correct factual statements about the image.
2. **`[Candidate Checklist]`**: Model-generated visual fact statements to be audited.

**[Output Format]**
Output a strict JSON code block with three keys:
1. `errors_and_hallucinations`: array of objects, each representing a direct contradiction.
   - `faulty_statement`: the contradicting statement from the Candidate Checklist.
   - `correction_or_reason`: brief explanation citing the Ground-Truth fact.
   - `severity`: `"High"` or `"Low"`.
   - Empty array `[]` if no contradictions are found.
2. `omissions`: string array. **This field does NOT affect scoring and should almost always be `[]`.**
3. `p_score`: float in [0.0, 1.0] representing the final reliability score.

No text outside the JSON block.

**[Scoring Rules]**
- Start at 1.0.
- Each **High** severity contradiction: deduct 0.5.
- Each **Low** severity contradiction: deduct 0.2.
- Minimum score: 0.0.

**[High vs Low Severity]**
- High: contradicts a primary visual fact (object count, key spatial relationship,
  presence/absence of a main object).
- Low: contradicts a secondary detail (background color, non-critical attribute).

---
**[Ground-Truth Checklist]**:
```
{golden_standard_text}
```

**[Candidate Checklist]**:
```
{checklist_to_review_text}
```
"""


# =============================================================================
# 3. Step Reward Judgment
# =============================================================================

# Used in EVPV-PRM inference pipeline (natural-language image description).
PROMPT_STEP_REWARD = """
You are a professional expert in mathematical reasoning.
You will judge whether the CURRENT solution step is correct given the image and the problem.
You MUST output ONLY a JSON integer: 1 or -1.
- 1 means the step is correct.
- -1 means the step is incorrect (contradicts the image/question/previous steps, or is invalid reasoning).

Important:
- Use the image + question + previous steps as context.
- Judge ONLY the CURRENT step relative to the full solution so far.
- Do NOT output any other text, keys, markdown, or explanation.

Problem:
```
{question_text}
```

Image description:
```
{image_description_text}
```

Previous steps:
```json
{history_steps_text}
```

CURRENT step to evaluate:
```
{current_step_text}
```

Problem image:
"""

# Used in step verifiers and ablation runner (structured JSON description).
JUDGE_PREFIX_STRICT = (
    "Now that you are a professional expert in mathematical reasoning, there is a multimodal "
    "mathematical problem and its solution steps, which require you to judge the solution steps. "
    "If the step is correct output 1, if the step is wrong output -1. "
    "Output ONLY one number: 1 or -1."
)

JUDGE_PREFIX_LENIENT = (
    "You are an expert reviewer. Output 1 if the step seems correct, otherwise -1. "
    "Only output 1 or -1."
)

JUDGE_PREFIX_NO_VISION = (
    "You are an expert in mathematical reasoning on text only. "
    "Judge the CURRENT step given the question and previous steps. "
    "Output ONLY 1 or -1."
)

# Alias used by step_verifier_local.py / step_verifier_api.py
JUDGE_PREFIX = JUDGE_PREFIX_STRICT

# Used in perception_intervention_eval for step splitting + judgement.
PROMPT_SPLIT_REASONING_TO_STEPS = r"""
You are an expert math/logic tutor.

Task:
Rewrite the given reasoning into a step-by-step solution.

STRICT OUTPUT FORMAT:
You MUST output ONLY a valid JSON array of strings.
- Use double quotes for strings.
- Use commas between items.
- No markdown, no code fences, no extra keys, no explanations.

Each string MUST start with "Step k: " where k starts at 1 and increases by 1.

If the reasoning is short, output a single step.

Example output:
["Step 1: ...", "Step 2: ..."]

Problem:
{question_text}

Reasoning:
{reasoning_text}

Problem image:
"""

PROMPT_STEP_JUDGE = r"""
You are a professional expert in mathematical reasoning.
You will judge whether the CURRENT solution step is correct given the image and the problem.

STRICT OUTPUT FORMAT:
You MUST output ONLY a valid JSON integer: 1 or -1.
No other text.

Examples:
1
-1

Problem:
{question_text}

Image description:
{image_description_text}

Previous steps:
{history_steps_text}

CURRENT step to evaluate:
{current_step_text}

Problem image:
"""

# Batch judging (step_verifier_api.py)
JUDGE_BATCH_SYSTEM = (
    "You are a professional expert in mathematical reasoning. "
    "You will judge whether each solution step is correct given the image and the problem. "
    "You MUST output ONLY a JSON array of integers, each element is 1 or -1."
)


# =============================================================================
# 4. Policy Inference
# =============================================================================

def create_inference_prompt(user_query: str, nonce: str, variant_id: int) -> str:
    """
    Build a structured prompt for visual mathematical reasoning.

    Parameters
    ----------
    user_query : str
        The question text (including any multiple-choice options).
    nonce : str
        A unique token injected to encourage diverse generations.
    variant_id : int
        Index of this candidate (1–8) shown to the model for diversity.
    """
    return f"""
You are a meticulous and precise AI assistant, an expert in visual mathematical reasoning. Your primary goal is to solve the user's query by providing a detailed, step-by-step thought process.

You MUST provide your entire response in a single, valid JSON code block. Do not include any text, explanations, or markdown formatting outside of the JSON object.

---
### DIVERSITY REQUIREMENTS (VERY IMPORTANT)
- This is reasoning variant #{variant_id}. Your reasoning path should be meaningfully different from other variants.
- Try a different logical decomposition, use different intermediate variables, or vary the order of non-dependent steps.
- Use this nonce strictly as a randomness anchor for this specific generation: {nonce}
---

### JSON OUTPUT SPECIFICATION (CRITICAL)

Your entire output must conform to this JSON schema. Follow the rules for each field precisely.

```json
{{
  "reasoningprocess": [
    {{
      "steptext": "A single, clear step of reasoning...",
      "visualdependency": "A specific, observable fact from the image, or null."
    }}
  ],
  "finalanswer": "The final answer, formatted according to the rules below."
}}
```

#### Field-Specific Rules:

1.  **`reasoningprocess`** (List of Objects):
    -   **`steptext`**: Break down your thinking into small, logical steps. Each step should represent a single calculation, observation, or deduction. Be detailed and explicit.
    -   **`visualdependency`** (String or `null`):
        -   Include a description if the step directly reads a value/label from the image, identifies a shape/spatial relationship, or performs a calculation using numbers taken directly from the image.
        -   Use `null` ONLY for purely abstract steps that reference no image content.
        -   **CRITICAL**: When no visual dependency exists, use the JSON literal `null`, NEVER an empty string `""`.

2.  **`finalanswer`** (String):
    -   **Multiple-Choice**: output the option label only (e.g., `"A"`, `"B"`). No parentheses or option text.
    -   **Fill-in / Open-ended**: output the final numerical result only (e.g., `"42"`, `"3.14"`). No units or explanations.

---
### USER QUERY

{user_query}

---
Now, generate the single, complete, and valid JSON object.
""".strip()


# =============================================================================
# 5. Perception Intervention
# =============================================================================

NOISE_INJECT_SYSTEM = (
    "You are an adversarial perturbation module. Given a concise structured "
    "image description JSON, inject a small number of plausible but misleading "
    "errors. Output ONLY a single JSON object with no extra text."
)


def noise_inject_user(structured_json: dict) -> str:
    """Prompt asking a model to corrupt a structured description with subtle errors."""
    return (
        "Without changing the JSON schema, inject 3-5 subtle but critical errors "
        "(e.g., off-by-one counts, reversed correspondences, removed/replaced key "
        "constraints, flipped spatial directions, swapped labels).\n"
        "Output JSON:\n"
        "{\n"
        '  "caption": "...",\n'
        '  "key_facts": [...],\n'
        '  "uncertain": [...],\n'
        '  "noise_notes": ["brief list of what was changed"]\n'
        "}\n"
        "Original JSON:\n"
        + json.dumps(structured_json, ensure_ascii=False)
    )


def local_solve_prompt(question: str, extra_desc: Optional[dict]) -> str:
    """
    Solving prompt for the local policy model used in perception interventions.

    Parameters
    ----------
    question : str
        The question text.
    extra_desc : dict or None
        Structured image description to inject (oracle or noisy); pass None to omit.
    """
    extra_block = "None" if extra_desc is None else json.dumps(extra_desc, ensure_ascii=False)
    return (
        "You are a rigorous mathematical reasoning model. Solve the question below.\n"
        "You will receive: the question text, an optional structured image description, "
        "and an optional image.\n\n"
        "Output ONLY a single JSON object (no markdown):\n"
        "{\n"
        '  "reasoning": "concise but complete step-by-step reasoning",\n'
        '  "final": "final answer: for multiple-choice output the option letter '
        '(A/B/C/D/...); for others output a number/expression/short text"\n'
        "}\n\n"
        "Question:\n"
        "<<<" + question + ">>>\n\n"
        "Structured image description (may contain noise):\n"
        + extra_block
    )
