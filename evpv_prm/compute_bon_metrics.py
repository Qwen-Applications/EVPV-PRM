# -*- coding: utf-8 -*-
"""
Compute Best-of-N (BoN) reranking metrics across multimodal benchmarks.

Given the EVPV-PRM evaluation output (from evpv_prm_inference.py or
evpv_prm_inference_mmmu.py), this script computes Pass@k and BoN@k accuracy
for k = 1..8 using five different step-score aggregation strategies:

  1. Correctness Rate     – fraction of steps with score == 1
  2. Weighted Correctness – later steps receive higher weights
  3. Streak Score         – rewards consecutive correct steps
  4. Geometric Mean       – sensitive to any single incorrect step
  5. First-Error Position – prioritizes solutions where first error appears late

Results are reported per datasource and overall.

MathVista special handling:
  MathVista records in the evaluation JSONL may not carry ground-truth option
  labels. This script optionally loads the original MathVista test.jsonl to
  match and add the correct option labels before evaluation.

Requirements:
  pip install latex2sympy2

Input:
  - EVPV-PRM output JSONL (vlmresponseN.eval.raw_step_scores populated).
  - (Optional) MathVista test.jsonl for ground-truth option labels.

Output:
  - Printed tables: Pass@k / BoN@k per datasource for each aggregation method.
"""

import json
import math
import re
import os
from typing import List, Dict, Any, Union, Callable
from latex2sympy2 import latex2sympy


# ----------------------- MathVista Preprocessing -----------------------

def extract_image_id(image_path: str):
    """Extract numeric image ID from a file path (e.g., 'images/123.png' -> '123')."""
    if not image_path or not isinstance(image_path, str):
        return None
    base = os.path.basename(image_path)
    stem, _ = os.path.splitext(base)
    m = re.search(r"\d+", stem)
    return m.group(0) if m else None


def index_to_option(idx: int) -> str:
    """Map 0-based index to option letter (0 -> 'A', 1 -> 'B', ...)."""
    return chr(ord('A') + idx) if 0 <= idx < 26 else "N/A"


def build_choice_map(choices):
    """Build a mapping from option text to option letter."""
    if not choices or not isinstance(choices, list):
        return None
    return {text: index_to_option(i) for i, text in enumerate(choices)}


def preprocess_mathvista_records(test_file_path: str, vlm_records: List[Dict]) -> List[Dict]:
    """
    Enrich MathVista VLM records with ground-truth option labels from the
    original MathVista test.jsonl (matched by numeric image ID).
    """
    print(f"\n{'='*60}")
    print("Preprocessing MathVista records ...")
    print(f"Loading ground truth from: {test_file_path}")

    test_map      = {}
    test_total    = 0
    test_bad_json = 0

    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            test_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                test_bad_json += 1
                continue
            img_id = extract_image_id(rec.get("image"))
            if img_id and img_id not in test_map:
                test_map[img_id] = rec

    print(f"  Total lines: {test_total}, parse errors: {test_bad_json}")
    print(f"  Unique image IDs available: {len(test_map)}")

    processed = []
    matched   = 0
    unmatched = 0

    for vlm_rec in vlm_records:
        img_id = extract_image_id(vlm_rec.get("image"))
        if not img_id or img_id not in test_map:
            unmatched += 1
            processed.append(vlm_rec)
            continue

        test_rec  = test_map[img_id]
        choices   = test_rec.get("choices")
        choice_map = build_choice_map(choices)

        if choice_map:
            gt_text = test_rec.get("answer")
            vlm_rec["ground_truth_option"] = choice_map.get(gt_text)
            vlm_rec["choices"] = choices
            ans_text = vlm_rec.get("answer")
            if isinstance(ans_text, str) and ans_text in choice_map:
                vlm_rec["answer"] = choice_map[ans_text]

        if "query" in test_rec and "query" not in vlm_rec:
            vlm_rec["query"] = test_rec["query"]

        processed.append(vlm_rec)
        matched += 1

    print(f"  Matched: {matched}, unmatched: {unmatched}")
    print(f"{'='*60}\n")
    return processed


# ----------------------- Math Answer Normalization -----------------------

def fixfracs(string: str) -> str:
    substrs = string.split("\\frac")
    newstr  = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            newstr += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                newstr += substr
            else:
                try:
                    assert len(substr) >= 2
                    a, b = substr[0], substr[1]
                    post = substr[2:] if len(substr) > 2 else ""
                    if b != "{":
                        newstr += "{" + a + "}{" + b + "}" + post
                    else:
                        newstr += "{" + a + "}" + b + post
                except Exception:
                    return string
    return newstr


def fixaslashb(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        ai, bi = int(a), int(b)
        assert string == "{}/{}".format(ai, bi)
        return "\\frac{" + str(ai) + "}{" + str(bi) + "}"
    except Exception:
        return string


def removerightunits(string: str) -> str:
    return string.split("\\text{ ")[0]


def fixsqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits    = string.split("\\sqrt")
    newstring = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            newstring += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            newstring += "\\sqrt" + split
    return newstring


def stripstring(string: str) -> str:
    """Normalize a math string for comparison."""
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = removerightunits(string)
    string = string.replace("\\%", "").replace("\%", "")
    if string.startswith(" ."):
        string = " 0" + string[1:]
    if len(string) > 0 and string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    if len(string.split("\\approx")) == 2:
        string = string.split("\\approx")[-1]
    if "sqrt" in string:
        string = fixsqrt(string)
    string = string.replace(" ", "")
    if "sqrt" in string:
        string = fixfracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fixaslashb(string)
    return string


def findmathanswer(s: Any) -> str:
    """Extract and normalize a math answer from model output."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    if "{}" in s:
        s = s.replace("{}", "")

    pattern = re.compile(r"\\boxed\{([^}]*)\}", flags=re.S)
    match   = pattern.findall(s)
    if match:
        ans = match[-1]
    else:
        for flag in ["the final answer is", "the answer is",
                     "the correct answer is", "the answer should be"]:
            if flag in s:
                ans = s.split(flag)[-1].strip().split("\n")[0].strip()
                break
        else:
            ans = s

    if ans.find("}") != -1 and (ans.find("{") == -1 or ans.find("}") < ans.find("{")):
        ans = ans.split("}")[0]

    ans = ans.split("=")[-1].split("\\approx")[-1]
    ans = ans.replace("\\,", "").replace("∞", "\\infty").replace("+\infty", "\infty")
    ans = ans.replace("\\text", "").replace("\\mbox", "").replace("bmatrix", "pmatrix")
    ans = ans.replace("^{\\circ}", "").replace("^\\circ", "")
    ans = ans.replace("{m}^3", "").replace("m^3", "").replace("{units}", "").replace("units", "")
    ans = ans.replace("{km}", "").replace("km", "")
    return stripstring(ans.rstrip(".").strip())


def isequal(pred: str, gt: str) -> bool:
    """Check mathematical equivalence between predicted and ground-truth answers."""
    if pred is None or gt is None:
        return False
    if pred.replace(" ", "") == "" or gt.replace(" ", "") == "":
        return False
    if gt.strip() == pred.strip():
        return True
    try:
        gt_eval   = round(eval(str(latex2sympy(gt))),   2)
        pred_eval = round(eval(str(latex2sympy(pred))), 2)
        return abs(gt_eval - pred_eval) < 1e-5
    except Exception:
        return False


# ----------------------- Five Aggregation Strategies -----------------------

def calculate_correctness_rate(scores: List[int]) -> float:
    """Strategy 1: Fraction of steps with score == 1 (simple and robust)."""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s == 1) / len(scores)


def calculate_weighted_correctness(scores: List[int]) -> float:
    """Strategy 2: Weighted correctness – later steps receive higher weight."""
    if not scores:
        return 0.0
    weighted_sum  = sum(s * (i + 1) for i, s in enumerate(scores))
    weight_total  = sum(range(1, len(scores) + 1))
    max_possible  = weight_total
    min_possible  = -weight_total
    if max_possible == min_possible:
        return 0.0
    return (weighted_sum - min_possible) / (max_possible - min_possible)


def calculate_streak_score(scores: List[int]) -> float:
    """Strategy 3: Streak score – rewards consecutive correct steps."""
    if not scores:
        return 0.0
    score  = 0.0
    streak = 0
    for s in scores:
        if s == 1:
            streak += 1
            score  += streak
        else:
            streak  = 0
            score  -= 1
    max_score = sum(range(1, len(scores) + 1))
    min_score = -len(scores)
    if max_score == min_score:
        return 0.0
    return (score - min_score) / (max_score - min_score)


def calculate_geometric_mean(scores: List[int]) -> float:
    """Strategy 4: Geometric mean – sensitive to any single incorrect step."""
    if not scores:
        return 0.0
    mapped  = [1.0 if s == 1 else 0.1 for s in scores]
    product = 1.0
    for s in mapped:
        product *= s
    return product ** (1.0 / len(mapped))


def calculate_first_error_position(scores: List[int]) -> float:
    """Strategy 5: Position of first error – earlier errors penalized more."""
    if not scores:
        return 0.0
    for i, s in enumerate(scores):
        if s == -1:
            return i / len(scores)
    return 1.0


AGGREGATION_METHODS = {
    "correctness_rate": {
        "name":        "Strategy 1: Correctness Rate",
        "func":        calculate_correctness_rate,
        "description": "Simple fraction of correct steps; stable and general.",
    },
    "weighted_correctness": {
        "name":        "Strategy 2: Weighted Correctness",
        "func":        calculate_weighted_correctness,
        "description": "Later steps receive higher weight.",
    },
    "streak_score": {
        "name":        "Strategy 3: Streak Score",
        "func":        calculate_streak_score,
        "description": "Rewards consecutive correct-step runs.",
    },
    "geometric_mean": {
        "name":        "Strategy 4: Geometric Mean",
        "func":        calculate_geometric_mean,
        "description": "Sensitive to any single incorrect step (strict).",
    },
    "first_error_position": {
        "name":        "Strategy 5: First-Error Position",
        "func":        calculate_first_error_position,
        "description": "Prioritizes solutions with later first errors.",
    },
}


# ----------------------- Evaluation Framework -----------------------

def get_normalized_gts(record: Dict[str, Any]) -> set:
    """Build a set of normalized acceptable ground-truth strings."""
    answer  = record.get("answer")
    choices = record.get("choices")

    possible_gts = [str(answer) if answer is not None else ""]
    if (
        isinstance(choices, list) and choices
        and isinstance(answer, str) and len(answer) == 1
        and "A" <= answer <= "Z"
    ):
        idx = ord(answer) - ord("A")
        if 0 <= idx < len(choices):
            possible_gts.append(str(choices[idx]))

    normalized = {findmathanswer(gt) for gt in possible_gts}
    return {gt for gt in normalized if gt.strip()}


def is_correct(model_answer: str, normalized_gts: set) -> bool:
    """Check if model_answer matches any normalized ground-truth value."""
    if model_answer is None or not normalized_gts:
        return False
    normalized_pred = findmathanswer(model_answer)
    if not normalized_pred.strip():
        return False
    for norm_gt in normalized_gts:
        if len(norm_gt) == 1 and norm_gt.isalpha():
            if normalized_pred == norm_gt:
                return True
        else:
            if isequal(normalized_pred, norm_gt):
                return True
    return False


def extract_responses(record: Dict[str, Any], max_k: int, score_func: Callable) -> List[Dict[str, Any]]:
    """Extract responses with their aggregated scores."""
    responses = []
    for i in range(1, max_k + 1):
        key = f"vlmresponse{i}"
        if key in record and isinstance(record[key], dict):
            final_answer = record[key].get("finalanswer")
            scores       = record[key].get("eval", {}).get("raw_step_scores", [])
            if not isinstance(scores, list):
                scores = []
            responses.append({"answer": final_answer, "score": score_func(scores)})
        else:
            responses.append({"answer": None, "score": -1.0})
    return responses


def compute_pass_arrays(responses: List[Dict[str, Any]], normalized_gts: set, max_k: int):
    """Compute Pass@k and BoN@k arrays for k = 1..max_k."""
    pass_k = [False] * max_k
    for k in range(1, max_k + 1):
        for i in range(k):
            if is_correct(responses[i]["answer"], normalized_gts):
                pass_k[k - 1] = True
                break

    bon_k = [False] * max_k
    for k in range(1, max_k + 1):
        cand      = responses[:k]
        max_score = max(r["score"] for r in cand)
        best      = [r for r in cand if math.isclose(r["score"], max_score)]
        bon_k[k - 1] = any(is_correct(r["answer"], normalized_gts) for r in best)

    return pass_k, bon_k


def pct(x: float) -> str:
    return f"{x:.2f}%"


def print_results_table(
    stats: Dict[str, Dict[str, Any]],
    datasources: List[str],
    max_k: int,
    method_name: str,
    description: str,
):
    """Print a formatted Pass@k / BoN@k table for one aggregation method."""
    print("\n" + "=" * 140)
    print(f"{method_name}")
    print(f"Description: {description}")
    print("=" * 140)

    header1 = ["k"]
    header2 = [" "]
    for ds in datasources:
        header1 += [ds, ""]
        header2 += ["Std Pass@k", "BoN@k"]

    rows = [header1, header2]
    for k in range(1, max_k + 1):
        row = [f"k={k}"]
        for ds in datasources:
            total    = stats[ds]["total"]
            std_rate = (stats[ds]["pass_correct"][k - 1] / total * 100.0) if total else 0.0
            bon_rate = (stats[ds]["bon_correct"][k - 1]  / total * 100.0) if total else 0.0
            row += [pct(std_rate), pct(bon_rate)]
        rows.append(row)

    col_widths = [max(len(str(r[j])) for r in rows) for j in range(len(rows[0]))]

    def fmt(r):
        return " | ".join(str(r[j]).ljust(col_widths[j]) for j in range(len(r)))

    print(fmt(rows[0]))
    print(fmt(rows[1]))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows[2:]:
        print(fmt(r))
    print()


def evaluate_with_method(records: List[Dict], max_k: int, method_key: str, method_info: Dict):
    """Run BoN evaluation for one aggregation method and print the result table."""
    score_func  = method_info["func"]
    method_name = method_info["name"]
    description = method_info["description"]

    stats: Dict[str, Dict[str, Any]] = {}

    def ensure(ds: str):
        if ds not in stats:
            stats[ds] = {"total": 0, "pass_correct": [0] * max_k, "bon_correct": [0] * max_k}

    # Determine the best single response index (for Pass@1 baseline)
    overall_total         = len(records)
    overall_single_correct = [0] * max_k

    for record in records:
        normalized_gts = get_normalized_gts(record)
        responses      = extract_responses(record, max_k=max_k, score_func=score_func)
        for i in range(max_k):
            if is_correct(responses[i]["answer"], normalized_gts):
                overall_single_correct[i] += 1

    if overall_total == 0:
        print("Error: no valid records found.")
        return

    best_idx_for_pass1 = max(range(max_k), key=lambda i: overall_single_correct[i] / overall_total)

    for record in records:
        ds = str(record.get("datasource", "UNKNOWN"))
        ensure(ds)
        ensure("OVERALL")

        normalized_gts = get_normalized_gts(record)
        responses      = extract_responses(record, max_k=max_k, score_func=score_func)
        pass_k, bon_k  = compute_pass_arrays(responses, normalized_gts, max_k)
        pass_k[0]      = is_correct(responses[best_idx_for_pass1]["answer"], normalized_gts)

        for key in (ds, "OVERALL"):
            stats[key]["total"] += 1
            for i in range(max_k):
                stats[key]["pass_correct"][i] += int(pass_k[i])
                stats[key]["bon_correct"][i]  += int(bon_k[i])

    datasources = sorted([k for k in stats if k != "OVERALL"]) + ["OVERALL"]
    print_results_table(stats, datasources, max_k, method_name, description)


# ----------------------- Main -----------------------

def main():
    # ---- Path configuration ----
    # Update these paths to point to your evaluation output files.
    evpv_results_path    = "output/evpv_prm_results.jsonl"
    mathvista_test_path  = "data/mathvista/test.jsonl"   # optional; set to "" to skip
    max_k                = 8

    print("=" * 140)
    print("EVPV-PRM Best-of-N Evaluation  (5 aggregation strategies)")
    print("=" * 140)
    print(f"EVPV results file   : {evpv_results_path}")
    print(f"MathVista test file : {mathvista_test_path}")
    print(f"Max k               : {max_k}")
    print("=" * 140)

    # Load and split records by datasource
    print("\nLoading EVPV results ...")
    all_records       = []
    mathvista_records = []
    other_records     = []

    with open(evpv_results_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                all_records.append(record)
                if record.get("datasource") == "MathVista":
                    mathvista_records.append(record)
                else:
                    other_records.append(record)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON at line {line_num}.")

    print(f"Total records: {len(all_records)}")
    print(f"  MathVista : {len(mathvista_records)}")
    print(f"  Other     : {len(other_records)}")

    # Optionally enrich MathVista records with ground-truth option labels
    if mathvista_records and mathvista_test_path and os.path.exists(mathvista_test_path):
        processed_mathvista = preprocess_mathvista_records(mathvista_test_path, mathvista_records)
    else:
        if mathvista_records:
            print("\nMathVista test file not found or not specified; skipping preprocessing.\n")
        processed_mathvista = mathvista_records

    final_records = other_records + processed_mathvista
    print(f"Records for evaluation: {len(final_records)}")

    datasource_counts: Dict[str, int] = {}
    for rec in final_records:
        ds = rec.get("datasource", "UNKNOWN")
        datasource_counts[ds] = datasource_counts.get(ds, 0) + 1
    print("\nRecords per datasource:")
    for ds in sorted(datasource_counts):
        print(f"  {ds}: {datasource_counts[ds]}")

    # Run all five aggregation strategies
    print(f"\n{'='*140}")
    print(f"Running {len(AGGREGATION_METHODS)} aggregation strategies ...")
    print(f"{'='*140}")

    for method_key, method_info in AGGREGATION_METHODS.items():
        evaluate_with_method(final_records, max_k, method_key, method_info)

    print("\n" + "=" * 140)
    print("Evaluation complete.")
    print("=" * 140)


if __name__ == "__main__":
    main()
