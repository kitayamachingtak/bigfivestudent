from typing import Dict, Tuple
import re
from fractions import Fraction


class AnswerEvaluator:

    @staticmethod
    def normalize_answer(answer: str) -> str:
        answer = str(answer).strip()

        answer = answer.replace('\u2212', '-')   
        answer = answer.replace('\u2013', '-')
        answer = answer.replace('π', 'pi')
        answer = answer.replace('√', 'sqrt')

        answer = re.sub(r'^\(?[A-Ea-e]\)?[\.\)\s]+', '', answer).strip()

        answer = re.sub(r'\\boxed\{([^}]+)\}', r'\1', answer)
        answer = re.sub(r'\\text[a-z]*\{([^}]*)\}', r'\1', answer)
        answer = re.sub(r'\\(?:left|right)\s*', '', answer)
        answer = answer.replace('\\cdot', '*').replace('\\times', '*')
        answer = answer.replace('\\pi', 'pi')

        answer = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', answer)
        answer = re.sub(r'\\sqrt\s+(\S+)', r'sqrt(\1)', answer)
        answer = re.sub(r'\\sqrt', 'sqrt', answer)

        for _ in range(5):
            new = re.sub(r'\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', answer)
            if new == answer:
                break
            answer = new

        unit_pattern = (
            r'(?<![a-zA-Z])'
            r'(meters?|centimeters?|kilometres?|kilometers?|millimeters?'
            r'|cm|km|mm'
            r'|feet|foot|inches?|yards?'
            r'|seconds?|minutes?|hours?|days?'
            r'|kilograms?|grams?|milligrams?|kg|mg'
            r'|liters?|litres?|milliliters?|ml'
            r'|radians?|degrees?'
            r'|units?)'
            r'(?![a-zA-Z])'
        )
        answer = re.sub(unit_pattern, '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'(?<=\d)\s*\bm\b(?![a-zA-Z])', '', answer)
        answer = re.sub(r'\s+', '', answer)
        answer = answer.lower()
        answer = answer.replace('\\', '')
        answer = answer.replace('{', '(').replace('}', ')')
        answer = re.sub(r'\((\d+(?:\.\d+)?)\)', r'\1', answer)

        # -(a)/(b) → -a/b
        answer = re.sub(r'-\(([^()]+)\)/\(([^()]+)\)', r'-\1/\2', answer)
        answer = re.sub(r'\((-[^()]+)\)', r'\1', answer)

        # -(sqrt2)/2  →  -sqrt2/2
        answer = re.sub(r'\(sqrt([^()]*)\)', r'sqrt\1', answer)
        answer = answer.strip('+')

        return answer

    @staticmethod
    def try_parse_fraction(text: str) -> str:
        try:
            normalized = AnswerEvaluator.normalize_answer(text)
            if any(s in normalized for s in ['sqrt', 'pi', 'sin', 'cos', 'tan', 'log']):
                return normalized
            if '/' in normalized:
                parts = normalized.split('/')
                if len(parts) == 2:
                    try:
                        frac = Fraction(int(parts[0]), int(parts[1]))
                        return str(frac)
                    except Exception:
                        pass
            return normalized
        except Exception:
            return AnswerEvaluator.normalize_answer(text)

    @staticmethod
    def _numeric_value(norm: str):
 
        import math
        if any(s in norm for s in ['sqrt', 'pi', 'sin', 'cos', 'tan', 'log']):
            try:
                safe_env = {
                    'sqrt': math.sqrt, 'pi': math.pi,
                    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                    'log': math.log, '__builtins__': {}
                }
                if re.fullmatch(r'[-+*/()0-9.sqrt pi]+', norm.replace('sqrt', 'SQRT')):
                    return float(eval(norm, safe_env))
            except Exception:
                pass
            return None
        try:
            if '/' in norm:
                parts = norm.split('/')
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            return float(norm)
        except Exception:
            return None

    @staticmethod
    def answers_match(student: str, gold: str) -> bool:
        student_norm = AnswerEvaluator.normalize_answer(student)
        gold_norm = AnswerEvaluator.normalize_answer(gold)

        if not student_norm or not gold_norm:
            return False

        if student_norm == gold_norm:
            return True

        student_frac = AnswerEvaluator.try_parse_fraction(student)
        gold_frac = AnswerEvaluator.try_parse_fraction(gold)
        if student_frac == gold_frac:
            return True

        sv = AnswerEvaluator._numeric_value(student_norm)
        gv = AnswerEvaluator._numeric_value(gold_norm)
        if sv is not None and gv is not None:
            if abs(sv - gv) < 1e-6:
                return True

        return False

    @staticmethod
    def tokenize(text: str) -> set:
        normalized = AnswerEvaluator.normalize_answer(text)
        tokens = re.split(r'[,\s\+\-\*\/\(\)\[\]\{\}=]+', normalized)
        return set(t for t in tokens if t)

    @staticmethod
    def calculate_f1_precision_recall(pred_tokens: set, gold_tokens: set) -> Tuple[float, float, float]:
        if not pred_tokens or not gold_tokens:
            return 0.0, 0.0, 0.0
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1, precision, recall

    @staticmethod
    def exact_match(student_answer: str, gold_answer: str) -> bool:
        return AnswerEvaluator.answers_match(student_answer, gold_answer)

    @staticmethod
    def evaluate_single_answer(student_answer: str, gold_answer: str) -> Dict:
        student_tokens = AnswerEvaluator.tokenize(student_answer)
        gold_tokens = AnswerEvaluator.tokenize(gold_answer)
        exact = AnswerEvaluator.exact_match(student_answer, gold_answer)
        f1, precision, recall = AnswerEvaluator.calculate_f1_precision_recall(student_tokens, gold_tokens)
        return {'exact_match': exact, 'f1': f1, 'precision': precision, 'recall': recall}

    @staticmethod
    def evaluate_answer(student_answer: str, extracted_answer: str, raw_answer: str) -> Dict:
        eval_extracted = AnswerEvaluator.evaluate_single_answer(student_answer, str(extracted_answer))
        eval_raw = AnswerEvaluator.evaluate_single_answer(student_answer, str(raw_answer))
        best_exact = eval_extracted['exact_match'] or eval_raw['exact_match']
        best_f1 = max(eval_extracted['f1'], eval_raw['f1'])
        best_precision = max(eval_extracted['precision'], eval_raw['precision'])
        best_recall = max(eval_extracted['recall'], eval_raw['recall'])
        return {
            'exact_match': best_exact,
            'f1': best_f1,
            'precision': best_precision,
            'recall': best_recall,
            'extracted_scores': eval_extracted,
            'raw_scores': eval_raw
        }

    @staticmethod
    def calculate_micro_macro_f1(exam_results: list) -> Dict:
        if not exam_results:
            return {
                'micro_f1': 0.0, 'macro_f1': 0.0,
                'micro_precision': 0.0, 'micro_recall': 0.0,
                'macro_precision': 0.0, 'macro_recall': 0.0
            }
        total_precision = total_recall = total_f1 = 0.0
        total_tp = total_predicted = total_gold = 0
        for result in exam_results:
            metrics = result.get('metrics', {})
            total_f1 += metrics.get('f1', 0.0)
            total_precision += metrics.get('precision', 0.0)
            total_recall += metrics.get('recall', 0.0)
            student_tokens = AnswerEvaluator.tokenize(result.get('answer', ''))
            extracted_tokens = AnswerEvaluator.tokenize(str(result.get('extracted_answer', '')))
            raw_tokens = AnswerEvaluator.tokenize(str(result.get('raw_answer', '')))
            gold_tokens = extracted_tokens | raw_tokens
            common = student_tokens & gold_tokens
            total_tp += len(common)
            total_predicted += len(student_tokens)
            total_gold += len(gold_tokens)
        n = len(exam_results)
        macro_f1 = total_f1 / n
        macro_precision = total_precision / n
        macro_recall = total_recall / n
        micro_precision = total_tp / total_predicted if total_predicted > 0 else 0.0
        micro_recall = total_tp / total_gold if total_gold > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                    if (micro_precision + micro_recall) > 0 else 0.0)
        return {
            'micro_f1': micro_f1, 'macro_f1': macro_f1,
            'micro_precision': micro_precision, 'micro_recall': micro_recall,
            'macro_precision': macro_precision, 'macro_recall': macro_recall
        }

