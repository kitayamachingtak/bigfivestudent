from typing import Dict, Tuple
import re
from fractions import Fraction

class AnswerEvaluator:
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize mathematical expressions"""
        answer = str(answer).strip()
        
        # Unicode/misc substitutions
        answer = answer.replace('\u2212', '-')
        answer = answer.replace('\u2013', '-')
        answer = answer.replace('π', 'pi')
        answer = answer.replace('√', 'sqrt')
        
        # Strip multiple-choice prefix (A), B), C., etc.
        answer = re.sub(r'^\(?[A-Ea-e]\)?[\.\)\s]+', '', answer).strip()
        
        # LaTeX wrappers
        answer = re.sub(r'\\boxed\{([^}]+)\}', r'\1', answer)
        answer = re.sub(r'\\text[a-z]*\{([^}]*)\}', r'\1', answer)
        answer = re.sub(r'\\(?:left|right)\s*', '', answer)
        answer = answer.replace('\\cdot', '*').replace('\\times', '*')
        answer = answer.replace('\\pi', 'pi')
        
        # Square roots
        answer = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', answer)
        answer = re.sub(r'\\sqrt\s+(\S+)', r'sqrt(\1)', answer)
        answer = re.sub(r'\\sqrt', 'sqrt', answer)
        
        # LaTeX fractions
        for _ in range(5):
            new = re.sub(r'\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', answer)
            if new == answer:
                break
            answer = new
        
        # Strip units
        unit_pattern = (
            r'(?<![a-zA-Z])'
            r'(meters?|centimeters?|kilometres?|kilometers?|millimeters?'
            r'|cm|km|mm|m'
            r'|feet|foot|inches?|yards?'
            r'|seconds?|minutes?|hours?|days?'
            r'|kilograms?|grams?|milligrams?|kg|mg'
            r'|liters?|litres?|milliliters?|ml'
            r'|radians?|degrees?'
            r'|units?)'
            r'(?![a-zA-Z])'
        )
        answer = re.sub(unit_pattern, '', answer, flags=re.IGNORECASE)
        
        # Remove whitespace
        answer = re.sub(r'\s+', '', answer)
        
        # Lowercase
        answer = answer.lower()
        
        # Remove backslashes
        answer = answer.replace('\\', '')
        
        # Braces to parens
        answer = answer.replace('{', '(').replace('}', ')')
        
        # Collapse redundant parens
        answer = re.sub(r'\((\d+(?:\.\d+)?)\)', r'\1', answer)
        
        # Normalize negative fractions
        answer = re.sub(r'-\(([^()]+)\)/\(([^()]+)\)', r'-\1/\2', answer)
        answer = re.sub(r'\((-[^()]+)\)', r'\1', answer)
        
        # Remove parens around sqrt
        answer = re.sub(r'\(sqrt([^()]*)\)', r'sqrt\1', answer)
        
        answer = answer.strip('+')
        return answer
    
    @staticmethod
    def try_parse_fraction(text: str) -> str:
        """Try to parse and normalize fractions"""
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
                    except:
                        pass
            
            return normalized
        except:
            return AnswerEvaluator.normalize_answer(text)
    
    @staticmethod
    def _numeric_value(norm: str):
        """Try to compute a float from a normalized expression"""
        import math
        
        if any(s in norm for s in ['sqrt', 'pi', 'sin', 'cos', 'tan', 'log']):
            try:
                safe_env = {
                    'sqrt': math.sqrt, 'pi': math.pi,
                    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                    'log': math.log, '__builtins__': {}
                }
                # Simple safety check
                if re.fullmatch(r'[-+*/()0-9.sqrtpi ]+', norm.replace('sqrt', 'SQRT').replace('pi', 'PI')):
                    return float(eval(norm, safe_env))
            except:
                pass
            return None
        
        try:
            if '/' in norm:
                parts = norm.split('/')
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            return float(norm)
        except:
            return None
    
    @staticmethod
    def answers_match(student: str, gold: str) -> bool:
        """Check if two answers match"""
        student_norm = AnswerEvaluator.normalize_answer(student)
        gold_norm = AnswerEvaluator.normalize_answer(gold)
        
        if not student_norm or not gold_norm:
            return False
        
        # Direct exact match
        if student_norm == gold_norm:
            return True
        
        # Fraction canonical form
        student_frac = AnswerEvaluator.try_parse_fraction(student)
        gold_frac = AnswerEvaluator.try_parse_fraction(gold)
        
        if student_frac == gold_frac:
            return True
        
        # Numeric value comparison
        sv = AnswerEvaluator._numeric_value(student_norm)
        gv = AnswerEvaluator._numeric_value(gold_norm)
        
        if sv is not None and gv is not None:
            if abs(sv - gv) < 1e-6:
                return True
        
        return False
    
    @staticmethod
    def tokenize(text: str) -> set:
        """
        Tokenize normalized answer into meaningful tokens.
        改进：忽略单字母tokens（除非是有意义的答案如选项A/B/C/D）
        """
        normalized = AnswerEvaluator.normalize_answer(text)
        
        # Split by operators and delimiters
        tokens = re.split(r'[,\s\+\-\*\/\(\)\[\]\{\}=]+', normalized)
        
        # Filter: remove empty and single-letter tokens (except a-e for choices)
        meaningful_tokens = set()
        for t in tokens:
            if not t:
                continue
            # Keep multi-char tokens
            if len(t) > 1:
                meaningful_tokens.add(t)
            # Keep single letters only if they're choice options
            elif t in 'abcde':
                meaningful_tokens.add(t)
            # Keep single digits
            elif t.isdigit():
                meaningful_tokens.add(t)
        
        return meaningful_tokens
    
    @staticmethod
    def calculate_f1_precision_recall(pred_tokens: set, gold_tokens: set) -> Tuple[float, float, float]:
        """Calculate F1, precision, recall from token sets"""
        if not pred_tokens or not gold_tokens:
            return 0.0, 0.0, 0.0
        
        common = pred_tokens & gold_tokens
        
        precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = len(common) / len(gold_tokens) if len(gold_tokens) > 0 else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return f1, precision, recall
    
    @staticmethod
    def exact_match(student_answer: str, gold_answer: str) -> bool:
        """Exact match using answers_match"""
        return AnswerEvaluator.answers_match(student_answer, gold_answer)
    
    @staticmethod
    def evaluate_single_answer(student_answer: str, gold_answer: str) -> Dict:
        """Evaluate student answer against one gold answer"""
        # Exact match (整体匹配)
        exact = AnswerEvaluator.exact_match(student_answer, gold_answer)
        
        # Token-based metrics
        student_tokens = AnswerEvaluator.tokenize(student_answer)
        gold_tokens = AnswerEvaluator.tokenize(gold_answer)
        
        f1, precision, recall = AnswerEvaluator.calculate_f1_precision_recall(
            student_tokens, gold_tokens
        )
        
        return {
            'exact_match': exact,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    @staticmethod
    def evaluate_answer(student_answer: str, extracted_answer: str, raw_answer: str) -> Dict:
        """
        Evaluate student answer against both standard answers.
        如果与任一标准答案exact_match，则F1也应该是满分。
        """
        eval_extracted = AnswerEvaluator.evaluate_single_answer(student_answer, str(extracted_answer))
        eval_raw = AnswerEvaluator.evaluate_single_answer(student_answer, str(raw_answer))
        
        # 如果任一exact_match为True，F1应该是1.0
        if eval_extracted['exact_match'] or eval_raw['exact_match']:
            return {
                'exact_match': True,
                'f1': 1.0,
                'precision': 1.0,
                'recall': 1.0,
                'extracted_scores': eval_extracted,
                'raw_scores': eval_raw
            }
        
        # 否则取两者中的最大值
        best_f1 = max(eval_extracted['f1'], eval_raw['f1'])
        best_precision = max(eval_extracted['precision'], eval_raw['precision'])
        best_recall = max(eval_extracted['recall'], eval_raw['recall'])
        
        return {
            'exact_match': False,
            'f1': best_f1,
            'precision': best_precision,
            'recall': best_recall,
            'extracted_scores': eval_extracted,
            'raw_scores': eval_raw
        }
    
    @staticmethod
    def calculate_micro_macro_f1(exam_results: list) -> Dict:
        """Calculate micro and macro F1 scores"""
        if not exam_results:
            return {
                'micro_f1': 0.0,
                'macro_f1': 0.0,
                'micro_precision': 0.0,
                'micro_recall': 0.0,
                'macro_precision': 0.0,
                'macro_recall': 0.0
            }
        
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        
        total_true_positives = 0
        total_predicted = 0
        total_gold = 0
        
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
            
            total_true_positives += len(common)
            total_predicted += len(student_tokens)
            total_gold += len(gold_tokens)
        
        n = len(exam_results)
        macro_f1 = total_f1 / n
        macro_precision = total_precision / n
        macro_recall = total_recall / n
        
        micro_precision = total_true_positives / total_predicted if total_predicted > 0 else 0.0
        micro_recall = total_true_positives / total_gold if total_gold > 0 else 0.0
        
        if micro_precision + micro_recall > 0:
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        else:
            micro_f1 = 0.0
        
        return {
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall
        }