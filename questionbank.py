import pandas as pd
import random
from typing import List, Dict, Optional

class QuestionBank:
    def __init__(self, csv_path: str = "questions/NuminaMath_with_answers_and_category.csv"):
        self.df = pd.read_csv(csv_path)
        self.df['id'] = self.df.index
        self.categories = self.df['predicted_category'].unique().tolist()
    
    def get_questions_by_category(self, category: str, limit: Optional[int] = None) -> pd.DataFrame:
        questions = self.df[self.df['predicted_category'] == category].copy()
        if limit:
            questions = questions.head(limit)
        return questions
    
    def get_question_by_id(self, question_id: int) -> Dict:
        row = self.df[self.df['id'] == question_id].iloc[0]
        return {
            'id': row['id'],
            'question': row['problem'],
            'solution': row['solution'],
            'extracted_answer': row['extracted_answer'],
            'raw_answer': row['raw_answer'],
            'category': row['predicted_category']
        }
    
    def sample_questions(self, category: str, n: int, exclude_ids: List[int] = None) -> List[Dict]:
        questions = self.df[self.df['predicted_category'] == category].copy()
        if exclude_ids:
            questions = questions[~questions['id'].isin(exclude_ids)]
        
        sampled = questions.sample(n=min(n, len(questions)), random_state=random.randint(0, 10000))
        return [self.get_question_by_id(qid) for qid in sampled['id'].tolist()]
    
    def get_all_ids_by_category(self, category: str) -> List[int]:
        return self.df[self.df['predicted_category'] == category]['id'].tolist()