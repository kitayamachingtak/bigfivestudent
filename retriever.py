import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

class RetrieverModel:
    def __init__(self, model_path: str = "infly/inf-retriever-v1-1.5b", device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.device = device
        self.model.to(device)
        self.max_length = 8192
    
    def encode(self, texts: List[str]) -> Tensor:
        batch_dict = self.tokenizer(texts, max_length=self.max_length, padding=True, 
                                    truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

class QuestionRetriever:
    def __init__(self, question_bank, retriever_model: RetrieverModel, category: str, 
                 cache_dir: str = "cache"):
        self.question_bank = question_bank
        self.retriever = retriever_model
        self.category = category
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.question_ids = []
        self.question_embeddings = None
        self._load_or_compute_embeddings()
    
    def _load_or_compute_embeddings(self):
        cache_file = os.path.join(self.cache_dir, f"{self.category}_embeddings.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.question_ids = cache_data['ids']
                self.question_embeddings = cache_data['embeddings']
            print(f"Loaded embeddings for {len(self.question_ids)} questions from cache")
        else:
            questions_df = self.question_bank.get_questions_by_category(self.category)
            self.question_ids = questions_df['id'].tolist()
            question_texts = questions_df['problem'].tolist()
            
            print(f"Computing embeddings for {len(question_texts)} questions...")
            batch_size = 32
            all_embeddings = []
            for i in range(0, len(question_texts), batch_size):
                batch = question_texts[i:i+batch_size]
                embeddings = self.retriever.encode(batch)
                all_embeddings.append(embeddings.cpu())
            
            self.question_embeddings = torch.cat(all_embeddings, dim=0)
            
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'ids': self.question_ids,
                    'embeddings': self.question_embeddings
                }, f)
            print(f"Saved embeddings to cache")
    
    def retrieve(self, query: str, top_k: int = 5, task_description: str = None) -> List[int]:
        if task_description is None:
            task_description = "Given a student's learning query, retrieve relevant math problems"
        
        query_with_instruct = get_detailed_instruct(task_description, query)
        query_embedding = self.retriever.encode([query_with_instruct])
        
        scores = (query_embedding @ self.question_embeddings.to(query_embedding.device).T).squeeze(0)
        top_k_indices = torch.topk(scores, k=min(top_k, len(scores))).indices.cpu().tolist()
        
        return [self.question_ids[idx] for idx in top_k_indices]

class MemoryRetriever:
    def __init__(self, retriever_model: RetrieverModel):
        self.retriever = retriever_model
        self.memories = []
        self.memory_embeddings = None
    
    def add_memory(self, content: str, metadata: Dict = None):
        self.memories.append({
            'content': content,
            'metadata': metadata or {}
        })
        
        embedding = self.retriever.encode([content])
        if self.memory_embeddings is None:
            self.memory_embeddings = embedding.cpu()
        else:
            self.memory_embeddings = torch.cat([self.memory_embeddings, embedding.cpu()], dim=0)
    
    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.0, 
                 task_description: str = None) -> List[Dict]:
        if len(self.memories) == 0:
            return []
        
        if task_description is None:
            task_description = "Given a student's recall query, retrieve relevant memories from past learning"
        
        query_with_instruct = get_detailed_instruct(task_description, query)
        query_embedding = self.retriever.encode([query_with_instruct])
        
        scores = (query_embedding @ self.memory_embeddings.to(query_embedding.device).T).squeeze(0)
        
        filtered_indices = (scores >= threshold).nonzero(as_tuple=True)[0].cpu().tolist()
        if not filtered_indices:
            return []
        
        filtered_scores = scores[filtered_indices]
        top_k_local = torch.topk(filtered_scores, k=min(top_k, len(filtered_scores)))
        top_k_indices = [filtered_indices[idx] for idx in top_k_local.indices.cpu().tolist()]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'content': self.memories[idx]['content'],
                'metadata': self.memories[idx]['metadata'],
                'score': scores[idx].item()
            })
        
        return results
    
    def clear(self):
        self.memories = []
        self.memory_embeddings = None