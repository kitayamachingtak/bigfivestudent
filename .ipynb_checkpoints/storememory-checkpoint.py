from typing import List, Dict, Optional
from retriever import MemoryRetriever

class MemoryStore:
    def __init__(self, retriever_model):
        self.memory_retriever = MemoryRetriever(retriever_model)
        self.memories = []
        self.interaction_logs = []
    
    def log_interaction(self, round_num: int, timestamp: int, interaction_type: str,
                       prompt: str, response: str, metadata: Dict = None):
        log_entry = {
            'round': round_num,
            'timestamp': timestamp,
            'interaction_type': interaction_type,
            'prompt': prompt,
            'response': response,
            'metadata': metadata or {}
        }
        self.interaction_logs.append(log_entry)
    
    def add_memory(self, content: str, source: str, timestamp: int, 
                   round_num: int, decision_info: Dict):
        memory_entry = {
            'content': content,
            'source': source,
            'timestamp': timestamp,
            'round': round_num,
            'decision_info': decision_info
        }
        self.memories.append(memory_entry)
        
        if content:
            self.memory_retriever.add_memory(content, metadata={
                'source': source,
                'timestamp': timestamp,
                'round': round_num,
                'decision_info': decision_info
            })
    
    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[str]:
        if not query or len(self.memories) == 0:
            return []
        
        retrieved = self.memory_retriever.retrieve(query, top_k, threshold)
        return [item['content'] for item in retrieved]
    
    def get_all_memories(self) -> List[Dict]:
        return self.memories
    
    def get_all_interaction_logs(self) -> List[Dict]:
        return self.interaction_logs
    
    def clear(self):
        self.memories = []
        self.interaction_logs = []
        self.memory_retriever.clear()