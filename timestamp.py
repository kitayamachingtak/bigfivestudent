class TimestampManager:
    def __init__(self, config):
        self.config = config
        self.current_timestamp = 0
        self.action_history = []
    
    def consume(self, action: str, description: str = "") -> int:
        cost = self.config.get_cost(action)
        self.current_timestamp += cost
        self.action_history.append({
            'action': action,
            'cost': cost,
            'timestamp': self.current_timestamp,
            'description': description
        })
        return cost
    
    def get_total_time(self) -> int:
        return self.current_timestamp
    
    def get_history(self) -> list:
        return self.action_history
    
    def reset(self):
        self.current_timestamp = 0
        self.action_history = []