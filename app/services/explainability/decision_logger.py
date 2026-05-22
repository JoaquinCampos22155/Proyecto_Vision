from typing import List, Dict, Optional


class DecisionLogger:

    def __init__(self):
        self.logs: List[Dict] = []

    def add(
        self,
        image: str,
        reason_type: str,
        message: str,
        score: Optional[float] = None,
        metadata: Optional[dict] = None
    ):

        self.logs.append({
            "image": image,
            "reason_type": reason_type,
            "message": message,
            "score": round(score, 4) if score is not None else None,
            "metadata": metadata or {}
        })

    def get_logs(self) -> List[Dict]:
        return self.logs

    def clear(self):
        self.logs = []

    def summary(self):

        return {
            "total_logs": len(self.logs),
            "logs": self.logs
        }