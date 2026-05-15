from app.schemas.response_schemas import ConditionEstimate


class ConditionService:
    @staticmethod
    def estimate_condition() -> ConditionEstimate:
        return ConditionEstimate(
            label="unknown",
            confidence=0.0,
            note="Condition estimation is only a suggestion and should not be used as a final decision.",
        )
