from typing import Union
from pydantic import BaseModel, field_validator, Field


class AnalyzeRequest(BaseModel):
    features: list[str] = Field(min_length=2)
    contamination: float = Field(ge=0.01, le=0.20)
    n_estimators: int = Field(ge=50, le=300)
    max_samples: Union[str, int]

    @field_validator("max_samples")
    @classmethod
    def validate_max_samples(cls, v: Union[str, int]) -> Union[str, int]:
        """Validate max_samples is one of the allowed values.

        Args:
            v: The max_samples value to validate.

        Returns:
            The validated max_samples value.

        Raises:
            ValueError: If v is not in {"auto", 256, 512, 1024}.
        """
        valid = {"auto", 256, 512, 1024}
        if v not in valid:
            raise ValueError(f"max_samples must be one of {valid}")
        return v


class AnomalyRow(BaseModel):
    row_id: int
    air_temp: float
    process_temp: float
    rotational_speed: float
    torque: float
    tool_wear: float
    anomaly_score: float
    ground_truth_failure: int
    failure_types: str


class AnalyzeResponse(BaseModel):
    total_rows: int
    anomaly_count: int
    contamination_used: float
    features: list[str]
    anomalies: list[AnomalyRow]
    all_scores: list[float]
    cached: bool


class ExplainRequest(BaseModel):
    anomalies: list[dict]


class ExplainResponse(BaseModel):
    explanations: list[dict]


class QueryRequest(BaseModel):
    question: str
    context_rows: int = Field(default=20, le=50)


class QueryResponse(BaseModel):
    answer: str
