from pydantic import BaseModel, Field
from typing import Optional
class TitanicPayload(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str
    Age: Optional[float] = None
    SibSp: Optional[int] = 0
    Parch: Optional[int] = 0
    Fare: Optional[float] = None
    Embarked: Optional[str] = None