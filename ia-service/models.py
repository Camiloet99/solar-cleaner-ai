from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class Params(BaseModel):
    brushRpm: Optional[float] = None
    waterPressure: Optional[float] = None
    detergentFlowRate: Optional[float] = None
    robotSpeed: Optional[float] = None
    passOverlap: Optional[float] = None
    dwellTime: Optional[float] = None

class Point(BaseModel):
    sessionId: str
    panelId: str
    timestamp: str
    temperature: float = Field(..., description="Panel surface temperature (°C)")
    humidity: float = Field(..., description="Ambient humidity (%)")
    dustIndex: float = Field(..., description="0..1 dirtiness estimator")
    powerOutput: float = Field(..., description="Watt instantaneous")
    vibration: Optional[float] = None
    microFractureRisk: Optional[float] = None
    location: Optional[Dict[str, float]] = None
    params: Optional[Params] = None

class PredictReq(BaseModel):
    points: List[Point]

class PredictResp(BaseModel):
    sessionId: str
    timestamp: str
    predictedEfficiencyLoss: float
    recommendedCleaningFrequency: str  # now | hold_20s | after_2_windows
    cleaningRouteAdjustment: str
    alerts: List[str] = []
    proposedCommands: Optional[Dict] = None
    explain: Optional[str] = None
