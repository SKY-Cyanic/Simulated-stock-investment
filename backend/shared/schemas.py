"""
Pydantic 스키마 정의
API 요청/응답 데이터 검증
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Emotion(str, Enum):
    CONFIDENT = "확신"
    GREEDY = "욕심"
    FEARFUL = "두려움"
    CALM = "냉정"


class AgentType(str, Enum):
    MOMENTUM_CHASER = "momentum_chaser"
    SMART_MONEY = "smart_money"
    LOSS_AVERSION = "loss_aversion"


# User Schemas
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    level: int
    learning_points: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# Trade Schemas
class TradeCreate(BaseModel):
    symbol: str = Field(..., max_length=20)
    action: TradeAction
    price: float = Field(..., gt=0)
    quantity: int = Field(..., gt=0)
    emotion: Optional[Emotion] = None
    reason: Optional[str] = None


class TradeResponse(BaseModel):
    id: int
    user_id: str
    symbol: str
    action: str
    price: float
    quantity: int
    timestamp: datetime
    emotion: Optional[str]
    unrealized_pnl: Optional[float]
    
    class Config:
        from_attributes = True


# Market Data Schemas
class CandleData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class MarketDataResponse(BaseModel):
    symbol: str
    timeframe: str
    candles: List[CandleData]


# AI Simulation Schemas
class SimulationRequest(BaseModel):
    symbol: str
    current_price: float
    recent_candles: List[CandleData]
    news_events: Optional[List[Dict[str, Any]]] = []
    simulation_duration: int = Field(default=60, ge=5, le=300)  # 5분~5시간
    agent_counts: Dict[str, int] = Field(
        default={
            "momentum_chaser": 1000,
            "smart_money": 50,
            "loss_aversion": 800
        }
    )


class PricePrediction(BaseModel):
    time_offset_min: int
    price: float
    confidence: float


class AgentActionSummary(BaseModel):
    buy_ratio: float
    sell_ratio: Optional[float] = 0.0
    hold_ratio: Optional[float] = 0.0
    avg_confidence: float


class RiskAssessment(BaseModel):
    volatility_index: float  # 0-1
    manipulation_risk: float  # 0-1


class SimulationResponse(BaseModel):
    simulation_id: str
    predicted_prices: List[PricePrediction]
    agent_actions: Dict[str, AgentActionSummary]
    risk_assessment: RiskAssessment


# Psychology Analysis Schemas
class PsychologyScores(BaseModel):
    fomo_index: int = Field(..., ge=0, le=100)
    loss_cut_delay_index: int = Field(..., ge=0, le=100)
    impulsive_trading_index: int = Field(..., ge=0, le=100)
    risk_management_score: int = Field(..., ge=0, le=100)


class FeedbackContent(BaseModel):
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    action_items: List[str]
    next_week_goal: str


class SimilarPastSituation(BaseModel):
    date: str
    situation: str
    outcome: str
    lesson_learned: str


class AgentFeedbackResponse(BaseModel):
    user_id: str
    analysis_period: str
    psychology_scores: PsychologyScores
    feedback: FeedbackContent
    historical_trend: Dict[str, Any]
    similar_past_situations: List[SimilarPastSituation]


# RAG Schemas
class HistoricalSearchRequest(BaseModel):
    query: str
    symbol: str
    search_period: str = "2020-01-01~2025-10-20"
    top_k: int = Field(default=5, ge=1, le=20)
    return_price_data: bool = True


class PriceImpact(BaseModel):
    before_price: float
    after_1d: Optional[float]
    after_3d: Optional[float]
    after_7d: Optional[float]
    after_30d: Optional[float]
    max_gain: float
    max_drawdown: float


class VolumeImpact(BaseModel):
    avg_volume_before: int
    peak_volume: int
    spike_ratio: float


class HistoricalEvent(BaseModel):
    rank: int
    similarity_score: float
    event: Dict[str, Any]
    price_impact: PriceImpact
    volume_impact: VolumeImpact
    context: str


class AggregateAnalysis(BaseModel):
    avg_price_impact_7d: float
    avg_price_impact_30d: float
    success_rate: float
    confidence_interval_7d: List[float]
    recommendation: str


class HistoricalSearchResponse(BaseModel):
    query: str
    results: List[HistoricalEvent]
    aggregate_analysis: AggregateAnalysis


# Game Schemas
class GameStateResponse(BaseModel):
    user_id: str
    current_level: int
    balance: int
    total_profit_loss: int
    portfolio: List[Dict[str, Any]]
    achievements: List[str]
