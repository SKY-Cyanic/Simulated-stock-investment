"""
공유 데이터베이스 모델
모든 마이크로서비스에서 사용하는 공통 모델 정의
"""
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, 
    BigInteger, Boolean, JSON, Index, ForeignKey, DECIMAL
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class User(Base):
    """사용자 테이블"""
    __tablename__ = 'users'
    
    id = Column(String(50), primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    level = Column(Integer, default=1)
    learning_points = Column(Integer, default=10000)  # 가상 게임 머니
    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime)
    
    __table_args__ = (
        Index('idx_username', 'username'),
        Index('idx_email', 'email'),
    )


class UserTrade(Base):
    """사용자 거래 이력"""
    __tablename__ = 'user_trades'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey('users.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    action = Column(String(10), nullable=False)  # buy/sell
    price = Column(DECIMAL(10, 2), nullable=False)
    quantity = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=func.now())
    emotion = Column(String(20))  # 확신, 욕심, 두려움, 냉정
    reason = Column(Text)
    price_change_before_trade = Column(Float)  # 직전 변화율
    news_sentiment = Column(Float)  # 당시 뉴스 감성
    unrealized_pnl = Column(Float)  # 미실현 손익 (매도 시)
    hold_duration_minutes = Column(Integer)  # 보유 기간 (분)
    
    __table_args__ = (
        Index('idx_user_time', 'user_id', 'timestamp'),
        Index('idx_symbol', 'symbol'),
    )


class AgentSimulation(Base):
    """AI 에이전트 시뮬레이션 결과"""
    __tablename__ = 'agent_simulations'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    simulation_time = Column(DateTime, nullable=False, default=func.now())
    agent_type = Column(String(50), nullable=False)  # momentum_chaser, smart_money, loss_aversion
    action_distribution = Column(JSON)  # {"buy": 0.7, "hold": 0.2, "sell": 0.1}
    predicted_price = Column(DECIMAL(10, 2))
    actual_price = Column(DECIMAL(10, 2))  # 나중에 업데이트
    accuracy_score = Column(Float)
    confidence = Column(Float)
    
    __table_args__ = (
        Index('idx_symbol_time', 'symbol', 'simulation_time'),
    )


class MarketData(Base):
    """시장 가격 데이터 (캔들)"""
    __tablename__ = 'market_data'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 15m, 1h, 1d
    open = Column(DECIMAL(10, 2), nullable=False)
    high = Column(DECIMAL(10, 2), nullable=False)
    low = Column(DECIMAL(10, 2), nullable=False)
    close = Column(DECIMAL(10, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)
    
    __table_args__ = (
        Index('idx_symbol_time', 'symbol', 'timestamp'),
        Index('idx_timeframe', 'timeframe'),
    )


class NewsEvent(Base):
    """뉴스 이벤트"""
    __tablename__ = 'news_events'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    event_date = Column(DateTime, nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text)
    category = Column(String(50))  # 신제품 발표, 실적 발표, 인수합병 등
    sentiment_score = Column(Float)  # -1 ~ 1
    impact_score = Column(Float)  # 0 ~ 1 (영향력 크기)
    price_impact_1d = Column(Float)
    price_impact_3d = Column(Float)
    price_impact_7d = Column(Float)
    price_impact_30d = Column(Float)
    embedding_id = Column(String(100))  # Vector DB 참조 ID
    
    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'event_date'),
        Index('idx_category', 'category'),
    )


class UserPsychologyProfile(Base):
    """사용자 심리 프로필"""
    __tablename__ = 'user_psychology_profiles'
    
    user_id = Column(String(50), ForeignKey('users.id'), primary_key=True)
    fomo_index = Column(Integer, default=0)  # 0-100
    loss_cut_delay_index = Column(Integer, default=0)  # 0-100
    impulsive_trading_index = Column(Integer, default=0)  # 0-100
    risk_management_score = Column(Integer, default=50)  # 0-100
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    trend_data = Column(JSON)  # 주간/월간 추이 데이터
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    avg_hold_duration_minutes = Column(Float, default=0.0)


class GameLevel(Base):
    """게임 레벨 정의"""
    __tablename__ = 'game_levels'
    
    level = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    difficulty = Column(String(20))  # easy, medium, hard
    success_rate_target = Column(Float)  # 목표 승률
    volatility_factor = Column(Float)  # 변동성 계수
    agent_complexity = Column(JSON)  # 각 Agent 활성화 비율
    unlock_points_required = Column(Integer)  # 언락 필요 포인트


class UserProgress(Base):
    """사용자 진행 상황"""
    __tablename__ = 'user_progress'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(String(50), ForeignKey('users.id'), nullable=False)
    current_level = Column(Integer, default=1)
    total_learning_points = Column(Integer, default=10000)
    current_balance = Column(Integer, default=10000)
    total_profit_loss = Column(Integer, default=0)
    achievements = Column(JSON)  # 달성한 업적 목록
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_user_progress', 'user_id'),
    )
