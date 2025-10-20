"""
User Service - 사용자 관리 및 심리 분석 서비스
"""
import os
import sys
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
import redis.asyncio as redis
import json
from typing import List, Optional
import numpy as np

# 공유 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from models import Base, User, UserTrade, UserPsychologyProfile, UserProgress
from schemas import (
    UserCreate, UserLogin, UserResponse, TradeCreate, TradeResponse,
    AgentFeedbackResponse, PsychologyScores, FeedbackContent,
    SimilarPastSituation
)

app = FastAPI(title="Mind Trader User Service", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터베이스 설정
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres123@localhost:5432/mind_trader")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 패스워드 해싱
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT 설정
SECRET_KEY = os.getenv("JWT_SECRET", "your-secret-key-change-this")
ALGORITHM = "HS256"
security = HTTPBearer()

# Redis 클라이언트
redis_client = None


@app.on_event("startup")
async def startup_event():
    """서비스 시작"""
    global redis_client
    
    # 테이블 생성
    Base.metadata.create_all(bind=engine)
    
    # Redis 연결
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = await redis.from_url(redis_url, decode_responses=True)
    
    print("✅ User Service started successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """서비스 종료"""
    if redis_client:
        await redis_client.close()


def get_db():
    """DB 세션 생성"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_access_token(data: dict, expires_delta: timedelta = timedelta(days=7)):
    """JWT 토큰 생성"""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """현재 사용자 인증"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}


@app.post("/api/v1/users/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """사용자 등록"""
    # 중복 확인
    existing_user = db.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    
    # 사용자 생성
    user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    hashed_password = pwd_context.hash(user_data.password)
    
    new_user = User(
        id=user_id,
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password,
        level=1,
        learning_points=10000
    )
    
    db.add(new_user)
    
    # 심리 프로필 초기화
    psychology_profile = UserPsychologyProfile(
        user_id=user_id,
        fomo_index=0,
        loss_cut_delay_index=0,
        impulsive_trading_index=0,
        risk_management_score=50,
        trend_data={"weekly": [], "monthly": []}
    )
    db.add(psychology_profile)
    
    # 진행 상황 초기화
    progress = UserProgress(
        user_id=user_id,
        current_level=1,
        total_learning_points=10000,
        current_balance=10000,
        achievements=[]
    )
    db.add(progress)
    
    db.commit()
    db.refresh(new_user)
    
    return new_user


@app.post("/api/v1/users/login")
async def login_user(login_data: UserLogin, db: Session = Depends(get_db)):
    """사용자 로그인"""
    user = db.query(User).filter(User.username == login_data.username).first()
    
    if not user or not pwd_context.verify(login_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # 마지막 로그인 시간 업데이트
    user.last_login = datetime.now()
    db.commit()
    
    # JWT 토큰 생성
    access_token = create_access_token(data={"sub": user.id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.from_orm(user)
    }


@app.get("/api/v1/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """현재 사용자 정보 조회"""
    return current_user


@app.post("/api/v1/users/trades", response_model=TradeResponse)
async def create_trade(
    trade_data: TradeCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """거래 기록 생성"""
    # 거래 생성
    trade = UserTrade(
        user_id=current_user.id,
        symbol=trade_data.symbol,
        action=trade_data.action.value,
        price=float(trade_data.price),
        quantity=trade_data.quantity,
        emotion=trade_data.emotion.value if trade_data.emotion else None,
        reason=trade_data.reason,
        timestamp=datetime.now()
    )
    
    db.add(trade)
    
    # 잔액 업데이트
    progress = db.query(UserProgress).filter(UserProgress.user_id == current_user.id).first()
    if progress:
        if trade_data.action.value == "buy":
            cost = trade_data.price * trade_data.quantity
            progress.current_balance -= cost
        elif trade_data.action.value == "sell":
            revenue = trade_data.price * trade_data.quantity
            progress.current_balance += revenue
        
        db.commit()
    
    db.commit()
    db.refresh(trade)
    
    # 백그라운드에서 심리 프로필 업데이트
    await update_psychology_profile(current_user.id, db)
    
    return trade


@app.get("/api/v1/users/trades", response_model=List[TradeResponse])
async def get_user_trades(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """사용자 거래 이력 조회"""
    trades = db.query(UserTrade).filter(
        UserTrade.user_id == current_user.id
    ).order_by(desc(UserTrade.timestamp)).limit(limit).all()
    
    return trades


@app.get("/api/v1/ai/agent_feedback/{user_id}", response_model=AgentFeedbackResponse)
async def get_agent_feedback(
    user_id: str,
    trades_count: int = 10,
    db: Session = Depends(get_db)
):
    """사용자 심리 분석 및 피드백"""
    # 캐시 확인
    cache_key = f"feedback:{user_id}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 최근 거래 조회
    trades = db.query(UserTrade).filter(
        UserTrade.user_id == user_id
    ).order_by(desc(UserTrade.timestamp)).limit(trades_count).all()
    
    if not trades:
        raise HTTPException(status_code=404, detail="No trades found")
    
    # 심리 지표 계산
    psychology_scores = calculate_psychology_scores(trades)
    
    # 심리 프로필 조회
    profile = db.query(UserPsychologyProfile).filter(
        UserPsychologyProfile.user_id == user_id
    ).first()
    
    # 피드백 생성
    feedback = generate_feedback(trades, psychology_scores)
    
    # 과거 추이
    historical_trend = {
        "fomo_index_4weeks": profile.trend_data.get("fomo_4weeks", [0, 0, 0, psychology_scores.fomo_index]) if profile else [0],
        "improvement_rate": calculate_improvement_rate(profile) if profile else 0.0
    }
    
    # 유사 과거 상황 (간단한 예시)
    similar_situations = [
        SimilarPastSituation(
            date="2025-09-15",
            situation="급등 종목 고점 매수",
            outcome="손실 -8%",
            lesson_learned="단기 급등 후에는 조정 대기 전략이 유효"
        )
    ]
    
    response = AgentFeedbackResponse(
        user_id=user_id,
        analysis_period=f"{trades[-1].timestamp.date()} ~ {trades[0].timestamp.date()}",
        psychology_scores=psychology_scores,
        feedback=feedback,
        historical_trend=historical_trend,
        similar_past_situations=similar_situations
    )
    
    # 캐시 저장 (1시간)
    await redis_client.setex(cache_key, 3600, json.dumps(response.dict(), default=str))
    
    return response


def calculate_psychology_scores(trades: List[UserTrade]) -> PsychologyScores:
    """심리 지표 계산"""
    fomo_signals = []
    delay_signals = []
    impulsive_signals = []
    
    for i, trade in enumerate(trades):
        # FOMO 지수
        if trade.action == "buy" and trade.price_change_before_trade:
            if trade.price_change_before_trade > 2.0:
                fomo_score = min(trade.price_change_before_trade * 10, 100)
                fomo_signals.append(fomo_score)
        
        # 손절 지연
        if trade.action == "sell" and trade.unrealized_pnl and trade.unrealized_pnl < 0:
            loss_percent = abs(trade.unrealized_pnl)
            hold_hours = trade.hold_duration_minutes / 60 if trade.hold_duration_minutes else 0
            expected_hours = loss_percent * 1.0
            if hold_hours > expected_hours:
                delay_score = min((hold_hours / (expected_hours + 1e-8)) * 50, 100)
                delay_signals.append(delay_score)
        
        # 충동 매매
        if i < len(trades) - 1:
            time_diff = abs((trades[i].timestamp - trades[i+1].timestamp).total_seconds() / 60)
            if time_diff < 10:
                impulsive_signals.append(1)
    
    fomo_index = int(np.mean(fomo_signals)) if fomo_signals else 0
    loss_cut_delay = int(np.mean(delay_signals)) if delay_signals else 0
    impulsive = min(len(impulsive_signals) * 20, 100)
    
    return PsychologyScores(
        fomo_index=fomo_index,
        loss_cut_delay_index=loss_cut_delay,
        impulsive_trading_index=impulsive,
        risk_management_score=max(100 - (fomo_index + loss_cut_delay + impulsive) // 3, 0)
    )


def generate_feedback(trades: List[UserTrade], scores: PsychologyScores) -> FeedbackContent:
    """피드백 생성"""
    summary = f"이번 주 분석 결과, "
    
    if scores.fomo_index > 60:
        summary += "FOMO 지수가 높게 나타났습니다. 급등 종목 매수 전 10분 대기 전략을 권장합니다."
    elif scores.loss_cut_delay_index > 60:
        summary += "손절 타이밍이 지연되는 패턴이 보입니다. -3% 손절선 자동 설정을 활용해보세요."
    else:
        summary += "전반적으로 안정적인 투자 패턴을 보이고 있습니다!"
    
    strengths = []
    weaknesses = []
    action_items = []
    
    # 강점 찾기
    for trade in trades[:3]:
        if trade.action == "sell" and trade.unrealized_pnl and trade.unrealized_pnl > 0.05:
            strengths.append(f"{trade.timestamp.strftime('%m월 %d일')} {trade.symbol} 익절 타이밍이 우수했습니다 (+{trade.unrealized_pnl*100:.1f}%)")
    
    # 약점 찾기
    if scores.fomo_index > 50:
        weaknesses.append("급등 후 고점 매수 패턴이 반복되고 있습니다")
        action_items.append("매수 전 '10분 냉각' 타이머 활성화")
    
    if scores.loss_cut_delay_index > 50:
        weaknesses.append("손실 종목을 너무 오래 보유하는 경향이 있습니다")
        action_items.append("손절선 -3% 자동 설정 활용")
    
    if not action_items:
        action_items = ["현재 패턴 유지하기", "주간 매매 일지 작성하기"]
    
    next_goal = "FOMO 지수 60 이하로 낮추기" if scores.fomo_index > 60 else "안정적인 수익률 유지하기"
    
    return FeedbackContent(
        summary=summary,
        strengths=strengths if strengths else ["꾸준히 학습하고 계십니다!"],
        weaknesses=weaknesses if weaknesses else [],
        action_items=action_items,
        next_week_goal=next_goal
    )


async def update_psychology_profile(user_id: str, db: Session):
    """심리 프로필 업데이트"""
    trades = db.query(UserTrade).filter(
        UserTrade.user_id == user_id
    ).order_by(desc(UserTrade.timestamp)).limit(10).all()
    
    if not trades:
        return
    
    scores = calculate_psychology_scores(trades)
    
    profile = db.query(UserPsychologyProfile).filter(
        UserPsychologyProfile.user_id == user_id
    ).first()
    
    if profile:
        profile.fomo_index = scores.fomo_index
        profile.loss_cut_delay_index = scores.loss_cut_delay_index
        profile.impulsive_trading_index = scores.impulsive_trading_index
        profile.risk_management_score = scores.risk_management_score
        profile.total_trades = len(trades)
        profile.last_updated = datetime.now()
        
        db.commit()


def calculate_improvement_rate(profile: UserPsychologyProfile) -> float:
    """개선율 계산"""
    if not profile or not profile.trend_data:
        return 0.0
    
    fomo_trend = profile.trend_data.get("fomo_4weeks", [])
    if len(fomo_trend) >= 2:
        improvement = (fomo_trend[0] - fomo_trend[-1]) / (fomo_trend[0] + 1e-8) * 100
        return round(improvement, 1)
    
    return 0.0


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
