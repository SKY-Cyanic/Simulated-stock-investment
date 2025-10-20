"""
Game Service - 게임 로직 및 레벨 관리
"""
import os
import sys
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import redis.asyncio as redis
import httpx
from typing import List, Dict, Any
from datetime import datetime

# 공유 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from models import Base, User, UserProgress, GameLevel
from schemas import GameStateResponse

app = FastAPI(title="Mind Trader Game Service", version="1.0.0")

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

# Redis
redis_client = None

# 외부 서비스 URL
AI_ENGINE_URL = os.getenv("AI_ENGINE_URL", "http://localhost:8002")
MARKET_SERVICE_URL = os.getenv("MARKET_SERVICE_URL", "http://localhost:8004")


@app.on_event("startup")
async def startup_event():
    """서비스 시작"""
    global redis_client
    
    # 테이블 생성
    Base.metadata.create_all(bind=engine)
    
    # Redis 연결
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = await redis.from_url(redis_url, decode_responses=True)
    
    # 게임 레벨 초기화
    db = SessionLocal()
    init_game_levels(db)
    db.close()
    
    print("✅ Game Service started successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_game_levels(db: Session):
    """게임 레벨 초기화"""
    existing = db.query(GameLevel).first()
    if existing:
        return
    
    levels = [
        GameLevel(
            level=1,
            name="입문자의 첫 걸음",
            description="간단한 상승 패턴으로 성공 경험 쌓기",
            difficulty="easy",
            success_rate_target=0.8,
            volatility_factor=0.3,
            agent_complexity={"momentum_chaser": 0.8, "smart_money": 0.1, "loss_aversion": 0.1},
            unlock_points_required=0
        ),
        GameLevel(
            level=2,
            name="패턴 학습",
            description="상승/하락 패턴 구분하기",
            difficulty="easy",
            success_rate_target=0.75,
            volatility_factor=0.4,
            agent_complexity={"momentum_chaser": 0.7, "smart_money": 0.2, "loss_aversion": 0.1},
            unlock_points_required=5000
        ),
        GameLevel(
            level=3,
            name="변동성 체험",
            description="횡보장과 페이크 신호 대응하기",
            difficulty="medium",
            success_rate_target=0.6,
            volatility_factor=0.6,
            agent_complexity={"momentum_chaser": 0.6, "smart_money": 0.2, "loss_aversion": 0.2},
            unlock_points_required=12000
        ),
        GameLevel(
            level=4,
            name="심리 함정",
            description="FOMO와 손절 회피 극복하기",
            difficulty="medium",
            success_rate_target=0.55,
            volatility_factor=0.7,
            agent_complexity={"momentum_chaser": 0.5, "smart_money": 0.3, "loss_aversion": 0.2},
            unlock_points_required=20000
        ),
        GameLevel(
            level=5,
            name="현실 시장 시뮬레이션",
            description="실제 시장 수준의 복잡도 체험",
            difficulty="hard",
            success_rate_target=0.5,
            volatility_factor=1.0,
            agent_complexity={"momentum_chaser": 0.4, "smart_money": 0.3, "loss_aversion": 0.3},
            unlock_points_required=35000
        )
    ]
    
    for level in levels:
        db.add(level)
    
    db.commit()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/v1/game/levels")
async def get_game_levels(db: Session = Depends(get_db)):
    """게임 레벨 목록 조회"""
    levels = db.query(GameLevel).all()
    return [
        {
            "level": l.level,
            "name": l.name,
            "description": l.description,
            "difficulty": l.difficulty,
            "unlock_points_required": l.unlock_points_required
        }
        for l in levels
    ]


@app.get("/api/v1/game/state/{user_id}", response_model=GameStateResponse)
async def get_game_state(user_id: str, db: Session = Depends(get_db)):
    """사용자 게임 상태 조회"""
    progress = db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
    
    if not progress:
        raise HTTPException(status_code=404, detail="User progress not found")
    
    return GameStateResponse(
        user_id=user_id,
        current_level=progress.current_level,
        balance=progress.current_balance,
        total_profit_loss=progress.total_profit_loss,
        portfolio=[],  # 실제로는 포트폴리오 조회
        achievements=progress.achievements or []
    )


@app.post("/api/v1/game/start_scenario/{user_id}")
async def start_scenario(
    user_id: str,
    symbol: str = "SAMPLE_STOCK",
    db: Session = Depends(get_db)
):
    """시나리오 시작 - 특정 종목의 시뮬레이션 시작"""
    progress = db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
    
    if not progress:
        raise HTTPException(status_code=404, detail="User not found")
    
    # 현재 레벨 정보 가져오기
    level_info = db.query(GameLevel).filter(GameLevel.level == progress.current_level).first()
    
    if not level_info:
        raise HTTPException(status_code=404, detail="Level not found")
    
    # 시장 데이터 가져오기 (Market Service 호출)
    async with httpx.AsyncClient() as client:
        try:
            market_response = await client.get(
                f"{MARKET_SERVICE_URL}/api/v1/market/candles/{symbol}",
                params={"timeframe": "5m", "limit": 50},
                timeout=10.0
            )
            
            if market_response.status_code == 200:
                market_data = market_response.json()
            else:
                # 샘플 데이터 생성
                market_data = generate_sample_market_data(level_info)
        except:
            market_data = generate_sample_market_data(level_info)
    
    scenario = {
        "scenario_id": f"scenario_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "user_id": user_id,
        "symbol": symbol,
        "level": progress.current_level,
        "level_name": level_info.name,
        "difficulty": level_info.difficulty,
        "starting_balance": progress.current_balance,
        "starting_price": market_data.get("current_price", 10000),
        "market_data": market_data,
        "hints": generate_hints(level_info),
        "success_criteria": {
            "target_profit_rate": 0.05,  # 5% 수익
            "max_loss_rate": -0.03  # -3% 손실까지 허용
        }
    }
    
    # Redis에 시나리오 저장 (30분 TTL)
    await redis_client.setex(
        f"scenario:{scenario['scenario_id']}",
        1800,
        str(scenario)
    )
    
    return scenario


def generate_sample_market_data(level: GameLevel) -> Dict[str, Any]:
    """샘플 시장 데이터 생성"""
    import random
    
    base_price = 10000
    volatility = level.volatility_factor
    
    candles = []
    current_price = base_price
    
    for i in range(50):
        change = random.gauss(0, volatility * 0.01)
        
        # 레벨에 따라 트렌드 부여
        if level.level <= 2:
            change += 0.003  # 약한 상승 트렌드
        
        current_price *= (1 + change)
        
        candles.append({
            "timestamp": (datetime.now().timestamp() - (50 - i) * 300) * 1000,  # 5분 간격
            "open": current_price,
            "high": current_price * 1.005,
            "low": current_price * 0.995,
            "close": current_price,
            "volume": random.randint(100000, 500000)
        })
    
    return {
        "symbol": "SAMPLE_STOCK",
        "current_price": current_price,
        "candles": candles,
        "trend": "상승" if level.level <= 2 else "변동성"
    }


def generate_hints(level: GameLevel) -> List[str]:
    """레벨별 힌트 생성"""
    hints = {
        1: [
            "💡 상승 추세를 보이고 있습니다",
            "💡 적정한 타이밍에 매수해보세요",
            "💡 5% 수익이 나면 익절하는 것을 권장합니다"
        ],
        2: [
            "💡 이동평균선을 참고하세요",
            "💡 거래량이 증가하면 트렌드가 강해집니다"
        ],
        3: [
            "💡 횡보장에서는 단타보다 관망이 유리할 수 있습니다",
            "💡 페이크 브레이크아웃을 조심하세요"
        ],
        4: [
            "💡 급등 후 매수는 위험할 수 있습니다",
            "💡 손실이 -3%를 넘으면 손절을 고려하세요"
        ],
        5: [
            "💡 시장은 예측 불가능합니다. 리스크 관리가 핵심입니다"
        ]
    }
    
    return hints.get(level.level, ["💡 신중하게 투자하세요"])


@app.post("/api/v1/game/level_up/{user_id}")
async def level_up(user_id: str, db: Session = Depends(get_db)):
    """레벨 업"""
    progress = db.query(UserProgress).filter(UserProgress.user_id == user_id).first()
    
    if not progress:
        raise HTTPException(status_code=404, detail="User not found")
    
    next_level = progress.current_level + 1
    next_level_info = db.query(GameLevel).filter(GameLevel.level == next_level).first()
    
    if not next_level_info:
        return {"message": "최고 레벨에 도달했습니다!", "current_level": progress.current_level}
    
    # 포인트 확인
    if progress.total_learning_points < next_level_info.unlock_points_required:
        raise HTTPException(
            status_code=400,
            detail=f"레벨 업에 필요한 포인트: {next_level_info.unlock_points_required}, 현재: {progress.total_learning_points}"
        )
    
    progress.current_level = next_level
    db.commit()
    
    return {
        "message": f"레벨 {next_level}로 승급했습니다!",
        "level_name": next_level_info.name,
        "current_level": next_level
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
