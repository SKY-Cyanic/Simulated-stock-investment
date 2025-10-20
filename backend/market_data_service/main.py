"""
Market Data Service - 시장 데이터 제공
"""
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
import redis.asyncio as redis
from datetime import datetime, timedelta
import random
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from models import Base, MarketData
from schemas import MarketDataResponse, CandleData

app = FastAPI(title="Mind Trader Market Data Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres123@localhost:5432/mind_trader")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

redis_client = None


@app.on_event("startup")
async def startup_event():
    global redis_client
    Base.metadata.create_all(bind=engine)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = await redis.from_url(redis_url, decode_responses=True)
    
    # 샘플 데이터 생성
    db = SessionLocal()
    generate_sample_data(db)
    db.close()
    
    print("✅ Market Data Service started!")


def generate_sample_data(db: Session):
    """샘플 시장 데이터 생성"""
    existing = db.query(MarketData).first()
    if existing:
        return
    
    symbols = ["SAMSUNG", "LG", "HYUNDAI", "SAMPLE_STOCK"]
    
    for symbol in symbols:
        base_price = 10000
        now = datetime.now()
        
        for i in range(100):
            timestamp = now - timedelta(minutes=5 * (100 - i))
            change = random.gauss(0, 0.01)
            base_price *= (1 + change)
            
            candle = MarketData(
                symbol=symbol,
                timestamp=timestamp,
                timeframe="5m",
                open=base_price * 0.999,
                high=base_price * 1.003,
                low=base_price * 0.997,
                close=base_price,
                volume=random.randint(100000, 500000)
            )
            db.add(candle)
        
        db.commit()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/v1/market/candles/{symbol}", response_model=MarketDataResponse)
async def get_candles(symbol: str, timeframe: str = "5m", limit: int = 50):
    """캔들 데이터 조회"""
    db = SessionLocal()
    
    candles_db = db.query(MarketData).filter(
        MarketData.symbol == symbol,
        MarketData.timeframe == timeframe
    ).order_by(desc(MarketData.timestamp)).limit(limit).all()
    
    db.close()
    
    if not candles_db:
        # 샘플 데이터 생성
        candles = generate_realtime_sample(symbol, limit)
    else:
        candles = [
            CandleData(
                timestamp=c.timestamp,
                open=float(c.open),
                high=float(c.high),
                low=float(c.low),
                close=float(c.close),
                volume=int(c.volume)
            )
            for c in reversed(candles_db)
        ]
    
    return MarketDataResponse(
        symbol=symbol,
        timeframe=timeframe,
        candles=candles
    )


def generate_realtime_sample(symbol: str, count: int) -> List[CandleData]:
    """실시간 샘플 데이터 생성"""
    candles = []
    base_price = 10000
    now = datetime.now()
    
    for i in range(count):
        change = random.gauss(0, 0.01)
        base_price *= (1 + change)
        
        candles.append(CandleData(
            timestamp=now - timedelta(minutes=5 * (count - i)),
            open=base_price * 0.999,
            high=base_price * 1.003,
            low=base_price * 0.997,
            close=base_price,
            volume=random.randint(100000, 500000)
        ))
    
    return candles


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
