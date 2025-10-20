"""
RAG Service - 뉴스 검색 및 역사적 영향 분석
"""
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
from datetime import datetime
from typing import List, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from models import Base, NewsEvent
from schemas import (
    HistoricalSearchRequest, HistoricalSearchResponse,
    HistoricalEvent, PriceImpact, VolumeImpact, AggregateAnalysis
)

app = FastAPI(title="Mind Trader RAG Service", version="1.0.0")

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


@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)
    
    # 샘플 뉴스 데이터 생성
    db = SessionLocal()
    init_sample_news(db)
    db.close()
    
    print("✅ RAG Service started!")


def init_sample_news(db):
    """샘플 뉴스 데이터 초기화"""
    existing = db.query(NewsEvent).first()
    if existing:
        return
    
    sample_news = [
        {
            "symbol": "SAMSUNG",
            "event_date": datetime(2023, 8, 15),
            "title": "삼성전자, 갤럭시 Z 플립5 공개",
            "category": "신제품 발표",
            "sentiment_score": 0.75,
            "price_impact_1d": 2.1,
            "price_impact_7d": 5.0,
            "price_impact_30d": -1.5
        },
        {
            "symbol": "SAMSUNG",
            "event_date": datetime(2023, 6, 10),
            "title": "삼성, 5나노 공정 양산",
            "category": "기술 혁신",
            "sentiment_score": 0.68,
            "price_impact_1d": 1.8,
            "price_impact_7d": 5.2,
            "price_impact_30d": 2.1
        },
        {
            "symbol": "SAMSUNG",
            "event_date": datetime(2024, 2, 15),
            "title": "삼성, GAA 기술 발표",
            "category": "기술 혁신",
            "sentiment_score": 0.82,
            "price_impact_1d": 3.2,
            "price_impact_7d": 6.1,
            "price_impact_30d": 4.3
        }
    ]
    
    for news_data in sample_news:
        news = NewsEvent(**news_data)
        db.add(news)
    
    db.commit()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/v1/rag/search_historical_impact", response_model=HistoricalSearchResponse)
async def search_historical_impact(request: HistoricalSearchRequest):
    """과거 유사 이벤트 검색 및 영향 분석"""
    db = SessionLocal()
    
    # 간단한 키워드 매칭 (실제로는 임베딩 유사도 검색)
    query_keywords = request.query.lower().split()
    
    news_events = db.query(NewsEvent).filter(
        NewsEvent.symbol == request.symbol
    ).all()
    
    # 유사도 계산 (간단한 키워드 매칭)
    scored_events = []
    for event in news_events:
        title_lower = event.title.lower()
        score = sum(1 for keyword in query_keywords if keyword in title_lower) / len(query_keywords)
        if score > 0:
            scored_events.append((score, event))
    
    # 상위 K개 선택
    scored_events.sort(key=lambda x: x[0], reverse=True)
    top_events = scored_events[:request.top_k]
    
    results = []
    for rank, (score, event) in enumerate(top_events, 1):
        results.append(
            HistoricalEvent(
                rank=rank,
                similarity_score=round(score, 2),
                event={
                    "date": event.event_date.strftime("%Y-%m-%d"),
                    "title": event.title,
                    "category": event.category or "기타",
                    "source": "샘플 데이터"
                },
                price_impact=PriceImpact(
                    before_price=10000.0,
                    after_1d=10000 * (1 + event.price_impact_1d / 100) if event.price_impact_1d else None,
                    after_7d=10000 * (1 + event.price_impact_7d / 100) if event.price_impact_7d else None,
                    after_30d=10000 * (1 + event.price_impact_30d / 100) if event.price_impact_30d else None,
                    max_gain=max(event.price_impact_1d or 0, event.price_impact_7d or 0),
                    max_drawdown=min(0, event.price_impact_30d or 0)
                ),
                volume_impact=VolumeImpact(
                    avg_volume_before=12000000,
                    peak_volume=28000000,
                    spike_ratio=2.33
                ),
                context="신제품 발표로 단기 모멘텀 형성, 이후 조정"
            )
        )
    
    # 집계 분석
    if results:
        avg_7d = np.mean([e.price_impact.after_7d / 10000 - 1 for e in results if e.price_impact.after_7d]) * 100
        avg_30d = np.mean([e.price_impact.after_30d / 10000 - 1 for e in results if e.price_impact.after_30d]) * 100
        success_rate = sum(1 for e in results if e.price_impact.after_7d and e.price_impact.after_7d > 10000) / len(results)
        
        aggregate = AggregateAnalysis(
            avg_price_impact_7d=round(avg_7d, 2),
            avg_price_impact_30d=round(avg_30d, 2),
            success_rate=round(success_rate, 2),
            confidence_interval_7d=[round(avg_7d - 2, 2), round(avg_7d + 2, 2)],
            recommendation="단기 상승 가능성 높으나, 중기적으로는 변동성 존재. 7일 이내 익절 전략 권장."
        )
    else:
        aggregate = AggregateAnalysis(
            avg_price_impact_7d=0.0,
            avg_price_impact_30d=0.0,
            success_rate=0.0,
            confidence_interval_7d=[0.0, 0.0],
            recommendation="유사 사례가 부족합니다."
        )
    
    db.close()
    
    return HistoricalSearchResponse(
        query=request.query,
        results=results,
        aggregate_analysis=aggregate
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
