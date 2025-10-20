"""
AI Engine Service - PyTorch 모델 추론 서비스
"""
import os
import sys
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from datetime import datetime, timedelta
import redis.asyncio as redis
import json
from typing import Dict, List
import uuid

# 공유 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from schemas import (
    SimulationRequest, SimulationResponse, PricePrediction,
    AgentActionSummary, RiskAssessment
)

from models.agents import (
    MomentumChaserAgent, SmartMoneyAgent, LossAversionAgent,
    MultiAgentSimulator
)

app = FastAPI(title="Mind Trader AI Engine", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
redis_client = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simulator = None


@app.on_event("startup")
async def startup_event():
    """서비스 시작 시 모델 로드 및 Redis 연결"""
    global redis_client, simulator
    
    # Redis 연결
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = await redis.from_url(redis_url, decode_responses=True)
    
    # AI 모델 초기화
    print(f"Using device: {device}")
    
    momentum_model = MomentumChaserAgent(input_size=10, hidden_size=64, num_layers=2)
    smart_money_model = SmartMoneyAgent(input_size=15, hidden_size=128, num_layers=3)
    loss_aversion_model = LossAversionAgent(d_model=32, nhead=4, num_layers=2)
    
    # 모델을 평가 모드로 설정
    momentum_model.eval()
    smart_money_model.eval()
    loss_aversion_model.eval()
    
    # Multi-Agent 시뮬레이터 초기화
    simulator = MultiAgentSimulator(
        momentum_model=momentum_model,
        smart_money_model=smart_money_model,
        loss_aversion_model=loss_aversion_model,
        device=device
    )
    
    print("✅ AI Engine started successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """서비스 종료 시 정리"""
    global redis_client
    if redis_client:
        await redis_client.close()


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "device": str(device)}


@app.post("/api/v1/ai/simulate_market_impact", response_model=SimulationResponse)
async def simulate_market_impact(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
):
    """
    AI Agent 기반 시장 영향 시뮬레이션
    """
    try:
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # 캔들 데이터를 NumPy 배열로 변환
        price_history = np.array([
            [c.close, c.high, c.low, c.open, c.volume / 1000000]
            for c in request.recent_candles[-20:]  # 최근 20개만 사용
        ])
        
        # 정규화 (변화율로 변환)
        price_normalized = np.zeros((len(price_history), 10))
        for i in range(1, len(price_history)):
            price_normalized[i, 0] = (price_history[i, 0] - price_history[i-1, 0]) / price_history[i-1, 0]  # close 변화율
            price_normalized[i, 1] = (price_history[i, 1] - price_history[i, 0]) / price_history[i, 0]  # high-close
            price_normalized[i, 2] = (price_history[i, 2] - price_history[i, 0]) / price_history[i, 0]  # low-close
            price_normalized[i, 3:10] = 0  # 나머지 특징은 0으로 (실제로는 기술 지표 추가 가능)
        
        volume_history = price_history[:, 4:5]  # 거래량
        
        # 뉴스 감성 점수 평균
        news_sentiment = 0.0
        if request.news_events:
            sentiments = [e.get('sentiment', 0) for e in request.news_events]
            news_sentiment = np.mean(sentiments) if sentiments else 0.0
        
        # 공포-탐욕 지수 (간단한 계산: 최근 변화율 기반)
        recent_changes = price_normalized[-5:, 0]
        avg_change = np.mean(recent_changes)
        fear_greed_index = 50 + (avg_change * 1000)  # 0-100 사이로 스케일링
        fear_greed_index = np.clip(fear_greed_index, 0, 100)
        
        # 시뮬레이션 실행
        predicted_prices = []
        current_price = request.current_price
        
        # 5분 간격으로 시뮬레이션
        time_steps = request.simulation_duration // 5
        
        for i in range(time_steps):
            result = simulator.simulate_market_step(
                price_history=price_normalized,
                volume_history=volume_history,
                news_sentiment=news_sentiment,
                fear_greed_index=fear_greed_index,
                agent_counts=request.agent_counts,
                current_price=current_price
            )
            
            # 예측 가격 업데이트
            current_price = result['predicted_price']
            confidence = result['agent_actions']['momentum_chaser']['confidence']
            
            predicted_prices.append(
                PricePrediction(
                    time_offset_min=(i + 1) * 5,
                    price=round(current_price, 2),
                    confidence=round(confidence, 2)
                )
            )
            
            # 다음 스텝을 위해 가격 이력 업데이트
            new_price_row = np.array([[
                current_price,
                current_price * 1.005,
                current_price * 0.995,
                price_history[-1, 0],
                volume_history[-1, 0]
            ]])
            price_history = np.vstack([price_history[1:], new_price_row])
            
            # 정규화 재계산
            last_change = (current_price - price_normalized[-1, 0]) / (price_normalized[-1, 0] + 1e-8)
            new_normalized = np.zeros((1, 10))
            new_normalized[0, 0] = last_change
            price_normalized = np.vstack([price_normalized[1:], new_normalized])
        
        # Agent 행동 요약 (마지막 스텝 기준)
        final_actions = result['agent_actions']
        
        agent_actions = {
            "momentum_chaser": AgentActionSummary(
                buy_ratio=round(final_actions['momentum_chaser']['buy_ratio'], 3),
                sell_ratio=round(final_actions['momentum_chaser']['sell_ratio'], 3),
                hold_ratio=round(final_actions['momentum_chaser']['hold_ratio'], 3),
                avg_confidence=round(final_actions['momentum_chaser']['confidence'], 3)
            ),
            "smart_money": AgentActionSummary(
                buy_ratio=1.0 if final_actions['smart_money']['action'] == 'buy' else 0.0,
                avg_confidence=round(final_actions['smart_money']['position_size'], 3)
            ),
            "loss_aversion": AgentActionSummary(
                buy_ratio=round(final_actions['loss_aversion']['avg_down_prob'], 3),
                sell_ratio=round(final_actions['loss_aversion']['sell_prob'], 3),
                hold_ratio=round(final_actions['loss_aversion']['hold_prob'], 3),
                avg_confidence=0.6
            )
        }
        
        # 리스크 평가
        price_volatility = np.std([p.price for p in predicted_prices]) / request.current_price
        manipulation_risk = final_actions['smart_money']['position_size'] * 0.5
        
        risk_assessment = RiskAssessment(
            volatility_index=round(min(price_volatility * 10, 1.0), 2),
            manipulation_risk=round(min(manipulation_risk, 1.0), 2)
        )
        
        # 결과를 Redis에 캐싱 (TTL 5분)
        cache_key = f"simulation:{simulation_id}"
        await redis_client.setex(
            cache_key,
            300,  # 5분
            json.dumps({
                "simulation_id": simulation_id,
                "symbol": request.symbol,
                "timestamp": datetime.now().isoformat(),
                "predicted_prices": [p.dict() for p in predicted_prices],
                "agent_actions": {k: v.dict() for k, v in agent_actions.items()},
                "risk_assessment": risk_assessment.dict()
            })
        )
        
        return SimulationResponse(
            simulation_id=simulation_id,
            predicted_prices=predicted_prices,
            agent_actions=agent_actions,
            risk_assessment=risk_assessment
        )
        
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@app.get("/api/v1/ai/simulation/{simulation_id}")
async def get_simulation_result(simulation_id: str):
    """시뮬레이션 결과 조회"""
    cache_key = f"simulation:{simulation_id}"
    result = await redis_client.get(cache_key)
    
    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found or expired")
    
    return json.loads(result)


@app.get("/api/v1/ai/models/info")
async def get_models_info():
    """모델 정보 조회"""
    return {
        "models": {
            "momentum_chaser": {
                "type": "LSTM + Attention",
                "parameters": sum(p.numel() for p in simulator.momentum_model.parameters()),
                "description": "추종 매매 개미 모델"
            },
            "smart_money": {
                "type": "GRU + Value Estimator",
                "parameters": sum(p.numel() for p in simulator.smart_money_model.parameters()),
                "description": "저가 매집 세력 모델"
            },
            "loss_aversion": {
                "type": "Transformer Encoder",
                "parameters": sum(p.numel() for p in simulator.loss_aversion_model.parameters()),
                "description": "손절 회피 개미 모델"
            }
        },
        "device": str(device),
        "total_parameters": sum(
            p.numel() for model in [
                simulator.momentum_model,
                simulator.smart_money_model,
                simulator.loss_aversion_model
            ] for p in model.parameters()
        )
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
