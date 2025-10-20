# 프로젝트 마인드 트레이더(Mind Trader) 최종 설계 마스터플랜

> **심리 시뮬레이션 기반 가상 주식 투자 웹 게임**  
> Chief Strategy Officer's Design Document  
> Date: 2025-10-20

---

## 📋 목차

1. [게임의 비전 및 교육 목표](#1-게임의-비전-및-교육-목표-the-vision-fear-to-confidence)
2. [PyTorch 기반 심리 시뮬레이션 엔진 설계](#2-pytorch-기반-심리-시뮬레이션-엔진-설계-aiml-core)
3. [컨텍스트 엔지니어링 킬러 콘텐츠](#3-컨텍스트-엔지니어링-킬러-콘텐츠-rag--agent-features)
4. [기술 구현 및 데이터 관리 지침](#4-기술-구현-및-데이터-관리-지침-technical-architecture)
5. [최종 마스터플랜 통합 요약](#5-최종-마스터플랜-통합-요약)

---

## 1. 게임의 비전 및 교육 목표 (The Vision: Fear to Confidence)

### 1.1 핵심 미션 선언문

**"주식 투자에 대한 공포를 자신감으로, 감정적 투자를 체계적 전략으로 전환시키는 안전한 심리 훈련장"**

### 1.2 타겟 유저의 공포 극복을 위한 3가지 핵심 설계 방안

#### 방안 1: 단계별 성공 경험 축적 시스템 (Progressive Success Framework)

**설계 원리:**
- **레벨 1-3 (안전지대)**: 승률 80% 이상의 간단한 시나리오
  - 명확한 상승/하락 패턴만 제공
  - 손실 발생 시 즉시 "교육적 힌트" 제공
  - 가상 멘토가 매 거래 전 조언 제공
  
- **레벨 4-7 (도전지대)**: 승률 60%로 하락, 변동성 증가
  - 횡보 패턴, 페이크 신호 등장
  - 심리적 함정(FOMO, 손절 회피) 시뮬레이션 도입
  
- **레벨 8+ (현실지대)**: 실제 시장 수준의 불확실성
  - AI 세력 모델의 복잡한 상호작용
  - 예측 불가능한 이벤트 발생

**공포 극복 메커니즘:**
- 작은 성공 → 도파민 보상 → 자신감 구축
- 실패 시에도 "무엇을 배웠는지" 명확히 피드백
- 손실을 "학습 포인트"로 환산하여 긍정적 프레이밍

#### 방안 2: 안전망 시뮬레이션 (Safety Net Simulation)

**설계 원리:**
- **타임 트래블 기능**: 과거 10회 거래로 되돌아가 다시 시도 가능
  - 단, 되돌릴 때마다 "학습 코인" 소모 (무한 반복 방지)
  - 각 선택지의 결과를 A/B로 비교할 수 있는 "평행 우주" UI
  
- **가상 손실 시각화**: 실제 돈이 아님을 지속적으로 상기
  - 게임 머니 단위를 "학습 포인트(LP)"로 표시
  - "실제 투자 시 이만큼의 손실" vs "게임에서의 안전한 학습" 대비 표시

**공포 극복 메커니즘:**
- "실패해도 괜찮다"는 심리적 안전망 제공
- 실험적 투자 전략 시도 유도
- 위험 감수에 대한 긍정적 경험 축적

#### 방안 3: 감정 인지 훈련 시스템 (Emotional Awareness Training)

**설계 원리:**
- **실시간 감정 체크인**: 매매 결정 전/후 감정 상태 기록
  - "지금 어떤 감정으로 이 결정을 내리시나요?" (두려움/욕심/냉정/확신)
  - 감정 데이터와 투자 성과의 상관관계 시각화
  
- **심리 패턴 대시보드**: 
  - "욕심으로 산 주식의 평균 수익률: -12%"
  - "두려움으로 판 주식의 기회 손실: +18%"
  - 개인별 감정-성과 히트맵 제공

**공포 극복 메커니즘:**
- 감정을 인지하고 객관화하는 메타인지 능력 향상
- "공포는 나쁜 것이 아니라 관리할 대상"이라는 인식 전환
- 데이터 기반 의사결정 습관 형성

### 1.3 차별점: 실시간 데이터 + 경쟁의 교육적 조화

#### 차별화 전략 1: 협력적 경쟁 (Cooperative Competition)

**기존 문제:** 순위 경쟁 → 투기적 행동 조장 → 교육 목표 훼손

**해결 방안:**
- **팀 리그 시스템**: 4-6명이 한 팀을 이루어 "평균 학습 성장률" 경쟁
  - 개인 수익률이 아닌 "심리 점수 개선도" 측정
  - 팀원끼리 전략 공유 시 보너스 점수
  
- **교육 목표 연동 리더보드**:
  - "손절 타이밍 정확도" 순위
  - "감정 컨트롤 안정성" 순위
  - "리스크 관리 우수성" 순위

#### 차별화 전략 2: 실시간 데이터의 교육적 재해석

**기존 문제:** 실시간 데이터 → 과도한 긴장감 → 충동 매매

**해결 방안:**
- **지연 모드 옵션**: 초보자는 5분/15분 지연 데이터로 시작
  - "실시간성"보다 "패턴 학습"을 우선시
  - 레벨 업 시 점차 실시간으로 전환
  
- **뉴스-가격 연관성 하이라이트**:
  - 실시간 뉴스 발생 시 "과거 유사 사례"를 RAG로 검색
  - "2023년 유사한 뉴스 발생 시 주가는 이렇게 움직였습니다" 교육 팝업
  
- **쿨다운 시스템**: 
  - 10분 이내 3회 이상 매매 시 "감정 트레이딩 경고" 및 30초 대기 시간
  - 충동적 의사결정 방지

#### 차별화 전략 3: 경쟁을 학습 동기로 전환

**설계:**
- **주간 챌린지**: "이번 주 목표: 손절 3회 성공하기"
  - 수익이 아닌 "올바른 행동" 목표
  - 달성 시 새로운 AI 분석 도구 언락
  
- **멘토-멘티 매칭**:
  - 상위 레벨 플레이어가 하위 레벨 플레이어 코칭
  - 멘토는 멘티 성장에 따라 보상 획득

---

## 2. PyTorch 기반 심리 시뮬레이션 엔진 설계 (AI/ML Core)

### 2.1 가상 시장 참여자 Agent 모델 (3가지 핵심 타입)

#### Agent Type 1: 추종 매매 개미 (Momentum Chaser)

**심리적 특성:**
- 강한 상승세에 FOMO 발동 → 고점 매수
- 하락세에 공포 → 저점 매도
- 뉴스/소셜 미디어 감성에 민감

**PyTorch 모델 구조:**
```python
class MomentumChaserAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=2)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.fc = nn.Linear(64, 3)  # Buy/Hold/Sell
        
    def forward(self, price_history, sentiment_score, volume_trend):
        # Input: 최근 20봉 가격, 뉴스 감성 점수, 거래량 추세
        lstm_out, _ = self.lstm(price_history)
        
        # Attention: 최근 5봉에 가중치 집중 (단기 추세 추종)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Sentiment 반영: 긍정 뉴스 + 상승 → 강한 매수 확률
        combined = attn_out * (1 + sentiment_score * 0.5)
        
        action_prob = self.fc(combined)
        return F.softmax(action_prob, dim=-1)
```

**가격 영향 메커니즘:**
- 매수/매도 압력 = Agent 수 × 평균 거래량 × 확신도
- 확신도 = sigmoid(가격 변화율 × 2) → 변화율이 클수록 강한 확신
- 시장 가격 변동: `ΔP = (총 매수량 - 총 매도량) / 유동성 계수`

**훈련 목표:**
- 실제 개미 투자자 거래 데이터로 학습 (손실 함수: 실제 거래 패턴과의 KL Divergence)
- 보상: 실제 개미 투자자의 평균 수익률(-15~-20%)과 유사하게 행동 시 +1

#### Agent Type 2: 저가 매집 세력 (Smart Money Accumulator)

**심리적 특성:**
- 공포 시점에 분산 매수 (contrarian)
- 장기 목표 가격 설정 후 체계적 접근
- 거래량 분석을 통한 타이밍 포착

**PyTorch 모델 구조:**
```python
class SmartMoneyAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=15, hidden_size=128, num_layers=3)
        self.value_estimator = nn.Linear(128, 1)  # 내재 가치 추정
        self.position_manager = nn.Linear(128, 1)  # 포지션 크기 결정
        
    def forward(self, price_history, volume_profile, fear_greed_index):
        gru_out, _ = self.gru(torch.cat([price_history, volume_profile], dim=-1))
        
        # 내재 가치 추정 (장기 이동평균 기반)
        intrinsic_value = self.value_estimator(gru_out)
        current_price = price_history[-1]
        
        # 저평가 = 매수 신호
        undervaluation = (intrinsic_value - current_price) / current_price
        
        # 공포 지수 높을수록 매수 의지 증가 (역발상)
        contrarian_signal = fear_greed_index < 30  # Fear zone
        
        if undervaluation > 0.1 and contrarian_signal:
            position_size = self.position_manager(gru_out) * undervaluation
            return {'action': 'buy', 'size': position_size}
        else:
            return {'action': 'hold', 'size': 0}
```

**가격 영향 메커니즘:**
- 분산 매수: 하루 거래량의 5-10%씩 나눠서 매수 → 급격한 가격 상승 방지
- 지지선 형성: 특정 가격대에 대량 매수 호가 → 하락 저지
- 시장 안정화 효과: `지지선 강도 = 누적 매수량 × 평균 보유 기간`

**훈련 목표:**
- 기관 투자자/외국인 매매 패턴 학습
- 보상: 장기 수익률 극대화 (6개월 홀딩 후 +20% 이상)

#### Agent Type 3: 손절 회피 개미 (Loss Aversion Trader)

**심리적 특성:**
- 수익 나면 빨리 실현 (5% 익절)
- 손실 나면 계속 보유 (손절 거부)
- 물타기 전략 선호

**PyTorch 모델 구조:**
```python
class LossAversionAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=4), 
            num_layers=2
        )
        self.loss_aversion_coef = nn.Parameter(torch.tensor(2.5))  # 손실 2.5배 민감
        
    def forward(self, portfolio_state, unrealized_pnl):
        # 미실현 손익에 따른 의사결정
        if unrealized_pnl > 0.05:  # 5% 이익
            sell_prob = sigmoid(unrealized_pnl * 10)  # 빠른 익절
        elif unrealized_pnl < -0.03:  # 3% 손실
            sell_prob = sigmoid(unrealized_pnl / self.loss_aversion_coef)  # 손절 회피
            # 추가 매수 (물타기) 확률
            avg_down_prob = sigmoid(-unrealized_pnl * 5)
            return {'sell_prob': sell_prob, 'avg_down_prob': avg_down_prob}
        else:
            return {'sell_prob': 0.1, 'avg_down_prob': 0}
```

**가격 영향 메커니즘:**
- 상승장: 조기 익절 → 상승 동력 약화
- 하락장: 손절 지연 → 하락 지속, 일시적 반등 시 물량 출회 → 재하락
- 물타기 효과: 일시적 매수세 유입 → 단기 반등 → 장기적 하락 압력 증가

**훈련 목표:**
- Prospect Theory 기반 손실 회피 편향 구현
- 실제 개인 투자자의 보유 기간 분포 학습 (손실 종목 2배 이상 길게 보유)

### 2.2 Multi-Agent 상호작용 시뮬레이션

**시장 가격 결정 메커니즘:**

```
P(t+1) = P(t) + Σ[각 Agent의 매수/매도 압력] + 노이즈

매수/매도 압력(i) = Agent수(i) × 평균거래량(i) × 행동확률(i) × 시장영향계수(i)

시장영향계수:
- 추종 개미: 0.3 (많지만 영향력 작음)
- 세력: 2.0 (소수지만 영향력 큼)
- 손절 회피 개미: 0.5 (중간)

최종 가격 = P(t+1) × (1 + random_walk_noise)
```

**Agent 간 상호작용:**
- 추종 개미가 세력의 매집을 감지 → 따라 매수 → 가격 상승 가속
- 세력이 목표가 도달 → 분산 매도 → 추종 개미 손실 → 손절 회피 개미 발생
- 손절 회피 개미의 물량 출회 → 추가 하락 → 새로운 저가 매집 기회

### 2.3 데이터 요구사항 (3가지 핵심 데이터)

#### 데이터 1: 고빈도 가격/거래량 시계열 데이터
- **형식**: 1분/5분/15분/일봉 OHLCV
- **기간**: 최소 5년치 (다양한 시장 국면 포함)
- **종목**: 코스피/코스닥 상위 200개 종목
- **용도**: 
  - Agent 모델의 가격 패턴 학습
  - 변동성, 추세, 지지/저항선 학습
  - 백테스팅 환경 구축

#### 데이터 2: 뉴스/소셜 미디어 감성 데이터
- **형식**: (timestamp, 종목코드, 뉴스 제목, 감성 점수[-1~1], 임팩트 점수[0~1])
- **출처**: 
  - 네이버 금융 뉴스
  - 증권사 리포트
  - 트위터/커뮤니티 게시글
- **용도**:
  - 추종 개미 Agent의 FOMO/공포 트리거 학습
  - RAG 시스템의 과거 뉴스-가격 영향 DB 구축
  - 감성 지표와 가격 변동 상관관계 모델링

#### 데이터 3: 실제 개인/기관 투자자 거래 패턴 데이터
- **형식**: 익명화된 거래 로그 (매수/매도 시점, 보유 기간, 손익률, 거래 빈도)
- **출처**: 
  - 공개 데이터: 금융감독원 투자자별 매매 동향
  - 합성 데이터: 기존 연구 논문의 통계 기반 생성
- **용도**:
  - Agent 모델의 현실적 행동 패턴 학습
  - 손절/익절 타이밍 분포 학습
  - 사용자 행동과 비교하여 "당신은 상위 몇 %의 투자자입니다" 피드백

---

## 3. 컨텍스트 엔지니어링 킬러 콘텐츠 (RAG & Agent Features)

### 3.1 심리 스코프(Mind Scope) AI 멘토 에이전트

#### 3.1.1 에이전트 아키텍처

**핵심 구성 요소:**
```
[사용자 매매 이력] → [행동 분석 모듈] → [심리 지표 추출] → [LLM 기반 피드백 생성] → [대화형 인터페이스]
                            ↓
                    [장기 메모리 벡터 DB]
                    (사용자 심리 프로필 누적)
```

**작동 방식:**

1. **데이터 수집 (최근 10회 매매 기록)**
   ```json
   {
     "trades": [
       {
         "timestamp": "2025-10-15 09:35:00",
         "action": "buy",
         "symbol": "삼성전자",
         "price": 75000,
         "quantity": 10,
         "reason_emotion": "확신",  // 사용자가 직접 선택
         "price_change_before_trade": "+3.2%",  // 직전 10분간 변화율
         "news_sentiment": 0.75,  // 당시 뉴스 감성 점수
       },
       {
         "timestamp": "2025-10-15 14:20:00",
         "action": "sell",
         "symbol": "삼성전자",
         "price": 74200,
         "quantity": 10,
         "reason_emotion": "두려움",
         "unrealized_pnl": -1.07,
         "hold_duration": "4h 45m"
       }
     ]
   }
   ```

2. **심리 지표 추출 알고리즘**

   **FOMO 지수 (0-100):**
   ```python
   def calculate_fomo_index(trades):
       fomo_signals = []
       for trade in trades:
           if trade['action'] == 'buy':
               # 급등 후 매수 = FOMO 신호
               if trade['price_change_before_trade'] > 2.0:
                   fomo_score = min(trade['price_change_before_trade'] * 10, 100)
                   fomo_signals.append(fomo_score)
               
               # 긍정 뉴스 + 감정="욕심" = FOMO
               if trade['news_sentiment'] > 0.6 and trade['reason_emotion'] == '욕심':
                   fomo_signals.append(70)
       
       return np.mean(fomo_signals) if fomo_signals else 0
   ```

   **손절 타이밍 지연 지수 (0-100):**
   ```python
   def calculate_loss_cut_delay(trades):
       delays = []
       for trade in trades:
           if trade['action'] == 'sell' and trade['unrealized_pnl'] < 0:
               # 손실 폭 대비 보유 기간
               loss_percent = abs(trade['unrealized_pnl'])
               hold_hours = parse_duration(trade['hold_duration'])
               
               # 손실 1%당 1시간 이상 보유 = 지연
               expected_hours = loss_percent * 1.0
               if hold_hours > expected_hours:
                   delay_score = min((hold_hours / expected_hours) * 50, 100)
                   delays.append(delay_score)
       
       return np.mean(delays) if delays else 0
   ```

   **충동 매매 지수 (0-100):**
   ```python
   def calculate_impulsive_trading(trades):
       # 10분 이내 연속 거래 횟수
       impulsive_count = 0
       for i in range(1, len(trades)):
           time_diff = (trades[i]['timestamp'] - trades[i-1]['timestamp']).seconds / 60
           if time_diff < 10:
               impulsive_count += 1
       
       return min(impulsive_count * 20, 100)
   ```

3. **LLM 기반 맞춤형 피드백 생성**

   **프롬프트 템플릿:**
   ```
   당신은 투자 심리 전문 멘토입니다. 다음 사용자의 매매 이력과 심리 지표를 분석하여 
   구체적이고 실행 가능한 조언을 제공하세요.
   
   ## 사용자 데이터
   - FOMO 지수: {fomo_index}/100
   - 손절 지연 지수: {loss_cut_delay}/100
   - 충동 매매 지수: {impulsive_trading}/100
   
   ## 최근 거래 예시
   {trade_examples}
   
   ## 과거 멘토링 이력 (RAG 검색 결과)
   {retrieved_past_feedback}
   
   ## 요구사항
   1. 가장 심각한 심리적 약점 1가지 지적
   2. 구체적인 개선 행동 3가지 제시 (예: "매수 전 10분 기다리기 루틴")
   3. 다음 주 실천 목표 1가지 설정
   4. 따뜻하고 격려하는 톤 유지
   ```

   **피드백 예시:**
   ```
   안녕하세요, {사용자명}님! 이번 주 거래를 분석해봤어요.
   
   😊 잘한 점: 
   10월 17일 현대차 매도는 훌륭했어요! 5% 수익에서 욕심 부리지 않고 
   목표대로 익절하신 게 인상적입니다.
   
   ⚠️ 개선 포인트:
   FOMO 지수가 73으로 높게 나왔어요. 특히 삼성전자 매수 건을 보면, 
   3% 급등 후 "이대로 놓치면 안 될 것 같아서" 매수하셨는데, 
   결국 -5% 손실로 이어졌습니다.
   
   📋 이번 주 실천 과제:
   1. "매수 버튼 10분 지연 타이머" 활성화하기
   2. 매수 전 체크리스트 작성: "왜 사는가? 목표가는? 손절선은?"
   3. 급등 종목은 일단 '관심 목록'에만 담고 하루 뒤 재평가
   
   다음 주 목표: FOMO 지수 60 이하로 낮추기! 
   할 수 있어요! 💪
   ```

#### 3.1.2 장기 대화 관리 및 심리 이력 보존

**벡터 DB 기반 메모리 시스템:**

```python
# Embedding 저장 구조
class UserPsychologyMemory:
    def __init__(self, user_id):
        self.user_id = user_id
        self.vector_db = ChromaDB(collection=f"user_{user_id}_psychology")
        
    def store_session(self, date, trades, psychology_scores, feedback):
        # 세션 요약 텍스트 생성
        summary = f"""
        날짜: {date}
        FOMO 지수: {psychology_scores['fomo']}
        주요 행동: {self._summarize_trades(trades)}
        멘토 피드백: {feedback}
        """
        
        # 임베딩 생성 및 저장
        embedding = self.embedding_model.encode(summary)
        self.vector_db.add(
            embeddings=[embedding],
            documents=[summary],
            metadatas=[{"date": date, "fomo": psychology_scores['fomo']}]
        )
    
    def retrieve_similar_past(self, current_trades, top_k=3):
        # 현재 거래와 유사한 과거 패턴 검색
        current_summary = self._summarize_trades(current_trades)
        current_embedding = self.embedding_model.encode(current_summary)
        
        results = self.vector_db.query(
            query_embeddings=[current_embedding],
            n_results=top_k
        )
        
        return results  # 과거 유사 상황에서의 피드백 반환
```

**장기 트렌드 추적:**
- 주간/월간 심리 지표 변화 그래프
- "3개월 전 FOMO 지수 85 → 현재 62, 27% 개선!" 
- 성장 스토리 내러티브 생성: "당신의 투자 심리 성장기"

### 3.2 RAG 기반 뉴스 역추적 시뮬레이션

#### 3.2.1 기능 개요

**목적:** 사용자가 특정 주식 매매 결정 전, 과거 유사한 뉴스/이벤트가 실제 가격에 미쳤던 영향을 검색하고 시뮬레이션

**사용 시나리오:**
```
사용자: "삼성전자가 신제품 발표했는데 살까요?"
       ↓
시스템: RAG 검색 → "삼성전자 신제품 발표" 유사 과거 사례 5개 검색
       ↓
시뮬레이션: 과거 사례의 평균 가격 변동 패턴 시각화
       ↓
교육적 인사이트: "신제품 발표 후 단기 급등(+5%) → 1주일 내 조정(-3%)
                패턴이 60% 확률로 발생했습니다"
```

#### 3.2.2 RAG 시스템 아키텍처

**데이터 파이프라인:**

1. **뉴스 임베딩 DB 구축**
   ```python
   # 과거 5년간 주요 뉴스 전처리
   news_db = [
       {
           "date": "2023-08-15",
           "title": "삼성전자, 갤럭시 Z 플립5 공개",
           "symbol": "005930",
           "category": "신제품 발표",
           "price_impact_7d": +4.2,  # 7일 후 가격 변화율
           "price_impact_30d": -1.5,  # 30일 후 가격 변화율
           "volume_spike": 2.3  # 평소 대비 거래량 증가 배수
       },
       ...
   ]
   
   # Sentence Transformer로 임베딩
   embeddings = model.encode([news['title'] for news in news_db])
   
   # Vector DB 저장 (FAISS 또는 Milvus)
   index = faiss.IndexFlatL2(embedding_dim)
   index.add(embeddings)
   ```

2. **유사 뉴스 검색**
   ```python
   def search_similar_news(query, symbol, top_k=5):
       query_embedding = model.encode([query])
       
       # 벡터 유사도 검색
       distances, indices = index.search(query_embedding, top_k * 3)
       
       # 필터링: 동일 종목만
       results = [news_db[i] for i in indices[0] if news_db[i]['symbol'] == symbol]
       
       return results[:top_k]
   ```

3. **가격 영향 시뮬레이션**
   ```python
   def simulate_price_impact(similar_news):
       # 과거 사례들의 가격 변화 패턴 추출
       patterns = []
       for news in similar_news:
           pattern = {
               'days_after': [1, 3, 7, 14, 30],
               'price_changes': [
                   news['price_impact_1d'],
                   news['price_impact_3d'],
                   news['price_impact_7d'],
                   news['price_impact_14d'],
                   news['price_impact_30d']
               ]
           }
           patterns.append(pattern)
       
       # 평균 패턴 및 신뢰구간 계산
       avg_pattern = np.mean([p['price_changes'] for p in patterns], axis=0)
       std_pattern = np.std([p['price_changes'] for p in patterns], axis=0)
       
       return {
           'average': avg_pattern,
           'confidence_interval': (avg_pattern - std_pattern, avg_pattern + std_pattern),
           'sample_size': len(patterns)
       }
   ```

#### 3.2.3 UI/UX 설계

**인터페이스 구성:**

```
┌─────────────────────────────────────────────────────────┐
│  뉴스 역추적 시뮬레이터                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  📰 현재 뉴스: "삼성전자, 3나노 공정 양산 돌입"          │
│  📅 발표일: 2025-10-20                                  │
│                                                         │
│  🔍 유사한 과거 사례 5개를 찾았습니다:                   │
│                                                         │
│  ┌───────────────────────────────────────────────┐    │
│  │ 1. 2023-06-10: "삼성, 5나노 공정 양산"        │    │
│  │    → 7일 후: +5.2% | 30일 후: +2.1%           │    │
│  │                                                │    │
│  │ 2. 2022-11-03: "삼성, 4나노 고객사 확보"      │    │
│  │    → 7일 후: +3.8% | 30일 후: -1.5%           │    │
│  │                                                │    │
│  │ 3. 2024-02-15: "삼성, GAA 기술 발표"          │    │
│  │    → 7일 후: +6.1% | 30일 후: +4.3%           │    │
│  └───────────────────────────────────────────────┘    │
│                                                         │
│  📊 예상 가격 변동 패턴 (과거 평균):                    │
│                                                         │
│      %                                                  │
│     +8│         •                                       │
│     +6│       •   •                                     │
│     +4│     •       •                                   │
│     +2│   •           •                                 │
│      0├─•───────────────•─────                          │
│     -2│                   •                             │
│       └─1d──3d──7d──14d──30d                            │
│                                                         │
│  💡 인사이트:                                           │
│  • 단기(7일): 평균 +5.0% 상승 (신뢰구간: +3% ~ +7%)   │
│  • 중기(30일): 평균 +1.6% 상승 (변동성 큼)            │
│  • 거래량: 평소의 2.1배 급증 예상                      │
│                                                         │
│  ⚠️ 주의사항:                                           │
│  과거 사례의 60%는 초기 급등 후 조정을 겪었습니다.     │
│  단기 차익 목적이라면 7일 이내 익절 전략 권장합니다.   │
│                                                         │
│  [이 정보로 시뮬레이션 해보기]  [매매 전략 짜기]        │
└─────────────────────────────────────────────────────────┘
```

**인터랙티브 요소:**

1. **시간대별 토글**: 사용자가 1일/3일/7일/30일 구간을 선택하여 상세 분석
2. **사례 클릭 시 상세 보기**: 해당 시점의 차트, 추가 뉴스, 시장 상황 표시
3. **시뮬레이션 모드**: 
   - "만약 그때 샀다면?" 버튼 → 과거 사례 각각에서 매수 후 결과 체험
   - 타임라인 플레이어: 일자별로 가격 변화 애니메이션

**교육적 강화 요소:**

- **확률적 사고 훈련**: "100% 확실한 것은 없습니다. 과거 사례의 40%는 하락했어요"
- **리스크 관리 제안**: "최대 손실 -X% 예상 시나리오도 존재합니다. 손절선 설정하세요"
- **장기 관점 유도**: 30일/90일 장기 데이터도 함께 표시

#### 3.2.4 기술적 고려사항

**성능 최적화:**
- 뉴스 임베딩 캐싱: 자주 검색되는 뉴스는 메모리에 캐시
- 비동기 검색: 사용자가 뉴스 클릭 시 백그라운드에서 미리 검색 시작
- 결과 페이지네이션: 초기 3개 사례만 표시, "더 보기"로 확장

**데이터 품질 관리:**
- 뉴스 카테고리 라벨링: "신제품 발표", "실적 발표", "인수합병" 등 분류
- 중복 제거: 동일 이벤트의 여러 뉴스는 하나로 통합
- 이상치 제거: 특수한 시장 상황(코로나, 전쟁 등) 발생 시기 데이터는 별도 표시

---

## 4. 기술 구현 및 데이터 관리 지침 (Technical Architecture)

### 4.1 마이크로서비스 아키텍처 (MSA) 설계

#### 전체 시스템 구조

```
┌──────────────┐
│   Frontend   │  (React/Vue.js)
│   Web App    │
└──────┬───────┘
       │ REST API / WebSocket
       │
┌──────▼──────────────────────────────────────────────┐
│            API Gateway (Kong/AWS API Gateway)       │
└──────┬──────────────────────────────────────────────┘
       │
       ├─────────┬─────────┬──────────┬──────────────┐
       │         │         │          │              │
┌──────▼──┐ ┌───▼────┐ ┌──▼─────┐ ┌─▼──────┐  ┌────▼────┐
│  Game   │ │  AI    │ │  RAG   │ │ User   │  │ Market  │
│ Service │ │ Engine │ │Service │ │Service │  │  Data   │
│         │ │        │ │        │ │        │  │ Service │
└────┬────┘ └───┬────┘ └───┬────┘ └───┬────┘  └────┬────┘
     │          │           │          │            │
     └──────────┴───────────┴──────────┴────────────┘
                           │
                    ┌──────▼─────┐
                    │  Message   │
                    │   Queue    │
                    │ (RabbitMQ) │
                    └────────────┘
                           │
                ┌──────────┴──────────┐
         ┌──────▼─────┐        ┌─────▼──────┐
         │ PostgreSQL │        │   Redis    │
         │  (주 DB)   │        │  (캐시)    │
         └────────────┘        └────────────┘
```

### 4.2 PyTorch 모델 훈련/추론 분리 - Context Offloading

#### 4.2.1 API 인터페이스 설계 (3가지 핵심 API)

**API 1: POST /api/v1/ai/simulate_market_impact**

**목적:** 현재 시장 상황에서 AI Agent들의 행동을 시뮬레이션하여 향후 가격 변동 예측

**요청:**
```json
{
  "symbol": "005930",
  "current_price": 75000,
  "recent_candles": [
    {"timestamp": "2025-10-20T09:00:00", "open": 74500, "high": 75200, "low": 74300, "close": 75000, "volume": 1500000},
    // ... 최근 20개 봉
  ],
  "news_events": [
    {"timestamp": "2025-10-20T08:30:00", "title": "삼성전자 3분기 실적 발표", "sentiment": 0.65}
  ],
  "simulation_duration": 60,  // 시뮬레이션할 미래 시간 (분)
  "agent_counts": {
    "momentum_chaser": 1000,
    "smart_money": 50,
    "loss_aversion": 800
  }
}
```

**응답:**
```json
{
  "simulation_id": "sim_20251020_093022",
  "predicted_prices": [
    {"time_offset_min": 5, "price": 75300, "confidence": 0.72},
    {"time_offset_min": 10, "price": 75450, "confidence": 0.68},
    // ...
    {"time_offset_min": 60, "price": 75800, "confidence": 0.45}
  ],
  "agent_actions": {
    "momentum_chaser": {"buy_ratio": 0.73, "avg_confidence": 0.81},
    "smart_money": {"buy_ratio": 0.22, "avg_confidence": 0.65},
    "loss_aversion": {"sell_ratio": 0.15, "avg_confidence": 0.51}
  },
  "risk_assessment": {
    "volatility_index": 0.68,  // 0-1, 높을수록 변동성 큼
    "manipulation_risk": 0.32  // 세력 주도 가능성
  }
}
```

**백엔드 처리:**
- 요청 수신 → Message Queue에 작업 등록
- AI Engine 워커가 비동기로 PyTorch 모델 추론 실행
- 결과를 Redis 캐시에 저장 (TTL 5분)
- 클라이언트는 WebSocket 또는 polling으로 결과 수신

---

**API 2: GET /api/v1/ai/agent_feedback/{user_id}**

**목적:** 사용자의 최근 매매 기록을 분석하여 심리 스코프 피드백 제공

**요청:**
```
GET /api/v1/ai/agent_feedback/user_12345?trades_count=10&language=ko
```

**응답:**
```json
{
  "user_id": "user_12345",
  "analysis_period": "2025-10-13 ~ 2025-10-20",
  "psychology_scores": {
    "fomo_index": 73,
    "loss_cut_delay_index": 58,
    "impulsive_trading_index": 42,
    "risk_management_score": 65
  },
  "feedback": {
    "summary": "이번 주 FOMO 지수가 높게 나타났습니다. 급등 종목 매수 전 10분 대기 전략을 권장합니다.",
    "strengths": [
      "10월 17일 현대차 익절 타이밍이 우수했습니다 (+5.2%)"
    ],
    "weaknesses": [
      "삼성전자 매수 시 3% 급등 후 고점 매수 → FOMO 패턴",
      "LG화학 손절 지연 (손실 -7%까지 보유)"
    ],
    "action_items": [
      "매수 전 '10분 냉각' 타이머 활성화",
      "손절선 -3% 자동 설정 활용",
      "주간 매매 일지 작성하기"
    ],
    "next_week_goal": "FOMO 지수 60 이하로 낮추기"
  },
  "historical_trend": {
    "fomo_index_4weeks": [85, 78, 75, 73],  // 4주간 추이
    "improvement_rate": 14.1  // % 개선
  },
  "similar_past_situations": [
    {
      "date": "2025-09-15",
      "situation": "급등 종목 고점 매수",
      "outcome": "손실 -8%",
      "lesson_learned": "단기 급등 후에는 조정 대기 전략이 유효"
    }
  ]
}
```

**백엔드 처리:**
- User Service에서 최근 거래 이력 조회
- AI Engine으로 심리 분석 요청
- RAG Service에서 과거 유사 상황 검색 (벡터 DB)
- LLM으로 피드백 텍스트 생성 (GPT-4 또는 Claude)
- 결과 캐싱 (사용자당 일 1회 갱신)

---

**API 3: POST /api/v1/rag/search_historical_impact**

**목적:** 특정 뉴스/이벤트와 유사한 과거 사례 검색 및 가격 영향 분석

**요청:**
```json
{
  "query": "삼성전자 신제품 발표",
  "symbol": "005930",
  "search_period": "2020-01-01~2025-10-20",
  "top_k": 5,
  "return_price_data": true
}
```

**응답:**
```json
{
  "query": "삼성전자 신제품 발표",
  "results": [
    {
      "rank": 1,
      "similarity_score": 0.92,
      "event": {
        "date": "2023-08-15",
        "title": "삼성전자, 갤럭시 Z 플립5 공개",
        "category": "신제품 발표",
        "source": "연합뉴스"
      },
      "price_impact": {
        "before_price": 70000,
        "after_1d": 71500,
        "after_3d": 72100,
        "after_7d": 73500,
        "after_30d": 71800,
        "max_gain": 5.0,
        "max_drawdown": -2.3
      },
      "volume_impact": {
        "avg_volume_before": 12000000,
        "peak_volume": 28000000,
        "spike_ratio": 2.33
      },
      "context": "플래그십 스마트폰 신제품 발표로 단기 모멘텀 형성, 이후 실적 발표 전 조정"
    },
    // ... 나머지 4개 결과
  ],
  "aggregate_analysis": {
    "avg_price_impact_7d": 4.8,
    "avg_price_impact_30d": 1.2,
    "success_rate": 0.6,  // 상승한 비율
    "confidence_interval_7d": [3.2, 6.4],
    "recommendation": "단기 상승 가능성 높으나, 중기적으로는 변동성 존재. 7일 이내 익절 전략 권장."
  }
}
```

**백엔드 처리:**
- 쿼리 임베딩 생성 (Sentence Transformer)
- Vector DB에서 유사도 검색 (FAISS/Milvus)
- 검색된 이벤트의 가격 데이터를 Market Data Service에서 조회
- 통계 분석 (평균, 표준편차, 신뢰구간)
- 결과 JSON 반환

### 4.3 데이터 동기화 전략

#### 4.3.1 실시간 주가 데이터 파이프라인

**데이터 플로우:**
```
[증권사 API] → [Market Data Service] → [Redis Stream] → [Game Service]
                                              ↓
                                        [AI Engine 구독]
```

**구현 방식:**

1. **Market Data Service**
   ```python
   # WebSocket으로 실시간 데이터 수신
   async def stream_market_data():
       async with websockets.connect('wss://api.stock.com/stream') as ws:
           async for message in ws:
               data = json.loads(message)
               
               # Redis Stream에 발행
               redis_client.xadd(
                   'market:stream',
                   {
                       'symbol': data['symbol'],
                       'price': data['price'],
                       'timestamp': data['timestamp']
                   }
               )
               
               # 1분 캔들 집계
               aggregate_candle(data)
   ```

2. **AI Engine 구독 및 추론**
   ```python
   # Redis Stream 소비자
   async def consume_market_stream():
       last_id = '0'
       while True:
           messages = redis_client.xread(
               {'market:stream': last_id},
               count=100,
               block=1000
           )
           
           for message in messages:
               price_data = message['data']
               
               # 주기적으로 AI 모델 추론 트리거 (1분마다)
               if should_run_inference(price_data['timestamp']):
                   await run_agent_simulation(price_data)
   ```

3. **데이터 일관성 보장**
   - **Write-Through 캐시**: Market Data Service가 PostgreSQL에 저장 후 Redis 갱신
   - **Event Sourcing**: 모든 가격 변화를 이벤트 로그로 저장 (재현 가능)
   - **버전 관리**: 각 데이터에 타임스탬프 + 시퀀스 번호 부여

#### 4.3.2 AI 모델 추론 결과 동기화

**문제:** AI 모델 추론은 느리다 (1-3초) → 실시간 게임 지연 발생

**해결책: 예측 캐싱 + 점진적 업데이트**

```python
class SimulationCache:
    def __init__(self):
        self.cache = {}  # {symbol: {timestamp: prediction}}
        
    async def get_or_compute(self, symbol, current_time):
        # 캐시 확인 (30초 이내 결과면 재사용)
        if symbol in self.cache:
            cached = self.cache[symbol]
            if current_time - cached['timestamp'] < 30:
                return cached['prediction']
        
        # 캐시 미스 → 비동기 추론 시작
        task_id = await submit_inference_task(symbol)
        
        # 이전 결과가 있으면 우선 반환 (stale data)
        if symbol in self.cache:
            asyncio.create_task(self._update_cache(symbol, task_id))
            return self.cache[symbol]['prediction']
        
        # 첫 요청이면 대기
        result = await wait_for_inference(task_id)
        self.cache[symbol] = {'timestamp': current_time, 'prediction': result}
        return result
```

**최적화 전략:**
- **배치 추론**: 여러 종목을 한 번에 모델에 입력 (GPU 효율 향상)
- **모델 경량화**: ONNX 변환 또는 TensorRT로 추론 속도 3-5배 향상
- **우선순위 큐**: 사용자가 많이 보는 종목 우선 추론

#### 4.3.3 데이터베이스 스키마 설계

**핵심 테이블:**

```sql
-- 사용자 거래 이력
CREATE TABLE user_trades (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,  -- buy/sell
    price DECIMAL(10, 2),
    quantity INT,
    timestamp TIMESTAMPTZ NOT NULL,
    emotion VARCHAR(20),  -- 사용자가 선택한 감정
    reason TEXT,
    INDEX idx_user_time (user_id, timestamp DESC)
);

-- AI 에이전트 시뮬레이션 결과
CREATE TABLE agent_simulations (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    simulation_time TIMESTAMPTZ NOT NULL,
    agent_type VARCHAR(50),
    action_distribution JSONB,  -- {"buy": 0.7, "hold": 0.2, "sell": 0.1}
    predicted_price DECIMAL(10, 2),
    actual_price DECIMAL(10, 2),  -- 나중에 업데이트
    accuracy_score FLOAT,
    INDEX idx_symbol_time (symbol, simulation_time DESC)
);

-- 뉴스 임베딩 메타데이터 (벡터는 별도 Vector DB)
CREATE TABLE news_events (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    event_date TIMESTAMPTZ NOT NULL,
    title TEXT NOT NULL,
    category VARCHAR(50),
    sentiment_score FLOAT,
    price_impact_7d FLOAT,
    price_impact_30d FLOAT,
    embedding_id VARCHAR(100),  -- Vector DB 참조 ID
    INDEX idx_symbol_date (symbol, event_date DESC)
);

-- 사용자 심리 프로필
CREATE TABLE user_psychology_profiles (
    user_id VARCHAR(50) PRIMARY KEY,
    fomo_index INT,
    loss_cut_delay_index INT,
    impulsive_trading_index INT,
    last_updated TIMESTAMPTZ,
    trend_data JSONB  -- 주간/월간 추이 데이터
);
```

### 4.4 확장성 및 성능 고려사항

#### 4.4.1 부하 분산 전략

- **Game Service**: Stateless 설계 → 수평 확장 용이 (k8s HPA)
- **AI Engine**: GPU 워커 풀 (NVIDIA Triton Inference Server 활용)
  - 동적 스케일링: 요청 대기 큐 길이 기준
- **Redis 클러스터**: Sharding (종목별 분산)

#### 4.4.2 모니터링 지표

- AI 추론 레이턴시: p50, p95, p99 추적
- 캐시 히트율: 목표 80% 이상
- WebSocket 동시 접속자 수
- 데이터 신선도: 최신 가격 데이터 지연 시간 (목표 1초 이내)

---

## 5. 최종 마스터플랜 통합 요약

### 5.1 프로젝트 핵심 가치 제안

**"안전한 실패를 통한 투자 심리 마스터리"**

- **타겟 달성**: 주식 초보가 실전 투자 전 **가상 환경에서 100회 이상의 안전한 실패 경험**을 통해 심리적 면역력 구축
- **교육 효과**: 감정 기반 투자 → 데이터 기반 의사결정으로 전환 (FOMO 지수 평균 40% 감소 목표)
- **몰입 요소**: 실시간 AI 시뮬레이션 + 개인화된 멘토 피드백으로 높은 리텐션 유도

### 5.2 구현 로드맵 (3단계)

#### Phase 1: MVP (3개월)
- [ ] 기본 게임 시스템 (레벨 1-5, 단순 가격 패턴)
- [ ] 3가지 AI Agent 모델 초기 버전 (PyTorch 훈련)
- [ ] 심리 스코프 기본 피드백 (FOMO, 손절 지연 지수)
- [ ] 실시간 주가 데이터 연동 (5분 지연)
- [ ] RESTful API 기본 3종 구축

#### Phase 2: 고도화 (3개월)
- [ ] RAG 뉴스 역추적 시뮬레이션 기능
- [ ] AI Agent 모델 고도화 (Multi-Agent 상호작용)
- [ ] 협력적 경쟁 시스템 (팀 리그)
- [ ] 실시간 데이터 (1분 지연으로 개선)
- [ ] 장기 메모리 벡터 DB (사용자 성장 추적)

#### Phase 3: 완성 (3개월)
- [ ] 레벨 6-15 고난이도 시나리오
- [ ] 멘토-멘티 매칭 시스템
- [ ] 모바일 앱 출시
- [ ] 실시간 랭킹 및 주간 챌린지
- [ ] A/B 테스트 기반 교육 효과 최적화

### 5.3 성공 지표 (KPI)

**사용자 행동 지표:**
- DAU 1만명, MAU 5만명 (6개월 후)
- 평균 세션 시간: 25분 이상
- 7일 리텐션: 40% 이상

**교육 효과 지표:**
- 사용자 FOMO 지수 평균 40% 감소 (3개월 사용 후)
- 손절 타이밍 정확도 60% 향상
- 게임 졸업 후 실전 투자 시작률: 35%

**기술 성능 지표:**
- AI 추론 레이턴시: p95 < 2초
- 시스템 가용성: 99.5% 이상
- 동시 접속자 지원: 10,000명

### 5.4 리스크 관리

**기술 리스크:**
- AI 모델 정확도 부족 → 실제 투자자 데이터 기반 지속 재훈련
- 실시간 데이터 비용 → 초기에는 5분 지연 데이터 활용, 단계적 전환

**사업 리스크:**
- 사행성 논란 → 교육 목표 명확화, 현금 보상 절대 금지
- 경쟁사 출현 → AI 심리 분석 기능으로 차별화 강화

**운영 리스크:**
- 서버 과부하 → 클라우드 오토스케일링, CDN 활용
- 데이터 보안 → 사용자 거래 데이터 암호화, GDPR 준수

### 5.5 개발팀 액션 아이템

**백엔드 팀:**
1. MSA 아키텍처 기반 5개 서비스 구축 (Game, AI Engine, RAG, User, Market Data)
2. Redis Stream 기반 실시간 데이터 파이프라인 구현
3. PostgreSQL + Vector DB (ChromaDB/Milvus) 스키마 설계

**AI/ML 팀:**
1. PyTorch로 3가지 Agent 모델 구현 및 훈련
2. ONNX 변환 후 추론 서버 최적화 (TensorRT/Triton)
3. LLM 프롬프트 엔지니어링 (피드백 생성)

**프론트엔드 팀:**
1. React 기반 게임 UI/UX 구현
2. WebSocket 실시간 차트 라이브러리 (TradingView 연동)
3. 뉴스 역추적 시뮬레이션 인터랙티브 인터페이스

**데이터 팀:**
1. 과거 5년 주가/뉴스 데이터 수집 및 전처리
2. 감성 분석 모델 구축 (뉴스 제목 → 감성 점수)
3. 데이터 품질 모니터링 대시보드

---

## 📌 결론

본 마스터플랜은 **PyTorch 기반 AI 시뮬레이션**, **RAG 컨텍스트 엔지니어링**, **교육 중심 게임 설계**를 결합하여 주식 투자 초보자의 심리적 장벽을 허무는 혁신적인 웹 게임 시스템을 제시합니다.

**핵심 성공 요인:**
1. ✅ 안전한 실패 경험을 통한 점진적 자신감 구축
2. ✅ AI 심리 분석으로 개인화된 학습 경로 제공
3. ✅ 실시간 데이터와 교육 목표의 절묘한 균형
4. ✅ 확장 가능한 MSA 아키텍처로 빠른 반복 개발 지원

개발팀은 본 문서를 기반으로 **Phase 1 MVP를 3개월 내 출시**할 수 있으며, 이후 사용자 피드백을 반영한 지속적 개선을 통해 **대한민국 No.1 투자 교육 플랫폼**으로 성장할 수 있습니다.

**"공포를 자신감으로, 감정을 전략으로 - Mind Trader가 함께합니다."** 🚀

---

*Document Version: 1.0*  
*Last Updated: 2025-10-20*  
*Contact: CSO Office*
