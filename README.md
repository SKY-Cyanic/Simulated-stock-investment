# 🧠 Mind Trader - 심리 시뮬레이션 기반 가상 주식 투자 웹 게임

> **"공포를 자신감으로, 감정을 전략으로 전환시키는 안전한 심리 훈련장"**

Mind Trader는 PyTorch 기반 AI Agent 시뮬레이션과 RAG(Retrieval-Augmented Generation) 기술을 활용하여 주식 투자 초보자의 심리적 장벽을 극복하도록 돕는 교육용 웹 게임입니다.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)](https://reactjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)

---

## 📋 목차

- [주요 기능](#-주요-기능)
- [시스템 아키텍처](#-시스템-아키텍처)
- [기술 스택](#-기술-스택)
- [설치 및 실행](#-설치-및-실행)
- [API 문서](#-api-문서)
- [프로젝트 구조](#-프로젝트-구조)
- [개발 로드맵](#-개발-로드맵)

---

## ✨ 주요 기능

### 1. PyTorch 기반 AI 심리 시뮬레이션 엔진

3가지 타입의 AI Agent가 실제 투자자 심리를 모델링하여 가상 시장을 생성합니다:

- **추종 매매 개미 (Momentum Chaser)**: LSTM + Attention 기반, FOMO 심리 구현
- **저가 매집 세력 (Smart Money)**: GRU + 역발상 전략, 체계적 분산 매수
- **손절 회피 개미 (Loss Aversion)**: Transformer 기반, Prospect Theory 적용

### 2. 개인화된 심리 분석 AI 멘토

사용자의 매매 기록을 분석하여 다음 지표를 제공합니다:
- **FOMO 지수**: 공포와 욕심에 의한 충동 매매 경향
- **손절 타이밍 지연 지수**: 손실 발생 시 손절 지연 정도
- **충동 매매 지수**: 단기간 내 반복 거래 빈도
- **리스크 관리 점수**: 전체적인 리스크 관리 능력

### 3. RAG 기반 뉴스 역추적 시뮬레이션

과거 유사 뉴스 이벤트를 검색하고 실제 가격 영향을 시뮬레이션합니다:
- 유사도 기반 과거 사례 검색
- 7일/30일 가격 변동 패턴 분석
- 확률적 예측 및 리스크 평가

### 4. 단계별 성장 시스템

레벨 1(안전지대) → 레벨 5(현실 시장) 까지 점진적 난이도 상승:
- 레벨별 목표 승률 및 변동성 조정
- AI Agent 복잡도 점진적 증가
- 교육적 힌트 및 피드백 제공

---

## 🏗 시스템 아키텍처

```
┌──────────────┐
│   Frontend   │  (React + TailwindCSS)
│   Web App    │
└──────┬───────┘
       │ REST API
       │
┌──────▼──────────────────────────────────────────────┐
│            API Gateway (Kong/Nginx)                 │
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

### 마이크로서비스 구성

1. **Game Service** (Port 8001): 게임 로직, 레벨 관리, 시나리오 생성
2. **AI Engine Service** (Port 8002): PyTorch 모델 추론, 시장 시뮬레이션
3. **User Service** (Port 8003): 사용자 인증, 거래 기록, 심리 분석
4. **Market Data Service** (Port 8004): 가격 데이터 제공
5. **RAG Service** (Port 8005): 뉴스 검색, 역사적 영향 분석

---

## 🛠 기술 스택

### Backend
- **Framework**: FastAPI 0.104+
- **ML/AI**: PyTorch 2.1, NumPy
- **Database**: PostgreSQL 15, Redis 7
- **Message Queue**: RabbitMQ 3
- **Authentication**: JWT (PyJWT)

### Frontend
- **Framework**: React 18.2
- **Styling**: TailwindCSS 3.3
- **Charts**: Recharts 2.10
- **Routing**: React Router DOM 6.20
- **HTTP Client**: Axios

### Infrastructure
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Kubernetes (Optional)
- **CI/CD**: GitHub Actions (Optional)

---

## 🚀 설치 및 실행

### 사전 요구사항

- Docker 20.10+
- Docker Compose 2.0+
- (선택) Node.js 18+ (로컬 개발 시)
- (선택) Python 3.10+ (로컬 개발 시)

### 빠른 시작 (Docker Compose)

```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/mind-trader.git
cd mind-trader

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 필요한 값 수정 (선택사항)

# 3. Docker Compose로 전체 스택 실행
docker-compose up -d

# 4. 서비스 상태 확인
docker-compose ps

# 5. 로그 확인
docker-compose logs -f
```

### 서비스 접속

- **Frontend**: http://localhost:3000
- **Game Service**: http://localhost:8001/docs
- **AI Engine**: http://localhost:8002/docs
- **User Service**: http://localhost:8003/docs
- **Market Data Service**: http://localhost:8004/docs
- **RAG Service**: http://localhost:8005/docs
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **RabbitMQ Management**: http://localhost:15672 (admin/admin123)

### 로컬 개발 환경 설정

#### Backend 개발

```bash
# 각 서비스 디렉토리로 이동
cd backend/ai_engine

# 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서비스 실행
uvicorn main:app --reload --port 8000
```

#### Frontend 개발

```bash
cd frontend/mind-trader-app

# 의존성 설치
npm install

# 개발 서버 실행
npm start
```

### 데이터베이스 초기화

서비스 첫 실행 시 자동으로 테이블이 생성됩니다. 수동 초기화가 필요한 경우:

```bash
docker-compose exec postgres psql -U postgres -d mind_trader

# SQL 실행
\dt  # 테이블 목록 확인
```

---

## 📚 API 문서

### 주요 API 엔드포인트

#### 1. User Service

**사용자 등록**
```http
POST /api/v1/users/register
Content-Type: application/json

{
  "username": "testuser",
  "email": "test@example.com",
  "password": "password123"
}
```

**로그인**
```http
POST /api/v1/users/login
Content-Type: application/json

{
  "username": "testuser",
  "password": "password123"
}

Response:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "user": {...}
}
```

**심리 분석 피드백**
```http
GET /api/v1/ai/agent_feedback/{user_id}?trades_count=10
Authorization: Bearer {token}

Response:
{
  "psychology_scores": {
    "fomo_index": 73,
    "loss_cut_delay_index": 58,
    ...
  },
  "feedback": {
    "summary": "...",
    "strengths": [...],
    "weaknesses": [...],
    "action_items": [...]
  }
}
```

#### 2. AI Engine Service

**시장 영향 시뮬레이션**
```http
POST /api/v1/ai/simulate_market_impact
Content-Type: application/json

{
  "symbol": "SAMSUNG",
  "current_price": 75000,
  "recent_candles": [...],
  "simulation_duration": 60,
  "agent_counts": {
    "momentum_chaser": 1000,
    "smart_money": 50,
    "loss_aversion": 800
  }
}

Response:
{
  "simulation_id": "sim_20251020_093022",
  "predicted_prices": [...],
  "agent_actions": {...},
  "risk_assessment": {...}
}
```

#### 3. Game Service

**시나리오 시작**
```http
POST /api/v1/game/start_scenario/{user_id}

Response:
{
  "scenario_id": "...",
  "level": 1,
  "starting_balance": 10000,
  "market_data": {...},
  "hints": [...]
}
```

전체 API 문서는 각 서비스의 `/docs` 엔드포인트에서 Swagger UI로 확인 가능합니다.

---

## 📁 프로젝트 구조

```
mind-trader/
├── backend/
│   ├── shared/              # 공유 모듈
│   │   ├── models.py        # SQLAlchemy 모델
│   │   └── schemas.py       # Pydantic 스키마
│   ├── ai_engine/           # AI 추론 서비스
│   │   ├── models/
│   │   │   └── agents.py    # PyTorch Agent 모델
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── user_service/        # 사용자 관리 서비스
│   ├── game_service/        # 게임 로직 서비스
│   ├── market_data_service/ # 시장 데이터 서비스
│   └── rag_service/         # RAG 검색 서비스
├── frontend/
│   └── mind-trader-app/     # React 앱
│       ├── src/
│       │   ├── components/
│       │   │   ├── Login.js
│       │   │   ├── Dashboard.js
│       │   │   ├── Game.js
│       │   │   └── Psychology.js
│       │   ├── App.js
│       │   └── index.js
│       ├── package.json
│       └── Dockerfile
├── docker-compose.yml       # Docker Compose 설정
├── .env.example             # 환경 변수 예시
├── MIND_TRADER_MASTERPLAN.md  # 프로젝트 설계 문서
└── README.md                # 이 파일
```

---

## 🎯 개발 로드맵

### Phase 1: MVP (✅ 완료)
- [x] 기본 게임 시스템 (레벨 1-5)
- [x] 3가지 AI Agent 모델 구현
- [x] 심리 스코프 기본 피드백
- [x] 실시간 주가 데이터 연동
- [x] RESTful API 기본 구축
- [x] React 프론트엔드 구현

### Phase 2: 고도화 (진행 중)
- [ ] RAG 뉴스 역추적 시뮬레이션 고도화
- [ ] AI Agent 모델 실제 데이터로 재훈련
- [ ] 협력적 경쟁 시스템 (팀 리그)
- [ ] 장기 메모리 벡터 DB 통합
- [ ] 모바일 반응형 개선

### Phase 3: 완성
- [ ] 레벨 6-15 고난이도 시나리오
- [ ] 멘토-멘티 매칭 시스템
- [ ] 모바일 앱 출시 (React Native)
- [ ] 실시간 랭킹 및 주간 챌린지
- [ ] A/B 테스트 기반 최적화

---

## 🧪 테스트

### 단위 테스트 실행

```bash
# Backend 테스트
cd backend/ai_engine
pytest tests/

# Frontend 테스트
cd frontend/mind-trader-app
npm test
```

### 통합 테스트

```bash
# 전체 스택 테스트
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

---

## 📊 모니터링

### 서비스 헬스 체크

```bash
# 모든 서비스 헬스 체크
curl http://localhost:8001/health  # Game Service
curl http://localhost:8002/health  # AI Engine
curl http://localhost:8003/health  # User Service
curl http://localhost:8004/health  # Market Data Service
curl http://localhost:8005/health  # RAG Service
```

### 로그 확인

```bash
# 특정 서비스 로그
docker-compose logs -f ai_engine

# 모든 서비스 로그
docker-compose logs -f
```

---

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

---

## 👥 제작자

- **Chief Strategy Officer**: Mind Trader Development Team

---

## 📞 문의

- 이슈 제보: [GitHub Issues](https://github.com/yourusername/mind-trader/issues)
- 이메일: contact@mindtrader.com

---

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트의 도움을 받았습니다:
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [TailwindCSS](https://tailwindcss.com/)

---

**"공포를 자신감으로, 감정을 전략으로 - Mind Trader가 함께합니다."** 🚀
