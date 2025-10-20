# 🚀 Mind Trader 빠른 시작 가이드

## 1분 안에 시작하기

### 1단계: Docker Compose로 전체 시스템 실행

```bash
# 프로젝트 디렉토리로 이동
cd /workspace

# 모든 서비스 시작 (최초 실행 시 5-10분 소요)
docker-compose up -d

# 서비스 시작 확인
docker-compose ps
```

### 2단계: 웹 브라우저에서 접속

http://localhost:3000 으로 접속하세요!

### 3단계: 회원가입 및 로그인

1. 회원가입 페이지에서 계정 생성
   - 사용자 이름: `testuser`
   - 이메일: `test@example.com`
   - 비밀번호: `password123`

2. 로그인 후 대시보드 확인

### 4단계: 게임 시작

1. 대시보드에서 "🎮 게임 시작" 클릭
2. 시나리오 자동 생성 (레벨 1부터 시작)
3. 가격 차트 확인 및 매매 시도
4. 감정 상태 선택 후 매수/매도 실행

### 5단계: 심리 분석 확인

1. 최소 3-5회 거래 후
2. "🧠 심리 분석" 메뉴 클릭
3. FOMO 지수, 손절 지연 지수 등 확인
4. AI 멘토의 피드백 확인

---

## 🔍 문제 해결

### 서비스가 시작되지 않는 경우

```bash
# 로그 확인
docker-compose logs -f

# 특정 서비스 재시작
docker-compose restart ai_engine

# 전체 재시작
docker-compose down
docker-compose up -d
```

### 포트 충돌 발생 시

`docker-compose.yml` 파일에서 포트 변경:

```yaml
services:
  game_service:
    ports:
      - "8081:8000"  # 8001 → 8081로 변경
```

### 데이터베이스 초기화

```bash
# PostgreSQL 컨테이너 접속
docker-compose exec postgres psql -U postgres -d mind_trader

# 테이블 확인
\dt

# 종료
\q
```

---

## 📱 주요 기능 테스트

### AI 시뮬레이션 테스트

```bash
curl -X POST "http://localhost:8002/api/v1/ai/simulate_market_impact" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SAMSUNG",
    "current_price": 75000,
    "recent_candles": [],
    "simulation_duration": 30
  }'
```

### 사용자 등록 테스트

```bash
curl -X POST "http://localhost:8003/api/v1/users/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser2",
    "email": "test2@example.com",
    "password": "password123"
  }'
```

---

## 🎮 게임 플레이 팁

1. **레벨 1-2**: 안전지대 - 성공 경험 쌓기
   - 승률 80% 이상의 간단한 패턴
   - 실수해도 괜찮습니다!

2. **레벨 3-4**: 도전지대 - 심리 훈련
   - FOMO 유혹, 손절 회피 극복
   - 감정 체크가 중요합니다

3. **레벨 5**: 현실 시장 - 실전 연습
   - 실제 시장 수준의 불확실성
   - 리스크 관리가 핵심!

---

## 🛠 개발자 모드

### 로컬에서 개발하기

```bash
# Backend - AI Engine
cd backend/ai_engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8002

# Frontend
cd frontend/mind-trader-app
npm install
npm start
```

### API 문서 확인

- AI Engine: http://localhost:8002/docs
- User Service: http://localhost:8003/docs
- Game Service: http://localhost:8001/docs

---

## 📊 모니터링

### RabbitMQ 관리 콘솔

http://localhost:15672
- Username: `admin`
- Password: `admin123`

### Redis CLI 접속

```bash
docker-compose exec redis redis-cli
> KEYS *
> GET "user:*"
```

---

## 🆘 도움말

문제가 발생하면:
1. `docker-compose logs -f` 로그 확인
2. GitHub Issues에 문의
3. README.md 상세 문서 참고

---

**즐거운 게임 되세요! 🎉**
