"""
PyTorch 기반 AI Agent 모델 구현
3가지 타입: 추종 매매 개미, 저가 매집 세력, 손절 회피 개미
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class MomentumChaserAgent(nn.Module):
    """
    추종 매매 개미 (Momentum Chaser)
    - 강한 상승세에 FOMO 발동 → 고점 매수
    - 하락세에 공포 → 저점 매도
    - 뉴스/소셜 미디어 감성에 민감
    """
    
    def __init__(self, input_size=10, hidden_size=64, num_layers=2):
        super(MomentumChaserAgent, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True
        )
        
        # 감성 점수 통합 레이어
        self.sentiment_layer = nn.Linear(1, hidden_size)
        
        # 최종 행동 결정
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # Buy/Hold/Sell
        )
        
    def forward(
        self, 
        price_history: torch.Tensor,  # (batch, seq_len, features)
        sentiment_score: torch.Tensor,  # (batch, 1)
        volume_trend: torch.Tensor  # (batch, seq_len, 1)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            price_history: 최근 20봉 가격 데이터 (정규화된 변화율)
            sentiment_score: 뉴스 감성 점수 (-1 ~ 1)
            volume_trend: 거래량 추세
            
        Returns:
            action_prob: Buy/Hold/Sell 확률 분포
            confidence: 행동에 대한 확신도
        """
        # 가격과 거래량 결합
        combined_input = torch.cat([price_history, volume_trend], dim=-1)
        
        # LSTM으로 시계열 패턴 학습
        lstm_out, (hidden, cell) = self.lstm(combined_input)
        
        # Attention: 최근 5봉에 가중치 집중
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 최근 시점 특징 추출
        recent_features = attn_out[:, -1, :]  # (batch, hidden_size)
        
        # 감성 점수 임베딩
        sentiment_emb = self.sentiment_layer(sentiment_score)
        sentiment_emb = F.relu(sentiment_emb)
        
        # 감성 점수 반영: 긍정 뉴스 + 상승 → 강한 매수 확률
        # FOMO 효과: sentiment_score가 높으면 매수 확률 증폭
        fomo_amplifier = 1.0 + torch.clamp(sentiment_score * 0.5, -0.5, 0.5)
        amplified_features = recent_features * fomo_amplifier
        
        # 최종 결합
        final_features = torch.cat([amplified_features, sentiment_emb], dim=-1)
        
        # 행동 확률 예측
        logits = self.fc(final_features)
        action_prob = F.softmax(logits, dim=-1)
        
        # 확신도: 최대 확률값
        confidence = torch.max(action_prob, dim=-1)[0]
        
        return {
            'action_prob': action_prob,  # [buy_prob, hold_prob, sell_prob]
            'confidence': confidence,
            'attention_weights': attn_weights
        }
    
    def calculate_market_impact(
        self,
        action_prob: torch.Tensor,
        num_agents: int,
        avg_volume: int,
        liquidity_coefficient: float = 0.3
    ) -> float:
        """
        시장 가격에 미치는 영향 계산
        
        Args:
            action_prob: Buy/Hold/Sell 확률
            num_agents: 이 타입의 Agent 수
            avg_volume: 평균 거래량
            liquidity_coefficient: 시장 영향 계수 (작을수록 영향 큼)
        """
        buy_prob = action_prob[0].item()
        sell_prob = action_prob[2].item()
        
        # 순 매수/매도 압력
        net_pressure = (buy_prob - sell_prob) * num_agents * avg_volume
        
        # 가격 변동 (%)
        price_impact = net_pressure * liquidity_coefficient / 10000
        
        return price_impact


class SmartMoneyAgent(nn.Module):
    """
    저가 매집 세력 (Smart Money Accumulator)
    - 공포 시점에 분산 매수 (contrarian)
    - 장기 목표 가격 설정 후 체계적 접근
    - 거래량 분석을 통한 타이밍 포착
    """
    
    def __init__(self, input_size=15, hidden_size=128, num_layers=3):
        super(SmartMoneyAgent, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 내재 가치 추정 네트워크
        self.value_estimator = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 포지션 크기 결정 네트워크
        self.position_manager = nn.Sequential(
            nn.Linear(hidden_size + 2, 64),  # +2 for undervaluation & fear_index
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 사이 포지션 비율
        )
        
    def forward(
        self,
        price_history: torch.Tensor,  # (batch, seq_len, features)
        volume_profile: torch.Tensor,  # (batch, seq_len, features)
        fear_greed_index: torch.Tensor  # (batch, 1) 0-100
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            price_history: 가격 이력
            volume_profile: 거래량 프로파일
            fear_greed_index: 공포-탐욕 지수 (0=극도 공포, 100=극도 탐욕)
        """
        # 가격과 거래량 결합
        combined = torch.cat([price_history, volume_profile], dim=-1)
        
        # GRU로 장기 패턴 학습
        gru_out, hidden = self.gru(combined)
        
        # 마지막 hidden state 사용
        last_hidden = gru_out[:, -1, :]  # (batch, hidden_size)
        
        # 내재 가치 추정 (장기 이동평균 기반)
        intrinsic_value = self.value_estimator(last_hidden)
        
        # 현재 가격
        current_price = price_history[:, -1, 0:1]  # (batch, 1)
        
        # 저평가 정도 계산
        undervaluation = (intrinsic_value - current_price) / (current_price + 1e-8)
        
        # 공포 지수 정규화 (0-1)
        fear_normalized = fear_greed_index / 100.0
        
        # 역발상 신호: 공포 지수가 낮을수록 (공포 많을수록) 매수 의지 증가
        contrarian_signal = 1.0 - fear_normalized
        
        # 포지션 결정 입력
        position_input = torch.cat([
            last_hidden,
            undervaluation,
            contrarian_signal
        ], dim=-1)
        
        # 포지션 크기 결정 (0-1)
        position_size = self.position_manager(position_input)
        
        # 매수 조건: 저평가 + 공포 시점
        buy_threshold = 0.1  # 10% 이상 저평가
        fear_threshold = 0.3  # 공포 지수 30 이하
        
        should_buy = (undervaluation > buy_threshold) & (fear_normalized < fear_threshold)
        
        # 행동 결정
        action = torch.where(should_buy, torch.ones_like(position_size), torch.zeros_like(position_size))
        
        return {
            'action': action,  # 1=buy, 0=hold
            'position_size': position_size,
            'intrinsic_value': intrinsic_value,
            'undervaluation': undervaluation,
            'contrarian_signal': contrarian_signal
        }
    
    def calculate_support_level(
        self,
        position_size: float,
        num_agents: int,
        avg_volume: int,
        avg_holding_period: int = 30  # days
    ) -> Dict[str, float]:
        """
        지지선 형성 계산
        
        Returns:
            support_strength: 지지선 강도
            accumulated_volume: 누적 매수량
        """
        accumulated_volume = position_size * num_agents * avg_volume
        support_strength = accumulated_volume * avg_holding_period / 1000
        
        return {
            'support_strength': support_strength,
            'accumulated_volume': accumulated_volume,
            'stabilization_effect': min(support_strength * 0.01, 5.0)  # 최대 5% 안정화
        }


class LossAversionAgent(nn.Module):
    """
    손절 회피 개미 (Loss Aversion Trader)
    - 수익 나면 빨리 실현 (5% 익절)
    - 손실 나면 계속 보유 (손절 거부)
    - 물타기 전략 선호
    """
    
    def __init__(self, d_model=32, nhead=4, num_layers=2):
        super(LossAversionAgent, self).__init__()
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 포트폴리오 상태 임베딩
        self.portfolio_embedding = nn.Linear(5, d_model)  # 보유 수량, 평균 매수가, 현재가, PnL, 보유 기간
        
        # 손실 회피 계수 (학습 가능한 파라미터)
        # Prospect Theory에 따르면 손실이 이익보다 2-2.5배 크게 느껴짐
        self.loss_aversion_coef = nn.Parameter(torch.tensor(2.5))
        
        # 행동 결정 네트워크
        self.decision_net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # sell_prob, hold_prob, avg_down_prob
        )
        
    def forward(
        self,
        portfolio_state: torch.Tensor,  # (batch, features) [qty, avg_price, current_price, pnl, hold_days]
        unrealized_pnl: torch.Tensor  # (batch, 1) 미실현 손익률
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            portfolio_state: 포트폴리오 상태
            unrealized_pnl: 미실현 손익률 (-1 ~ 1)
        """
        # 포트폴리오 임베딩
        portfolio_emb = self.portfolio_embedding(portfolio_state)
        portfolio_emb = portfolio_emb.unsqueeze(1)  # (batch, 1, d_model)
        
        # Transformer 인코딩
        encoded = self.encoder(portfolio_emb)
        encoded = encoded.squeeze(1)  # (batch, d_model)
        
        # 행동 확률 예측
        logits = self.decision_net(encoded)
        
        # 손익률에 따른 행동 조정
        pnl_value = unrealized_pnl.squeeze(-1)
        
        # 조건별 확률 조정
        sell_base = torch.sigmoid(logits[:, 0])
        hold_base = torch.sigmoid(logits[:, 1])
        avg_down_base = torch.sigmoid(logits[:, 2])
        
        # 이익 상황 (5% 이상): 빠른 익절
        profit_mask = pnl_value > 0.05
        sell_prob = torch.where(
            profit_mask,
            torch.sigmoid(pnl_value * 10),  # 이익 클수록 매도 확률 증가
            torch.sigmoid(pnl_value / self.loss_aversion_coef)  # 손실은 회피
        )
        
        # 손실 상황 (-3% 이하): 손절 회피 + 물타기
        loss_mask = pnl_value < -0.03
        avg_down_prob = torch.where(
            loss_mask,
            torch.sigmoid(-pnl_value * 5),  # 손실 클수록 물타기 확률 증가
            torch.zeros_like(pnl_value)
        )
        
        # 홀딩 확률
        hold_prob = 1.0 - sell_prob - avg_down_prob
        hold_prob = torch.clamp(hold_prob, 0.0, 1.0)
        
        # 확률 정규화
        total = sell_prob + hold_prob + avg_down_prob + 1e-8
        sell_prob = sell_prob / total
        hold_prob = hold_prob / total
        avg_down_prob = avg_down_prob / total
        
        return {
            'sell_prob': sell_prob,
            'hold_prob': hold_prob,
            'avg_down_prob': avg_down_prob,
            'loss_aversion_coef': self.loss_aversion_coef.item()
        }
    
    def calculate_market_impact(
        self,
        sell_prob: float,
        avg_down_prob: float,
        num_agents: int,
        avg_volume: int,
        current_trend: str  # 'up' or 'down'
    ) -> Dict[str, float]:
        """
        시장 영향 계산
        """
        if current_trend == 'up':
            # 상승장: 조기 익절 → 상승 동력 약화
            selling_pressure = sell_prob * num_agents * avg_volume
            price_impact = -selling_pressure * 0.5 / 10000
        else:
            # 하락장: 손절 지연 + 물타기 → 일시 반등 후 재하락
            buying_pressure = avg_down_prob * num_agents * avg_volume * 0.3  # 물타기는 작은 규모
            selling_pressure = sell_prob * num_agents * avg_volume * 0.2  # 손절은 미미
            
            # 단기 반등
            short_term_impact = (buying_pressure - selling_pressure) * 0.5 / 10000
            # 장기 하락 압력
            long_term_pressure = -num_agents * avg_volume * 0.1 / 10000
            
            price_impact = short_term_impact
            
        return {
            'immediate_impact': price_impact,
            'selling_pressure': sell_prob * num_agents * avg_volume,
            'buying_pressure': avg_down_prob * num_agents * avg_volume
        }


class MultiAgentSimulator:
    """
    여러 Agent를 통합하여 시장 시뮬레이션 수행
    """
    
    def __init__(
        self,
        momentum_model: MomentumChaserAgent,
        smart_money_model: SmartMoneyAgent,
        loss_aversion_model: LossAversionAgent,
        device: str = 'cpu'
    ):
        self.momentum_model = momentum_model.to(device)
        self.smart_money_model = smart_money_model.to(device)
        self.loss_aversion_model = loss_aversion_model.to(device)
        self.device = device
        
    def simulate_market_step(
        self,
        price_history: np.ndarray,
        volume_history: np.ndarray,
        news_sentiment: float,
        fear_greed_index: float,
        agent_counts: Dict[str, int],
        current_price: float
    ) -> Dict[str, any]:
        """
        한 스텝의 시장 시뮬레이션
        
        Returns:
            predicted_price: 예측 가격
            agent_actions: 각 Agent의 행동
            net_pressure: 순 매수/매도 압력
        """
        # NumPy → PyTorch 변환
        price_tensor = torch.FloatTensor(price_history).unsqueeze(0).to(self.device)
        volume_tensor = torch.FloatTensor(volume_history).unsqueeze(0).to(self.device)
        sentiment_tensor = torch.FloatTensor([[news_sentiment]]).to(self.device)
        fear_tensor = torch.FloatTensor([[fear_greed_index]]).to(self.device)
        
        with torch.no_grad():
            # 1. 추종 개미 행동
            momentum_out = self.momentum_model(
                price_tensor,
                sentiment_tensor,
                volume_tensor.unsqueeze(-1)
            )
            momentum_impact = self.momentum_model.calculate_market_impact(
                momentum_out['action_prob'][0],
                agent_counts['momentum_chaser'],
                int(volume_history[-1, 0])
            )
            
            # 2. 세력 행동
            smart_out = self.smart_money_model(
                price_tensor,
                volume_tensor,
                fear_tensor
            )
            smart_support = self.smart_money_model.calculate_support_level(
                smart_out['position_size'][0].item(),
                agent_counts['smart_money'],
                int(volume_history[-1, 0])
            )
            
            # 3. 손절 회피 개미 (PnL 가정: -5% ~ +10% 랜덤)
            pnl_sample = torch.FloatTensor([[np.random.uniform(-0.05, 0.10)]]).to(self.device)
            portfolio_sample = torch.FloatTensor([[100, current_price*0.95, current_price, pnl_sample.item(), 10]]).to(self.device)
            
            loss_aversion_out = self.loss_aversion_model(portfolio_sample, pnl_sample)
            
            # 종합 가격 영향
            total_impact = (
                momentum_impact +
                smart_support.get('stabilization_effect', 0) * 0.5
            )
            
            predicted_price = current_price * (1 + total_impact / 100)
            
        return {
            'predicted_price': predicted_price,
            'price_change_pct': total_impact,
            'agent_actions': {
                'momentum_chaser': {
                    'buy_ratio': momentum_out['action_prob'][0][0].item(),
                    'hold_ratio': momentum_out['action_prob'][0][1].item(),
                    'sell_ratio': momentum_out['action_prob'][0][2].item(),
                    'confidence': momentum_out['confidence'][0].item()
                },
                'smart_money': {
                    'action': 'buy' if smart_out['action'][0].item() > 0.5 else 'hold',
                    'position_size': smart_out['position_size'][0].item(),
                    'undervaluation': smart_out['undervaluation'][0].item()
                },
                'loss_aversion': {
                    'sell_prob': loss_aversion_out['sell_prob'][0].item(),
                    'hold_prob': loss_aversion_out['hold_prob'][0].item(),
                    'avg_down_prob': loss_aversion_out['avg_down_prob'][0].item()
                }
            },
            'market_impact': total_impact
        }
