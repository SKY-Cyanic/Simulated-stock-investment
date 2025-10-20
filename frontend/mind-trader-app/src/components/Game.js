import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function Game({ user, token }) {
  const [scenario, setScenario] = useState(null);
  const [loading, setLoading] = useState(false);
  const [position, setPosition] = useState(null); // { quantity, avgPrice }
  const [balance, setBalance] = useState(10000);
  const [emotion, setEmotion] = useState('확신');

  useEffect(() => {
    startScenario();
  }, []);

  const startScenario = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `${process.env.REACT_APP_GAME_SERVICE_URL || 'http://localhost:8001'}/api/v1/game/start_scenario/${user.id}`,
        {
          method: 'POST',
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        setScenario(data);
        setBalance(data.starting_balance);
      }
    } catch (error) {
      console.error('Failed to start scenario:', error);
    } finally {
      setLoading(false);
    }
  };

  const executeTrade = async (action) => {
    if (!scenario) return;

    const currentPrice = scenario.market_data.current_price;
    const quantity = 10; // 고정 수량

    try {
      // 거래 기록
      await fetch(
        `${process.env.REACT_APP_USER_SERVICE_URL || 'http://localhost:8003'}/api/v1/users/trades`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({
            symbol: scenario.symbol,
            action: action,
            price: currentPrice,
            quantity: quantity,
            emotion: emotion,
            reason: ''
          })
        }
      );

      // 포지션 업데이트
      if (action === 'buy') {
        const cost = currentPrice * quantity;
        if (balance >= cost) {
          setBalance(balance - cost);
          if (position) {
            const totalQty = position.quantity + quantity;
            const newAvgPrice = (position.avgPrice * position.quantity + currentPrice * quantity) / totalQty;
            setPosition({ quantity: totalQty, avgPrice: newAvgPrice });
          } else {
            setPosition({ quantity, avgPrice: currentPrice });
          }
          alert(`✅ 매수 완료! ${quantity}주를 ${currentPrice.toFixed(0)}원에 매수했습니다.`);
        } else {
          alert('❌ 잔액이 부족합니다!');
        }
      } else if (action === 'sell') {
        if (position && position.quantity >= quantity) {
          const revenue = currentPrice * quantity;
          setBalance(balance + revenue);
          const pnl = ((currentPrice - position.avgPrice) / position.avgPrice * 100).toFixed(2);
          setPosition({
            quantity: position.quantity - quantity,
            avgPrice: position.avgPrice
          });
          alert(`✅ 매도 완료! ${quantity}주를 ${currentPrice.toFixed(0)}원에 매도했습니다. 손익: ${pnl}%`);
        } else {
          alert('❌ 보유 수량이 부족합니다!');
        }
      }
    } catch (error) {
      console.error('Trade execution failed:', error);
      alert('❌ 거래 실행에 실패했습니다.');
    }
  };

  if (loading || !scenario) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-white text-2xl">시나리오 로딩 중...</div>
      </div>
    );
  }

  // 차트 데이터 변환
  const chartData = scenario.market_data.candles.map((candle, idx) => ({
    time: idx,
    price: candle.close
  }));

  const unrealizedPnL = position
    ? ((scenario.market_data.current_price - position.avgPrice) / position.avgPrice * 100).toFixed(2)
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 text-white">
      <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        {/* 헤더 */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold">🎮 투자 시뮬레이션</h1>
            <p className="text-gray-300 mt-2">{scenario.level_name} - {scenario.difficulty}</p>
          </div>
          <Link
            to="/dashboard"
            className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-md"
          >
            ← 대시보드
          </Link>
        </div>

        {/* 힌트 */}
        <div className="bg-blue-900 bg-opacity-50 p-4 rounded-lg mb-6">
          <h3 className="font-semibold mb-2">💡 힌트</h3>
          <ul className="space-y-1">
            {scenario.hints.map((hint, idx) => (
              <li key={idx} className="text-sm text-blue-200">{hint}</li>
            ))}
          </ul>
        </div>

        {/* 상태 정보 */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white bg-opacity-10 p-4 rounded-lg">
            <p className="text-gray-300 text-sm">잔액</p>
            <p className="text-2xl font-bold">{balance.toLocaleString()}원</p>
          </div>
          <div className="bg-white bg-opacity-10 p-4 rounded-lg">
            <p className="text-gray-300 text-sm">현재가</p>
            <p className="text-2xl font-bold">{scenario.market_data.current_price.toFixed(0)}원</p>
          </div>
          <div className="bg-white bg-opacity-10 p-4 rounded-lg">
            <p className="text-gray-300 text-sm">보유 수량</p>
            <p className="text-2xl font-bold">{position?.quantity || 0}주</p>
          </div>
          <div className={`bg-white bg-opacity-10 p-4 rounded-lg`}>
            <p className="text-gray-300 text-sm">미실현 손익</p>
            <p className={`text-2xl font-bold ${unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {unrealizedPnL >= 0 ? '+' : ''}{unrealizedPnL}%
            </p>
          </div>
        </div>

        {/* 차트 */}
        <div className="bg-white bg-opacity-10 p-6 rounded-lg mb-6">
          <h3 className="text-xl font-semibold mb-4">가격 차트</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis dataKey="time" stroke="#999" />
              <YAxis stroke="#999" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#333', border: 'none' }}
                labelStyle={{ color: '#fff' }}
              />
              <Legend />
              <Line type="monotone" dataKey="price" stroke="#8884d8" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* 거래 패널 */}
        <div className="bg-white bg-opacity-10 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4">거래 실행</h3>
          
          <div className="mb-4">
            <label className="block text-sm mb-2">감정 상태 (매매 전 감정 체크)</label>
            <select
              value={emotion}
              onChange={(e) => setEmotion(e.target.value)}
              className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2"
            >
              <option value="확신">확신</option>
              <option value="욕심">욕심</option>
              <option value="두려움">두려움</option>
              <option value="냉정">냉정</option>
            </select>
          </div>

          <div className="flex gap-4">
            <button
              onClick={() => executeTrade('buy')}
              className="flex-1 bg-blue-600 hover:bg-blue-700 py-3 rounded-lg font-semibold"
            >
              📈 매수 (10주)
            </button>
            <button
              onClick={() => executeTrade('sell')}
              className="flex-1 bg-red-600 hover:bg-red-700 py-3 rounded-lg font-semibold"
              disabled={!position || position.quantity === 0}
            >
              📉 매도 (10주)
            </button>
          </div>

          <div className="mt-4 text-sm text-gray-300">
            <p>💡 감정 체크를 통해 나의 투자 심리 패턴을 분석합니다</p>
            <p>💡 목표: +{scenario.success_criteria.target_profit_rate * 100}% 수익 달성</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Game;
