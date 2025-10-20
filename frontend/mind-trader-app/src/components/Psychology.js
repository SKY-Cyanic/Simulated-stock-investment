import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

function Psychology({ user, token }) {
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchFeedback();
  }, []);

  const fetchFeedback = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `${process.env.REACT_APP_USER_SERVICE_URL || 'http://localhost:8003'}/api/v1/ai/agent_feedback/${user.id}?trades_count=10`
      );
      
      if (response.ok) {
        const data = await response.json();
        setFeedback(data);
      }
    } catch (error) {
      console.error('Failed to fetch feedback:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !feedback) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-900 to-indigo-900">
        <div className="text-white text-2xl">심리 분석 중...</div>
      </div>
    );
  }

  const { psychology_scores, feedback: feedbackContent, historical_trend } = feedback;

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-indigo-900 to-blue-900 text-white">
      <div className="max-w-6xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        {/* 헤더 */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">🧠 투자 심리 분석 리포트</h1>
          <Link
            to="/dashboard"
            className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-md"
          >
            ← 대시보드
          </Link>
        </div>

        {/* 심리 지표 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <ScoreCard
            title="FOMO 지수"
            score={psychology_scores.fomo_index}
            description="공포와 욕심에 의한 충동 매매 경향"
          />
          <ScoreCard
            title="손절 지연 지수"
            score={psychology_scores.loss_cut_delay_index}
            description="손실 발생 시 손절 타이밍 지연"
          />
          <ScoreCard
            title="충동 매매 지수"
            score={psychology_scores.impulsive_trading_index}
            description="단기간 내 반복 거래 빈도"
          />
          <ScoreCard
            title="리스크 관리 점수"
            score={psychology_scores.risk_management_score}
            description="전체적인 리스크 관리 능력"
            isPositive={true}
          />
        </div>

        {/* 피드백 */}
        <div className="bg-white bg-opacity-10 p-6 rounded-lg mb-6">
          <h2 className="text-2xl font-bold mb-4">📊 분석 요약</h2>
          <p className="text-lg text-gray-200">{feedbackContent.summary}</p>
        </div>

        {/* 강점 */}
        {feedbackContent.strengths.length > 0 && (
          <div className="bg-green-900 bg-opacity-30 p-6 rounded-lg mb-6">
            <h3 className="text-xl font-semibold mb-3 text-green-300">😊 잘한 점</h3>
            <ul className="space-y-2">
              {feedbackContent.strengths.map((strength, idx) => (
                <li key={idx} className="flex items-start">
                  <span className="text-green-400 mr-2">✓</span>
                  <span>{strength}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* 약점 */}
        {feedbackContent.weaknesses.length > 0 && (
          <div className="bg-yellow-900 bg-opacity-30 p-6 rounded-lg mb-6">
            <h3 className="text-xl font-semibold mb-3 text-yellow-300">⚠️ 개선 포인트</h3>
            <ul className="space-y-2">
              {feedbackContent.weaknesses.map((weakness, idx) => (
                <li key={idx} className="flex items-start">
                  <span className="text-yellow-400 mr-2">!</span>
                  <span>{weakness}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* 실천 과제 */}
        <div className="bg-blue-900 bg-opacity-30 p-6 rounded-lg mb-6">
          <h3 className="text-xl font-semibold mb-3 text-blue-300">📋 이번 주 실천 과제</h3>
          <ol className="space-y-2 list-decimal list-inside">
            {feedbackContent.action_items.map((item, idx) => (
              <li key={idx} className="text-blue-100">{item}</li>
            ))}
          </ol>
        </div>

        {/* 다음 주 목표 */}
        <div className="bg-purple-900 bg-opacity-30 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-2 text-purple-300">🎯 다음 주 목표</h3>
          <p className="text-lg font-semibold text-purple-100">{feedbackContent.next_week_goal}</p>
          <p className="text-sm text-purple-200 mt-2">할 수 있어요! 💪</p>
        </div>

        {/* 개선율 */}
        {historical_trend && historical_trend.improvement_rate > 0 && (
          <div className="mt-6 bg-green-900 bg-opacity-20 p-4 rounded-lg text-center">
            <p className="text-sm text-green-300">지난 4주 대비 개선율</p>
            <p className="text-3xl font-bold text-green-400">+{historical_trend.improvement_rate}%</p>
          </div>
        )}
      </div>
    </div>
  );
}

function ScoreCard({ title, score, description, isPositive = false }) {
  const getColor = () => {
    if (isPositive) {
      return score >= 70 ? 'bg-green-600' : score >= 40 ? 'bg-yellow-600' : 'bg-red-600';
    } else {
      return score >= 70 ? 'bg-red-600' : score >= 40 ? 'bg-yellow-600' : 'bg-green-600';
    }
  };

  return (
    <div className="bg-white bg-opacity-10 p-4 rounded-lg">
      <h3 className="text-sm font-semibold mb-2">{title}</h3>
      <div className="flex items-end mb-2">
        <span className="text-4xl font-bold">{score}</span>
        <span className="text-gray-400 ml-1">/100</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
        <div
          className={`h-2 rounded-full ${getColor()}`}
          style={{ width: `${score}%` }}
        ></div>
      </div>
      <p className="text-xs text-gray-400">{description}</p>
    </div>
  );
}

export default Psychology;
