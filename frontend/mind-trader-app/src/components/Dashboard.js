import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

function Dashboard({ user, onLogout }) {
  const [progress, setProgress] = useState(null);
  const [levels, setLevels] = useState([]);

  useEffect(() => {
    if (user) {
      fetchGameState();
      fetchLevels();
    }
  }, [user]);

  const fetchGameState = async () => {
    try {
      const response = await fetch(
        `${process.env.REACT_APP_GAME_SERVICE_URL || 'http://localhost:8001'}/api/v1/game/state/${user.id}`
      );
      if (response.ok) {
        const data = await response.json();
        setProgress(data);
      }
    } catch (error) {
      console.error('Failed to fetch game state:', error);
    }
  };

  const fetchLevels = async () => {
    try {
      const response = await fetch(
        `${process.env.REACT_APP_GAME_SERVICE_URL || 'http://localhost:8001'}/api/v1/game/levels`
      );
      if (response.ok) {
        const data = await response.json();
        setLevels(data);
      }
    } catch (error) {
      console.error('Failed to fetch levels:', error);
    }
  };

  if (!user) return null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500">
      {/* 헤더 */}
      <nav className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">🧠 Mind Trader</h1>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-gray-700">환영합니다, {user.username}님!</span>
              <button
                onClick={onLogout}
                className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-md"
              >
                로그아웃
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* 메인 컨텐츠 */}
      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        {/* 사용자 통계 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-xl p-6">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">현재 레벨</h3>
            <p className="text-4xl font-bold text-indigo-600">
              {progress?.current_level || user.level || 1}
            </p>
          </div>
          <div className="bg-white rounded-lg shadow-xl p-6">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">학습 포인트</h3>
            <p className="text-4xl font-bold text-green-600">
              {progress?.balance?.toLocaleString() || user.learning_points?.toLocaleString() || '10,000'}
            </p>
          </div>
          <div className="bg-white rounded-lg shadow-xl p-6">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">총 손익</h3>
            <p className={`text-4xl font-bold ${progress?.total_profit_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {progress?.total_profit_loss >= 0 ? '+' : ''}{progress?.total_profit_loss?.toLocaleString() || '0'}
            </p>
          </div>
        </div>

        {/* 레벨 목록 */}
        <div className="bg-white rounded-lg shadow-xl p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">게임 레벨</h2>
          <div className="space-y-4">
            {levels.map((level) => (
              <div
                key={level.level}
                className={`p-4 rounded-lg border-2 ${
                  progress?.current_level >= level.level
                    ? 'border-green-500 bg-green-50'
                    : 'border-gray-300 bg-gray-50'
                }`}
              >
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="text-lg font-semibold">
                      레벨 {level.level}: {level.name}
                    </h3>
                    <p className="text-sm text-gray-600">{level.description}</p>
                    <span className={`inline-block mt-2 px-3 py-1 rounded-full text-xs font-semibold ${
                      level.difficulty === 'easy' ? 'bg-green-200 text-green-800' :
                      level.difficulty === 'medium' ? 'bg-yellow-200 text-yellow-800' :
                      'bg-red-200 text-red-800'
                    }`}>
                      {level.difficulty === 'easy' ? '초급' : level.difficulty === 'medium' ? '중급' : '고급'}
                    </span>
                  </div>
                  <div>
                    {progress?.current_level >= level.level ? (
                      <span className="text-green-600 font-semibold">✓ 완료</span>
                    ) : (
                      <span className="text-gray-500">필요 포인트: {level.unlock_points_required}</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 액션 버튼 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Link
            to="/game"
            className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white p-8 rounded-lg shadow-xl hover:shadow-2xl transition-shadow text-center"
          >
            <h3 className="text-2xl font-bold mb-2">🎮 게임 시작</h3>
            <p className="text-blue-100">시뮬레이션에서 실전 투자 연습하기</p>
          </Link>
          <Link
            to="/psychology"
            className="bg-gradient-to-r from-purple-500 to-pink-600 text-white p-8 rounded-lg shadow-xl hover:shadow-2xl transition-shadow text-center"
          >
            <h3 className="text-2xl font-bold mb-2">🧠 심리 분석</h3>
            <p className="text-purple-100">나의 투자 심리 패턴 분석받기</p>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
