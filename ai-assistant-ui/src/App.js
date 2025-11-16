import React, { useState, useEffect } from 'react';
import './App.css';
import SyncForm from './components/SyncForm';
import QueryForm from './components/QueryForm';

function App() {
  const [activeTab, setActiveTab] = useState('query');
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="App">
      {/* Animated background */}
      <div className="background-animated">
        <div className="gradient-orb orb-1"></div>
        <div className="gradient-orb orb-2"></div>
        <div className="gradient-orb orb-3"></div>
        <div className="grid-overlay"></div>
      </div>

      {/* Cursor glow effect */}
      <div 
        className="cursor-glow" 
        style={{ 
          left: `${mousePosition.x}px`, 
          top: `${mousePosition.y}px` 
        }}
      ></div>

      <header className="App-header">
        <div className="logo-container">
          <div className="logo-bot">
            <img src="/ceira-logo.svg" alt="CEIRA Bot" className="bot-svg" />
          </div>
          <div className="logo-text-container">
            <h1 className="logo-title">
              <span className="letter-c glow">C</span>
              <span className="letter-e">E</span>
              <span className="letter-i">I</span>
              <span className="letter-r">R</span>
              <span className="letter-a glow">A</span>
              {/* CloudExtend X Logo beside CEIRA */}
              <span className="cloudextend-x-badge">
                <svg viewBox="0 0 50 50" xmlns="http://www.w3.org/2000/svg">
                  <defs>
                    <linearGradient id="xBadgeCyanGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" style={{ stopColor: '#00E5CC', stopOpacity: 1 }} />
                      <stop offset="100%" style={{ stopColor: '#00D4CC', stopOpacity: 1 }} />
                    </linearGradient>
                    <linearGradient id="xBadgeBlueGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" style={{ stopColor: '#0066FF', stopOpacity: 1 }} />
                      <stop offset="100%" style={{ stopColor: '#0052CC', stopOpacity: 1 }} />
                    </linearGradient>
                  </defs>
                  {/* Cyan stroke (top-left to bottom-right) */}
                  <path d="M 8 8 L 42 42" stroke="url(#xBadgeCyanGradient)" strokeWidth="6" strokeLinecap="round" fill="none"/>
                  {/* Blue stroke (top-right to bottom-left) */}
                  <path d="M 42 8 L 8 42" stroke="url(#xBadgeBlueGradient)" strokeWidth="6" strokeLinecap="round" fill="none"/>
                </svg>
              </span>
            </h1>
            <p className="logo-subtitle">CloudExtend Intelligent Response Assistant</p>
          </div>
        </div>
        <div className="header-stats">
          <div className="stat-item">
            <span className="stat-icon">⚡</span>
            <span className="stat-label">Ultra-Fast</span>
          </div>
          <div className="stat-item">
            <span className="stat-icon">🔒</span>
            <span className="stat-label">Secure</span>
          </div>
          <div className="stat-item">
            <span className="stat-icon">🤖</span>
            <span className="stat-label">AI-Powered</span>
          </div>
        </div>
      </header>

      <div className="tabs">
        <button 
          className={activeTab === 'query' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('query')}
        >
          <span className="tab-icon">💬</span>
          <span className="tab-text">Query CEIRA</span>
          <div className="tab-glow"></div>
        </button>
        <button 
          className={activeTab === 'sync' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('sync')}
        >
          <span className="tab-icon">🔄</span>
          <span className="tab-text">Sync Data</span>
          <div className="tab-glow"></div>
        </button>
      </div>

      <div className="content">
        {activeTab === 'query' && <QueryForm />}
        {activeTab === 'sync' && <SyncForm />}
      </div>

      <footer className="App-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <span className="footer-logo">CloudExtend</span>
            <span className="footer-separator">×</span>
            <span className="footer-tech">AWS Bedrock</span>
          </div>
          <div className="footer-version">CEIRA v1.0.0</div>
        </div>
        <div className="footer-wave"></div>
      </footer>
    </div>
  );
}

export default App;
