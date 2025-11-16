import React, { useState } from 'react';
import './QueryForm.css';

function QueryForm() {
  const [question, setQuestion] = useState('');
  const [userRole, setUserRole] = useState('developer');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleQuery = async () => {
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const requestBody = {
      question: question,
      user_role: userRole
    };

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();
      
      if (response.ok) {
        setResult(data);
      } else {
        setError(data.detail || 'Query failed');
      }
    } catch (err) {
      setError('Failed to connect to server: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleQuery();
    }
  };

  return (
    <div className="query-form">
      <div className="form-header">
        <h2>
          <span className="header-icon">🧠</span>
          Query CEIRA
        </h2>
        <p className="form-subtitle">Ask anything about your organization's knowledge base</p>
      </div>

      <div className="form-section">
        <label className="section-label">
          <span className="label-icon">👤</span>
          Select Your Role
        </label>
        <select 
          value={userRole} 
          onChange={(e) => setUserRole(e.target.value)}
          className="select-input futuristic-select"
        >
          <option value="developer">🔧 Developer</option>
          <option value="support">🎧 Support Engineer</option>
          <option value="manager">📊 Manager</option>
          <option value="general">🌐 General User</option>
        </select>
      </div>

      <div className="form-section">
        <label className="section-label">
          <span className="label-icon">💭</span>
          Your Question
        </label>
        <div className="textarea-container">
          <textarea
            placeholder="Type your question here... (Ctrl+Enter to submit)"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyPress={handleKeyPress}
            className="textarea-input question-textarea futuristic-textarea"
            rows="6"
          />
          <div className="textarea-glow"></div>
        </div>
      </div>

      <button 
        className={`query-button ${loading ? 'loading' : ''}`}
        onClick={handleQuery}
        disabled={loading}
      >
        <span className="button-content">
          {loading ? (
            <>
              <span className="spinner"></span>
              <span>CEIRA is thinking...</span>
            </>
          ) : (
            <>
              <span className="button-icon">⚡</span>
              <span>Ask CEIRA</span>
            </>
          )}
        </span>
        <div className="button-glow"></div>
      </button>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {result && (
        <div className="result-box">
          <div className="answer-section">
            <h3>
              <span className="section-icon">🤖</span>
              CEIRA's Response
              <span className="ai-badge">AI</span>
            </h3>
            <div className="answer-content">
              {result.answer}
            </div>
          </div>

          <div className="metadata-section">
            <div className="metadata-item">
              <span className="metadata-label">⚡ Processing Time:</span>
              <span className="metadata-value">{result.processing_time_seconds?.toFixed(2)}s</span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">📊 Confidence:</span>
              <span className="metadata-value">{Math.abs(result.confidence_score)?.toFixed(2)}</span>
            </div>
          </div>

          {result.sources && result.sources.length > 0 && (
            <div className="sources-section">
              <h3>
                <span className="section-icon">📚</span>
                Knowledge Sources ({result.sources.length})
              </h3>
              <div className="sources-list">
                {result.sources.map((source, index) => (
                  <div key={index} className="source-item">
                    <div className="source-header">
                      <span className="source-type">
                        {source.type === 'github' && '💻'}
                        {source.type === 'confluence' && '📄'}
                        {source.type === 'jira' && '🎫'}
                        {' ' + source.type}
                      </span>
                      <span className="source-similarity">
                        Match: {Math.abs(source.similarity_score)?.toFixed(2)}
                      </span>
                    </div>
                    <div className="source-title">
                      {source.title}
                    </div>
                    {source.url && (
                      <a 
                        href={source.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="source-link"
                      >
                        <span>🔗</span>
                        <span>View Source</span>
                      </a>
                    )}
                    {source.repository && (
                      <div className="source-meta">
                        <span className="meta-icon">📦</span>
                        Repository: {source.repository}
                      </div>
                    )}
                    {source.file_path && (
                      <div className="source-meta">
                        <span className="meta-icon">📁</span>
                        Path: {source.file_path}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.suggested_actions && result.suggested_actions.length > 0 && (
            <div className="suggestions-section">
              <h3>
                <span className="section-icon">💡</span>
                Suggested Actions
              </h3>
              <ul className="suggestions-list">
                {result.suggested_actions.map((action, index) => (
                  <li key={index}>{action}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default QueryForm;
