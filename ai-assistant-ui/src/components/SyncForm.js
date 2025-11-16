import React, { useState } from 'react';
import './SyncForm.css';

function SyncForm() {
  const [sources, setSources] = useState({
    github: false,
    confluence: false,
    jira: false
  });
  const [repositories, setRepositories] = useState('');
  const [spaces, setSpaces] = useState('');
  const [includePaths, setIncludePaths] = useState('');
  const [excludePaths, setExcludePaths] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSourceChange = (source) => {
    setSources(prev => ({ ...prev, [source]: !prev[source] }));
  };

  const handleSync = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    // Build selected sources array
    const selectedSources = Object.keys(sources).filter(key => sources[key]);

    if (selectedSources.length === 0) {
      setError('Please select at least one source');
      setLoading(false);
      return;
    }

    // Build request body
    const requestBody = {
      sources: selectedSources
    };

    // Add optional fields
    if (repositories.trim()) {
      requestBody.repositories = repositories.split(',').map(r => r.trim());
    }
    if (spaces.trim()) {
      requestBody.spaces = spaces.split(',').map(s => s.trim());
    }
    if (includePaths.trim()) {
      requestBody.include_paths = includePaths.split('\n').map(p => p.trim()).filter(p => p);
    }
    if (excludePaths.trim()) {
      requestBody.exclude_paths = excludePaths.split('\n').map(p => p.trim()).filter(p => p);
    }

    try {
      const response = await fetch('http://localhost:8000/sync', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();
      
      if (response.ok) {
        setResult(data);
        // Poll for status
        pollSyncStatus();
      } else {
        setError(data.detail || 'Sync failed');
      }
    } catch (err) {
      setError('Failed to connect to server: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const pollSyncStatus = async () => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8000/sync/status');
        const data = await response.json();
        
        setResult(data);
        
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Error polling status:', err);
      }
    }, 3000); // Poll every 3 seconds
  };

  return (
    <div className="sync-form">
      <div className="form-header">
        <h2>
          <span className="header-icon">🔄</span>
          Sync Knowledge Base
        </h2>
        <p className="form-subtitle">Update CEIRA's knowledge from your data sources</p>
      </div>

      <div className="form-section">
        <label className="section-label">
          <span className="label-icon">📂</span>
          Data Sources
        </label>
        <div className="checkbox-group">
          <label className={`checkbox-label futuristic-checkbox ${sources.github ? 'checked' : ''}`}>
            <input
              type="checkbox"
              checked={sources.github}
              onChange={() => handleSourceChange('github')}
            />
            <span className="checkbox-custom"></span>
            <span className="checkbox-text">
              <span className="checkbox-icon">💻</span>
              GitHub
            </span>
          </label>
          <label className={`checkbox-label futuristic-checkbox ${sources.confluence ? 'checked' : ''}`}>
            <input
              type="checkbox"
              checked={sources.confluence}
              onChange={() => handleSourceChange('confluence')}
            />
            <span className="checkbox-custom"></span>
            <span className="checkbox-text">
              <span className="checkbox-icon">📄</span>
              Confluence
            </span>
          </label>
          <label className={`checkbox-label futuristic-checkbox ${sources.jira ? 'checked' : ''}`}>
            <input
              type="checkbox"
              checked={sources.jira}
              onChange={() => handleSourceChange('jira')}
            />
            <span className="checkbox-custom"></span>
            <span className="checkbox-text">
              <span className="checkbox-icon">🎫</span>
              Jira
            </span>
          </label>
        </div>
      </div>

      {sources.github && (
        <div className="form-section animated-section">
          <label className="section-label">
            <span className="label-icon">🔗</span>
            GitHub Repositories
          </label>
          <input
            type="text"
            placeholder="e.g., backend-services, frontend-app (comma-separated)"
            value={repositories}
            onChange={(e) => setRepositories(e.target.value)}
            className="text-input futuristic-input"
          />
        </div>
      )}

      {sources.confluence && (
        <div className="form-section animated-section">
          <label className="section-label">
            <span className="label-icon">📚</span>
            Confluence Spaces
          </label>
          <input
            type="text"
            placeholder="e.g., PRP, DOCS (comma-separated)"
            value={spaces}
            onChange={(e) => setSpaces(e.target.value)}
            className="text-input futuristic-input"
          />
        </div>
      )}

      <div className="form-section">
        <label className="section-label">
          <span className="label-icon">✅</span>
          Include Paths (one per line)
        </label>
        <textarea
          placeholder="e.g.,&#10;src/&#10;docs/&#10;README.md"
          value={includePaths}
          onChange={(e) => setIncludePaths(e.target.value)}
          className="textarea-input futuristic-textarea"
          rows="4"
        />
      </div>

      <div className="form-section">
        <label className="section-label">
          <span className="label-icon">❌</span>
          Exclude Paths (one per line)
        </label>
        <textarea
          placeholder="e.g.,&#10;node_modules/&#10;tests/&#10;*.test.js"
          value={excludePaths}
          onChange={(e) => setExcludePaths(e.target.value)}
          className="textarea-input futuristic-textarea"
          rows="4"
        />
      </div>

      <button 
        className={`sync-button ${loading ? 'loading' : ''}`}
        onClick={handleSync}
        disabled={loading}
      >
        <span className="button-content">
          {loading ? (
            <>
              <span className="spinner"></span>
              <span>Syncing Data...</span>
            </>
          ) : (
            <>
              <span className="button-icon">🚀</span>
              <span>Start Sync</span>
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
          <h3>
            <span className="section-icon">📊</span>
            Sync Status
          </h3>
          <div className="status-grid">
            <div className="status-item">
              <span className="status-label">Status:</span>
              <span className={`status-value ${result.status}`}>
                {result.status === 'running' && (
                  <>
                    <span className="status-spinner"></span>
                    Running
                  </>
                )}
                {result.status === 'completed' && '✅ Completed'}
                {result.status === 'failed' && '❌ Failed'}
              </span>
            </div>
            <div className="status-item">
              <span className="status-label">Documents:</span>
              <span className="status-value documents">{result.processed_documents || 0}</span>
            </div>
            <div className="status-item">
              <span className="status-label">Chunks:</span>
              <span className="status-value chunks">{result.total_chunks || 0}</span>
            </div>
            <div className="status-item">
              <span className="status-label">Errors:</span>
              <span className="status-value errors">{result.errors || 0}</span>
            </div>
          </div>
          {result.message && (
            <div className="status-message">
              <span className="message-icon">💬</span>
              {result.message}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default SyncForm;
