import { useReducer, useEffect } from 'react';
import { BACKEND_BASE } from '../config/api';

const initialAnalyticsState = {
  data: null,
  loading: true,
  error: '',
};

function analyticsReducer(state, action) {
  switch (action.type) {
    case 'loading':
      return { ...state, loading: true, error: '' };
    case 'success':
      return { data: action.data, loading: false, error: '' };
    case 'error':
      return { ...state, loading: false, error: action.error || 'Data retrieval failed' };
    default:
      return state;
  }
}

function useTabAnalytics(endpoint, tickers, startDate, endDate, refreshKey = 0) {
  const [state, dispatch] = useReducer(analyticsReducer, initialAnalyticsState);

  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();
    dispatch({ type: 'loading' });
    const formattedTickers = Array.isArray(tickers) ? tickers.join(',') : tickers;
    const url = `${BACKEND_BASE}/api/analytics/${endpoint}?tickers=${encodeURIComponent(formattedTickers)}&start_date=${startDate}&end_date=${endDate}`;
    const token = localStorage.getItem('portfolio-governance-auth-token');
    const headers = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    fetch(url, { headers, signal: controller.signal })
      .then(async res => {
        if (!res.ok) {
          const text = await res.text().catch(() => '');
          throw new Error(text || `Failed to fetch ${endpoint} data: ${res.status}`);
        }
        return res.json();
      })
      .then(result => {
        if (isMounted) {
          dispatch({ type: 'success', data: result });
        }
      })
      .catch(err => {
        if (err.name === 'AbortError') return;
        console.error(`Error loading analytics tab [${endpoint}]:`, err);
        if (isMounted) {
          dispatch({ type: 'error', error: err.message });
        }
      });

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [endpoint, tickers, startDate, endDate, refreshKey]);

  return state;
}

export function useEdaAnalytics(tickers, startDate, endDate, refreshKey) {
  return useTabAnalytics('eda', tickers, startDate, endDate, refreshKey);
}

export function useInstabilityAnalytics(tickers, startDate, endDate, refreshKey) {
  return useTabAnalytics('instability', tickers, startDate, endDate, refreshKey);
}

export function useAdvisoryAllocationAnalytics(tickers, startDate, endDate, refreshKey) {
  return useTabAnalytics('advisory-allocation', tickers, startDate, endDate, refreshKey);
}

export function useDiversificationAnalytics(tickers, startDate, endDate, refreshKey) {
  return useTabAnalytics('diversification', tickers, startDate, endDate, refreshKey);
}

export function useRiskGovernanceAnalytics(tickers, startDate, endDate, refreshKey) {
  return useTabAnalytics('risk-governance', tickers, startDate, endDate, refreshKey);
}

export function useContagionAnalytics(tickers, startDate, endDate, refreshKey) {
  return useTabAnalytics('contagion', tickers, startDate, endDate, refreshKey);
}

export function useAgentGovernanceAnalytics(tickers, startDate, endDate, refreshKey) {
  return useTabAnalytics('agent-governance', tickers, startDate, endDate, refreshKey);
}

export function useBacktestingAnalytics(tickers, startDate, endDate, refreshKey) {
  return useTabAnalytics('backtesting', tickers, startDate, endDate, refreshKey);
}
