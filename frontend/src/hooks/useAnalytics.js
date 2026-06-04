import { useState, useEffect } from 'react';
import { BACKEND_BASE } from '../config/api';

function useTabAnalytics(endpoint, tickers, startDate, endDate) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError('');

    const formattedTickers = Array.isArray(tickers) ? tickers.join(',') : tickers;
    const url = `${BACKEND_BASE}/api/analytics/${endpoint}?tickers=${encodeURIComponent(formattedTickers)}&start_date=${startDate}&end_date=${endDate}`;

    fetch(url)
      .then(async res => {
        if (!res.ok) {
          const text = await res.text().catch(() => '');
          throw new Error(text || `Failed to fetch ${endpoint} data: ${res.status}`);
        }
        return res.json();
      })
      .then(result => {
        if (isMounted) {
          setData(result);
          setLoading(false);
        }
      })
      .catch(err => {
        console.error(`Error loading analytics tab [${endpoint}]:`, err);
        if (isMounted) {
          setError(err.message || 'Data retrieval failed');
          setLoading(false);
        }
      });

    return () => { isMounted = false; };
  }, [endpoint, tickers, startDate, endDate]);

  return { data, loading, error };
}

export function useEdaAnalytics(tickers, startDate, endDate) {
  return useTabAnalytics('eda', tickers, startDate, endDate);
}

export function useInstabilityAnalytics(tickers, startDate, endDate) {
  return useTabAnalytics('instability', tickers, startDate, endDate);
}

export function useAdvisoryAllocationAnalytics(tickers, startDate, endDate) {
  return useTabAnalytics('advisory-allocation', tickers, startDate, endDate);
}

export function useDiversificationAnalytics(tickers, startDate, endDate) {
  return useTabAnalytics('diversification', tickers, startDate, endDate);
}

export function useRiskGovernanceAnalytics(tickers, startDate, endDate) {
  return useTabAnalytics('risk-governance', tickers, startDate, endDate);
}

export function useContagionAnalytics(tickers, startDate, endDate) {
  return useTabAnalytics('contagion', tickers, startDate, endDate);
}

export function useAgentGovernanceAnalytics(tickers, startDate, endDate) {
  return useTabAnalytics('agent-governance', tickers, startDate, endDate);
}

export function useBacktestingAnalytics(tickers, startDate, endDate) {
  return useTabAnalytics('backtesting', tickers, startDate, endDate);
}
