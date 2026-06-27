import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  chatHistoryCacheKeys,
  readCachedMessages,
  readCachedSessions,
  writeCachedMessages,
  writeCachedSessions,
} from './chatHistoryCache.js';

function createStorage() {
  const values = new Map();
  return {
    getItem(key) {
      return values.has(key) ? values.get(key) : null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
    removeItem(key) {
      values.delete(key);
    },
  };
}

describe('chat history cache', () => {
  it('scopes cache keys by user id', () => {
    assert.deepEqual(chatHistoryCacheKeys('user_123'), {
      sessions: 'portfolio-ai-chat-cache:sessions:user_123',
      messagesPrefix: 'portfolio-ai-chat-cache:messages:user_123:',
    });
  });

  it('stores and reads session summaries with newest ordering intact', () => {
    const storage = createStorage();
    const sessions = [
      { session_id: 's2', title: 'Second', updated_at: '2026-06-27T10:00:00Z' },
      { session_id: 's1', title: 'First', updated_at: '2026-06-27T09:00:00Z' },
    ];

    writeCachedSessions('user_123', sessions, storage);

    assert.deepEqual(readCachedSessions('user_123', storage), sessions);
    assert.deepEqual(readCachedSessions('user_456', storage), []);
  });

  it('stores and reads messages per user and session', () => {
    const storage = createStorage();
    const messages = [
      { id: 'm1', role: 'user', content: 'hello' },
      { id: 'm2', role: 'assistant', content: 'hi' },
    ];

    writeCachedMessages('user_123', 'session-1', messages, storage);

    assert.deepEqual(readCachedMessages('user_123', 'session-1', storage), messages);
    assert.deepEqual(readCachedMessages('user_123', 'session-2', storage), []);
    assert.deepEqual(readCachedMessages('user_456', 'session-1', storage), []);
  });

  it('returns empty arrays for corrupt cached payloads', () => {
    const storage = createStorage();
    storage.setItem(chatHistoryCacheKeys('user_123').sessions, '{bad json');

    assert.deepEqual(readCachedSessions('user_123', storage), []);
  });
});
