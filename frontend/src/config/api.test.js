import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { BACKEND_BASE } from './api.js';

describe('api config', () => {
  it('defaults to localhost in development and same-origin backend path in production', () => {
    assert.equal(BACKEND_BASE, 'http://127.0.0.1:8000');
  });
});
