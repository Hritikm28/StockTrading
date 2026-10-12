"""
NSEFetcher must stop calling an endpoint that keeps failing: NSE answers
blocked datacenter requests with an HTML page (HTTP 200), and retrying that
for every stock in the universe once pushed the daily run past its time
limit.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class _Resp:
    def __init__(self, status, text):
        self.status_code = status
        self.text = text

    def json(self):
        import json
        return json.loads(self.text)


@pytest.fixture
def fetcher(monkeypatch, tmp_path):
    import nse_fetcher
    monkeypatch.setattr(nse_fetcher.time, 'sleep', lambda s: None)
    monkeypatch.setattr(nse_fetcher, 'CACHE_DIR', tmp_path)
    f = nse_fetcher.NSEFetcher()
    f._initialized = True
    monkeypatch.setattr(f, '_init_session', lambda: None)
    return f


def _serve(monkeypatch, f, responder):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return responder(url)
    monkeypatch.setattr(f.session, 'get', fake_get)
    return calls


def test_html_block_page_is_a_failure(monkeypatch, fetcher):
    _serve(monkeypatch, fetcher,
           lambda url: _Resp(200, '<!DOCTYPE html><html>blocked</html>'))
    assert fetcher._get('https://www.nseindia.com/api/x?symbol=A') is None


def test_endpoint_skipped_after_repeated_failures(monkeypatch, fetcher):
    calls = _serve(monkeypatch, fetcher,
                   lambda url: _Resp(200, '<html>blocked</html>'))
    for sym in ['A', 'B', 'C']:
        fetcher._get(f'https://www.nseindia.com/api/ann?symbol={sym}')
    made = len(calls)
    for sym in ['D', 'E', 'F', 'G']:
        assert fetcher._get(f'https://www.nseindia.com/api/ann?symbol={sym}') is None
    assert len(calls) == made                     # no more network calls
    # a different endpoint is unaffected
    fetcher._get('https://www.nseindia.com/api/other?symbol=A')
    assert len(calls) > made


def test_success_resets_the_count(monkeypatch, fetcher):
    state = {'ok': False}
    _serve(monkeypatch, fetcher, lambda url: (
        _Resp(200, '{"data": []}') if state['ok'] else _Resp(403, '')))
    url = 'https://www.nseindia.com/api/ann?symbol=A'
    fetcher._get(url)
    fetcher._get(url)
    state['ok'] = True
    assert fetcher._get(url) is not None
    state['ok'] = False
    fetcher._get(url)
    fetcher._get(url)
    state['ok'] = True
    assert fetcher._get(url) is not None          # 2 fails < limit: still tried


def test_yfinance_options_skipped_after_empty_streak(monkeypatch, fetcher):
    yf_calls = []
    monkeypatch.setattr(fetcher, '_get_option_chain_yf',
                        lambda s: yf_calls.append(s) or None)
    monkeypatch.setattr(fetcher, '_get_option_chain_nse', lambda s: None)
    for i in range(fetcher.YF_OPTIONS_LIMIT + 10):
        fetcher.get_option_chain(f'S{i}')
    assert len(yf_calls) == fetcher.YF_OPTIONS_LIMIT
