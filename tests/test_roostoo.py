import json
import unittest

from bot.roostoo import (RateLimiter, RoostooAuthError, RoostooClient, RoostooError,
                         build_query, sign)

# The worked example from the Roostoo API documentation.
DOC_SECRET = "S1XP1e3UZj6A7H5fATj0jNhqPxxdSJYdInClVN65XAbvqqMKjVHjA7PZj4W12oep"
DOC_PARAMS = {"pair": "BNB/USD", "quantity": "2000", "side": "BUY",
              "timestamp": "1580774512000", "type": "MARKET"}
DOC_SIGNATURE = "20b7fd5550b67b3bf0c1684ed0f04885261db8fdabd38611e9e6af23c19b7fff"
API_KEY = "test-api-key-4f2a"


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return json.loads(self.text)


class FakeSession:
    """Records requests and replays canned responses in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, None, headers))
        return self.responses.pop(0)

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append(("POST", url, data, headers))
        return self.responses.pop(0)


def make_client(*responses):
    session = FakeSession(*responses)
    client = RoostooClient(API_KEY, DOC_SECRET, session=session, max_retries=2, backoff_sec=0)
    return client, session


class SigningTest(unittest.TestCase):
    def test_query_is_sorted_by_key(self):
        params = dict(reversed(list(DOC_PARAMS.items())))
        self.assertEqual(build_query(params),
                         "pair=BNB/USD&quantity=2000&side=BUY&timestamp=1580774512000&type=MARKET")

    def test_signature_matches_documented_example(self):
        self.assertEqual(sign(build_query(DOC_PARAMS), DOC_SECRET), DOC_SIGNATURE)


class RateLimiterTest(unittest.TestCase):
    def test_waits_until_oldest_call_leaves_the_window(self):
        now = [0.0]
        slept = []

        def sleep(seconds):
            slept.append(seconds)
            now[0] += seconds

        limiter = RateLimiter(3, period=60.0, clock=lambda: now[0], sleep=sleep)
        for _ in range(3):
            limiter.acquire()
            now[0] += 1.0
        self.assertEqual(slept, [])
        limiter.acquire()  # 4th call at t=3 must wait for the call at t=0 to expire
        self.assertAlmostEqual(sum(slept), 57.0)


class ClientTest(unittest.TestCase):
    def test_signed_post_sends_exactly_the_signed_body(self):
        client, session = make_client(FakeResponse(200, {"Success": True, "OrderDetail": {"OrderID": 7}}))
        detail = client.place_order("BNB/USD", "BUY", "2000")
        self.assertEqual(detail["OrderID"], 7)
        method, url, body, headers = session.calls[0]
        self.assertEqual(method, "POST")
        self.assertTrue(url.endswith("/v3/place_order"))
        self.assertIn("pair=BNB/USD&quantity=2000&side=BUY&timestamp=", body)
        self.assertEqual(headers["MSG-SIGNATURE"], sign(body, DOC_SECRET))
        self.assertEqual(headers["RST-API-KEY"], API_KEY)
        self.assertEqual(headers["Content-Type"], "application/x-www-form-urlencoded")

    def test_signed_get_puts_the_signed_string_in_the_url(self):
        wallet = {"USD": {"Free": 50000, "Lock": 0}}
        client, session = make_client(FakeResponse(200, {"Success": True, "SpotWallet": wallet}))
        self.assertEqual(client.balance(), wallet)
        _, url, _, headers = session.calls[0]
        query = url.split("?", 1)[1]
        self.assertTrue(query.startswith("timestamp="))
        self.assertEqual(headers["MSG-SIGNATURE"], sign(query, DOC_SECRET))

    def test_balance_falls_back_to_documented_wallet_key(self):
        wallet = {"USD": {"Free": 1, "Lock": 0}}
        client, _ = make_client(FakeResponse(200, {"Success": True, "Wallet": wallet}))
        self.assertEqual(client.balance(), wallet)

    def test_success_false_raises_with_message(self):
        client, _ = make_client(FakeResponse(200, {"Success": False, "ErrMsg": "insufficient balance"}))
        with self.assertRaisesRegex(RoostooError, "insufficient balance"):
            client.place_order("BTC/USD", "BUY", "1")

    def test_empty_results_are_not_errors(self):
        client, _ = make_client(
            FakeResponse(200, {"Success": False, "ErrMsg": "no pending order under this account",
                               "TotalPending": 0, "OrderPairs": {}}),
            FakeResponse(200, {"Success": False, "ErrMsg": "no order matched"}),
        )
        self.assertEqual(client.pending_count(), 0)
        self.assertEqual(client.query_order(pending_only=True), [])

    def test_401_with_plain_text_body_raises_auth_error(self):
        client, _ = make_client(FakeResponse(401, "Unauthorized"))
        with self.assertRaises(RoostooAuthError):
            client.balance()

    def test_server_errors_are_retried_then_succeed(self):
        client, session = make_client(FakeResponse(502, "Bad Gateway"),
                                      FakeResponse(200, {"ServerTime": 123}))
        self.assertEqual(client.server_time(), 123)
        self.assertEqual(len(session.calls), 2)

    def test_place_order_is_never_retried(self):
        client, session = make_client(FakeResponse(502, "Bad Gateway"),
                                      FakeResponse(200, {"Success": True, "OrderDetail": {}}))
        with self.assertRaises(RoostooError):
            client.place_order("BTC/USD", "BUY", "1")
        self.assertEqual(len(session.calls), 1)

    def test_query_by_order_id_sends_no_other_filters(self):
        client, session = make_client(FakeResponse(200, {"Success": True, "OrderMatched": [{"OrderID": 5}]}))
        self.assertEqual(client.query_order(order_id=5, pair="BTC/USD"), [{"OrderID": 5}])
        body = session.calls[0][2]
        self.assertNotIn("pair=", body)
        self.assertIn("order_id=5", body)

    def test_keys_never_reach_the_api_log(self):
        client, _ = make_client(FakeResponse(200, {"Success": True, "SpotWallet": {}}))
        with self.assertLogs("roostoo.api", level="INFO") as captured:
            client.balance()
        text = "\n".join(captured.output)
        self.assertNotIn(API_KEY, text)
        self.assertNotIn(DOC_SECRET, text)
        self.assertNotIn("MSG-SIGNATURE", text)


if __name__ == "__main__":
    unittest.main()
