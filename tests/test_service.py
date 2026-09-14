from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import unittest

from fastapi.testclient import TestClient

from src.serve import RecentRequestMetrics, SingleFlight, create_app
from src.settings import Settings


class FakeRuntime:
    version = "fixture-v1"
    def __init__(self, settings):
        self.entered, self.release = threading.Event(), threading.Event()
        self.ready = True
    def info(self):
        return {"index_version": self.version, "provider": "fixture"}
    def backend_ready(self):
        return self.ready
    def answer(self, question):
        if question == "slow":
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError("fixture timeout")
        if question == "error":
            raise RuntimeError("SECRET_SHOULD_NOT_LEAK")
        return {"answer": "fixture answer", "index_version": self.version}
    def stream_answer(self, question, cancel_event):
        yield {"type": "metadata", "index_version": self.version}
        yield {"type": "token", "text": "fixture "}
        yield {"type": "token", "text": "answer"}
        yield {"type": "final", "complete": True, "answer": "fixture answer",
               "index_version": self.version}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(Path("fixture-notes").resolve(), Path("fixture-store").resolve(),
                                 request_timeout=1, max_question_chars=40)
        self.app = create_app(self.settings, runtime_factory=FakeRuntime)
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8000")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.runtime = self.app.state.runtime
        self.addCleanup(self.runtime.release.set)

    def test_health_version_and_valid_answer(self):
        self.assertEqual(self.client.get("/health/live").status_code, 200)
        ready = self.client.get("/health/ready").json()
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["backend_check"], "runtime_self_check")
        self.assertEqual(self.client.get("/v1/index").json()["index_version"], "fixture-v1")
        response = self.client.post("/v1/answer", json={"question": "RAG是什么？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-request-id"], response.json()["request_id"])
        self.assertIn("service_elapsed_ms", response.json()["phase_timings_ms"])
        metrics = self.client.get("/v1/metrics").json()
        self.assertEqual(metrics["requests"]["observed_total"], 1)
        self.assertEqual(metrics["requests"]["outcomes"], {"completed": 1})
        self.assertIn("service_elapsed_ms", metrics["requests"]["latency_ms"])
        self.assertNotIn("fixture answer", json.dumps(metrics))
        self.runtime.ready = False
        self.assertEqual(self.client.get("/health/ready").status_code, 503)

    def test_invalid_requests_do_not_echo_input(self):
        for value in ({"question": "  "}, {"question": 12}, {"question": "x", "api_key": "SECRET"},
                      {"question": "x" * 41}):
            response = self.client.post("/v1/answer", json=value)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("SECRET", response.text)
        response = self.client.post("/v1/answer", content="x" * 2000,
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.client.post("/v1/answer", content="{}").status_code, 415)

    def test_cross_origin_and_host_are_rejected(self):
        self.assertEqual(self.client.get("/v1/index", headers={"Host": "evil.example"}).status_code, 403)
        self.assertEqual(self.client.post("/v1/answer", json={"question": "x"},
                                         headers={"Origin": "https://evil.example"}).status_code, 403)

    def test_model_error_is_sanitized(self):
        response = self.client.post("/v1/answer", json={"question": "error"})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("SECRET_SHOULD_NOT_LEAK", response.text)
        self.assertEqual(self.client.post("/v1/answer", json={"question": "x"}).status_code, 200)

    def test_ndjson_stream_has_explicit_final_event(self):
        response = self.client.post("/v1/answer/stream", json={"question": "RAG是什么？"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/x-ndjson")
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual([event["type"] for event in events],
                         ["metadata", "token", "token", "final"])
        self.assertTrue(events[-1]["complete"])
        self.assertEqual(events[-1]["answer"], "fixture answer")
        self.assertTrue(all(event["request_id"] for event in events))

    def test_busy_and_timeout_keep_slot_until_worker_finishes(self):
        with ThreadPoolExecutor(1) as pool:
            first = pool.submit(self.client.post, "/v1/answer", json={"question": "slow"})
            self.assertTrue(self.runtime.entered.wait(2))
            second = self.client.post("/v1/answer", json={"question": "x"})
            self.assertEqual(second.status_code, 429)
            self.assertEqual(first.result(timeout=3).status_code, 504)
            self.assertEqual(self.client.post("/v1/answer", json={"question": "x"}).status_code, 429)
            self.runtime.release.set()
            with self.app.state.flight.lock:  # 等真实工作释放，而不是用 sleep 猜测。
                pass
            self.assertEqual(self.client.post("/v1/answer", json={"question": "x"}).status_code, 200)

    def test_queued_http_request_times_out_and_is_removed(self):
        app = create_app(self.settings, runtime_factory=FakeRuntime,
                         max_queue_size=1, queue_wait_timeout=0.05)
        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            runtime = app.state.runtime
            with ThreadPoolExecutor(1) as pool:
                first = pool.submit(client.post, "/v1/answer", json={"question": "slow"})
                self.assertTrue(runtime.entered.wait(2))
                queued = client.post("/v1/answer", json={"question": "queued"})
                self.assertEqual(queued.status_code, 503)
                self.assertEqual(queued.json()["error"], "queue_wait_timeout")
                self.assertEqual(app.state.flight.snapshot()["waiting"], 0)
                self.assertEqual(app.state.flight.snapshot()["queue_timeout_total"], 1)
                runtime.release.set()
                self.assertEqual(first.result(timeout=3).status_code, 200)


class SingleFlightQueueTests(unittest.TestCase):
    def test_fifo_queue_full_and_handoff(self):
        flight = SingleFlight(max_waiting=2)
        self.addCleanup(flight.close)
        first = flight.request_slot("first")
        second = flight.request_slot("second")
        third = flight.request_slot("third")
        fourth = flight.request_slot("fourth")
        self.assertEqual((first.state, second.state, third.state, fourth.reason),
                         ("admitted", "waiting", "waiting", "queue_full"))
        self.assertEqual((second.initial_position, third.initial_position), (1, 2))

        flight.release(first)
        self.assertTrue(flight.wait_for_slot(second, 0.1))
        self.assertEqual(second.state, "admitted")
        self.assertEqual(third.state, "waiting")
        flight.release(second)
        self.assertTrue(flight.wait_for_slot(third, 0.1))
        self.assertEqual(third.state, "admitted")
        flight.release(third)
        self.assertFalse(flight.snapshot()["busy"])

    def test_cancelled_waiter_is_not_admitted(self):
        flight = SingleFlight(max_waiting=2)
        self.addCleanup(flight.close)
        first = flight.request_slot("first")
        cancelled = flight.request_slot("cancelled")
        next_ticket = flight.request_slot("next")
        flight.cancel_ticket(cancelled)
        self.assertEqual(cancelled.state, "cancelled")
        flight.release(first)
        self.assertTrue(flight.wait_for_slot(next_ticket, 0.1))
        self.assertEqual(next_ticket.state, "admitted")
        flight.release(next_ticket)


class RecentRequestMetricsTests(unittest.TestCase):
    def test_bounded_window_and_nearest_rank_percentiles(self):
        metrics = RecentRequestMetrics(capacity=2)
        metrics.observe("completed", 200, {"service_elapsed_ms": 10})
        metrics.observe("completed", 200, {"service_elapsed_ms": 20})
        metrics.observe("failed", 502, {"service_elapsed_ms": 30, "queue_wait_ms": -1})

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["observed_total"], 3)
        self.assertEqual(snapshot["retained"], 2)
        self.assertEqual(snapshot["outcomes"], {"completed": 2, "failed": 1})
        self.assertEqual(snapshot["latency_ms"]["service_elapsed_ms"], {
            "count": 2, "avg": 25.0, "p50": 20.0, "p95": 30.0, "max": 30.0,
        })
        self.assertNotIn("queue_wait_ms", snapshot["latency_ms"])
