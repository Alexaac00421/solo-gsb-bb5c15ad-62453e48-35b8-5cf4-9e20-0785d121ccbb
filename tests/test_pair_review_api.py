"""配对唯一性核查 HTTP API 集成测试（临时 SQLite + TestClient）。"""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setenv("QSO_DB_PATH", tmp.name)
    from app.main import app

    with TestClient(app) as c:
        yield c
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


def _log(no, station, counterpart, t="2026-10-04T00:00:00+00:00",
         band="40m", mode="SSB"):
    return {
        "record_no": no,
        "station": station,
        "counterpart": counterpart,
        "time": t,
        "band": band,
        "mode": mode,
    }


def _make_drill(client, drill="RV", allowed=10):
    r = client.post(
        "/drills",
        json={"drill_id": drill, "allowed_time_diff_seconds": allowed},
    )
    assert r.status_code == 201, r.text


def test_empty_drill_returns_empty_checks(client):
    _make_drill(client)
    r = client.get("/drills/RV/pair-review")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["drill_id"] == "RV"
    assert body["version"] == 0
    assert body["sealed"] is False
    assert body["optimal_pair_count"] == 0
    assert body["minimum_total_time_diff_seconds"] == 0.0
    assert body["pair_checks"] == []


def test_unknown_drill_404(client):
    r = client.get("/drills/NOPE/pair-review")
    assert r.status_code == 404
    assert r.json()["error"] == "drill_not_found"


def test_unique_pair_marked_in_all_schemes(client):
    _make_drill(client)
    client.post("/drills/RV/logs", json={"logs": [
        _log("1", "A", "B", "2026-10-04T00:00:00+00:00"),
        _log("2", "B", "A", "2026-10-04T00:00:05+00:00"),
        _log("3", "C", "D"),  # 无互指
    ]})
    r = client.get("/drills/RV/pair-review")
    body = r.json()
    assert body["optimal_pair_count"] == 1
    assert body["minimum_total_time_diff_seconds"] == 5.0
    (chk,) = body["pair_checks"]
    assert chk["record_a"] == "1" and chk["record_b"] == "2"
    assert chk["locked"] is False
    assert chk["in_all_schemes"] is True
    assert chk["replaceable"] is False
    assert chk["alternative"] is None


def test_replaceable_pair_full_alternative(client):
    _make_drill(client)
    client.post("/drills/RV/logs", json={"logs": [
        _log("1", "A", "B"),
        _log("2", "A", "B"),
        _log("3", "B", "A"),
        _log("4", "B", "A"),
    ]})
    body = client.get("/drills/RV/pair-review").json()
    assert body["optimal_pair_count"] == 2
    assert body["minimum_total_time_diff_seconds"] == 0.0
    keys = [(c["record_a"], c["record_b"]) for c in body["pair_checks"]]
    assert keys == [("1", "3"), ("2", "4")]
    cur_key = ("1", "3"), ("2", "4")
    for chk in body["pair_checks"]:
        assert chk["in_all_schemes"] is False
        assert chk["replaceable"] is True
        alt = chk["alternative"]
        assert [(p["record_a"], p["record_b"]) for p in alt["pairs"]] == [
            ("1", "4"), ("2", "3")
        ]
        assert alt["unpaired"] == []
        assert alt["total_time_diff_seconds"] == 0.0
        # 增删相对当前完整结果：当前两对全部移除、替代两对全部新增。
        added = {(p["record_a"], p["record_b"]) for p in alt["changes"]["added"]}
        removed = {(p["record_a"], p["record_b"])
                   for p in alt["changes"]["removed"]}
        assert added == {("1", "4"), ("2", "3")}
        assert removed == set(cur_key)


def test_alternative_excludes_only_checked_pair_when_simple(client):
    # 1 左 2 右：当前 (1,2)，替代 (1,3)；增删各一对，未配对切换。
    _make_drill(client)
    client.post("/drills/RV/logs", json={"logs": [
        _log("1", "A", "B"),
        _log("2", "B", "A"),
        _log("3", "B", "A"),
    ]})
    body = client.get("/drills/RV/pair-review").json()
    (chk,) = body["pair_checks"]
    assert (chk["record_a"], chk["record_b"]) == ("1", "2")
    alt = chk["alternative"]
    assert [(p["record_a"], p["record_b"]) for p in alt["pairs"]] == [
        ("1", "3")
    ]
    assert alt["unpaired"] == ["2"]
    added = [(p["record_a"], p["record_b"]) for p in alt["changes"]["added"]]
    removed = [(p["record_a"], p["record_b"])
               for p in alt["changes"]["removed"]]
    assert added == [("1", "3")]
    assert removed == [("1", "2")]


def test_locked_pair_flagged_locked_and_not_reviewed(client):
    _make_drill(client)
    client.post("/drills/RV/logs", json={"logs": [
        _log("1", "A", "B", "2026-10-04T00:00:00+00:00"),
        _log("2", "B", "A", "2026-10-04T00:00:05+00:00"),
        _log("3", "C", "D"),
        _log("4", "D", "C"),
    ]})
    r = client.put("/drills/RV/locks", json={"record_a": "1", "record_b": "2"})
    assert r.status_code == 200, r.text

    body = client.get("/drills/RV/pair-review").json()
    by_key = {(c["record_a"], c["record_b"]): c for c in body["pair_checks"]}
    locked = by_key[("1", "2")]
    assert locked["locked"] is True
    assert locked["in_all_schemes"] is True
    assert locked["replaceable"] is False
    assert locked["alternative"] is None
    free = by_key[("3", "4")]
    assert free["locked"] is False
    assert free["in_all_schemes"] is True
    assert body["optimal_pair_count"] == 2


def test_alternative_complete_scheme_keeps_locked_pair(client):
    # 自由部分 2x2 可互换，另有一对独立锁定 (5,6)：替代方案必须完整，
    # 始终包含锁定对，且总时差含锁定时差 3 秒。
    _make_drill(client)
    client.post("/drills/RV/logs", json={"logs": [
        _log("1", "A", "B"),
        _log("2", "A", "B"),
        _log("3", "B", "A"),
        _log("4", "B", "A"),
        _log("5", "C", "D", "2026-10-04T00:00:03+00:00"),
        _log("6", "D", "C"),
    ]})
    r = client.put("/drills/RV/locks", json={"record_a": "5", "record_b": "6"})
    assert r.status_code == 200, r.text

    body = client.get("/drills/RV/pair-review").json()
    assert body["optimal_pair_count"] == 3
    assert body["minimum_total_time_diff_seconds"] == 3.0
    by_key = {(c["record_a"], c["record_b"]): c for c in body["pair_checks"]}
    assert by_key[("5", "6")]["locked"] is True
    for key in (("1", "3"), ("2", "4")):
        alt = by_key[key]["alternative"]
        alt_keys = {(p["record_a"], p["record_b"]) for p in alt["pairs"]}
        assert alt_keys == {("1", "4"), ("2", "3"), ("5", "6")}
        assert alt["total_time_diff_seconds"] == 3.0
        # 增删只发生在自由部分，锁定对既不新增也不移除。
        added = {(p["record_a"], p["record_b"])
                 for p in alt["changes"]["added"]}
        removed = {(p["record_a"], p["record_b"])
                   for p in alt["changes"]["removed"]}
        assert ("5", "6") not in added and ("5", "6") not in removed


def test_quarantined_logs_excluded(client):
    _make_drill(client)
    client.post("/drills/RV/logs", json={"logs": [
        _log("1", "A", "B"),
        _log("2", "B", "A"),
        _log("3", "B", "A"),
    ]})
    # 隔离 3 后只剩 (1,2) 唯一一对。
    r = client.put("/drills/RV/quarantine", json={"record_nos": ["3"]})
    assert r.status_code == 200, r.text
    body = client.get("/drills/RV/pair-review").json()
    assert body["optimal_pair_count"] == 1
    (chk,) = body["pair_checks"]
    assert (chk["record_a"], chk["record_b"]) == ("1", "2")
    assert chk["in_all_schemes"] is True


def test_review_is_read_only(client):
    _make_drill(client)
    client.post("/drills/RV/logs", json={"logs": [
        _log("1", "A", "B"),
        _log("2", "B", "A"),
    ]})
    before = client.get("/drills/RV/result").json()
    client.get("/drills/RV/pair-review")
    client.get("/drills/RV/pair-review")
    after = client.get("/drills/RV/result").json()
    assert before == after
    snaps = client.get("/drills/RV/snapshots").json()["versions"]
    assert snaps == [0, 1]


def test_review_available_after_seal(client):
    _make_drill(client)
    client.post("/drills/RV/logs", json={"logs": [
        _log("1", "A", "B"),
        _log("2", "B", "A"),
    ]})
    client.post("/drills/RV/seal")
    r = client.get("/drills/RV/pair-review")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sealed"] is True
    assert body["pair_checks"][0]["in_all_schemes"] is True
    # 封存后核查仍不改变版本。
    assert body["version"] == 1


def test_review_tracks_version(client):
    _make_drill(client)
    client.post("/drills/RV/logs", json={"logs": [
        _log("1", "A", "B"),
        _log("2", "B", "A"),
    ]})
    body = client.get("/drills/RV/pair-review").json()
    assert body["version"] == 1
    client.put("/drills/RV/quarantine", json={"record_nos": ["2"]})
    body = client.get("/drills/RV/pair-review").json()
    assert body["version"] == 2
    assert body["pair_checks"] == []
