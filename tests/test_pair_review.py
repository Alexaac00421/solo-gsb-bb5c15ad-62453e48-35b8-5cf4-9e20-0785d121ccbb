"""配对唯一性核查算法测试：构造场景 + 小规模随机穷举对照。"""

from __future__ import annotations

import random

from app.matching import (
    MICROS_PER_SECOND,
    InvalidLockError,
    Record,
    match_records,
    review_free_pairs,
)


def rec(no, station, counterpart, t, band="40m", mode="SSB"):
    return Record(
        record_no=no,
        station=station,
        counterpart=counterpart,
        ts_micros=int(t * MICROS_PER_SECOND),
        band=band,
        mode=mode,
    )


def _candidate_edges(records, allowed_micros):
    edges = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            if (
                a.station == b.counterpart
                and a.counterpart == b.station
                and a.band == b.band
                and a.mode == b.mode
                and abs(a.ts_micros - b.ts_micros) <= allowed_micros
            ):
                edges.append(
                    (
                        tuple(sorted((a.record_no, b.record_no))),
                        frozenset((i, j)),
                        abs(a.ts_micros - b.ts_micros),
                    )
                )
    return edges


def brute_review(records, allowed, locked_keys=()):
    """穷举前两级（最大基数、最小总时差）下的全部合法自由匹配。

    返回 (最大自由配对数, 最小自由总时差微秒, 当前三级最优自由配对键,
    {配对键: (是否在全部方案中, 字典序最小替代方案键元组)})。
    """
    locked_keys = {tuple(sorted(k)) for k in locked_keys}
    locked_ids = {x for k in locked_keys for x in k}
    active = [r for r in records if r.record_no not in locked_ids]
    allowed_micros = int(allowed * MICROS_PER_SECOND)
    edges = _candidate_edges(active, allowed_micros)

    schemes = []
    for mask in range(1 << len(edges)):
        used = set()
        keys: list[tuple[str, str]] = []
        total = 0
        ok = True
        for ei in range(len(edges)):
            if not (mask >> ei) & 1:
                continue
            key, verts, weight = edges[ei]
            if used & verts:
                ok = False
                break
            used |= verts
            keys.append(key)
            total += weight
        if ok:
            schemes.append((len(keys), total, tuple(sorted(keys))))

    max_k = max((s[0] for s in schemes), default=0)
    best = [s for s in schemes if s[0] == max_k]
    min_total = min((s[1] for s in best), default=0)
    optimal = [s[2] for s in best if s[1] == min_total]
    current = min(optimal) if optimal else ()

    expected = {}
    for key in current:
        alts = [s for s in optimal if key not in s]
        expected[key] = (
            not alts,
            min(alts, key=lambda ks: ks) if alts else None,
        )
    return max_k, min_total, current, expected


def assert_review_matches_bruteforce(records, allowed, locked_keys=()):
    review = review_free_pairs(records, allowed, locked_keys)
    locked_norm = {tuple(sorted(k)) for k in locked_keys}
    max_k, min_total, current, expected = brute_review(
        records, allowed, locked_keys
    )
    free_checks = {c.pair.as_tuple(): c for c in review.checks if not c.locked}
    locked_checks = {c.pair.as_tuple(): c for c in review.checks if c.locked}

    assert set(locked_checks) == locked_norm
    for c in locked_checks.values():
        assert c.in_all_schemes is True
        assert c.alternative is None
    assert set(free_checks) == set(current)
    assert review.max_pair_count == max_k + len(locked_norm)
    rmap = {r.record_no: r for r in records}
    locked_total = sum(
        abs(rmap[a].ts_micros - rmap[b].ts_micros) for a, b in locked_norm
    )
    assert (
        review.min_total_time_diff_seconds
        == (locked_total + min_total) / MICROS_PER_SECOND
    )

    # 当前三级最优自由结果与核查所用当前配对一致。
    current_match = match_records(records, allowed, locked_keys)
    assert {p.as_tuple() for p in current_match.pairs} - locked_norm == set(
        current
    )

    for key, (in_all, alt_keys) in expected.items():
        c = free_checks[key]
        assert c.in_all_schemes is in_all, (key, c.in_all_schemes, in_all)
        if in_all:
            assert c.alternative is None
        else:
            got = tuple(p.as_tuple() for p in c.alternative.pairs)
            # 替代方案是自由池方案（算法层不含锁定对）。
            assert got == alt_keys, (key, got, alt_keys)
            assert key not in got
            assert c.alternative.total_time_diff_seconds == (
                min_total / MICROS_PER_SECOND
            )
    return review


def test_single_pair_unique():
    records = [rec("1", "A", "B", 0), rec("2", "B", "A", 5)]
    r = assert_review_matches_bruteforce(records, 10)
    (c,) = r.checks
    assert c.in_all_schemes and c.alternative is None


def test_perfect_swap_all_replaceable():
    # 2x2 同时刻：两种完美匹配同总时差（0），字典序当前为 (1,3)(2,4)，
    # 两对都可替换，替代方案均为 (1,4)(2,3)。
    records = [
        rec("1", "A", "B", 0),
        rec("2", "A", "B", 0),
        rec("3", "B", "A", 0),
        rec("4", "B", "A", 0),
    ]
    r = assert_review_matches_bruteforce(records, 10)
    assert [c.pair.as_tuple() for c in r.checks] == [
        ("1", "3"), ("2", "4")
    ]
    for c in r.checks:
        assert not c.in_all_schemes
        alt = tuple(p.as_tuple() for p in c.alternative.pairs)
        assert alt == (("1", "4"), ("2", "3"))


def test_alternative_is_lexicographically_smallest():
    # 3 左 2 右：当前 (1,4)(2,5) 字典序最小；排除 (1,4) 后最小替代应
    # 让记录号 1 配 5（(1,5)(2,4) 与 (1,5)(3,4) 中取较小者）。
    records = [
        rec("1", "A", "B", 0),
        rec("2", "A", "B", 0),
        rec("3", "A", "B", 0),
        rec("4", "B", "A", 0),
        rec("5", "B", "A", 0),
    ]
    r = assert_review_matches_bruteforce(records, 10)
    by_key = {c.pair.as_tuple(): c for c in r.checks}
    alt14 = tuple(
        p.as_tuple() for p in by_key[("1", "4")].alternative.pairs
    )
    assert alt14 == (("1", "5"), ("2", "4"))


def test_min_total_diff_makes_unique():
    # 两种完美匹配时差和不同 => 当前最优方案的配对全部唯一。
    records = [
        rec("1", "A", "B", 0),
        rec("2", "A", "B", 10),
        rec("3", "B", "A", 2),
        rec("4", "B", "A", 8),
    ]
    r = assert_review_matches_bruteforce(records, 20)
    assert all(c.in_all_schemes for c in r.checks)
    assert r.min_total_time_diff_seconds == 4.0


def test_unmatched_records_remain_unmatched_in_alternative():
    # 1 左 2 右：当前 (1,2)（字典序更小），替代 (1,3)，另一记录落单。
    records = [
        rec("1", "A", "B", 0),
        rec("2", "B", "A", 0),
        rec("3", "B", "A", 0),
    ]
    r = assert_review_matches_bruteforce(records, 10)
    (c,) = r.checks
    assert c.pair.as_tuple() == ("1", "2")
    assert not c.in_all_schemes
    assert tuple(p.as_tuple() for p in c.alternative.pairs) == (("1", "3"),)
    assert c.alternative.unpaired == ("2",)


def test_empty_and_no_edges():
    r = review_free_pairs([], 10, ())
    assert r.checks == () and r.max_pair_count == 0
    assert r.min_total_time_diff_seconds == 0.0

    records = [rec("1", "A", "B", 0), rec("2", "B", "A", 11)]
    r = review_free_pairs(records, 10, ())
    assert r.checks == () and r.max_pair_count == 0


def test_locked_pairs_are_irreplaceable():
    # 2x2 同时刻，锁定 (2,4) 后只剩 (1,3) 可配；锁定对不可替换，
    # 唯一的自由对 (1,3) 在唯一方案中。
    records = [
        rec("1", "A", "B", 0),
        rec("2", "A", "B", 0),
        rec("3", "B", "A", 0),
        rec("4", "B", "A", 0),
    ]
    r = assert_review_matches_bruteforce(records, 10, locked_keys=[("2", "4")])
    by_key = {c.pair.as_tuple(): c for c in r.checks}
    assert by_key[("2", "4")].locked
    assert by_key[("2", "4")].in_all_schemes
    assert by_key[("2", "4")].alternative is None
    assert by_key[("1", "3")].in_all_schemes
    assert r.max_pair_count == 2


def test_lock_with_free_ambiguity():
    # 锁定 (1,3)，剩余 2/4/5（2 左，4、5 右）自由部分仍有替换空间。
    records = [
        rec("1", "A", "B", 0),
        rec("2", "A", "B", 0),
        rec("3", "B", "A", 0),
        rec("4", "B", "A", 0),
        rec("5", "B", "A", 0),
    ]
    assert_review_matches_bruteforce(
        records, 10, locked_keys=[("1", "3")]
    )


def test_invalid_lock_raises():
    records = [rec("1", "A", "B", 0), rec("2", "B", "A", 11)]
    try:
        review_free_pairs(records, 10, locked_keys=[("1", "2")])
    except InvalidLockError:
        pass
    else:  # pragma: no cover
        raise AssertionError("应当抛出 InvalidLockError")


def test_randomized_vs_bruteforce():
    rng = random.Random(20261007)
    calls = ["A", "B", "C", "D"]
    for _trial in range(400):
        n = rng.randint(0, 7)
        records = []
        for k in range(n):
            s = rng.choice(calls)
            c = rng.choice([x for x in calls if x != s])
            records.append(
                rec(
                    f"r{k}",
                    s,
                    c,
                    rng.randint(0, 60),
                    band=rng.choice(["40m", "20m"]),
                    mode=rng.choice(["SSB", "CW"]),
                )
            )
        assert_review_matches_bruteforce(
            records, rng.choice([0, 5, 30, 100])
        )


def test_randomized_with_locks():
    rng = random.Random(31337)
    calls = ["A", "B", "C"]
    for _trial in range(150):
        n = rng.randint(2, 6)
        records = []
        for k in range(n):
            s = rng.choice(calls)
            c = rng.choice([x for x in calls if x != s])
            records.append(
                rec(f"r{k}", s, c, rng.randint(0, 20),
                    band="40m", mode="SSB")
            )
        allowed = 30
        lock = None
        for i in range(n):
            for j in range(i + 1, n):
                a, b = records[i], records[j]
                if (
                    a.station == b.counterpart
                    and a.counterpart == b.station
                    and abs(a.ts_micros - b.ts_micros)
                    <= allowed * MICROS_PER_SECOND
                ):
                    lock = (a.record_no, b.record_no)
                    break
            if lock:
                break
        if lock is not None:
            assert_review_matches_bruteforce(
                records, allowed, locked_keys=[lock]
            )
