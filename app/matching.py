"""通联日志配对核心算法。

配对规则：
  1. 两条日志必须相互指认（A 本台/对方 == B 对方/本台）；
  2. 波段、模式相同；
  3. 两条带时区时间的绝对时差不超过演练允许时差；
  4. 每条记录最多参与一次配对。

优化目标（字典序优先级）：
  1. 最大化配对数量；
  2. 在配对数最大的前提下最小化总时差；
  3. 仍并列时取字典序最小的配对列表：每对的两个记录号先内部升序，
     再把所有配对整体升序后按列表字典序比较。

锁定（复核员人工互证）：
  ``locked_keys`` 中的记录对强制入选结果，即使因此拉低全场配对数也不
  替换；锁定对占用的记录从自由匹配池中剔除，剩余记录再按上述三级
  目标求解。调用方须保证锁定对合法（现存、相互指认、波段/模式一致、
  时差不超限），否则抛 :class:`InvalidLockError`。

实现方式：
  相互指认关系天然构成二分图——按 ``(min(本台,对方), max(本台,对方),
  波段, 模式)`` 分边，本台呼号较小的记录在左部 L，另一条在右部 R，
  所有候选边只可能跨部存在，因此是二分图。在此图上做最小费用最大流
  （min-cost max-flow, Dijkstra + potential）：
    * 每条流对应一对配对，最大流 => 配对数最大；
    * 费用 = 时差（微秒）* SCALE + OFFSET - 2**(E-1-rank)，
      其中边按配对键 (rec_a, rec_b) 升序赋予 rank 0..E-1，SCALE = 2**E。
      同一配对数下，时差之和的差至少为 SCALE，严格压倒字典序奖励
      （奖励之差的绝对值 <= 2**E - 1），因此最小费用依次满足
      “时差和最小”、“配对列表字典序最小”。OFFSET = 2**(E-1)
      保证所有边费用非负，初始 potential 为 0 即可用 Dijkstra。

复核唯一性核查（:func:`review_free_pairs`）只比较前两级目标
（配对数最大、总时差最小）下的全部合法方案：
    * 在同一张二分图上以“纯时差”为费用求最小费用最大流；
    * 若某条入选边与其右部端点在残余网络的零费用（约化费用为 0）
      强连通分量内，则沿该零费用环交换流量即可得到**同等配对数、
      同等总时差**的另一个完整方案——该对可替换；否则该对出现在
      所有此类方案中，为唯一选择；
    * 可替换时，排除该边后按上述三级目标（仍带字典序奖励）重算一次，
      所得即“不含该对的完整替代方案”中字典序最小的一份。
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Sequence

# 时差与微秒互相换算。
MICROS_PER_SECOND = 1_000_000


@dataclass(frozen=True)
class Record:
    """一条台站通联日志（算法所需字段）。"""

    record_no: str
    station: str       # 本台呼号（已规范化）
    counterpart: str   # 对方呼号（已规范化）
    ts_micros: int     # 带时区时间归一到 UTC 的微秒时间戳
    band: str          # 波段（已规范化）
    mode: str          # 模式（已规范化）


@dataclass(frozen=True)
class Pair:
    """一对互证成功的日志。"""

    record_a: str
    record_b: str
    time_diff_seconds: float

    def as_tuple(self) -> tuple[str, str]:
        return (self.record_a, self.record_b)


@dataclass(frozen=True)
class MatchResult:
    pairs: tuple[Pair, ...]
    unpaired: tuple[str, ...]
    total_time_diff_seconds: float


@dataclass(frozen=True)
class PairCheck:
    """单条当前配对在“最大配对数 + 最小总时差”全部方案中的核查结论。

    ``locked`` 为 ``True`` 时为人工锁定对：不参与自由匹配，标为不可
    替换（``in_all_schemes`` 恒为 ``True``），也不产生替代方案。
    自由配对可替换（``in_all_schemes=False``）时，``alternative`` 为
    不含该对的字典序最小完整自由匹配（:class:`MatchResult`）。
    """

    pair: Pair
    locked: bool
    in_all_schemes: bool
    alternative: MatchResult | None = None


@dataclass(frozen=True)
class ReviewResult:
    """复核核查结果：自由池最优基数/总时差，及逐条当前配对的结论。"""

    max_pair_count: int
    min_total_time_diff_seconds: float
    checks: tuple[PairCheck, ...]


class InvalidLockError(ValueError):
    """锁定对在当前日志集合上不再合法（缺失/不互证/波段模式/时差超限）。"""


def _pair_key(r1: Record, r2: Record) -> tuple[str, str]:
    a, b = r1.record_no, r2.record_no
    return (a, b) if a <= b else (b, a)


def pair_eligibility(
    record_map: dict[str, Record], allowed_micros: int, a: str, b: str
) -> tuple[str, str] | None:
    """检查 ``(a, b)`` 是否为可锁定的互证候选。

    合法返回 ``None``；否则返回 ``(原因码, 中文说明)``，原因码取值：
    ``same_record`` / ``record_missing`` / ``not_mutual`` /
    ``band_mode`` / ``time_diff``。
    """
    if a == b:
        return "same_record", f"两条记录号不能相同: {a}"
    ra = record_map.get(a)
    rb = record_map.get(b)
    if ra is None or rb is None:
        missing = a if ra is None else b
        return "record_missing", f"记录不存在: {missing}"
    if ra.station != rb.counterpart or ra.counterpart != rb.station:
        return "not_mutual", f"记录 {a} 与 {b} 未相互指认，不能锁定"
    if ra.band != rb.band or ra.mode != rb.mode:
        return "band_mode", f"记录 {a} 与 {b} 波段或模式不一致，不能锁定"
    diff_micros = abs(ra.ts_micros - rb.ts_micros)
    if diff_micros > allowed_micros:
        return (
            "time_diff",
            f"记录 {a} 与 {b} 时差 {diff_micros / MICROS_PER_SECOND:g} 秒"
            f"超过允许值 {allowed_micros / MICROS_PER_SECOND:g} 秒，不能锁定",
        )
    return None


def _candidate_edges(
    records: Sequence[Record], allowed_micros: int
) -> list[tuple[int, int, int, tuple[str, str]]]:
    """构造候选边 ``(左部下标, 右部下标, 时差微秒, 内部升序配对键)``。

    用右部索引 ``(本台, 对方, band, mode)`` 反查所有相互指认、波段/
    模式一致且时差不超限的候选；结果按配对键升序（rank 顺序）。
    """
    right_index: dict[tuple[str, str, str, str], list[int]] = {}
    left_flags: list[bool] = []
    for i, r in enumerate(records):
        is_left = (r.station, r.counterpart) < (r.counterpart, r.station)
        left_flags.append(is_left)
        if not is_left:
            key = (r.station, r.counterpart, r.band, r.mode)
            right_index.setdefault(key, []).append(i)

    edges: list[tuple[int, int, int, tuple[str, str]]] = []
    for i, r in enumerate(records):
        if not left_flags[i]:
            continue
        for j in right_index.get(
            (r.counterpart, r.station, r.band, r.mode), []
        ):
            diff_micros = abs(records[j].ts_micros - r.ts_micros)
            if diff_micros <= allowed_micros:
                edges.append((i, j, diff_micros, _pair_key(r, records[j])))
    edges.sort(key=lambda e: e[3])
    return edges


@dataclass
class _FlowNetwork:
    """二分图最小费用最大流网络（前向星残余图）。"""

    graph: list[list[list[int]]]
    source: int
    sink: int
    # 与候选边对齐：(左部网络节点, L->R 前向边下标)
    match_arc: list[tuple[int, int]]
    # 配对键 -> 候选边 rank（排除指定边重算时定位用）
    key_rank: dict[tuple[str, str], int]


def _build_network(
    edges: Sequence[tuple[int, int, int, tuple[str, str]]],
    costs: Sequence[int],
) -> _FlowNetwork:
    """按候选边与逐条边费用构造 s->L->R->t 容量 1 的流网络。"""
    left_nodes = sorted({i for (i, _j, _w, _k) in edges})
    right_nodes = sorted({j for (_i, j, _w, _k) in edges})
    lidx = {ri: k + 1 for k, ri in enumerate(left_nodes)}
    ridx = {rj: k + 1 + len(left_nodes) for k, rj in enumerate(right_nodes)}
    source = 0
    sink = 1 + len(left_nodes) + len(right_nodes)
    n_nodes = sink + 1

    graph: list[list[list[int]]] = [[] for _ in range(n_nodes)]

    def add_edge(u: int, v: int, cap: int, cost: int) -> int:
        # 返回“u 的前向边”下标；向同一 u 追加新边不会改变旧下标。
        fwd = len(graph[u])
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, fwd])
        return fwd

    # 每个左/右节点只连一条容量 1 的源/汇边，保证每条记录最多用一次。
    for u in range(1, 1 + len(left_nodes)):
        add_edge(source, u, 1, 0)
    for k in range(len(right_nodes)):
        v = 1 + len(left_nodes) + k
        add_edge(v, sink, 1, 0)

    match_arc: list[tuple[int, int]] = []
    key_rank: dict[tuple[str, str], int] = {}
    for rank, (i, j, _weight, key) in enumerate(edges):
        u = lidx[i]
        v = ridx[j]
        fwd = add_edge(u, v, 1, costs[rank])
        match_arc.append((u, fwd))
        key_rank[key] = rank
    return _FlowNetwork(graph, source, sink, match_arc, key_rank)


def _run_min_cost_flow(net: _FlowNetwork) -> tuple[int, list[int]]:
    """在网络上跑最小费用最大流（Dijkstra + potential）。

    返回 ``(最大流条数, 最终 potential)``。容量均为 1，每次增广一条。
    最终残余网络配合返回的 potential 满足“约化费用非负”，可直接据此
    判定零费用环。
    """
    graph = net.graph
    n_nodes = len(graph)
    potential = [0] * n_nodes
    flow = 0
    while True:
        inf = None  # Python 大整数无溢出风险，用 None 表示不可达
        dist = [inf] * n_nodes
        prev_v = [-1] * n_nodes
        prev_e = [-1] * n_nodes
        dist[net.source] = 0
        heap: list[tuple[int, int]] = [(0, net.source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d != dist[u]:
                continue
            for ei, arc in enumerate(graph[u]):
                v, cap, cost, _rev = arc
                if cap <= 0:
                    continue
                nd = d + cost + potential[u] - potential[v]
                if dist[v] is None or nd < dist[v]:
                    dist[v] = nd
                    prev_v[v] = u
                    prev_e[v] = ei
                    heapq.heappush(heap, (nd, v))
        if dist[net.sink] is None:
            break
        for v in range(n_nodes):
            if dist[v] is not None:
                potential[v] += dist[v]
        # 所有容量均为 1，每次增广 1 条流。
        v = net.sink
        while v != net.source:
            u = prev_v[v]
            ei = prev_e[v]
            arc = graph[u][ei]
            arc[1] -= 1
            rev = graph[v][arc[3]]
            rev[1] += 1
            v = u
        flow += 1
    return flow, potential


def _extract_match(
    records: Sequence[Record],
    edges: Sequence[tuple[int, int, int, tuple[str, str]]],
    net: _FlowNetwork,
) -> tuple[list[Pair], set[str], int]:
    """从求流后的残余网络提取入选配对、已配对记录号与总时差（微秒）。"""
    pairs: list[Pair] = []
    matched: set[str] = set()
    total_micros = 0
    for rank, (i, j, weight, key) in enumerate(edges):
        # L->R 的前向边容量被用尽 => 该边入选匹配。
        u, fwd = net.match_arc[rank]
        if net.graph[u][fwd][1] == 0:
            total_micros += weight
            pairs.append(
                Pair(
                    record_a=key[0],
                    record_b=key[1],
                    time_diff_seconds=weight / MICROS_PER_SECOND,
                )
            )
            matched.add(records[i].record_no)
            matched.add(records[j].record_no)
    return pairs, matched, total_micros


def _lex_costs(
    edges: Sequence[tuple[int, int, int, tuple[str, str]]]
) -> list[int]:
    """三级目标费用：时差*SCALE + OFFSET - 字典序奖励（见模块文档）。"""
    e_count = len(edges)
    scale = 1 << e_count          # 2**E，时差主项的放大倍数
    # 入选 rank 边的字典序奖励为 2**(E-1-rank)：rank 0 为最高位，
    # 最大化奖励之和等价于最大化“从 rank 0 起的含边掩码”的字典序。
    bonus = [1 << (e_count - 1 - rank) for rank in range(e_count)]
    # 费用 = 时差*SCALE + 偏移 - 奖励；最小化费用即最大化奖励。
    # 最大奖励为 2**(E-1)，减去奖励后该项恒 >= 0，故费用恒非负，
    # 初始 potential 全为 0 即可直接使用 Dijkstra。
    offset = 1 << (e_count - 1)
    return [
        weight * scale + offset - bonus[rank]
        for rank, (_i, _j, weight, _k) in enumerate(edges)
    ]


def _free_match_result(
    records: Sequence[Record],
    pairs: Sequence[Pair],
    matched: set[str],
    total_micros: int,
) -> MatchResult:
    """把自由匹配的配对/占用集合/总时差组装为 :class:`MatchResult`。"""
    ordered = sorted(pairs, key=lambda p: p.as_tuple())
    unpaired = tuple(
        r.record_no for r in records if r.record_no not in matched
    )
    return MatchResult(
        pairs=tuple(ordered),
        unpaired=unpaired,
        total_time_diff_seconds=total_micros / MICROS_PER_SECOND,
    )


def _match_free_with_edges(
    records: Sequence[Record],
    edges: Sequence[tuple[int, int, int, tuple[str, str]]],
    exclude_key: tuple[str, str] | None = None,
) -> tuple[list[Pair], set[str], int]:
    """在给定候选边上按三级目标求全场最优自由匹配。

    ``exclude_key`` 给出时从候选边中剔除该配对（用于复核替代方案：
    在不含当前某对的前提下取字典序最小方案）。
    """
    if exclude_key is not None:
        edges = [e for e in edges if e[3] != exclude_key]
    if not edges:
        return [], set(), 0
    net = _build_network(edges, _lex_costs(edges))
    _run_min_cost_flow(net)
    return _extract_match(records, edges, net)


def _match_free(
    records: Sequence[Record], allowed_micros: int
) -> tuple[list[Pair], set[str], int]:
    """在无锁定记录的集合上求全场最优匹配。

    返回 ``(配对列表, 已配对记录号集合, 总时差微秒)``。
    """
    edges = _candidate_edges(records, allowed_micros)
    return _match_free_with_edges(records, edges)


def match_records(
    records: Sequence[Record],
    allowed_diff_seconds: float,
    locked_keys: Sequence[tuple[str, str]] = (),
) -> MatchResult:
    """计算全场最优互证配对。

    ``records`` 内的记录号须唯一（服务层已保证）。``locked_keys`` 中的
    记录对强制入选（服务层已校验合法且互不占用），其记录不参与自由
    匹配；即使锁定使全场配对数下降，锁定对也保留不替换。
    """
    allowed_micros = int(round(allowed_diff_seconds * MICROS_PER_SECOND))
    record_map = {r.record_no: r for r in records}

    # --- 先固定全部锁定对，并把其记录移出自由匹配池 ---
    pairs: list[Pair] = []
    locked_ids: set[str] = set()
    total_micros = 0
    for raw_a, raw_b in locked_keys:
        a, b = (raw_a, raw_b) if raw_a <= raw_b else (raw_b, raw_a)
        problem = pair_eligibility(record_map, allowed_micros, a, b)
        if problem is not None:
            raise InvalidLockError(f"锁定 ({a}, {b}) 已失效: {problem[1]}")
        if a in locked_ids or b in locked_ids:
            raise InvalidLockError(f"记录被多个锁定对重复占用: {a}, {b}")
        locked_ids.add(a)
        locked_ids.add(b)
        weight = abs(record_map[a].ts_micros - record_map[b].ts_micros)
        total_micros += weight
        pairs.append(
            Pair(
                record_a=a,
                record_b=b,
                time_diff_seconds=weight / MICROS_PER_SECOND,
            )
        )

    free_records = [r for r in records if r.record_no not in locked_ids]
    free_pairs, free_matched, free_total = _match_free(
        free_records, allowed_micros
    )
    pairs.extend(free_pairs)
    total_micros += free_total

    pairs.sort(key=lambda p: p.as_tuple())
    matched = locked_ids | free_matched
    unpaired = tuple(
        r.record_no for r in records if r.record_no not in matched
    )
    return MatchResult(
        pairs=tuple(pairs),
        unpaired=unpaired,
        total_time_diff_seconds=total_micros / MICROS_PER_SECOND,
    )


# --------------------------------------------------------------------------
# 复核唯一性核查：当前每条非锁定配对是否出现在所有“配对数最大、总时差
# 最小”的合法完整方案中；可替换时给一份字典序最小的替代方案。
# --------------------------------------------------------------------------
def _zero_scc(
    net: _FlowNetwork, potential: Sequence[int]
) -> list[int]:
    """残余网络中“约化费用为 0”的弧构成的子图的强连通分量。

    返回每个网络节点的分量号（同一分量号相同；遍历顺序不影响“是否同
    分量”的判定）。用迭代 Kosaraju 避免大图递归深度问题。
    """
    graph = net.graph
    n_nodes = len(graph)
    zero_adj: list[list[int]] = [[] for _ in range(n_nodes)]
    reverse_adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, arcs in enumerate(graph):
        pu = potential[u]
        for v, cap, cost, _rev in arcs:
            if cap > 0 and cost + pu - potential[v] == 0:
                zero_adj[u].append(v)
                reverse_adj[v].append(u)

    # 第一遍 DFS：后序（迭代实现）。
    visited = [False] * n_nodes
    order: list[int] = []
    for start in range(n_nodes):
        if visited[start]:
            continue
        visited[start] = True
        stack: list[tuple[int, int]] = [(start, 0)]
        while stack:
            u, idx = stack[-1]
            if idx < len(zero_adj[u]):
                v = zero_adj[u][idx]
                stack[-1] = (u, idx + 1)
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, 0))
            else:
                order.append(u)
                stack.pop()

    # 第二遍：逆序在反图上 DFS 标分量。
    component = [-1] * n_nodes
    comp_id = 0
    for start in reversed(order):
        if component[start] != -1:
            continue
        component[start] = comp_id
        stack = [start]
        while stack:
            u = stack.pop()
            for v in reverse_adj[u]:
                if component[v] == -1:
                    component[v] = comp_id
                    stack.append(v)
        comp_id += 1
    return component


def review_free_pairs(
    records: Sequence[Record],
    allowed_diff_seconds: float,
    locked_keys: Sequence[tuple[str, str]],
) -> ReviewResult:
    """复核当前互证结果中每条配对的唯一性。

    以同一批日志、允许时差与锁定/隔离状态为准（被隔离记录已由调用方
    从 ``records`` 中剔除）：先固定全部锁定对（其记录移出自由匹配池，
    锁定对不可替换），再在自由匹配池上枚举所有满足“配对数全场最大、
    总时差在该数量下最小”的合法方案，判定当前每条自由配对是否出现在
    **所有** 这类方案中；可替换时给出**一份**不含该对的完整替代方案
    （按配对记录号字典序规则取最小）。
    """
    allowed_micros = int(round(allowed_diff_seconds * MICROS_PER_SECOND))
    record_map = {r.record_no: r for r in records}

    # --- 固定锁定对（合法性/互不占用），其记录移出自由匹配池 ---
    locked_pairs: list[Pair] = []
    locked_ids: set[str] = set()
    locked_total_micros = 0
    for raw_a, raw_b in locked_keys:
        a, b = (raw_a, raw_b) if raw_a <= raw_b else (raw_b, raw_a)
        problem = pair_eligibility(record_map, allowed_micros, a, b)
        if problem is not None:
            raise InvalidLockError(f"锁定 ({a}, {b}) 已失效: {problem[1]}")
        if a in locked_ids or b in locked_ids:
            raise InvalidLockError(f"记录被多个锁定对重复占用: {a}, {b}")
        locked_ids.add(a)
        locked_ids.add(b)
        weight = abs(record_map[a].ts_micros - record_map[b].ts_micros)
        locked_total_micros += weight
        locked_pairs.append(
            Pair(
                record_a=a,
                record_b=b,
                time_diff_seconds=weight / MICROS_PER_SECOND,
            )
        )

    free_records = [r for r in records if r.record_no not in locked_ids]
    edges = _candidate_edges(free_records, allowed_micros)

    # 空场（无候选边）：没有可核查的自由配对。
    if not edges:
        checks = tuple(
            PairCheck(pair=p, locked=True, in_all_schemes=True)
            for p in sorted(locked_pairs, key=lambda x: x.as_tuple())
        )
        return ReviewResult(
            max_pair_count=len(locked_pairs),
            min_total_time_diff_seconds=(
                locked_total_micros / MICROS_PER_SECOND
            ),
            checks=checks,
        )

    # --- 纯时差费用的最小费用最大流：刻画前两级目标下的全部最优方案 ---
    plain_costs = [weight for (_i, _j, weight, _k) in edges]
    net = _build_network(edges, plain_costs)
    max_count, potential = _run_min_cost_flow(net)
    _plain_pairs, _plain_matched, free_min_micros = _extract_match(
        free_records, edges, net
    )
    component = _zero_scc(net, potential)

    # 当前自由结果 = 三级目标（含字典序）下的自由匹配；其每条入选边
    # 两端在零费用 SCC 同分量 <=> 存在同基数同总时差的另一方案。
    current_pairs, _matched, _total = _match_free_with_edges(
        free_records, edges
    )
    current_keys = {p.as_tuple() for p in current_pairs}
    replaceable: set[tuple[str, str]] = set()
    for rank, (i, j, _weight, key) in enumerate(edges):
        if key not in current_keys:
            continue
        u, fwd = net.match_arc[rank]
        v = net.graph[u][fwd][0]
        if component[u] == component[v]:
            replaceable.add(key)

    checks: list[PairCheck] = [
        PairCheck(pair=p, locked=True, in_all_schemes=True)
        for p in locked_pairs
    ]
    for pair in sorted(current_pairs, key=lambda p: p.as_tuple()):
        key = pair.as_tuple()
        if key not in replaceable:
            checks.append(PairCheck(pair=pair, locked=False,
                                    in_all_schemes=True))
            continue
        # 不含该对的方案：排除该边后仍按三级目标重算 => 同基数、同最小
        # 总时差方案中配对列表字典序最小的一份。
        alt_pairs, alt_matched, alt_total = _match_free_with_edges(
            free_records, edges, exclude_key=key
        )
        checks.append(
            PairCheck(
                pair=pair,
                locked=False,
                in_all_schemes=False,
                alternative=_free_match_result(
                    free_records, alt_pairs, alt_matched, alt_total
                ),
            )
        )

    checks.sort(key=lambda c: c.pair.as_tuple())
    return ReviewResult(
        max_pair_count=len(locked_pairs) + max_count,
        min_total_time_diff_seconds=(
            (locked_total_micros + free_min_micros) / MICROS_PER_SECOND
        ),
        checks=tuple(checks),
    )


# --------------------------------------------------------------------------
# 未互证日志缺口诊断（复核员按演练查看未配对记录的缺口归因）
#
# 对每条当前结果中的未配对记录，按当前日志依次归因（前者命中即不再下探）：
#   1. candidate_occupied（候选已占用）：存在完全满足互指、波段、模式、
#      时差条件的对方记录。最大基数匹配下该候选必然已被别的配对占用
#      （否则增广这条边即可多配一对），故未入选的原因是候选被占用；
#   2. time_diff_exceeded（时差超限）：否则存在互指且波段、模式相同的
#      对方记录（时差必然都超过允许值，否则会落入上一类）；
#   3. band_or_mode_mismatch（波段或模式不符）：否则存在互指的对方记录；
#   4. missing_counterpart_log（缺少对方日志）：没有任何互指记录。
# 前三类返回该类候选中时差最小、再按记录号最小的对方记录及实际时差。
# --------------------------------------------------------------------------
GAP_CANDIDATE_OCCUPIED = "candidate_occupied"
GAP_TIME_DIFF_EXCEEDED = "time_diff_exceeded"
GAP_BAND_OR_MODE_MISMATCH = "band_or_mode_mismatch"
GAP_MISSING_COUNTERPART = "missing_counterpart_log"

GAP_REASON_LABELS: dict[str, str] = {
    GAP_CANDIDATE_OCCUPIED: "候选已占用",
    GAP_TIME_DIFF_EXCEEDED: "时差超限",
    GAP_BAND_OR_MODE_MISMATCH: "波段或模式不符",
    GAP_MISSING_COUNTERPART: "缺少对方日志",
}


@dataclass(frozen=True)
class GapItem:
    """一条未配对记录的缺口诊断结论。

    末类（缺少对方日志）``counterpart_record_no`` 与
    ``time_diff_seconds`` 均为 ``None``。
    """

    record_no: str
    reason: str
    counterpart_record_no: str | None
    time_diff_seconds: float | None


def _is_mutual(r1: Record, r2: Record) -> bool:
    """两条日志是否相互指认（本台/对方呼号互换）。"""
    return r1.station == r2.counterpart and r1.counterpart == r2.station


def diagnose_gaps(
    records: Sequence[Record],
    allowed_diff_seconds: float,
    unpaired: Sequence[str],
) -> list[GapItem]:
    """按当前日志为每条未配对记录归因缺口原因。

    ``unpaired`` 取当前互证结果的未配对记录（其记录必现存于
    ``records``）；结果按记录号升序返回。前三类的对方记录在各自候选
    集合内按“时差最小、再记录号最小”择一返回。
    """
    allowed_micros = int(round(allowed_diff_seconds * MICROS_PER_SECOND))
    record_map = {r.record_no: r for r in records}
    items: list[GapItem] = []
    for no in sorted(unpaired):
        u = record_map[no]
        # 三个逐层收窄的候选集（均含 (时差微秒, 对方记录号)）：
        # mutual ⊇ band_mode ⊇ eligible。
        mutual: list[tuple[int, str]] = []
        band_mode: list[tuple[int, str]] = []
        eligible: list[tuple[int, str]] = []
        for v in records:
            if v.record_no == no or not _is_mutual(u, v):
                continue
            diff_micros = abs(u.ts_micros - v.ts_micros)
            mutual.append((diff_micros, v.record_no))
            if v.band == u.band and v.mode == u.mode:
                band_mode.append((diff_micros, v.record_no))
                if diff_micros <= allowed_micros:
                    eligible.append((diff_micros, v.record_no))

        if eligible:
            reason, pool = GAP_CANDIDATE_OCCUPIED, eligible
        elif band_mode:
            reason, pool = GAP_TIME_DIFF_EXCEEDED, band_mode
        elif mutual:
            reason, pool = GAP_BAND_OR_MODE_MISMATCH, mutual
        else:
            items.append(GapItem(no, GAP_MISSING_COUNTERPART, None, None))
            continue
        diff_micros, other_no = min(pool, key=lambda x: (x[0], x[1]))
        items.append(
            GapItem(
                record_no=no,
                reason=reason,
                counterpart_record_no=other_no,
                time_diff_seconds=diff_micros / MICROS_PER_SECOND,
            )
        )
    return items
