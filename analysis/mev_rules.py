"""Transaction-level MEV evidence, separated from economic labels and cohorts.

Qin et al. (2022), section IV, supplies the ordered cycle and sandwich baseline.
Chi et al. (2024), algorithms 3/4, motivates the separately named graph-cycle
and combined-sandwich variants. Source versions are in config/mev_sources.json.

What decides a label
--------------------
The primary label is the literature pattern and nothing else. Arbitrage is
Qin's ordered cycle, heuristics H1 to H4; sandwich is Qin's direction, order and
90 to 110 percent quantity test, H1 to H5. Three added restrictions are kept and
named in every record: two distinct pools, strictly positive amounts, and an
independently supported shared executor before two different senders may be
treated as one actor.

Classification does not require measured profit or attribution to an EOA.
Qin's pattern can hold when proceeds remain in the executing contract or pass
through an internal ether transfer that emits no log. :func:`evaluate_candidate`
reports sender-side ERC-20 flows separately from the classification.
``profit_status`` is ``not_measured``; absent sender transfers do not establish
that a transaction was unprofitable.

Native value is never measured here. Internal calls, plain value transfers,
``selfdestruct`` refunds and coinbase payments emit no log, so a hidden ether
outflow is always possible. That possibility cannot erase a Qin pattern and
cannot establish that it is profitable.

A graph cycle that Qin's log order does not support is a separate, lesser claim.
It stays a candidate, because the graph search checks no amounts at all, so a
flash-loan-shaped or quantity-violating loop reaches it. A combined sandwich
keeps its constituent Qin evidence or stays a candidate; neither is a complete
Chi reproduction, which needs call traces this module never sees.

Inputs are decoded records, not pool-count aggregates. Missing swap metadata
keeps a pattern incomplete rather than letting it become a positive finding, and
missing cash-flow evidence never becomes a zero balance.
"""
from collections import defaultdict
from fractions import Fraction
from itertools import product
import hashlib
import json


RULE_VERSION = "qin_primary_labels_20261018"

#: Qin et al. (2022) IV-C. The ordered cycle, as heuristics, in its own terms.
QIN_ARBITRAGE = {
    "label": "qin_2022_arbitrage_ordered_cycle",
    "source": "Qin et al. (2022), section IV-C",
    "heuristics": {
        "H1": "all swap actions occur within one transaction",
        "H2": "the transaction contains more than one swap action",
        "H3": "the ordered swaps form a closed loop: each output asset is the "
              "next input asset, and the final output closes onto the first input",
        "H4": "each hop's input amount does not exceed the previous hop's "
              "output amount",
    },
}

#: Qin et al. (2022) IV-A. Direction, order and the 90 to 110 percent band.
QIN_SANDWICH = {
    "label": "qin_2022_sandwich_direction_order_quantity",
    "source": "Qin et al. (2022), section IV-A",
    "heuristics": {
        "H1": "front, victim and back transactions occur in that order within one block",
        "H2": "each front transaction maps to one back transaction",
        "H3": "front and victim swaps exchange the same asset pair in the same "
              "direction; the back swap reverses that direction",
        "H4": "the attacker legs share a sender or use the same executing contract",
        "H5": "the back swap's input is 90 to 110 percent of the front swap's "
              "output",
    },
}

#: Restrictions this module adds to the published heuristics, kept because each
#: removes a class of false positive, and named in every record so a reader can
#: tell the published rule from the local one.
ADDED_RESTRICTIONS = {
    "two_distinct_pools": "an arbitrage cycle must touch at least two "
                          "distinct pools; this excludes same-pool round trips "
                          "that Qin's four stated heuristics do not exclude",
    "positive_amounts": "a swap leg with a zero amount is not usable for a "
                        "pattern",
    "shared_executor_requires_explicit_support": "two different senders are "
        "treated as one actor only through a destination the caller has "
        "independently supported; a shared public router never suffices",
}

#: Which added restriction bears on which strategy. A record lists only the
#: restrictions that actually constrained it.
#:
#: ``two_distinct_pools`` is an arbitrage rule. A sandwich acts on one pool by
#: H3, so listing it on a sandwich record states the opposite of the pattern
#: the record describes. ``shared_executor_requires_explicit_support`` is a
#: sandwich rule: an arbitrage cycle is one transaction, so there are never two
#: senders to associate. ``positive_amounts`` is enforced in :func:`_swaps` and
#: so constrains both.
#:
#: Liquidations are absent on purpose. They are not built by :func:`_candidate`
#: and carry no such list; their own zero-amount rule lives in
#: :func:`liquidation_evidence` and shows up there as the
#: ``zero_amount_liquidation`` outcome.
STRATEGY_RESTRICTIONS = {
    "arbitrage": ("positive_amounts", "two_distinct_pools"),
    "sandwich": ("positive_amounts",
                 "shared_executor_requires_explicit_support"),
}

# Every declared restriction must be assigned to at least one strategy, and no
# strategy may cite one that is not declared. Checked at import so a new
# restriction cannot be added without deciding where it applies.
assert set().union(*STRATEGY_RESTRICTIONS.values()) == set(ADDED_RESTRICTIONS)

#: Restrictions removed from the label path by the 2026-10-18 decision. Kept
#: visible so a reader of an older result can see what changed and why.
REMOVED_RESTRICTIONS = {
    "sender_side_positive_erc20_flow": "an added restriction absent from Qin. "
        "Proceeds retained in the executing contract, or moved by an internal "
        "ether transfer that emits no log, are invisible to it, so it rejected "
        "patterns Qin identifies. Reported as a diagnostic, not a classification requirement.",
    "sender_in_balance_scope": "attribution to an observed EOA is reported, "
        "not required. Principal attribution stays unassigned without an "
        "explicit scope basis.",
}

#: Attached to every record. Each names something the label does not assert.
BASE_DEVIATIONS = (
    "profit_not_required_for_label",
    "native_value_not_measured",
    "principal_attribution_not_asserted",
)
#: Added wherever a Chi construction is referenced but not reproduced.
CHI_INCOMPLETE = "chi_2024_completeness_requires_call_traces"


class EvidenceError(ValueError):
    pass


def _integer(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{name} must be an exact integer")
    if value < (1 if positive else 0):
        raise EvidenceError(f"{name} is out of range")
    return value


def _identity(row):
    return (row["transaction_hash"], row["log_index"])


def _order(row):
    return (row["block_number"], row["transaction_index"], row["log_index"])


def _swaps(rows):
    rows = list(rows)
    seen, transactions = set(), {}
    for row in rows:
        if row.get("status") != "decoded":
            raise EvidenceError("Rules require complete decoded swaps; retain unresolved rows separately")
        for name in ("transaction_hash", "pool", "token_in", "token_out"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise EvidenceError(f"Missing swap field: {name}")
        for name in ("block_number", "transaction_index", "log_index"):
            _integer(row.get(name), name)
        for name in ("amount_in", "amount_out"):
            _integer(row.get(name), name, positive=True)
        if row["token_in"] == row["token_out"]:
            raise EvidenceError("A decoded swap must exchange distinct assets")
        key = _identity(row)
        if key in seen:
            raise EvidenceError("Duplicate swap identity")
        seen.add(key)
        metadata = (row["block_number"], row["transaction_index"], row.get("tx_from"), row.get("tx_to"))
        previous = transactions.setdefault(row["transaction_hash"], metadata)
        if previous != metadata:
            raise EvidenceError("Inconsistent transaction metadata across swaps")
    return sorted(rows, key=_order)


def _digest(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _candidate(strategy, variant, rows, *, victims=(), details=None,
               status="candidate", reason="pattern_incomplete",
               literature=None, heuristics=(), deviations=()):
    """One evidence record. ``status`` is the literature verdict, nothing else.

    ``actor_senders`` is the set of transaction senders observed on the
    pattern's own legs. It is an observation, not a principal: the sender of a
    transaction need not own the balances it moves, so
    ``principal_attribution`` stays ``unassigned`` until a caller supplies an
    explicit scope basis.
    """
    if strategy not in STRATEGY_RESTRICTIONS:
        raise EvidenceError(f"No restriction set declared for strategy {strategy!r}")
    rows = sorted(rows, key=_order)
    txs = sorted({row["transaction_hash"] for row in rows})
    senders = sorted({row["tx_from"] for row in rows if row.get("tx_from")})
    literature = literature or {}
    satisfied = tuple(heuristics)
    result = {
        "strategy": strategy, "variant": variant, "rule_version": RULE_VERSION,
        "status": status, "reason": reason,
        "literature_label": literature.get("label", "none"),
        "literature_source": literature.get("source", "none"),
        "literature_heuristics_satisfied": list(satisfied),
        "literature_heuristics": {key: literature.get("heuristics", {})[key]
                                  for key in satisfied
                                  if key in literature.get("heuristics", {})},
        # Only the restrictions that bear on this strategy, not the glossary.
        "added_restrictions": sorted(STRATEGY_RESTRICTIONS[strategy]),
        "deviations": sorted(set(BASE_DEVIATIONS) | set(deviations)),
        "block_numbers": sorted({row["block_number"] for row in rows}),
        "transaction_hashes": txs, "actor_senders": senders,
        "actor_senders_are_observed_not_principals": True,
        "principal_attribution": "unassigned",
        "swap_ids": [list(_identity(row)) for row in rows],
        "pools": sorted({row["pool"] for row in rows}),
        "victim_transaction_hashes": sorted(set(victims)),
        "details": details or {},
        "profit_status": "not_measured",
    }
    result["evidence_id"] = _digest(result)
    return result


def ordered_cycle(rows):
    """Qin-style log-order baseline; not a multi-pool proxy.

    The graph variant is separate because nested calls can emit swap logs in a
    different order from the asset path. Equality in the closing amount is kept
    as a pattern; strict positive cash flow is a later, distinct requirement.
    """
    rows = _swaps(rows)
    if len(rows) < 2 or len({r["transaction_hash"] for r in rows}) != 1:
        return False
    if len({r["pool"] for r in rows}) < 2:
        return False
    following = rows[1:] + rows[:1]
    return all(left["token_out"] == right["token_in"]
               and right["amount_in"] <= left["amount_out"]
               for left, right in zip(rows, following))


#: Criteria in the order the audit reports a first failure, using the names the
#: independent review used so its counts can be reconciled against ours.
CYCLE_CRITERIA = (
    "h1_single_transaction",
    "h2_more_than_one_swap_action",
    "h3_loop_fails_in_log_order",
    "h4_interior_fails",
    "h4_wraparound_fails_cycle_loses_value",
    "two_distinct_pools_not_satisfied",
)

_CYCLE_FAILURE_DEVIATIONS = {
    "h3_loop_fails_in_log_order": "qin_h3_log_order_not_satisfied",
    "h4_interior_fails": "qin_h4_interior_amount_bound_not_satisfied",
    "h4_wraparound_fails_cycle_loses_value":
        "qin_h4_closing_amount_bound_not_satisfied",
    "h1_single_transaction": "qin_h1_not_single_transaction",
    "h2_more_than_one_swap_action": "qin_h2_single_swap_action",
    "two_distinct_pools_not_satisfied": "two_distinct_pools_not_satisfied",
}


def cycle_qin_audit(cycle, *, transaction_swap_count=None):
    """Test one cycle's swap subset, read in log order, against Qin H1 to H4.

    Qin's H3 is stated over "the n swap actions s1..sn **of an arbitrage**", and
    section IV-C explicitly allows one transaction to carry two arbitrages. The
    unit of the test is therefore the cycle's own legs, not every swap the
    transaction happens to contain: a transaction that routes a three-leg cycle
    and two unrelated swaps still contains the arbitrage Qin describes.

    The legs are read in log order among themselves, so intervening non-cycle
    logs do not break the chain. Interior hops and the closing wraparound hop
    are reported separately, because a loop that chains correctly and only
    loses value on the close fails H4 alone, and saying H3 failed there would
    be untrue.
    """
    cycle = sorted(cycle, key=_order)
    hops = len(cycle)
    if hops == 0:
        raise EvidenceError("An empty cycle cannot be audited")
    following = cycle[1:]
    interior_chain = all(left["token_out"] == right["token_in"]
                         for left, right in zip(cycle, following))
    loop_closes = cycle[-1]["token_out"] == cycle[0]["token_in"]
    interior_amounts = all(right["amount_in"] <= left["amount_out"]
                           for left, right in zip(cycle, following))
    wraparound_amount = cycle[0]["amount_in"] <= cycle[-1]["amount_out"]
    swaps_in_transaction = (len(cycle) if transaction_swap_count is None
                            else int(transaction_swap_count))
    checks = {
        "h1_single_transaction": len({r["transaction_hash"] for r in cycle}) == 1,
        "h2_more_than_one_swap_action": swaps_in_transaction > 1,
        "h3_loop_fails_in_log_order": not (interior_chain and loop_closes),
        "h4_interior_fails": not interior_amounts,
        "h4_wraparound_fails_cycle_loses_value": not wraparound_amount,
        "two_distinct_pools_not_satisfied": len({r["pool"] for r in cycle}) < 2,
    }
    # h1 and h2 are stated positively; the rest name a failure.
    failures = [name for name in CYCLE_CRITERIA
                if (not checks[name] if name.startswith(("h1_", "h2_"))
                    else checks[name])]
    return {
        "swap_ids": [list(_identity(row)) for row in cycle],
        "log_indices": [row["log_index"] for row in cycle],
        "pools": sorted({row["pool"] for row in cycle}),
        "hops": hops,
        "cycle_asset": cycle[0]["token_in"],
        "transaction_swap_count": swaps_in_transaction,
        "h1_single_transaction": checks["h1_single_transaction"],
        "h2_more_than_one_swap_action": checks["h2_more_than_one_swap_action"],
        "h3_interior_chain": interior_chain,
        "h3_loop_closes": loop_closes,
        "h3_closed_loop_in_log_order": interior_chain and loop_closes,
        "h4_interior_hops": interior_amounts,
        "h4_wraparound_hop": wraparound_amount,
        "h4_all_hops_incl_wraparound": interior_amounts and wraparound_amount,
        "two_distinct_pools": not checks["two_distinct_pools_not_satisfied"],
        "qualifies": not failures,
        "failures": failures,
        "first_failing_criterion": failures[0] if failures else None,
    }


def _within_scc_adjacency(outgoing):
    """Keep only cycle-capable edges, using two iterative SCC passes."""
    reverse = defaultdict(list)
    for token, targets in outgoing.items():
        reverse.setdefault(token, [])
        for target in targets:
            reverse[target].append(token)
    visited, finished = set(), []
    for start in reverse:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, iter(outgoing.get(start, ())))]
        while stack:
            token, targets = stack[-1]
            try:
                target = next(targets)
            except StopIteration:
                stack.pop()
                finished.append(token)
                continue
            if target not in visited:
                visited.add(target)
                stack.append((target, iter(outgoing.get(target, ()))))
    component = {}
    for start in reversed(finished):
        if start in component:
            continue
        component[start] = start
        stack = [start]
        while stack:
            token = stack.pop()
            for predecessor in reverse[token]:
                if predecessor not in component:
                    component[predecessor] = start
                    stack.append(predecessor)
    return {token: [target for target in targets if component[token] == component[target]]
            for token, targets in outgoing.items()}


def graph_cycles(rows, *, max_states=100000):
    """Return deterministic simple token cycles, without imposing log order.

    Exhausting the search budget raises. It must never be interpreted as an
    absence of arbitrage. Parallel edges keep their individual swap identities.

    Walk distinct token pairs from each cycle's smallest token, then expand
    every leg combination at closure. A simple cycle visits each pair once,
    so this preserves all concrete cycles, canonical identities and ordering.
    The two-pool filter still applies to each concrete combination.

    ``max_states`` charges both collapsed edge examinations and expanded leg
    combinations. It bounds both search and output work; its units differ
    from the earlier per-leg traversal's state count. Linear SCC preprocessing,
    like adjacency construction, is outside this counter.
    """
    rows = _swaps(rows)
    _integer(max_states, "max_states", positive=True)
    if len({r["transaction_hash"] for r in rows}) > 1:
        raise EvidenceError("An atomic cycle cannot span transactions")
    # Parallel legs between one token pair are collapsed to a single edge for
    # the walk and expanded again when a cycle closes.
    legs = defaultdict(list)
    for index, row in enumerate(rows):
        legs[(row["token_in"], row["token_out"])].append(index)
    outgoing = defaultdict(list)
    for token_in, token_out in sorted(legs):
        outgoing[token_in].append(token_out)
    # A directed cycle cannot cross between strongly connected components.
    outgoing = _within_scc_adjacency(outgoing)
    cycles, seen, work = [], set(), 0

    def expand(pairs):
        """Every choice of one leg per hop, canonicalised exactly as before."""
        nonlocal work
        for combination in product(*(legs[pair] for pair in pairs)):
            work += 1
            if work > max_states:
                raise EvidenceError("Cycle search budget exhausted")
            cycle = list(combination)
            if len(cycle) < 2 or len({rows[i]["pool"] for i in cycle}) < 2:
                continue
            rotations = [tuple(cycle[i:] + cycle[:i]) for i in range(len(cycle))]
            key = min(rotations)
            if key not in seen:
                seen.add(key)
                cycles.append(key)

    def walk(start_token, token, pairs, tokens):
        nonlocal work
        for nxt in outgoing.get(token, ()):
            work += 1
            if work > max_states:
                raise EvidenceError("Cycle search budget exhausted")
            if nxt == start_token:
                if len(pairs) + 1 >= 2:
                    expand(pairs + [(token, nxt)])
            elif nxt not in tokens and nxt > start_token:
                # Only the smallest token of a cycle starts its walk.
                walk(start_token, nxt, pairs + [(token, nxt)], tokens | {nxt})

    for token in sorted(outgoing):
        walk(token, token, [], {token})
    return [[rows[index] for index in cycle] for cycle in sorted(cycles)]


def arbitrage_candidate(rows, *, variant="graph_cycle", max_states=100000):
    rows = _swaps(rows)
    if not rows:
        return None
    if len({r["transaction_hash"] for r in rows}) != 1:
        raise EvidenceError("Arbitrage input must contain exactly one transaction")
    if variant not in {"qin_log_order", "graph_cycle"}:
        raise EvidenceError("Unknown arbitrage variant")
    # Both variants enumerate candidate cycles the same way, because Qin's
    # heuristics are about an arbitrage's own legs and the only way to find
    # those legs in a transaction carrying other swaps is to enumerate. An
    # exhausted search budget still raises rather than reporting an absence.
    cycles = graph_cycles(rows, max_states=max_states)
    if not cycles:
        return None
    audits = [cycle_qin_audit(cycle, transaction_swap_count=len(rows))
              for cycle in cycles]
    qualifying = [i for i, audit in enumerate(audits) if audit["qualifies"]]
    if variant == "qin_log_order" and not qualifying:
        return None

    qualifying_swap_ids = {tuple(swap_id) for i in qualifying
                           for swap_id in audits[i]["swap_ids"]}
    details = {
        "cycles": [[list(_identity(row)) for row in cycle] for cycle in cycles],
        "cycle_tokens": sorted({r["token_in"] for cycle in cycles for r in cycle}),
        # Per cycle, in full, whether it qualified or not.
        "cycle_audits": audits,
        "enumerated_cycle_count": len(cycles),
        "qualifying_cycle_count": len(qualifying),
        "qualifying_cycles": [audits[i]["swap_ids"] for i in qualifying],
        "qualifying_cycle_failure_criteria": sorted(
            {name for audit in audits for name in audit["failures"]}),
        # Whether the heuristics hold on at least one enumerated cycle. The
        # whole-transaction reading is kept beside it because it is what an
        # earlier version of this module tested, and the two differ exactly
        # when a transaction carries swaps outside the arbitrage.
        "qin_log_order_match": bool(qualifying),
        "whole_transaction_ordered_cycle": ordered_cycle(rows),
        "transaction_swap_count": len(rows),
        "swap_legs_outside_qualifying_cycles": sum(
            1 for row in rows if _identity(row) not in qualifying_swap_ids),
        "amount_rule": "qin_h4_nonincreasing_hops_in_cycle_log_order"
                       if qualifying else "no_cycle_satisfied_qin_h3_and_h4",
    }
    if qualifying:
        # At least one enumerated cycle is the pattern Qin defines. That admits
        # the transaction, whichever search found the cycle.
        return _candidate(
            "arbitrage", variant, rows, details=details, status="identified",
            reason="qin_ordered_cycle_pattern_complete",
            literature=QIN_ARBITRAGE, heuristics=("H1", "H2", "H3", "H4"),
            deviations=(("graph_search_also_enumerated_cycles",)
                        if variant == "graph_cycle" else ()))
    # No cycle qualified. The deviations name what actually failed, per cycle,
    # rather than asserting a log-order failure that may not have occurred.
    observed = {name for audit in audits for name in audit["failures"]}
    return _candidate(
        "arbitrage", variant, rows, details=details, status="candidate",
        reason="no_enumerated_cycle_qualifies_under_qin_heuristics",
        literature={"label": "graph_cycle_candidate_not_qin_supported",
                    "source": "Chi et al. (2024), algorithm 3, partially",
                    "heuristics": {"H1": QIN_ARBITRAGE["heuristics"]["H1"],
                                   "H2": QIN_ARBITRAGE["heuristics"]["H2"]}},
        heuristics=("H1", "H2"),
        deviations=tuple(_CYCLE_FAILURE_DEVIATIONS[name]
                         for name in CYCLE_CRITERIA if name in observed)
                   + (CHI_INCOMPLETE,))


def strict_asset_flows(transfers, *, owners, transaction_hashes, complete,
                      scope="erc20", native_status="unmeasured"):
    """Exact observed asset flows for an explicitly supplied address set.

    Ownership is not inferred from sharing a router. Callers must supply the
    supported owner set and verify transfer coverage. The returned scope is
    evidence provenance, not an assertion of complete wealth or net profit.
    """
    if type(complete) is not bool:
        raise EvidenceError("Transfer completeness must be explicit")
    owners, txs = frozenset(owners), frozenset(transaction_hashes)
    if not owners or not txs or any(not isinstance(x, str) or not x for x in owners | txs):
        raise EvidenceError("Explicit owners and transaction identities are required")
    if scope not in {"erc20", "erc20_and_native"}:
        raise EvidenceError("Unknown balance scope")
    if native_status not in {"unmeasured", "complete", "not_required"}:
        raise EvidenceError("Unknown native-flow status")
    if scope == "erc20_and_native" and native_status != "complete":
        raise EvidenceError("Complete native flows required for the declared scope")
    net, inflow, outflow, seen = defaultdict(int), defaultdict(int), defaultdict(int), set()
    reasons = []
    for row in transfers:
        if row.get("transaction_hash") not in txs:
            raise EvidenceError("Transfer outside candidate transaction set")
        if row.get("status") != "decoded":
            reasons.append("undecoded_transfer")
            continue
        for name in ("token", "from", "to"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise EvidenceError(f"Missing transfer field: {name}")
        _integer(row.get("log_index"), "log_index")
        amount = _integer(row.get("amount"), "amount")
        key = _identity(row)
        if key in seen:
            raise EvidenceError("Duplicate transfer identity")
        seen.add(key)
        token = row["token"]
        enters, leaves = row["to"] in owners, row["from"] in owners
        if enters and not leaves:
            inflow[token] += amount
            net[token] += amount
        elif leaves and not enters:
            outflow[token] += amount
            net[token] -= amount
    if not complete:
        reasons.append("incomplete_transfer_coverage")
    covered = complete and not reasons
    return {
        "status": "complete" if covered else "incomplete",
        "reasons": sorted(set(reasons)), "owners": sorted(owners),
        "transaction_hashes": sorted(txs), "scope": scope,
        "native_status": native_status,
        "net": dict(sorted(net.items())), "inflow": dict(sorted(inflow.items())),
        "outflow": dict(sorted(outflow.items())),
        "nonnegative": covered and all(value >= 0 for value in net.values()),
        "positive": covered and any(value > 0 for value in net.values()),
    }


_AUDIT_NOTE = ("An unmeasured native leg, a balance retained by the executing "
               "contract, or a transfer outside the declared scope can hold "
               "the proceeds. This is an audit of observed ERC-20 movement, "
               "not a measurement of profit, and an absence of movement is not "
               "evidence that none occurred.")


def flow_audit(candidate, flows, *, require_native=False):
    """Describe observed asset movement for a candidate. Decides no label.

    Returns a verdict on the *evidence*, never on the pattern. ``erc20_net``
    reports the sign of the observed net for the declared scope, or
    ``not_evaluable`` when coverage is incomplete or the observed sender falls
    outside the scope. ``profit_established`` is always ``False``: the scope is
    ERC-20 only, native value is never measured, and fees and builder payments
    are out of scope.
    """
    owners = set(flows["owners"])
    senders = set(candidate["actor_senders"])
    sender_in_scope = bool(senders) and senders.issubset(owners)
    native_measured = flows["native_status"] == "complete"
    audit = {
        "scope": flows["scope"],
        "coverage": flows["status"],
        "coverage_reasons": list(flows["reasons"]),
        "observed_senders": sorted(senders),
        "observed_sender_in_scope": sender_in_scope,
        "native_measured": native_measured,
        "native_evidence_requested": bool(require_native),
        "profit_established": False,
        "absence_is_not_evidence_of_absence": True,
        "note": _AUDIT_NOTE,
    }
    if flows["status"] != "complete":
        audit.update(erc20_net="not_evaluable",
                     status="transfer_coverage_incomplete")
    elif not senders:
        audit.update(erc20_net="not_evaluable",
                     status="no_observed_transaction_sender")
    elif not sender_in_scope:
        audit.update(erc20_net="not_evaluable",
                     status="observed_sender_outside_declared_scope")
    elif flows["positive"] and flows["nonnegative"]:
        audit.update(erc20_net="positive",
                     status="positive_observed_erc20_net")
    elif not flows["nonnegative"]:
        audit.update(erc20_net="negative",
                     status="negative_observed_erc20_net")
    else:
        audit.update(erc20_net="zero", status="zero_observed_erc20_net")
    return audit


def evaluate_candidate(candidate, flows, *, require_native=False):
    """Attach the asset-flow audit to a candidate without relabelling it.

    The literature verdict in ``status`` is carried through untouched. This
    function adds ``balance_evidence`` (the exact observed numbers) and
    ``flow_audit`` (what they do and do not support). It cannot promote a graph
    cycle to the Qin label and cannot reject a Qin pattern for want of a
    positive sender-side balance, which is the restriction the method review
    found to be ours rather than Qin's.
    """
    if candidate is None:
        return None
    if candidate.get("strategy") not in {"arbitrage", "sandwich"}:
        raise EvidenceError("Flow audit expects an arbitrage or sandwich candidate")
    if set(candidate["transaction_hashes"]) != set(flows["transaction_hashes"]):
        raise EvidenceError("Cash-flow support differs from candidate support")
    audit = flow_audit(candidate, flows, require_native=require_native)
    result = dict(candidate)
    result["balance_evidence"] = flows
    result["flow_audit"] = audit
    result["variant"] = candidate["variant"] + "+strict_asset_flows"
    # status and reason are deliberately not touched.
    result["label_basis"] = "literature_pattern_only"
    result["profit_status"] = "not_measured"
    result["principal_attribution"] = (
        "declared_scope_basis" if flows.get("scope_basis") else "unassigned")
    result["deviations"] = sorted(set(candidate.get("deviations", ()))
                                  | {"flow_audit_is_independent_of_label"})
    result["parent_evidence_id"] = candidate["evidence_id"]
    result["evidence_id"] = _digest({k: v for k, v in result.items() if k != "evidence_id"})
    return result


def _same_actor(front, back, shared_executors):
    a, b = front.get("tx_from"), back.get("tx_from")
    if a and a == b:
        return "same_transaction_sender"
    executor = front.get("tx_to")
    if (executor and executor == back.get("tx_to") and executor in shared_executors
            and a and b):
        return "explicit_shared_executor"
    return None


def sandwich_candidates(rows, *, shared_executors=(), quantity_lower=Fraction(9, 10),
                        quantity_upper=Fraction(11, 10), combine=False):
    """Qin direction/quantity baseline, with explicit shared-executor evidence.

    Unrelated rows may occur between the legs. Multiple same-direction victims
    are retained. Shared public destinations do not establish an attacker:
    different senders require an independently supported shared executor.
    ``combine=True`` joins patterns sharing an attacker transaction (Chi's
    combined-attack construction). Each component keeps its own Qin label;
    aggregate cash flows remain a separate audit.
    """
    rows = _swaps(rows)
    if not isinstance(quantity_lower, Fraction) or not isinstance(quantity_upper, Fraction):
        raise EvidenceError("Quantity bounds must be exact Fractions")
    if not 0 < quantity_lower <= 1 <= quantity_upper:
        raise EvidenceError("Invalid sandwich quantity bounds")
    shared_executors = frozenset(shared_executors)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["block_number"], row["pool"])].append(row)

    # Pass one: every leg pair that satisfies H1, H3, H4 and H5. H2 is not a
    # property of a pair in isolation, so it is not decided here.
    legs = []
    for (block_number, _pool), swaps in sorted(grouped.items()):
        for i, front in enumerate(swaps):
            for back in swaps[i + 1:]:
                if front["transaction_index"] >= back["transaction_index"]:
                    continue
                if (back["token_in"], back["token_out"]) != (front["token_out"], front["token_in"]):
                    continue
                association = _same_actor(front, back, shared_executors)
                if association is None:
                    continue
                ratio = Fraction(back["amount_in"], front["amount_out"])
                if not quantity_lower <= ratio <= quantity_upper:
                    continue
                victims = [row for row in swaps
                           if front["transaction_index"] < row["transaction_index"] < back["transaction_index"]
                           and (row["token_in"], row["token_out"]) == (front["token_in"], front["token_out"])
                           and row.get("tx_from")
                           and row["tx_from"] not in {front.get("tx_from"), back.get("tx_from")}]
                if not victims:
                    continue
                legs.append({"block": block_number, "front": front, "back": back,
                             "association": association, "ratio": ratio,
                             "victims": victims})

    primary = _primary_transaction_pairs(legs)

    found = []
    for leg in legs:
        front, back = leg["front"], leg["back"]
        pair = (leg["block"], front["transaction_hash"], back["transaction_hash"])
        is_primary = pair in primary
        found.append(_candidate(
            "sandwich",
            "qin_direction_quantity" if is_primary else "relaxed_front_back_pairing",
            [front, back],
            victims=[row["transaction_hash"] for row in leg["victims"]],
            status="identified" if is_primary else "candidate",
            reason="qin_sandwich_pattern_complete" if is_primary
                   else "front_or_back_transaction_already_primarily_paired",
            literature=QIN_SANDWICH,
            # H2 is "each front transaction maps to one back transaction". A
            # pair admitted only by relaxing that mapping does not satisfy it
            # and must not claim it.
            heuristics=("H1", "H2", "H3", "H4", "H5") if is_primary
                       else ("H1", "H3", "H4", "H5"),
            deviations=() if is_primary else ("qin_h2_one_to_one_pairing_relaxed",),
            details={
                "association": leg["association"],
                "quantity_ratio": [leg["ratio"].numerator, leg["ratio"].denominator],
                "quantity_bounds": [[quantity_lower.numerator, quantity_lower.denominator],
                                    [quantity_upper.numerator, quantity_upper.denominator]],
                "token_in": front["token_in"], "token_out": front["token_out"],
                "front_swap_id": list(_identity(front)), "back_swap_id": list(_identity(back)),
                "primary_pairing": is_primary,
                "front_transaction": front["transaction_hash"],
                "back_transaction": back["transaction_hash"],
            }))

    primaries = [row for row in found if row["details"]["primary_pairing"]]
    if not combine:
        # Only the supported one-to-one pairs. A relaxed pair is not a Qin
        # sandwich and is not reported as an alternative reading of one.
        return _dedupe_by_evidence_id(primaries)
    # Always the primary pairs, and additionally the overlapping groups that
    # relaxing the mapping exposes. A group is only as strong as its parts.
    return _dedupe_by_evidence_id(
        primaries + found + combine_sandwich_candidates(found))


def _primary_transaction_pairs(legs):
    """Deterministic one-to-one front/back transaction matching, per block.

    Qin's H2 is a statement about transactions, not about swap logs: each front
    transaction maps to one back transaction. The old code enforced uniqueness
    on ``(transaction_hash, log_index)`` and reset it per pool, so one front
    transaction could take a second back through another pool or another log,
    and with ``combine=True`` it skipped the check altogether while still
    stamping H2 as satisfied.

    Several leg pairs between the *same* two transactions stay supported
    together: one attacker transaction can touch a pool more than once. What is
    excluded is a second, different counterpart for either side.

    The matching is greedy over a total order on transaction pairs, so it does
    not depend on row order, dict order or pool iteration: earliest front
    transaction first, then earliest back, then the hashes.
    """
    by_block = defaultdict(set)
    index = {}
    for leg in legs:
        front, back = leg["front"], leg["back"]
        pair = (front["transaction_hash"], back["transaction_hash"])
        by_block[leg["block"]].add(pair)
        index[(leg["block"], *pair)] = (front["transaction_index"],
                                        back["transaction_index"])
    primary = set()
    for block_number in sorted(by_block):
        used_fronts, used_backs = set(), set()
        ordered = sorted(by_block[block_number],
                         key=lambda pair: (*index[(block_number, *pair)], *pair))
        for front_tx, back_tx in ordered:
            if front_tx in used_fronts or back_tx in used_backs:
                continue
            used_fronts.add(front_tx)
            used_backs.add(back_tx)
            primary.add((block_number, front_tx, back_tx))
    return primary


def _dedupe_by_evidence_id(records):
    out = {}
    for record in records:
        out.setdefault(record["evidence_id"], record)
    return [out[key] for key in sorted(out)]


def combine_sandwich_candidates(candidates):
    """Connected components of shared attack legs; victims never become actors.

    This is Chi's combined-attack construction applied to Qin-identified
    components, and it is not a reproduction of Chi: that needs call traces.
    The combined record keeps every component's own Qin evidence, and if any
    component is not itself Qin-identified the combination stays a candidate
    rather than inheriting a label none of its parts earned.
    """
    candidates = list(candidates)
    if any(row.get("strategy") != "sandwich" for row in candidates):
        raise EvidenceError("Only sandwich candidates can be combined")
    if any(row.get("variant") == "combined_direction_quantity" for row in candidates):
        raise EvidenceError("An already-combined record cannot be recombined")
    groups = []
    for candidate in sorted(candidates, key=lambda r: r["evidence_id"]):
        scope = (tuple(candidate["block_numbers"]), tuple(candidate["pools"]),
                 candidate["details"]["token_in"], candidate["details"]["token_out"])
        txs = set(candidate["transaction_hashes"])
        matches = [i for i, (key, seen, members) in enumerate(groups) if key == scope and txs & seen]
        members = [candidate]
        for i in reversed(matches):
            _, seen, previous = groups.pop(i)
            txs |= seen
            members.extend(previous)
        groups.append((scope, txs, members))
    output = []
    for _, txs, members in groups:
        if len(members) == 1:
            output.append(members[0])
            continue
        result = dict(members[0])
        victims = {tx for row in members for tx in row["victim_transaction_hashes"]}
        members_qin = all(row.get("status") == "identified" for row in members)
        if victims & txs:
            status, reason = "unresolved", "actor_victim_role_conflict"
        elif members_qin:
            status, reason = "identified", "combined_qin_sandwich_components"
        else:
            status, reason = "candidate", "component_without_qin_sandwich_label"
        # A group may claim only what every component claims. Inheriting the
        # first member's list would let one primary pair lend H2 to a group
        # that contains a relaxed pairing.
        shared_heuristics = sorted(
            set.intersection(*(set(row["literature_heuristics_satisfied"])
                               for row in members)))
        result.update(
            variant="combined_direction_quantity",
            status=status, reason=reason,
            literature_label="combined_sandwich_of_qin_components" if members_qin
                             else "combined_sandwich_candidate",
            literature_source="Chi et al. (2024), algorithm 4, partially",
            literature_heuristics_satisfied=shared_heuristics,
            literature_heuristics={key: QIN_SANDWICH["heuristics"][key]
                                   for key in shared_heuristics
                                   if key in QIN_SANDWICH["heuristics"]},
            deviations=sorted(set(BASE_DEVIATIONS) | {CHI_INCOMPLETE}
                              | {"combination_is_not_a_chi_reproduction"}
                              | (set() if members_qin
                                 else {"component_lacks_qin_sandwich_label"})),
            transaction_hashes=sorted(txs),
            actor_senders=sorted({sender for row in members for sender in row["actor_senders"]}),
            victim_transaction_hashes=sorted(victims),
            swap_ids=[list(x) for x in sorted({tuple(x) for row in members for x in row["swap_ids"]})],
            # Every component's own Qin evidence is retained, not replaced by a
            # list of ids: the combined claim is only as good as its parts.
            details={"components": sorted(row["evidence_id"] for row in members),
                     "component_count": len(members),
                     "components_detail": [
                         {"evidence_id": row["evidence_id"],
                          "status": row["status"], "reason": row["reason"],
                          "literature_label": row["literature_label"],
                          "literature_heuristics_satisfied":
                              list(row["literature_heuristics_satisfied"]),
                          "transaction_hashes": list(row["transaction_hashes"]),
                          "actor_senders": list(row["actor_senders"]),
                          "victim_transaction_hashes":
                              list(row["victim_transaction_hashes"]),
                          "details": dict(row["details"])}
                         for row in sorted(members, key=lambda r: r["evidence_id"])]},
        )
        result["evidence_id"] = _digest({k: v for k, v in result.items() if k != "evidence_id"})
        output.append(result)
    return sorted(output, key=lambda row: row["evidence_id"])


def liquidation_evidence(event):
    """Protocol-defined liquidation activity; no inference of realized profit."""
    if event.get("status") != "decoded":
        return {"strategy": "liquidation", "status": "unresolved",
                "reason": event.get("reason", "liquidation_not_decoded"),
                "rule_version": RULE_VERSION, "profit_status": "not_measured"}
    required = ("transaction_hash", "protocol", "market", "liquidator", "borrower",
                "debt_asset", "collateral_asset", "collateral_amount_unit")
    if any(event.get(key) is None for key in required):
        raise EvidenceError("Incomplete liquidation evidence")
    debt = _integer(event.get("debt_amount"), "debt_amount")
    collateral = _integer(event.get("collateral_amount"), "collateral_amount")
    # Real protocol events can carry a zero leg. Preserve them, but keep the
    # positive-amount eligibility rule rather than crashing or counting them.
    zero_amount = debt == 0 or collateral == 0
    result = {
        "strategy": "liquidation", "variant": "protocol_event", "rule_version": RULE_VERSION,
        "status": "rejected" if zero_amount else "identified" if event.get("tx_from") else "unresolved",
        "reason": "zero_amount_liquidation" if zero_amount else (
            "decoded_liquidation_event" if event.get("tx_from") else "missing_transaction_sender"),
        "transaction_hashes": [event["transaction_hash"]],
        "actor_senders": [event["tx_from"]] if event.get("tx_from") else [],
        "event": dict(event), "profit_status": "not_measured",
    }
    result["evidence_id"] = _digest(result)
    return result
