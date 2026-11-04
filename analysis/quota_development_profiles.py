"""Fixed quota-announcement daily development profiles; no pooled inference.

Metadata and calendar checks run before reading creation counts. The complete
capture is read under a shared lock and matched exactly to the existing trusted
measurement receipt and independent raw audit. No collector or outage-window
count table is used. All displayed days survive, with explicit unavailable
comparison reasons where the frozen calendar has insufficient references.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'docs/quota_profile_measurement.md': 'b1196c8c88d153a684551b12385e24587af837912e0ef6b806b9ae5a4a54b9bf',
    'docs/quota_reset_protocol.md': '2727df9a5c706c1198932940cffcba9277b5a8422728799218a3c0298e14ce8e',
    'data/quota_reset/prospective/receipt.json': '6e7d450b67fdeb199228a65a9f6d5efc0e92a7f6e6059186e88ef2a9112bd771',
    'data/strategy_development/liquidation/receipt.json': '9715a959877e492f89940d4b1b298d8769adbef07c7161027c7892b691bc7470',
    'reviews/development_measurement/raw_creation_complete_audit.json': '92d059a38ca68315c1da3a45143959d988b34ff530cc7ab170ee6c0a3e8b44fd',
    'reviews/blocker_endpoint_curation/specification_compatible/overlay.json': '76aaeee55e25989b588c49db46465ab0b250849beac0a346c8eb0d0bec912df5',
    'reviews/blocker_source_screen/diagnostic/report.json': '926a5b48b00889e5e45c814a73de3280abfa7aa290f7bca4fb5326d5b3a7a205',
}
FOUR = frozenset({
    'AWS_MULTIPLE_SERVICES_OPERATIONAL_ISSUE_5E6B8_EF2498889B5',
    'AWS_MULTIPLE_SERVICES_OPERATIONAL_ISSUE_E6426_01AD72B4E1E',
    'AWS_EC2_OPERATIONAL_ISSUE_FFA94_A4DC515D30F',
    'AWS_BEDROCK_OPERATIONAL_ISSUE_588F5_70C147423A3',
})
METRICS = {'deployments': 'creation_tx_count', 'creating_senders': 'deploying_addresses'}
TIERS = ('core', 'small', 'total')


def sibling(name):
    key = 'quota_development_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


_qr = sibling('quota_reset_inputs')
_sd = sibling('strategy_development')
_ci = _qr._ci


class ProfileError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ProfileError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def json_value(value):
    return json.loads(json.dumps(value, sort_keys=True))


def fraction_record(value):
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'decimal': float(value)}


def metric_comparison(raw, references, *, minimum=2):
    require(type(raw) is int and raw >= 0, 'raw count must be a nonnegative integer')
    require(all(type(v) is int and v >= 0 for v in references), 'reference counts must be nonnegative integers')
    if len(references) < minimum:
        return {'raw_count': raw, 'reference_mean': None, 'count_difference': None,
                'ratio': None, 'comparison_unavailable_reason': 'fewer_than_two_clean_observed_baselines',
                'ratio_unavailable_reason': 'fewer_than_two_clean_observed_baselines'}
    mean = Fraction(sum(references), len(references))
    return {'raw_count': raw, 'reference_mean': fraction_record(mean),
            'count_difference': fraction_record(Fraction(raw) - mean),
            'ratio': fraction_record(Fraction(raw) / mean) if mean > 0 else None,
            'comparison_unavailable_reason': None,
            'ratio_unavailable_reason': None if mean > 0 else 'zero_reference_mean'}


def required_dates(episode):
    return sorted(set(_qr.day_range(episode['baseline_start'], episode['baseline_end_exclusive'])) |
                  set(_qr.day_range(episode['display_start'], episode['display_end_exclusive'])))


def validate_cohort(episode, cohort):
    require(cohort['available'] and cohort['rule'] == 'min_tx_1' and cohort['head_cut'] == 50,
            'frozen cohort unavailable or rule drift')
    members = cohort['members']
    senders = [m['sender'] for m in members]
    require(len(senders) == len(set(senders)), 'duplicate cohort sender')
    require(all(m['history_transactions'] >= 1 for m in members), 'unqualified historical sender')
    require(members == sorted(members, key=lambda m: (-m['history_transactions'], m['sender'])),
            'historical rank ordering drift')
    require([m['rank'] for m in members] == list(range(1, len(members) + 1)), 'historical ranks drift')
    require(all(m['tier'] == ('core' if m['rank'] <= 50 else 'small') for m in members), 'tier assignment drift')
    require(cohort['membership_digest'] == _qr.digest(members), 'membership digest drift')
    require(cohort['history_start'] == episode['history_start']
            and cohort['history_end_exclusive'] == episode['history_end_exclusive'], 'cohort history drift')
    for tier in ('core', 'small'):
        require(cohort['tiers'][tier] == [m['sender'] for m in members if m['tier'] == tier],
                'tier member list drift')


def source_aware_blockers(original, overlay, cfg):
    records = {r['event_id']: r for r in overlay['records']}
    selected = [r for r in records.values() if r['primary_screening_candidate']['boundary_role'] == 'computational_uncertainty_censor']
    require({r['event_id'] for r in selected} == FOUR, 'amended source population drift')
    require(overlay['fixed_screening_observation_end'] == cfg['panel_end_exclusive'], 'screen endpoint drift')
    by_id = {b.ident: b for b in original if b.source == 'cloud_twin_window' and b.ident in FOUR}
    require(set(by_id) == FOUR and sum(b.source == 'cloud_twin_window' and b.ident in FOUR for b in original) == 4,
            'four unique original blockers required')
    end = _ci.parse_utc_seconds(cfg['panel_end_exclusive'])
    for ident, blocker in by_id.items():
        screen = records[ident]['primary_screening_candidate']
        require(screen['source_class'] == 'cloud_twin_window' and screen['recovered_at_utc'] is None,
                'screening boundary mislabelled as recovery')
        require((_ci.parse_utc_seconds(screen['supplied_start_utc']), _ci.parse_utc_seconds(screen['supplied_end_utc']))
                == (blocker.start_s, blocker.end_s), 'source blocker interval drift')
        require(_ci.parse_utc_seconds(screen['screen_start_utc']) == blocker.start_s
                and _ci.parse_utc_seconds(screen['screen_end_exclusive_utc']) == end, 'specification interval drift')
    return [replace(b, end_s=end) if b.source == 'cloud_twin_window' and b.ident in FOUR else b for b in original]


def prepare_inputs(root):
    """Only source identities, cohorts and calendars; no creation/table values."""
    pinned = {}
    def checked(path, expected=None):
        path = Path(path)
        value = sha(path)
        require(expected is None or value == expected, f'input hash drift: {path}')
        pinned[str(path.resolve())] = value
        return path
    for rel, value in PINS.items():
        checked(root / rel, value)
    prospective = root / 'data/quota_reset/prospective'
    receipt = read(prospective / 'receipt.json')
    for source in receipt['sources']:
        path = Path(source['path'])
        relative = path.relative_to(ROOT)
        checked(root / relative, source['sha256'])
    for name, value in receipt['outputs'].items():
        checked(prospective / name, value)
    episodes = read(prospective / 'episodes.json')
    saved_calendars = read(prospective / 'calendars.json')
    cohorts = read(prospective / 'cohorts.json')
    inventory = _qr.load_frozen(root / 'docs/quota_reset_protocol.md', root / 'data/reference/quota_reset_events.json')
    require(episodes == _qr.derive_episodes(inventory['events']), 'source-defined episodes drift')
    require(len(episodes) == 6 and len(cohorts) == 18, 'prospective population inventory drift')
    cfg = read(root / 'config/strategy_study.json')
    require(cfg['registry_blocker_policy'] == 'availability_excluding_privacy_processing'
            and cfg['minimum_clean_controls'] == 2, 'frozen availability policy drift')
    original = _ci.build_blocker_set(cfg, root)
    overlay = read(root / 'reviews/blocker_endpoint_curation/specification_compatible/overlay.json')
    for path, value in overlay['sources'].items():
        checked(root / Path(path).relative_to(ROOT), value)
    altered = source_aware_blockers(original, overlay, cfg)
    intervals, undated = _qr.quota_intervals(inventory['events'])
    diagnostics = read(root / 'reviews/blocker_source_screen/diagnostic/report.json')
    expected_ready = {(r['episode_id'], r['strategy']): r for r in diagnostics['quota_readiness']}
    source_receipt = read(root / 'data/strategy_development/liquidation/receipt.json')
    raw_audit = read(root / 'reviews/development_measurement/raw_creation_complete_audit.json')
    require(source_receipt['raw_creation_audit_sha256'] == PINS['reviews/development_measurement/raw_creation_complete_audit.json'],
            'trusted measurement/raw audit binding differs')
    require(len(source_receipt['sources']) == 1, 'single complete capture required')
    for module, value in source_receipt['provenance']['reader_code_sha256'].items():
        checked(root / 'analysis' / (module + '.py'), value)
    checked(root / 'reviews/development_measurement/check_raw_creations.py', raw_audit['audit_source_sha256'])
    checked(root / 'data/reference/history_block_boundaries.json', source_receipt['provenance']['boundary_file']['sha256'])
    checked(Path(__file__))
    liquidation, unavailable, revised, checks = {}, [], {}, []
    for episode in episodes:
        ident = episode['episode_id']
        before = _qr.calendar_for(episode, intervals, original)
        require(before == saved_calendars[ident], 'original quota calendars not reproduced')
        after = _qr.calendar_for(episode, intervals, altered)
        revised[ident] = after
        rows = [r for r in cohorts if r['episode_id'] == ident]
        require(len(rows) == 3 and {r['strategy'] for r in rows} == {'liquidation', 'arbitrage', 'sandwich'}, 'strategy inventory drift')
        for row in rows:
            if row['strategy'] != 'liquidation':
                require(not row['cohort']['available'] and row['cohort']['members'] is None, 'unsupported history became a population')
                unavailable.append({'episode_id': ident, 'strategy': row['strategy'], 'available': False,
                                    'reason': row['cohort']['reason'], 'members': None})
                continue
            validate_cohort(episode, row['cohort'])
            liquidation[ident] = row['cohort']
            readiness = row['capture_readiness']
            require(readiness['outcomes_read'] is False and not readiness['missing_addresses'], 'prospective capture scope drift')
            require(len(before) == len(after) == len(readiness['days']), 'display calendar alignment drift')
            old_dates, new_dates = [], []
            for old, new, day in zip(before, after, readiness['days']):
                require(old['date'] == new['date'] == day['date'], 'daily calendar alignment drift')
                require(day['ready_for_raw_count_measurement'], 'display day not capture-ready')
                require(day['clean_time_covered_baseline_dates'] == old['calendar_clean_baseline_dates'], 'baseline time readiness drift')
                if day['ready_for_matched_profile_measurement']:
                    old_dates.append(day['date'])
                if len(set(day['clean_time_covered_baseline_dates']) & set(new['calendar_clean_baseline_dates'])) >= 2:
                    new_dates.append(day['date'])
            check = {'episode_id': ident, 'strategy': 'liquidation', 'before_ready_dates': old_dates,
                     'after_ready_dates': new_dates, 'raw_display_readiness_changed': False}
            require(check == expected_ready[(ident, 'liquidation')], 'source-aware ready dates differ from pre-outcome diagnostic')
            checks.append(check)
    require(sum(len(v) for v in revised.values()) == 137, 'fixed display has changed')
    require(sum(len(r['after_ready_dates']) for r in checks) == 3, 'matched-ready day count differs')
    return {'episodes': episodes, 'cohorts': liquidation, 'unavailable': unavailable,
            'calendars': revised, 'calendar_checks': checks, 'undated_quota_entries': undated,
            'overlay_records': [r for r in overlay['records'] if r['event_id'] in FOUR],
            'trusted_receipt': source_receipt, 'raw_audit': raw_audit, 'pins': pinned}


def verify_capture_bytes(capture, trusted, raw_audit):
    """Hash all original payloads before opening any creation values."""
    source = trusted['sources'][0]
    require(Path(source['name']).resolve() == capture.resolve(), 'unexpected capture root')
    require(raw_audit['capture_root'] == str(capture), 'raw audit capture root differs')
    require(raw_audit['request_digest'] == source['request_digest'], 'audit request differs')
    require(raw_audit['complete_declared_range'] is True and raw_audit['interval'] == raw_audit['declared_interval'] == source['declared_blocks'],
            'raw audit does not attest complete declared capture')
    require(all(v is True for v in raw_audit['checks'].values()), 'raw audit checks incomplete')
    digests = source['digests']
    require(digests['request.json'] == raw_audit['request_sha256'], 'request file pins disagree')
    expected = {r['page']: {**r['payload_sha256'], 'page.json': r['page_sha256']} for r in raw_audit['page_receipts']}
    require(len(expected) == len(raw_audit['page_receipts']) == source['pages'], 'audit page identities not unique')
    require(expected == digests['pages'], 'measurement receipt and independent raw audit payload identities differ')
    actual_names = sorted(p.name for p in (capture / 'pages').iterdir() if p.is_dir() and not p.name.startswith('.'))
    require(actual_names == sorted(expected), 'capture page population drift')
    for name in ('request.json', 'coverage.json'):
        require(sha(capture / name) == digests[name], f'capture metadata drift: {name}')
    cursor = source['declared_blocks'][0]
    for name in sorted(expected, key=lambda s: int(s.split('_')[0])):
        low, high = map(int, name.split('_'))
        require(low == cursor and low < high, 'page coverage gap or overlap')
        cursor = high
        for filename, value in expected[name].items():
            require(filename in ('page.json', 'blocks.parquet', 'transactions.parquet', 'creations.parquet'), 'unexpected capture payload')
            require(sha(capture / 'pages' / name / filename) == value, f'capture payload drift: {name}/{filename}')
    require(cursor == source['declared_blocks'][1], 'capture interval incomplete')
    request, coverage = read(capture / 'request.json'), read(capture / 'coverage.json')
    require(coverage['complete'] is True and coverage['scanned_intervals'] == source['scanned_blocks']
            and not coverage['blocks_not_yet_scanned'], 'capture incomplete')
    require(_qr.digest(request['addresses']) == request['address_list_digest'], 'capture address digest differs')
    require(request['digest'] == source['request_digest'] and len(request['addresses']) == source['declared_addresses'], 'capture scope differs')
    return request


def make_groups_and_windows(episodes, cohorts):
    groups, windows = {}, []
    for episode in episodes:
        ident = episode['episode_id']
        members = cohorts[ident]['members']
        for tier in ('core', 'small'):
            groups[(ident, tier)] = [m['sender'] for m in members if m['tier'] == tier]
        require(not set(groups[(ident, 'core')]) & set(groups[(ident, 'small')]), 'tiers overlap')
        for day in required_dates(episode):
            for tier in ('core', 'small'):
                windows.append({'event_id': ident, 'design': 'quota_utc_day', 'window': day,
                                'offset_days': 0, 'tier': tier, 'onset_shift_minutes': 0,
                                'window_start_s': _qr.stamp(day), 'window_end_s': _qr.stamp(day) + _qr.DAY,
                                'head_cut': 50, 'qualification_rule': 'min_tx_1', 'defined': True, 'reason': None})
    return groups, windows


def daily_records(episodes, cohorts, source, boundaries):
    groups, windows = make_groups_and_windows(episodes, cohorts)
    result = _sd.count_windows(groups, windows, [source], boundaries,
                               cohort_available=list(cohorts), membership={tier: list(cohorts) for tier in ('core', 'small')})
    require(result['windows_unavailable'] == 0, 'uncovered day or sender: refusing observed zeros')
    indexed = {}
    for row in result['windows']:
        key = (row['event_id'], row['window'], row['tier'])
        require(key not in indexed, 'duplicate daily key')
        indexed[key] = {'episode_id': key[0], 'date': key[1], 'tier': key[2],
                        'membership_digest': cohorts[key[0]]['membership_digest'],
                        **{k: row[k] for k in ('window_start_s', 'window_end_s', 'cohort_addresses',
                                              'coverage_available', 'coverage_reason', 'creation_tx_count', 'deploying_addresses')}}
    for episode in episodes:
        ident = episode['episode_id']
        for day in required_dates(episode):
            core, small = (indexed[(ident, day, tier)] for tier in ('core', 'small'))
            total = {**core, 'tier': 'total', 'cohort_addresses': core['cohort_addresses'] + small['cohort_addresses']}
            for column in METRICS.values():
                total[column] = core[column] + small[column]
            indexed[(ident, day, 'total')] = total
    return [indexed[k] for k in sorted(indexed)], result


def profiles(episodes, cohorts, calendars, daily):
    indexed = {(r['episode_id'], r['date'], r['tier']): r for r in daily}
    require(len(indexed) == len(daily), 'daily input has duplicate cells')
    output = []
    for episode in episodes:
        ident = episode['episode_id']
        days = []
        for calendar in calendars[ident]:
            day = calendar['date']
            candidates = calendar['baseline_candidates']
            refs = [r['date'] for r in candidates if r['calendar_clean']
                    and all(indexed[(ident, r['date'], tier)]['coverage_available'] for tier in ('core', 'small'))]
            tiers = {}
            for tier in TIERS:
                observed = indexed[(ident, day, tier)]
                require(observed['coverage_available'], 'display day uncovered')
                tiers[tier] = {'cohort_addresses': observed['cohort_addresses'], 'coverage_available': True,
                               'metrics': {metric: metric_comparison(observed[column], [indexed[(ident, d, tier)][column] for d in refs])
                                           for metric, column in METRICS.items()}}
            days.append({'date': day, 'day_from_first_announcement': calendar['day_from_first_announcement'],
                         'eligible_reference_dates': refs, 'reference_candidates_with_exclusions': candidates,
                         'matched_ready': len(refs) >= 2, 'quota_annotations': calendar['quota_annotations'],
                         'outage_annotations': calendar['outage_annotations'],
                         'outage_annotation_intervals_are_screening_bounds_not_measured_global_outages': True,
                         'tiers': tiers})
        output.append({'episode': episode, 'membership_digest': cohorts[ident]['membership_digest'],
                       'fixed_cohort_size': len(cohorts[ident]['members']), 'days': days})
    return output


def verify_creation_digest(index, raw_audit):
    rows = sorted([r['block_number'], r['ts'], r['transaction_hash'], r['sender'], 1] for r in index['rows'].values())
    require(_qr.digest(rows) == raw_audit['creation_record_digest'], 'creation evidence differs from independent raw audit')
    require(len(rows) == raw_audit['payload_rows']['creations'], 'unique creation count differs from raw audit')
    return rows


def run(root=ROOT, *, capture_root, measurement_dir, results_dir, review_dir):
    root, capture = Path(root), Path(capture_root).resolve()
    measurement_dir, results_dir, review_dir = map(Path, (measurement_dir, results_dir, review_dir))
    require(not measurement_dir.exists() and not results_dir.exists(), 'refusing to overwrite prior scientific outputs')
    require(not (review_dir / 'preoutcome_gates.json').exists(), 'input validation report already exists')
    phase = {'creation_values_allowed': False, 'parquet_reads': []}
    import pyarrow as pa
    import pyarrow.parquet as pq
    original_read = pq.read_table
    def guarded_read(path, *args, **kwargs):
        path = Path(path).resolve()
        require(phase['creation_values_allowed'], 'creation counts read before input validation completed')
        require(path.is_relative_to(capture / 'pages') and path.name == 'creations.parquet', 'unexpected outcome/table read')
        phase['parquet_reads'].append(str(path))
        return original_read(path, *args, **kwargs)
    def guard(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise ProfileError('network forbidden for existing-capture quota measurement')
        if event == 'open' and isinstance(args[0], (str, bytes)):
            p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
            if p.name == 'window_counts.parquet' or p.is_relative_to(root / 'results/development'):
                raise ProfileError('existing outage-window outcome files are not quota inputs')
    sys.addaudithook(guard)
    pq.read_table = guarded_read
    try:
        with (capture / '.lock').open('rb') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            prepared = prepare_inputs(root)
            trusted, raw_audit = prepared['trusted_receipt'], prepared['raw_audit']
            request = verify_capture_bytes(capture, trusted, raw_audit)
            addresses = {m['sender'] for cohort in prepared['cohorts'].values() for m in cohort['members']}
            require(len(addresses) == 576 and addresses <= set(request['addresses']), 'quota address union outside declared capture')
            boundary_path = root / 'data/reference/history_block_boundaries.json'
            strict = read(boundary_path)
            for boundary in strict['boundaries']:
                _qr.sibling('mev_history_plan').boundary_block(strict, boundary['date_utc'])
            boundaries = _sd.read_boundaries(boundary_path)
            source_meta = {**trusted['sources'][0], 'addresses': frozenset(request['addresses'])}
            proof = _sd.time_coverage(source_meta, boundaries)
            require(proof == trusted['sources'][0]['time_coverage'], 'coverage proof drift')
            for episode in prepared['episodes']:
                for day in required_dates(episode):
                    require(proof['covers_time'] and _qr.stamp(day) > proof['floor_exclusive']
                            and _qr.stamp(day) + _qr.DAY <= proof['ceiling_inclusive'], 'required quota day outside proven capture')
            gates = {'all_passed': True, 'creation_values_read': False,
                     'calendar_checks': prepared['calendar_checks'], 'sources': prepared['pins'],
                     'trusted_capture_digest': request['digest'], 'verified_pages': len(trusted['sources'][0]['digests']['pages']),
                     'all_capture_payload_hashes_match_both_trusted_records': True,
                     'quota_cohort_union_addresses': len(addresses), 'required_episode_days': sum(len(required_dates(e)) for e in prepared['episodes']),
                     'shared_capture_read_lock_held': True, 'time_coverage': proof}
            write_new(review_dir / 'preoutcome_gates.json', gates)
            phase['creation_values_allowed'] = True
            source = _sd.read_capture_source(capture)
            require(source['digests'] == trusted['sources'][0]['digests'], 'reader source digests differ from trusted capture')
            index = _sd.creation_index([source])
            canonical = verify_creation_digest(index, raw_audit)
            require(index['duplicate_rows_collapsed'] == 0, 'audited unique capture unexpectedly has duplicates')
            daily, counted = daily_records(prepared['episodes'], prepared['cohorts'], source, boundaries)
            result = profiles(prepared['episodes'], prepared['cohorts'], prepared['calendars'], daily)
            require(sum(len(e['days']) for e in result) == 137, 'profile display day loss')
            require(sum(d['matched_ready'] for e in result for d in e['days']) == 3, 'profile matched-day drift')
            for path, value in prepared['pins'].items():
                require(sha(path) == value, f'input changed during measurement: {path}')
            require(sha(capture / 'coverage.json') == source['digests']['coverage.json'], 'capture changed during shared read')
            measurement_dir.mkdir(parents=True, exist_ok=False)
            results_dir.mkdir(parents=True, exist_ok=False)
            evidence_rows = [{'block_number': b, 'timestamp': ts, 'transaction_hash': tx, 'sender': sender, 'status': status}
                             for b, ts, tx, sender, status in canonical]
            pq.write_table(pa.Table.from_pylist(evidence_rows), measurement_dir / 'creation_evidence.parquet', compression='zstd')
            write_new(measurement_dir / 'daily_counts.json', daily)
            write_new(measurement_dir / 'source_aware_calendars.json', {
                'calendars': prepared['calendars'], 'screen_overlay_records': prepared['overlay_records'],
                'annotation_scope': 'availability-screen intervals, not measured uniform outages'})
            measurement = {'schema': 'quota_daily_development_measurement_1', 'stage': 'observed_daily_counts',
                           'sources': prepared['pins'], 'preoutcome_gates_sha256': sha(review_dir / 'preoutcome_gates.json'),
                           'original_raw_audit_sha256': PINS['reviews/development_measurement/raw_creation_complete_audit.json'],
                           'original_measurement_receipt_sha256': PINS['data/strategy_development/liquidation/receipt.json'],
                           'capture_request_digest': source['request_digest'], 'creation_record_digest': raw_audit['creation_record_digest'],
                           'snapshot_rows': len(canonical), 'snapshot_scope': 'complete audited capture evidence, not a new strategy population',
                           'daily_tier_rows': len(daily), 'unique_episode_days': len(daily) // 3,
                           'frozen_cohort_union': len(addresses), 'read_source_pages': source['pages'],
                           'source_digests_exactly_equal_trusted_measurement': True, 'time_coverage': proof,
                           'shared_read_lock_held_through_publication': True,
                           'parquet_reads': phase['parquet_reads'], 'outage_window_counts_read': False,
                           'unavailable_strategies': prepared['unavailable'],
                           'native_outcome_columns': METRICS, 'scope_notes': list(_sd.SCOPE_NOTES),
                           'outputs': {p.name: sha(p) for p in sorted(measurement_dir.iterdir()) if p.is_file()}}
            write_new(measurement_dir / 'receipt.json', measurement)
            numerical = {'schema': 'quota_development_profiles_1', 'episodes': result,
                         'unavailable_strategies': prepared['unavailable'],
                         'measurement_receipt_sha256': sha(measurement_dir / 'receipt.json'),
                         'all_fixed_display_days': 137, 'matched_ready_days': 3,
                         'scope': 'descriptive daily top-level creations by frozen prior liquidation senders; no AI adoption or redemption observation',
                         'inference': {'pooled_coefficient': False, 'confidence_intervals': False, 'p_values': False, 'causal_claim': False}}
            write_new(results_dir / 'liquidation_profiles.json', numerical)
            write_new(results_dir / 'receipt.json', {
                'sources': prepared['pins'], 'measurement_receipt_sha256': sha(measurement_dir / 'receipt.json'),
                'outputs': {'liquidation_profiles.json': sha(results_dir / 'liquidation_profiles.json')},
                'reference_arithmetic': 'exact rational mean, difference and positive-denominator ratio',
                'minimum_clean_observed_references': 2, 'network_requests': 0})
    finally:
        pq.read_table = original_read
    return {'display_days': 137, 'matched_ready_days': 3, 'daily_tier_rows': len(daily),
            'snapshot_rows': len(canonical), 'measurement_dir': str(measurement_dir), 'results_dir': str(results_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--measurement-dir', type=Path, default=ROOT / 'data/quota_reset/measurement')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results/quota_reset')
    parser.add_argument('--review-dir', type=Path, default=ROOT / 'reviews/quota_development_measurement')
    args = parser.parse_args()
    print(json.dumps(run(capture_root=args.capture_root, measurement_dir=args.measurement_dir,
                         results_dir=args.results_dir, review_dir=args.review_dir), indent=2))


if __name__ == '__main__':
    main()
