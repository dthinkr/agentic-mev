"""Primary liquidation comparisons under the source-bound four-record specification.

Strict original loaders run before any derived view is constructed. Only screen
columns change in memory; existing inputs/receipts are not restamped. All matching,
paired inference and trading decomposition use the existing pure helpers.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PANEL = Path('.build/strategy_panels/liquidation')
NATIVE = Path('data/strategy_development/liquidation')
specification = 'docs/blocker_source_specification.md'
OVERLAY = 'reviews/blocker_endpoint_curation/specification_compatible/overlay.json'
OVERLAY_RECEIPT = 'reviews/blocker_endpoint_curation/specification_compatible/receipt.json'
AUDIT = Path('reviews/quota_blocker_provenance/all_matching_aws')
META = Path('reviews/blocker_source_screen')
NATIVE_PIN = '9715a959877e492f89940d4b1b298d8769adbef07c7161027c7892b691bc7470'
PINS = {
    specification: '49e0ef5666c7d2c9ac5d46f2eb80d88288732a81c527ef76d25db24a8f227163',
    OVERLAY: '76aaeee55e25989b588c49db46465ab0b250849beac0a346c8eb0d0bec912df5',
    OVERLAY_RECEIPT: '7799dd799c4487b87e17986f6cf1026b04cc6bf6ae8ef809b7a3fd94aed628c0',
    str(META / 'diagnostic/report.json'): '926a5b48b00889e5e45c814a73de3280abfa7aa290f7bca4fb5326d5b3a7a205',
    str(META / 'diagnostic/changed_cells.json'): '5a7d2eb1b42a2f9c6994cf3c374d2175ea90dbb2edb9deaed3571bdeb4227919',
    str(META / 'guarded_reproduction.json'): '98cd06a506504f3ee8c02c38a55d13ec4379cba6a2c5c1445eabb9f45edcbac7',
    str(AUDIT / 'verified/receipt.json'): '48878f7606a786e14074f5adc9241ae2e636347e166e6cd56ad73d904af75710',
    str(AUDIT / 'verified/classified_records.json'): '445c5e4d7c21e7d27113e6f5b36c9276192bd3d9513403c991c9bac422879519',
    str(AUDIT / 'match_inventory.json'): '092dce7652226eb023fa94c729c2b97cd3d28790ec6a948d7205ff60f6789429',
    'data/reference/blocker_cloud_twin_source.csv': '42d1db0291eb6bf8ad0d701d44bb224464dda36468d024a64c678259824b5c1b',
}
IDS = frozenset({
    'AWS_MULTIPLE_SERVICES_OPERATIONAL_ISSUE_5E6B8_EF2498889B5',
    'AWS_MULTIPLE_SERVICES_OPERATIONAL_ISSUE_E6426_01AD72B4E1E',
    'AWS_EC2_OPERATIONAL_ISSUE_FFA94_A4DC515D30F',
    'AWS_BEDROCK_OPERATIONAL_ISSUE_588F5_70C147423A3',
})
KEY = ('event_id', 'design', 'window', 'offset_days', 'tier', 'head_cut',
       'qualification_rule', 'window_start_s', 'window_end_s')
SCREEN_COLUMNS = ('screen_valid', 'screen_invalid_reason', 'analysis_valid',
                  'analysis_invalid_reason', 'contamination_hits', 'contamination_detail')
CALENDARS = ('prior_days', 'prior_weeks')
RESULT_FILE = 'source_screen_estimates.json'
RECEIPT_FILE = 'source_screen_receipt.json'
OUTPUTS = {RESULT_FILE, RECEIPT_FILE, 'screen_validation.json', 'applied_overlay.json'}


class ScreenError(RuntimeError):
    """Source binding, unchanged-input or derived-inference contract failed."""


def require(condition, message):
    if not condition:
        raise ScreenError(message)


def sibling(name):
    path = ROOT / 'analysis' / f'{name}.py'
    if name in sys.modules and Path(getattr(sys.modules[name], '__file__', '')).resolve() == path:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_es = sibling('estimate_strategy')
_ed = sibling('estimate_development')
_ss = sibling('strategy_study')
_di = sibling('development_inputs')
_ci = sibling('competition_inputs')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalized(value):
    return json.loads(_es.dump_json(value))


def scientific(value):
    return normalized({k: v for k, v in value.items() if k != 'receipt'})


class Evidence:
    def __init__(self, root, source_root=None):
        self.root = Path(root).resolve()
        self.origin = Path(source_root or root).resolve()
        self.files = {}

    def checked(self, path, expected=None):
        path = _ss.source_path(Path(path), root=self.root, source_root=self.origin)
        require(path.is_file() and not path.is_symlink(), f'missing or linked input: {path}')
        digest = sha(path)
        require(expected is None or digest == expected, f'input hash mismatch: {path}')
        relative = str(path.relative_to(self.root))
        require(relative not in self.files or self.files[relative] == digest, f'input changed: {path}')
        self.files[relative] = digest
        return path

    def read(self, path, expected=None):
        return json.loads(self.checked(path, expected).read_text())

    def unchanged(self):
        for path, digest in self.files.items():
            require(sha(self.root / path) == digest, f'input changed during inference: {path}')


def validate_overlay(overlay, classified, frozen, source_rows, events, end_utc):
    """Validate supplied endpoints without adopting any inferred local-clock time."""
    require(overlay['fixed_screening_observation_end'] == end_utc, 'overlay observation end differs')
    require(len(overlay['records']) == len(classified) == 16, 'exact-match population differs')
    require(overlay['unmatched'] == frozen['unmatched'] and len(overlay['unmatched']) == 147,
            'unverified source population changed')
    indexed = {r['source_row']['id']: r for r in classified}
    require(len(indexed) == 16, 'duplicate classified source identity')
    seen, selected = set(), []
    for item in overlay['records']:
        ident = item['event_id']
        require(ident in indexed and ident not in seen, 'duplicate or unexpected overlay identity')
        seen.add(ident)
        source = indexed[ident]
        require(item['source_row'] == source['source_row'], 'overlay source row differs')
        matches = [row for row in source_rows if row.get('id') == ident]
        require(matches == [source['source_row']], 'source CSV row missing, changed or duplicated')
        event = events[ident]
        object_sha = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        require(object_sha == source['event_object_sha256'] and event['arn'] == item['arn'],
                'complete event identity differs')
        message = event['event_log'][source['matching_update_index']]
        require(message['message'] == item['endpoint_message'] == source['matching_update']['message'],
                'endpoint message differs')
        message_sha = hashlib.sha256(message['message'].encode()).hexdigest()
        require(message_sha == item['endpoint_message_sha256'] == source['matching_update']['message_sha256'],
                'endpoint message hash differs')
        require(message['timestamp'] == source['matching_update']['timestamp'], 'message timestamp differs')
        require(item['source_classification'] == source['classification'], 'classification changed')
        require(item['preserve_source_start'] is True and item['candidate_applied'] is False
                and item['geographic_exclusion_proposed'] is False
                and item['local_milestones_are_primary_screen_endpoints'] is False,
                'overlay scope or clock policy differs')
        bounds = item['primary_screening_candidate']
        start = _ci.parse_utc_seconds(source['source_row']['start'])
        end = _ci.parse_utc_seconds(source['source_row']['end'])
        require(bounds['source_class'] == 'cloud_twin_window' and bounds['recovered_at_utc'] is None,
                'source class or recovered-at assertion differs')
        require(_ci.parse_utc_seconds(bounds['supplied_start_utc']) == start
                and _ci.parse_utc_seconds(bounds['screen_start_utc']) == start
                and _ci.parse_utc_seconds(bounds['supplied_end_utc']) == end,
                'supplied start/end or screen start changed')
        require(bounds['is_observed_recovery_or_uniform_outage_claim'] is False,
                'screen censor incorrectly labelled recovery')
        if ident in IDS:
            require(source['classification'] != 'explicit_recovery', 'selected record claims explicit recovery')
            require(bounds['screen_end_exclusive_utc'] == end_utc
                    and bounds['reference_availability'] == 'unresolved'
                    and bounds['boundary_role'] == 'computational_uncertainty_censor'
                    and bounds['main_specification_rule'] == 'unresolved_reference_availability_through_fixed_observation_end',
                    'four-record uncertainty screen differs from specification')
            require(start < end < _ci.parse_utc_seconds(end_utc), 'selected source interval is not extended')
            selected.append(item)
        else:
            require(source['classification'] == 'explicit_recovery'
                    and _ci.parse_utc_seconds(bounds['screen_end_exclusive_utc']) == end,
                    'nonselected source interval changed')
            if item['retrospective_onset_candidate']:
                candidate = item['retrospective_onset_candidate']
                require(start == end and candidate['onset_utc'] is None and candidate['recovery_utc'] is None,
                        'an unresolved retrospective onset was adopted')
    require({r['event_id'] for r in selected} == IDS, 'four-record selection differs')
    return sorted(selected, key=lambda r: r['event_id'])


def source_overlay(evidence, cfg):
    for path, digest in PINS.items():
        evidence.checked(path, digest)
    overlay = evidence.read(OVERLAY)
    overlay_receipt = evidence.read(OVERLAY_RECEIPT)
    require(overlay_receipt['outputs'] == {'overlay.json': PINS[OVERLAY]}, 'overlay receipt output differs')
    require(overlay['main_specification']['sha256'] == PINS[specification], 'overlay specification binding differs')
    for path, digest in overlay_receipt['source_hashes'].items():
        evidence.checked(path, digest)
    classified = evidence.read(AUDIT / 'verified/classified_records.json')
    source_receipt = evidence.read(AUDIT / 'verified/receipt.json')
    frozen = evidence.read(AUDIT / 'match_inventory.json')
    with evidence.checked('data/reference/blocker_cloud_twin_source.csv').open() as stream:
        source_rows = list(csv.DictReader(stream))
    event_objects = {}
    for row in classified:
        relative = row['complete_event_path']
        event_objects[row['source_row']['id']] = evidence.read(
            AUDIT / 'verified' / relative, source_receipt['artifacts'][relative])
    selected = validate_overlay(overlay, classified, frozen, source_rows, event_objects,
                                cfg['panel_end_exclusive'])
    original = _ci.build_blocker_set(cfg, evidence.root)
    altered, changed = [], []
    for blocker in original:
        candidates = [r for r in selected if blocker.source == 'cloud_twin_window'
                      and r['event_id'] == blocker.ident]
        if not candidates:
            altered.append(blocker)
            continue
        require(len(candidates) == 1 and blocker.ident not in changed, 'duplicate selected blocker')
        bounds = candidates[0]['primary_screening_candidate']
        require(blocker.klass == 'infra' and blocker.start_s == _ci.parse_utc_seconds(bounds['supplied_start_utc'])
                and blocker.end_s == _ci.parse_utc_seconds(bounds['supplied_end_utc']),
                'support blocker differs from source overlay')
        altered.append(replace(blocker, end_s=_ci.parse_utc_seconds(bounds['screen_end_exclusive_utc'])))
        changed.append(blocker.ident)
    require(set(changed) == IDS, 'not all four source blockers found exactly once')
    return original, altered, {'records': selected, 'source_class': 'cloud_twin_window',
                              'recovery_timestamp_assigned': False,
                              'screening_end_exclusive': cfg['panel_end_exclusive'],
                              'overlay_sha256': PINS[OVERLAY], 'specification_sha256': PINS[specification]}


def unchanged_except(before, after, allowed):
    require(list(before.columns) == list(after.columns) and before.index.equals(after.index),
            'frame shape, columns or order changed')
    columns = [c for c in before.columns if c not in allowed]
    require(before[columns].equals(after[columns]), 'non-screen geometry, identity, counts or coverage changed')


def derive_screen(counts, events, cfg, blockers, amended):
    """Reproduce all original screen cells; change six screen fields and nothing else."""
    require(not counts.duplicated(list(KEY)).any(), 'duplicate full cell identity')
    lookup = events.set_index('event_id')
    require(not events.event_id.duplicated().any(), 'duplicate episode identity')
    old, new = _ci.BlockerIndex(blockers), _ci.BlockerIndex(amended)
    bounds = _ci.study_bounds(cfg)
    revised = counts.copy(deep=True)
    changes, screens = [], []
    for row in counts.to_dict('records'):
        require(row['treatment_policy'] == 'retain_eligible', 'treatment-retention policy changed')
        a, b = row['window_start_s'], row['window_end_s']
        own = {f'registry:{x}' for x in lookup.loc[row['event_id'], 'source_ids'].split('|') if x}
        exempt = own if _ci.own_exposure_exempt(row['window'], row['offset_days']) else set()
        sa, sb = _ci.screen_interval(row['start_s'], row['end_s'], row['window'], row['offset_days'])
        hits, detail = old.screen(sa, sb, exempt)
        next_hits, next_detail = new.screen(sa, sb, exempt)
        outside = _ci.window_support_outside(a, b, bounds)
        strict = _ci.window_invalid_reasons(outside, bool(hits), False)
        next_strict = _ci.window_invalid_reasons(outside, bool(next_hits), False)
        raw, next_raw = ';'.join(strict), ';'.join(next_strict)
        reason = _ss._policy_reason(raw, row['window'], row['offset_days'])
        next_reason = _ss._policy_reason(next_raw, row['window'], row['offset_days'])
        expected = (not strict, raw, not reason, reason, hits, detail)
        require(tuple(row[c] for c in SCREEN_COLUMNS) == expected, f'original screen does not reproduce: {row["event_id"]}')
        screens.append((not next_strict, next_raw, not next_reason, next_reason, next_hits, next_detail))
        if detail != next_detail:
            changes.append({'key': {k: row[k] for k in KEY}, 'before_analysis_valid': row['analysis_valid'],
                            'after_analysis_valid': not next_reason, 'before_detail': detail, 'after_detail': next_detail})
    for i, column in enumerate(SCREEN_COLUMNS):
        revised[column] = [row[i] for row in screens]
    unchanged_except(counts, revised, SCREEN_COLUMNS)
    require(not ((~counts.analysis_valid) & revised.analysis_valid).any(), 'extended screen gained availability')
    isolation = []
    for row in events.to_dict('records'):
        a, b = int(row['start_s']), int(row['end_s'])
        old_infra = any(old.klass[i] == 'infra' for i in old.hits(a, b))
        new_infra = any(new.klass[i] == 'infra' for i in new.hits(a, b))
        require(bool(row['infra_overlap']) == old_infra, 'original isolation flag differs')
        if new_infra != old_infra:
            isolation.append(row['event_id'])
    return revised, changes, sorted(isolation)


def derive_native_frame(original, screen_counts):
    """After strict native verification, derive only its matcher screen view."""
    keys = list(_ed.JOIN_KEYS)
    screen = screen_counts.copy()
    if 'onset_shift_minutes' not in screen:
        screen['onset_shift_minutes'] = 0
    require(not screen.duplicated(keys).any() and not original.duplicated(keys).any(),
            'duplicate native screen join key')
    lookup = screen.set_index(keys)
    revised = original.copy(deep=True)
    values = []
    for row in original.to_dict('records'):
        key = tuple(row[k] for k in keys)
        require(key in lookup.index, 'native row has no screen identity')
        identity = lookup.loc[key]
        require(row['window_start_s'] == identity.window_start_s and row['window_end_s'] == identity.window_end_s,
                'native screen geometry differs')
        values.append((bool(identity.analysis_valid), str(identity.analysis_invalid_reason)))
    revised['valid'] = [x[0] for x in values]
    revised['invalid_reason'] = [x[1] for x in values]
    unchanged_except(original, revised, ('valid', 'invalid_reason'))
    return revised


def restricted_block(frame, loaded, settings, cfg, outcome, label):
    helper = _es if outcome == 'trading' else _ed
    result = {'n_matched_cells': int(len(frame)),
              'statistics': helper._statistics_on(frame, loaded, settings, label)}
    if outcome == 'trading':
        result['decomposition_total'] = _es._decompose_on(
            frame, loaded, cfg, settings, restrict=False, tiers=(_es.TOTAL_TIER,), label=label)
        result['decomposition_groups'] = _es._decompose_on(
            frame, loaded, cfg, settings, restrict=True, tiers=_es.GROUP_TIERS,
            label=label + '+historical_both_tier')
    else:
        result['decomposition'] = dict(_ed.DECOMPOSITION)
    return normalized(result)


def analyse(view, native_frame, cfg, settings, outcome):
    offsets = _es.calendar_offsets(cfg)
    blocks, frames = {}, {}
    for calendar in CALENDARS:
        if outcome == 'trading':
            block, frame = _es._calendar_block(view, calendar, offsets[calendar], settings, cfg)
        else:
            block, frame = _ed._calendar_block(native_frame, view, calendar, offsets[calendar], settings,
                                               _ed.minimum_clean_controls(cfg))
        blocks[calendar], frames[calendar] = normalized(block), frame
    sets = {c: set(blocks[c]['matched_events']) for c in CALENDARS}
    common = sorted(set.intersection(*sets.values()))
    cross = {'role': 'common_support_comparison', 'matched_in_both': common,
             'n_matched_in_both': len(common),
             'matched_only_in': {c: sorted(sets[c] - set(common)) for c in CALENDARS}, 'calendars': {}}
    for calendar, frame in frames.items():
        cross['calendars'][calendar] = restricted_block(
            frame[frame.event_id.isin(common)], view, settings, cfg, outcome, 'common_to_both_calendars')
    return {'calendars': blocks, 'cross_calendar': cross}, frames


def same_event_comparison(old_frames, new_frames, view, cfg, settings, outcome):
    def compare(calendar, events):
        old = old_frames[calendar][old_frames[calendar].event_id.isin(events)].reset_index(drop=True)
        new = new_frames[calendar][new_frames[calendar].event_id.isin(events)].reset_index(drop=True)
        sort = ['event_id', 'metric']
        old, new = old.sort_values(sort).reset_index(drop=True), new.sort_values(sort).reset_index(drop=True)
        reference_equal = old[sort + ['control_offsets']].equals(new[sort + ['control_offsets']])
        rows_equal = old.equals(new)
        if reference_equal:
            require(rows_equal, 'same events/references changed measured values')
        first = restricted_block(old, view, settings, cfg, outcome, 'identical_event_intersection')
        second = restricted_block(new, view, settings, cfg, outcome, 'identical_event_intersection')
        if reference_equal:
            require(first == second, 'same events/references produced different inference')
        return {'events': sorted(events), 'n_events': len(events),
                'reference_offsets_equal': reference_equal, 'matched_rows_equal': rows_equal,
                'all_estimates_equal': first == second, 'original': first, 'source_aware': second}
    intersections = {c: set(old_frames[c].event_id) & set(new_frames[c].event_id) for c in CALENDARS}
    common = set.intersection(*intersections.values())
    return {'calendar_specific_intersections': {c: compare(c, intersections[c]) for c in CALENDARS},
            'common_to_both_screens_and_calendars': {
                'events': sorted(common), 'n_events': len(common),
                'calendars': {c: compare(c, common) for c in CALENDARS}}}


def verify_matching(original, corrected, old_frames, new_frames, expected):
    for calendar in CALENDARS:
        before, after = original['calendars'][calendar]['matched_events'], corrected['calendars'][calendar]['matched_events']
        require(before == expected[calendar]['before'] and after == expected[calendar]['after'],
                f'matching differs from outcome-blind metadata check: {calendar}')
        require(set(after) <= set(before), 'source-aware matching gained events')
        changed = []
        for event in after:
            old = set(old_frames[calendar].loc[old_frames[calendar].event_id == event, 'control_offsets'])
            new = set(new_frames[calendar].loc[new_frames[calendar].event_id == event, 'control_offsets'])
            if old != new:
                changed.append(event)
        require(changed == expected[calendar]['retained_reference_changes'], 'retained reference sets differ')
    for block, key in ((original, 'before'), (corrected, 'after')):
        require(block['cross_calendar']['matched_in_both'] == expected['common'][key], 'common event set differs')


def verify_original_result(actual, canonical, name):
    require(scientific(actual) == scientific(canonical), f'{name} original scientific output does not reproduce canonical')


def read_canonical(evidence, outcome):
    directory, payload, receipt_name = (
        ('results/strategy/liquidation', 'strategy_estimates.json', 'estimation_receipt.json')
        if outcome == 'trading' else
        ('results/development/liquidation', 'development_estimates.json', 'development_receipt.json'))
    receipt = evidence.read(Path(directory) / receipt_name)
    records = {r['file']: r['sha256'] for r in receipt['outputs']}
    result = evidence.read(Path(directory) / payload, records[payload])
    require(result['receipt'] == {k: v for k, v in receipt.items() if k != 'outputs'}, 'canonical receipt pair differs')
    return result


def build(root=ROOT, source_root=None):
    root = Path(root).resolve()
    evidence = Evidence(root, source_root)
    cfg = evidence.read('config/strategy_study.json')
    extension = evidence.read('config/extension.json')
    settings = _es.bootstrap_config(cfg, extension)
    require({k: settings[k] for k in ('bootstrap_draws', 'bootstrap_seed', 'primary_cluster_days', 'sensitivity_cluster_days')}
            == {'bootstrap_draws': 4000, 'bootstrap_seed': 20261017, 'primary_cluster_days': 7, 'sensitivity_cluster_days': 14},
            'frozen bootstrap specification differs')
    require(cfg['minimum_clean_controls'] == 2 and cfg['during_treatment_policy'] == 'retain_eligible', 'frozen design differs')
    # These unchanged strict boundaries run FIRST. No adjusted manifest is passed
    # to either loader, and no old receipt is attached to the derived views below.
    loaded = _ss.load_scoped_inputs(root / PANEL, root=root, config_path=root / 'config/strategy_study.json', source_root=source_root)
    native, native_stage = _di.load_frozen(
        root / PANEL, directory=root / NATIVE, receipt_sha256=NATIVE_PIN,
        source_audit=root / 'reviews/development_measurement/raw_creation_complete_audit.json',
        loaded=loaded, root=root, source_root=source_root)
    native_frame, excluded = _ed.development_frame(native, loaded)
    snapshots = {k: loaded[k].copy(deep=True) for k in ('counts', 'events', 'cohorts')}
    native_snapshot = native_frame.copy(deep=True)
    evidence.checked(PANEL / 'strategy_input_manifest.json')
    for row in loaded['manifest']['outputs']:
        evidence.checked(PANEL / row['file'], row['sha256'])
    evidence.checked(NATIVE / 'receipt.json', NATIVE_PIN)
    evidence.checked(NATIVE / 'window_counts.parquet', native_stage['count_table_sha256'])
    evidence.checked(native_stage['source_audit']['path'], native_stage['source_audit']['sha256'])
    blockers, amended, overlay = source_overlay(evidence, cfg)
    metadata = evidence.read(META / 'diagnostic/report.json')
    metadata_guard = evidence.read(META / 'guarded_reproduction.json')
    require(metadata_guard['separate_guarded_run_byte_identical'] is True
            and metadata_guard['outputs']['report.json'] == PINS[str(META / 'diagnostic/report.json')],
            'metadata screen guard binding differs')
    require(metadata['original_screen_reproduced_for_all_cells'] is True and metadata['outcome_columns_read'] == []
            and metadata['effect_estimation_run'] is False and set(metadata['corrected_source_ids']) == IDS,
            'metadata screen preconditions differ')
    # The prior metadata diagnostic also checked quota readiness. This driver
    # binds its complete report but does not read or recompute that separate task.
    for path, digest in metadata['inputs'].items():
        if path.startswith('data/quota_reset/') or path in {'data/reference/quota_reset_events.json', 'analysis/quota_reset_inputs.py'}:
            continue
        evidence.checked(path, digest)
    revised_counts, changes, isolation = derive_screen(loaded['counts'], loaded['events'], cfg, blockers, amended)
    require(len(loaded['counts']) == metadata['tier_window_rows'], 'full tier-window count differs')
    require(normalized(changes) == evidence.read(META / 'diagnostic/changed_cells.json'), 'full changed-cell inventory differs')
    require(len(changes) == metadata['changed_screen_detail_rows'] and sum(c['before_analysis_valid'] and not c['after_analysis_valid'] for c in changes)
            == metadata['analysis_valid_to_invalid_rows'], 'full-screen transition counts differ')
    require(isolation == sorted(c['event_id'] for c in metadata['isolation_changes']), 'isolation change inventory differs')
    revised_native = derive_native_frame(native_frame, revised_counts)
    canonical = {name: read_canonical(evidence, name) for name in ('trading', 'development')}
    actual_original = {
        'trading': _es.estimate(root / PANEL, root=root, source_root=source_root),
        'development': _ed.estimate(root / PANEL, root=root, source_root=source_root,
                                   frozen_counts=root / NATIVE, frozen_receipt_sha256=NATIVE_PIN,
                                   source_audit=root / 'reviews/development_measurement/raw_creation_complete_audit.json')}
    for outcome in canonical:
        verify_original_result(actual_original[outcome], canonical[outcome], outcome)
    original_view = {k: loaded[k] for k in ('counts', 'events', 'cohorts')}
    amended_view = {**original_view, 'counts': revised_counts}
    results = {}
    for outcome, match_key in (('trading', 'trading_matching'), ('development', 'native_development_matching')):
        old, old_frames = analyse(original_view, native_frame, cfg, settings, outcome)
        require(old['calendars'] == canonical[outcome]['calendars']
                and old['cross_calendar'] == canonical[outcome]['cross_calendar'],
                f'{outcome} helper orchestration differs from canonical estimator')
        new, new_frames = analyse(amended_view, revised_native, cfg, settings, outcome)
        verify_matching(old, new, old_frames, new_frames, metadata[match_key])
        comparison = same_event_comparison(old_frames, new_frames, original_view, cfg, settings, outcome)
        results[outcome] = {'original': old, 'source_aware': new, 'identical_event_comparisons': comparison,
                            'matching_changes': metadata[match_key]}
    for k, snapshot in snapshots.items():
        require(loaded[k].equals(snapshot), f'original {k} mutated')
    require(native_frame.equals(native_snapshot), 'original native frame mutated')
    unchanged_except(loaded['counts'], revised_counts, SCREEN_COLUMNS)
    unchanged_except(native_frame, revised_native, ('valid', 'invalid_reason'))
    _ss.verify_scoped_inputs(root / PANEL, root=root, config_path=root / 'config/strategy_study.json', source_root=source_root)
    modules = sorted(set(_es.CODE_MODULES) | set(_ed.CODE_MODULES) | {'source_screen_estimates', 'competition_inputs'})
    code = {name: sha(evidence.checked(Path('analysis') / f'{name}.py')) for name in modules}
    evidence.unchanged()
    validation = {
        'original_screen_reproduced_for_all_cells': True, 'tier_window_rows': len(loaded['counts']),
        'changed_screen_detail_rows': len(changes),
        'analysis_valid_to_invalid_rows': metadata['analysis_valid_to_invalid_rows'],
        'isolation_flag_changes': len(isolation), 'isolation_event_ids': isolation,
        'original_estimates_reproduced_before_amended_inference': True,
        'raw_counts_geometry_coverage_episodes_cohorts_ranks_unchanged': True,
        'trading_screen_columns_changed_only': list(SCREEN_COLUMNS),
        'native_matcher_columns_changed_only': ['valid', 'invalid_reason'],
        'native_outcome_coverage_verified_separately': True,
        'events_and_cohorts_unchanged': True,
        'isolation_flags_not_used_for_new_strata': True,
        'canonical_inputs_modified': False,
        'original_native_excluded_input_rows': len(excluded),
        'metadata_check_sha256': PINS[str(META / 'diagnostic/report.json')],
        'changed_cells_sha256': PINS[str(META / 'diagnostic/changed_cells.json')],
    }
    receipt = {
        'driver': 'source_screen_estimates', 'derivation': 'explicit_in_memory_screen_view_after_strict_original_verification',
        'strategy': 'liquidation', 'primary_stratum': {'window': 'during', 'head_cut': 50, 'onset_shift_minutes': 0},
        'original_panel_manifest_sha256': sha(root / PANEL / 'strategy_input_manifest.json'),
        'original_native_input_stage': native_stage,
        'specification_sha256': PINS[specification], 'overlay_sha256': PINS[OVERLAY],
        'overlay_receipt_sha256': PINS[OVERLAY_RECEIPT],
        'full_screen_metadata_sha256': PINS[str(META / 'diagnostic/report.json')],
        'inputs': dict(sorted(evidence.files.items())), 'driver_code_hashes': code, 'bootstrap': settings,
        'old_receipt_certifies_new_screen': False,
        'trust_boundary': 'original receipts verify unchanged geometry, cohorts and measured counts; this receipt separately binds amended screening and estimates',
        'remaining_source_uncertainty': '147 AWS rows unverified; two retrospective zero-duration onsets not adopted; no inferred-clock endpoint shortening',
        'raw_capture_reread': False, 'quota_outcomes_read': False, 'timing_or_stratum_estimates_run': False,
        'causal_claim': 'none; conditional primary comparisons under two recorded screens',
    }
    result = {'receipt': receipt, 'strategy': 'liquidation', 'primary_stratum': receipt['primary_stratum'],
              'bootstrap': settings, 'results': results, 'screen_validation': validation,
              'native_outcome_definition': _ed.OUTCOME_DEFINITION,
              'native_decomposition': _ed.DECOMPOSITION, 'trading_joint_identity': 'Q = A * I',
              'comparison_note': 'Sample losses and identical-event comparisons are separate; unchanged retained references require exactly equal within-event inference.',
              'causal_claim': receipt['causal_claim']}
    files = {RESULT_FILE: _es.dump_json(result), 'screen_validation.json': _es.dump_json(validation),
             'applied_overlay.json': _es.dump_json(overlay)}
    standalone = {**receipt, 'outputs': {name: hashlib.sha256(body.encode()).hexdigest() for name, body in sorted(files.items())}}
    files[RECEIPT_FILE] = _es.dump_json(standalone)
    return {'files': files, 'result': normalized(result), 'receipt': standalone}


def publish(outdir, built):
    outdir = Path(outdir)
    require(not outdir.is_symlink(), 'linked output root refused')
    files = built['files']
    require(set(files) == OUTPUTS, 'unexpected output inventory')
    if outdir.exists():
        require(outdir.is_dir() and {p.name for p in outdir.iterdir()} == OUTPUTS, 'existing output inventory differs')
        for name, body in files.items():
            path = outdir / name
            require(path.is_file() and not path.is_symlink() and path.read_bytes() == body.encode(),
                    'existing output differs; never overwritten')
        return 'unchanged'
    outdir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.source-screen-', dir=outdir.parent))
    try:
        for name, body in files.items():
            (staging / name).write_text(body)
        require(not outdir.exists() and not outdir.is_symlink(), 'output appeared during publication')
        os.rename(staging, outdir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return 'written'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--outdir', type=Path)
    args = parser.parse_args(argv)
    try:
        built = build(args.root, args.source_root)
        outcome = publish(args.outdir, built) if args.outdir else 'not_published'
        print(json.dumps({'publication': outcome, 'canonical_inputs_modified': False,
                          'matching': {k: {c: [len(v['original']['calendars'][c]['matched_events']),
                                              len(v['source_aware']['calendars'][c]['matched_events'])]
                                           for c in CALENDARS} for k, v in built['result']['results'].items()}}, indent=2))
        return 0
    except (ScreenError, OSError) as exc:
        print(f'source_screen_estimates: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
