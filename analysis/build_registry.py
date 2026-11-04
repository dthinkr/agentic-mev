#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the AIDA registry and developer-disruption sample from provider tables.

Run with ``python3 -I src/build_registry.py``. Read provider CSVs under
``data/incidents_v2/providers/``, preferring ``*.verified.csv`` when available.
Write ``registry.csv``, ``registry_dev_60min_24h.csv`` and ``COVERAGE.md`` under
``data/incidents_v2/``.

Deduplicate within providers using incident IDs, affected services, timestamps
and titles. Prefer official sources for dates and causes; retain source URLs.
The development sample includes API and coding-tool disruptions lasting
60 minutes to 24 hours at model developers.
"""

import csv
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "data", "incidents_v2")
PROV_DIR = os.path.join(HERE, "providers")
DATA_DIR = os.path.abspath(os.path.join(HERE, os.pardir))

OLD_REGISTRY = os.path.join(DATA_DIR, "support", "incident_registry.csv")
OTHER_PROVIDERS = os.path.join(DATA_DIR, "reference", "other_providers_outages.csv")
SEED_INVENTORY = os.path.join(DATA_DIR, "support", "seed_in_panel_inventory.csv")

OUT_REGISTRY = os.path.join(HERE, "registry.csv")
OUT_DEV = os.path.join(HERE, "registry_dev_60min_24h.csv")
OUT_COVERAGE = os.path.join(HERE, "COVERAGE.md")

WIN_START = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
WIN_END = datetime(2026, 10, 28, 0, 0, tzinfo=timezone.utc)
TOUCH_MIN = 10          # "overlap or touch within 10 minutes"
EXPOSURE_LO, EXPOSURE_HI = 60, 1440

BASE_COLS = [
    "provider", "provider_group", "region", "own_model", "incident_id", "title",
    "status", "impact", "start_utc", "end_utc", "duration_min", "components",
    "surface", "dev_relevant", "cause", "cause_category", "source_kind",
    "source_url", "evidence", "verified", "verified_source_url", "notes",
]
OUT_COLS = BASE_COLS + ["dedup_group", "n_sources", "related_provider"]

# Provider names and groups.
PROVIDER_ALIAS = {"deepseek": "deepseek"}

# Supplementary tables use source types as groups; prefer the provider table.
SOURCE_GROUPS = {"aggregator", "official_backfill"}
GROUP_FALLBACK = {"cursor": "coding_tool"}

# Exclude model hosts, coding tools and infrastructure providers from model developers.
OWN_MODEL_FALSE = {
    "amazon-bedrock", "azure-openai", "cursor",
    "cloudflare", "quicknode", "alchemy", "github", "infura",
    "aws", "npm", "azure", "gcp", "pypi",
}

SOURCE_RANK = {
    "official_statuspage": 0,
    "official_incidentio": 0,
    "official_other": 1,
    "wayback": 2,
    "isdown": 3,
    "statusgator": 4,
    "news": 5,
    "community": 6,
}
OFFICIAL_KINDS = {"official_statuspage", "official_incidentio", "official_other"}

DEV_SURFACES = {"api", "coding_agent"}

# Source assessment grades and coverage gaps from providers/<slug>.notes.md.
TABLE_META = {
    "alibaba-qwen": ("C", "Status-page history is a rolling ~12-month window, so three known incident ids predate what the API will return; StatusGator renders 2 of ~32 Model Studio incidents behind its paywall."),
    "amazon-bedrock": ("C", "AWS publishes no closed-event history endpoint; the whole table is rebuilt from 50 Wayback captures of the Health Dashboard, and six StatusGator component incidents carry no AWS text at all."),
    "anthropic": ("A", "Only the Wayback CDX index was accessible; archived page bodies could not corroborate the live status page."),
    "azure-openai": ("C", "Per-incident records live in tenant-scoped Azure Service Health; 6 of 19 incidents are acknowledged only through a webcast signup page, and 5 rows still have no end_utc."),
    "baidu-qianfan": ("A", "Baidu publishes no per-incident timeline; the console message centre is login-gated, health-dashboard retention stops at 2025-08-12, and the 2026-01-20 ERNIE outage proves official under-reporting."),
    "cloud-infra-blockers": ("n/a", "Source verification is not recorded for this control set; all rows have verified=false. GCP and Azure publish limited histories; 16 of 92 Azure Wayback snapshots were available before rate limiting."),
    "cn-others": ("D", "All of 2025 and 2026-01 to 2026-05-24 are empty for all six vendors because no vendor-published record exists; iFlytek, Baichuan and SenseNova produced zero availability rows across 22 months."),
    "cohere": ("A", "Live API history is capped at ~2025-03; the pre-2025-03 span was closed from archived incident permalinks, and pre-2025 history was not pursued."),
    "deepseek": ("B", "Only 3 of ~90 incidents carry a vendor-stated root cause, and the 9 added StatusGator/IsDown rows lack vendor confirmation; the vendor changed status-page vendors mid-window."),
    "google-ai": ("B", "Google Cloud's public dashboard went nearly silent after 2025-03; no Gemini CLI status source exists; 63 of 82 rows have no stated cause; the AI Studio endpoint has no archive to diff against."),
    "meta-llama": ("C", "Meta's observed model-API incident history is empty. Both dated events come from external sources; no 2025 status-page capture was found."),
    "minimax": ("B", "2025-01-01 to 2026-06-22 (about 18 months) has no incident data to collect: the status page did not exist yet."),
    "mirror-sweep": ("B", "StatusGator paginates deep history for only 3 model vendors (anthropic, claude, openai); every other vendor page is capped at its 5 most recent incidents behind a paywall. IsDown contributed nothing (Cloudflare challenge)."),
    "mistral": ("B", "The vendor reports no incidents for 2025-01-01 to 2025-07-23, about 32% of the window; availability during that interval is unknown."),
    "moonshot-kimi": ("A", "2025-01-01 to 2025-04-03 predates the status page's creation; 2025-02 and 2025-03 produced nothing in any source; no postmortems exist, so 847 of 853 rows have no stated cause."),
    "openai": ("A", "Root-cause text is absent for 629 of 643 incidents (not published by the vendor); StatusGator's own OpenAI history starts 2025-01-07, and one incident id's content cannot be retrieved."),
    "perplexity": ("B", "Two Instatus notices could not be opened (one has a single Wayback capture, one none); isdown.app is Cloudflare-403; the Internet Archive was intermittently offline during the run."),
    "tencent-hunyuan": ("A", "The table uses the live Chinese-language status board. Wayback records were unavailable, and an EdgeOne challenge blocked www.tencentcloud.com."),
    "volcengine-doubao": ("C", "Wayback has zero snapshots of status.volcengine.com and the vendor JSON API is auth-gated; no vendor root cause exists for any Doubao consumer incident and three have no published start or end."),
    "xai": ("B", "2025-01-01 to 2025-01-15 predates the status page's own history; Wayback rate-limited the host for most collection attempts; 25 of 253 rows rest on a single dated source."),
    "zhipu-bigmodel": ("C", "23 of 27 rows verified, 8 rows have no end_utc, and roughly the first seven months of the window have no vendor record at all; most rows are community reports."),
}

# Primary tables for providers without a dedicated file.
PRIMARY_TABLE = {
    "cloudflare": "cloud-infra-blockers", "quicknode": "cloud-infra-blockers",
    "alchemy": "cloud-infra-blockers", "github": "cloud-infra-blockers",
    "infura": "cloud-infra-blockers", "aws": "cloud-infra-blockers",
    "npm": "cloud-infra-blockers", "azure": "cloud-infra-blockers",
    "gcp": "cloud-infra-blockers", "pypi": "cloud-infra-blockers",
    "cursor": "mirror-sweep",
}

# Providers without public incident histories.
# id_prefix selects rows within a shared table; empty selects the whole table.
NO_PUBLIC_HISTORY = [
    ("Meta (Llama API / Meta Model API)", "meta-llama", "",
     "Observed status-page captures have an empty incident_history array; the October 2026 "
     "revision removed it. Both events come from an executive's X posts or news reports."),
    ("iFlytek Spark 星火 / 讯飞星辰 MaaS", "cn-others", "xfyun",
     "No public incident or maintenance feed. Both records describe model deprecations."),
    ("Baichuan 百川", "cn-others", "__none__",
     "No records across 22 months; no vendor or aggregator incident channel was found."),
    ("SenseTime SenseNova 商汤日日新", "cn-others", "__none__",
     "No records. The only dated notice concerns a quota change whose details require login."),
    ("01.AI 零一万物", "cn-others", "01ai",
     "No incident channel. One shutdown notice is available through two reposts; "
     "the original page has no retrievable text."),
    ("StepFun 阶跃星辰", "cn-others", "stepfun",
     "No incident channel. Two records concern capacity or retirement announcements."),
    ("Kuaishou Kling 可灵", "cn-others", "sg-kling",
     "No vendor status page. Five StatusGator crowd reports provide the availability "
     "incidents in cn-others."),
]

# Providers outside the registry's coverage.
NOT_COVERED = [
    ("groq", "Present in ../reference/other_providers_outages.csv (25 rows, 2024-04 to 2026-07) "
             "but has no table under providers/, so none of it is in this registry."),
    ("github-copilot", "../reference/github_copilot_incidents.csv has no corresponding provider "
                       "table. The registry includes GitHub platform incidents under `github`, "
                       "but excludes Copilot-specific incidents."),
    ("together / fireworks / openrouter / replicate / anyscale",
     "Model hosting and routing services outside the registry's scope."),
]

# title tokenisation
# Remove common status terms before comparing product names.
GENERIC = set("""
a about above across affected affecting after all also an and any are area areas as at available
availability back be because been before being below between both brief but by can cannot
complete completely continued continuing could create created creating currently customers
degradation degrade degraded degrades degrading delay delayed delays disruption disruptions down
downtime due during elevated elevation encountered error errors experience experienced
experiencing fail failed failing failure failures for from full had has have high higher impact
impacted impacting in increase increased increasing incident incidents intermittent
intermittently investigating is issue issues it its latencies latency limited load long longer
major many may might minor more most multiple network networking new no none not now observed
of off on ongoing only operation operations or other others our out outage outages over partial
partially performance period post potential problem problems rate rates reduced region regional
regions report reported reporting reports request requests resolution resolved response
responding result resulting results rising risk see service services several severe short
significant since slow slower slowness small some spike spikes stall stalled stalling start
started still stuck sustained temporarily temporary than that the their then there this those
through time timeout timeouts timing to total unable unavailable unavailability under unexpected
up update updated updates upon us use user users using via was we were when which while widespread
will with within without work working would
""".split())

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
WORD_RE = re.compile(r"[a-z0-9][a-z0-9._+/]*")
JACCARD_SAME = 0.34   # >= this -> titles describe the same product


def product_tokens(title):
    """Tokens in a title that plausibly name a product, model, chain or component."""
    t = unicodedata.normalize("NFKC", title).lower()
    toks = set()
    for run in CJK_RE.findall(t):
        toks.add(run)
    t = CJK_RE.sub(" ", t)
    t = t.replace("-", " ").replace("(", " ").replace(")", " ").replace(",", " ")
    t = t.replace("[", " ").replace("]", " ").replace('"', " ").replace("'", " ")
    for w in WORD_RE.findall(t):
        w = w.strip("._+/")
        if not w or w in GENERIC or len(w) < 2:
            continue
        toks.add(w)
    return toks


def titles_name_different_products(a, b):
    """Compare product tokens using Jaccard overlap.

    Empty or nested token sets do not prevent merging.
    """
    ta, tb = product_tokens(a), product_tokens(b)
    if not ta or not tb:
        return False
    if ta == tb or ta <= tb or tb <= ta:
        return False
    j = len(ta & tb) / len(ta | tb)
    return j < JACCARD_SAME


# related_provider: vendor names that may appear in another vendor's title
VENDOR_PATTERNS = [
    ("anthropic", r"anthropic|claude|fable ?[0-9]|mythos ?[0-9]"),
    ("openai", r"openai|chatgpt|gpt-?[0-9]|codex|sora|dall-?e|whisper"),
    ("google-ai", r"google ai|gemini|gemma|aistudio|ai studio|notebooklm|vertex"),
    ("xai", r"\bxai\b|x\.ai|grok"),
    ("deepseek", r"deepseek"),
    ("mistral", r"mistral|mixtral|codestral|devstral|magistral|pixtral"),
    ("cohere", r"cohere|command r|rerank"),
    ("perplexity", r"perplexity|sonar"),
    ("minimax", r"minimax|hailuo|abab"),
    ("moonshot-kimi", r"moonshot|kimi|k2-thinking"),
    ("zhipu-bigmodel", r"zhipu|智谱|z\.ai|bigmodel|glm-?[0-9]"),
    ("alibaba-qwen", r"qwen|通义|tongyi|model studio|百炼|dashscope"),
    ("baidu-qianfan", r"qianfan|千帆|ernie|文心"),
    ("tencent-hunyuan", r"hunyuan|混元"),
    ("volcengine-doubao", r"doubao|豆包|volcengine|火山|\bark\b"),
    ("meta-llama", r"\bllama\b|meta ai"),
    ("cn-others", r"iflytek|讯飞|星火|stepfun|阶跃|baichuan|百川|sensenova|商汤|kling|可灵|零一万物"),
    ("amazon-bedrock", r"bedrock"),
    ("azure-openai", r"azure openai|ai foundry|foundry"),
    ("cursor", r"\bcursor\b|bugbot"),
    ("cloudflare", r"cloudflare"),
    ("github", r"github"),
    ("aws", r"\baws\b|amazon web services|cloudfront|dynamodb|\bec2\b|\bs3\b"),
    ("azure", r"\bazure\b|microsoft entra"),
    ("gcp", r"google cloud|\bgcp\b|apigee"),
    ("alchemy", r"\balchemy\b"),
    ("infura", r"\binfura\b"),
    ("quicknode", r"quicknode"),
    ("npm", r"\bnpm\b"),
    ("pypi", r"\bpypi\b"),
]
VENDOR_RE = [(p, re.compile(rx, re.I)) for p, rx in VENDOR_PATTERNS]


# helpers
def parse_ts(raw):
    s = (raw or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc, second=0, microsecond=0)
        except ValueError:
            continue
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})", s)
    if m:
        y, mo, d, h, mi = (int(x) for x in m.groups())
        return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)
    return None


def fmt_ts(dt):
    return "" if dt is None else dt.strftime("%Y-%m-%dT%H:%MZ")


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes", "t")


def rank_of(kind):
    return SOURCE_RANK.get((kind or "").strip(), 9)


def load_provider_rows():
    """One row per (source table, source row), with timestamps parsed and the window applied."""
    files = sorted(f for f in os.listdir(PROV_DIR) if f.endswith(".csv"))
    slugs = sorted({f[:-len(".verified.csv")] if f.endswith(".verified.csv") else f[:-4]
                    for f in files})
    rows, stats = [], {}
    for slug in slugs:
        vpath = os.path.join(PROV_DIR, slug + ".verified.csv")
        path = vpath if os.path.exists(vpath) else os.path.join(PROV_DIR, slug + ".csv")
        with open(path, newline="", encoding="utf-8") as fh:
            raw = list(csv.DictReader(fh))
        kept, dropped_window, dropped_ts = 0, 0, 0
        for i, r in enumerate(raw):
            rec = {k: (r.get(k) or "").strip() for k in BASE_COLS}
            rec["_slug"] = slug
            rec["_file"] = os.path.basename(path)
            rec["_i"] = i
            rec["_provider"] = PROVIDER_ALIAS.get(rec["provider"].lower(), rec["provider"].lower())
            st = parse_ts(rec["start_utc"])
            en = parse_ts(rec["end_utc"])
            if st is None:
                dropped_ts += 1
                continue
            if not (WIN_START <= st < WIN_END):
                dropped_window += 1
                continue
            if en is not None and en < st:
                en = None                       # inverted interval: keep start, drop bad end
            if en is not None and en >= WIN_END:
                en = WIN_END - timedelta(minutes=1)
            rec["_start"], rec["_end"] = st, en
            rec["_rank"] = rank_of(rec["source_kind"])
            rec["_verified"] = truthy(rec["verified"])
            rows.append(rec)
            kept += 1
        stats[slug] = dict(file=os.path.basename(path), raw=len(raw), kept=kept,
                           dropped_window=dropped_window, dropped_bad_ts=dropped_ts,
                           verified_file=os.path.exists(vpath))
    return rows, slugs, stats


def canonical_attrs(rows):
    """Resolve each provider's group, model ownership and region."""
    groups, regions, own = defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    for r in rows:
        p = r["_provider"]
        g = r["provider_group"]
        if g and g not in SOURCE_GROUPS:
            groups[p][g] += 1
        if r["region"]:
            regions[p][r["region"]] += 1
        own[p].add(truthy(r["own_model"]))
    attrs = {}
    for p in sorted({r["_provider"] for r in rows}):
        if groups[p]:
            grp = sorted(groups[p].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        else:
            grp = GROUP_FALLBACK.get(p, "other")
        reg = (sorted(regions[p].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
               if regions[p] else "")
        if reg == "global":
            reg = "intl"
        om = (p not in OWN_MODEL_FALSE) and (True in own[p] or grp == "model_api")
        attrs[p] = dict(provider_group=grp, region=reg, own_model=om)
    return attrs


# clustering
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if rb < ra:
            ra, rb = rb, ra
        self.p[rb] = ra
        return True


def authority_window(group):
    """Select dates for one incident ID, preferring official sources to mirrors."""
    srt = sorted(group, key=lambda r: (r["_rank"], 0 if r["_verified"] else 1,
                                       0 if r["_end"] else 1, r["_start"], r["_slug"], r["_i"]))
    start = srt[0]["_start"]
    end = srt[0]["_end"]
    if end is None:
        for r in srt:
            if r["_end"] is not None:
                end = r["_end"]
                break
    if end is not None and end < start:
        start, end = end, start
    return start, end


def cluster_provider(rows):
    """Group one provider's rows by incident ID, then use complete-linkage clustering.

    Groups must share an affected service, overlap or lie within 10 minutes,
    and have compatible product titles. Require pairwise compatibility to
    prevent long incidents from linking unrelated shorter incidents.
    """
    n = len(rows)
    dsu = DSU(n)
    by_id = defaultdict(list)
    for i, r in enumerate(rows):
        if r["incident_id"]:
            by_id[r["incident_id"]].append(i)
    for _id in sorted(by_id):
        idxs = by_id[_id]
        for j in idxs[1:]:
            dsu.union(idxs[0], j)

    atoms_map = defaultdict(list)
    for i in range(n):
        atoms_map[dsu.find(i)].append(i)

    atoms = []
    for k in sorted(atoms_map):
        idxs = sorted(atoms_map[k])
        grp = [rows[i] for i in idxs]
        st, en = authority_window(grp)
        atoms.append(dict(rows=grp, start=st, end=en if en is not None else st,
                          surfaces={r["surface"] for r in grp},
                          titles=[r["title"] for r in grp], key=k))
    atoms.sort(key=lambda a: (a["start"], a["end"], a["key"]))

    tol = timedelta(minutes=TOUCH_MIN)

    def compatible(a, b):
        if not (a["surfaces"] & b["surfaces"]):
            return False
        if a["start"] - b["end"] > tol or b["start"] - a["end"] > tol:
            return False
        if all(titles_name_different_products(x, y) for x in a["titles"] for y in b["titles"]):
            return False
        return True

    clusters = []   # each: dict(atoms=[...], max_end=datetime)
    for a in atoms:
        placed = False
        for c in clusters:
            if a["start"] - c["max_end"] > tol:
                continue                      # cannot touch anything already in there
            if all(compatible(a, b) for b in c["atoms"]):
                c["atoms"].append(a)
                if a["end"] > c["max_end"]:
                    c["max_end"] = a["end"]
                placed = True
                break
        if not placed:
            clusters.append(dict(atoms=[a], max_end=a["end"]))

    out = []
    for c in clusters:
        rs = []
        for a in sorted(c["atoms"], key=lambda a: (a["start"], a["end"], a["key"])):
            rs.extend(a["rows"])
        out.append(rs)
    out.sort(key=lambda rs: (min(r["_start"] for r in rs),
                             max((r["_end"] or r["_start"]) for r in rs),
                             sorted(r["incident_id"] for r in rs)[0]))
    return out


def pick(cluster, field, want=None):
    """First non-empty value of field, scanning the cluster in authority order."""
    for r in cluster:
        v = r[field]
        if v and (want is None or want(v)):
            return v
    return ""


def merge_cluster(cluster, attrs, group_id):
    """Collapse one cluster into a single registry row."""
    srt = sorted(cluster, key=lambda r: (r["_rank"], 0 if r["_verified"] else 1,
                                         0 if r["cause"] else 1,
                                         0 if r["_end"] else 1,
                                         r["_start"], r["_slug"], r["_i"]))
    primary = srt[0]
    prov = primary["_provider"]
    a = attrs[prov]

    # Prefer authoritative dates within each ID; take the interval hull across IDs.
    sub = defaultdict(list)
    for r in srt:
        sub[r["incident_id"] or ("__noid__%d_%s" % (r["_i"], r["_slug"]))].append(r)
    wins = [authority_window(g) for _k, g in sorted(sub.items())]
    start = min(w[0] for w in wins)
    ends = [w[1] for w in wins if w[1] is not None]
    end = max(ends) if ends else None

    if end is None:
        dur = ""
        status = "unresolved_end"
    else:
        dur = int((end - start).total_seconds() // 60)
        status = pick(srt, "status", want=lambda v: v != "unresolved_end") or "resolved"

    surface = primary["surface"] or pick(cluster, "surface") or "other"
    cause_row = next((r for r in srt if r["cause"]), None)
    cause = cause_row["cause"] if cause_row else ""
    if cause_row and cause_row["cause_category"]:
        cat = cause_row["cause_category"]
    else:
        cats = Counter(r["cause_category"] for r in srt
                       if r["cause_category"] and r["cause_category"] != "unknown")
        cat = (sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if cats
               else (primary["cause_category"] or "unknown"))

    # Determine development relevance from the highest-ranked sources for this service.
    peers = [r for r in srt if r["surface"] == surface and r["_rank"] == primary["_rank"]]
    dev = any(truthy(r["dev_relevant"]) for r in peers) if peers else truthy(primary["dev_relevant"])
    if surface not in DEV_SURFACES:
        dev = False
    if cat == "maintenance":
        dev = False

    urls, seen = [], set()
    for r in sorted(srt, key=lambda r: (r["_rank"], r["source_url"], r["_slug"])):
        u = r["source_url"]
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
    vurls, vseen = [], set()
    for r in sorted(srt, key=lambda r: (r["_rank"], r["verified_source_url"], r["_slug"])):
        u = r["verified_source_url"]
        if u and u not in vseen:
            vseen.add(u)
            vurls.append(u)

    kinds = sorted({r["source_kind"] for r in srt if r["source_kind"]}, key=lambda k: (rank_of(k), k))
    slugs = sorted({r["_slug"] for r in srt})
    title = primary["title"] or pick(cluster, "title")
    components = max((r["components"] for r in srt if r["components"]), key=len, default="")

    all_ids = sorted({r["incident_id"] for r in srt if r["incident_id"]})
    note = primary["notes"]
    prov_note = ("sources=%s; tables=%s; rows_merged=%d"
                 % (",".join(kinds), ",".join(slugs), len(srt)))
    if len(all_ids) > 1:
        prov_note += "; provider_ids=" + ",".join(all_ids)
    if len(srt) > 1:
        distinct = {(r["_start"], r["_end"]) for r in srt}
        if len(distinct) > 1:
            spans = sorted({"%s/%s" % (fmt_ts(r["_start"]), fmt_ts(r["_end"])) for r in srt})
            prov_note += "; source_windows=" + " ".join(spans)
    note = ("%s [merge: %s]" % (note, prov_note)) if note else "[merge: %s]" % prov_note

    rel = []
    low = title.lower()
    for p, rx in VENDOR_RE:
        if p == prov:
            continue
        if rx.search(low):
            rel.append(p)

    return {
        "provider": prov,
        "provider_group": a["provider_group"],
        "region": a["region"],
        "own_model": "true" if a["own_model"] else "false",
        "incident_id": pick(srt, "incident_id"),
        "title": title,
        "status": status,
        "impact": pick(srt, "impact"),
        "start_utc": fmt_ts(start),
        "end_utc": fmt_ts(end),
        "duration_min": "" if dur == "" else str(dur),
        "components": components,
        "surface": surface,
        "dev_relevant": "true" if dev else "false",
        "cause": cause,
        "cause_category": cat,
        "source_kind": kinds[0] if kinds else "",
        "source_url": " | ".join(urls),
        "evidence": pick(srt, "evidence"),
        "verified": "true" if any(r["_verified"] for r in srt) else "false",
        "verified_source_url": " | ".join(vurls),
        "notes": note,
        "dedup_group": group_id,
        "n_sources": str(len(urls) if urls else len(srt)),
        "related_provider": ";".join(sorted(set(rel))),
        "_start": start, "_end": end, "_dur": dur, "_rows": srt, "_ids": set(all_ids),
    }


# old-registry comparison
def load_old_registry():
    out = []
    if not os.path.exists(OLD_REGISTRY):
        return out
    with open(OLD_REGISTRY, newline="", encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            st, en = parse_ts(r.get("start")), parse_ts(r.get("end"))
            if st is None or not (WIN_START <= st < WIN_END):
                continue
            out.append(dict(i=i, provider=(r.get("provider") or "").strip().lower(),
                            code=(r.get("code") or "").strip(),
                            title=(r.get("title") or "").strip(),
                            surface=(r.get("surface") or "").strip(),
                            dev_relevant=truthy(r.get("dev_relevant")),
                            start=st, end=en))
    return out


def compare_old(new_rows, old_rows):
    """Match earlier registry rows within providers by ID, then by time.

    Allow interval overlap or starts within 30 minutes. Multiple earlier rows
    may match one merged incident.
    """
    providers = sorted({o["provider"] for o in old_rows})
    new_by_p = defaultdict(list)
    for n in new_rows:
        if n["provider"] in providers:
            new_by_p[n["provider"]].append(n)
    old_by_p = defaultdict(list)
    for o in old_rows:
        old_by_p[o["provider"]].append(o)

    matched_new, matched_old, pairs = set(), set(), []
    for p in providers:
        id_index = {}
        for n in new_by_p[p]:
            for _id in sorted(n["_ids"]):
                id_index.setdefault(_id, n)
        for o in sorted(old_by_p[p], key=lambda o: (o["start"], o["i"])):
            n = id_index.get(o["code"])
            how = "id"
            if n is None:
                how = "time"
                best, bestd = None, None
                for cand in new_by_p[p]:
                    d = abs((cand["_start"] - o["start"]).total_seconds()) / 60.0
                    oe = o["end"] or o["start"]
                    ne = cand["_end"] or cand["_start"]
                    overlap = (o["start"] <= ne) and (cand["_start"] <= oe)
                    if overlap or d <= 30:
                        if bestd is None or (d, cand["dedup_group"]) < (bestd, best["dedup_group"]):
                            best, bestd = cand, d
                n = best
            if n is None:
                continue
            pairs.append((o, n, how))
            matched_old.add((p, o["i"]))
            matched_new.add(n["dedup_group"])
    new_only = [n for p in providers for n in new_by_p[p] if n["dedup_group"] not in matched_new]
    old_only = [o for p in providers for o in old_by_p[p] if (p, o["i"]) not in matched_old]
    disagree = []
    for o, n, how in pairs:
        ds = (n["_start"] - o["start"]).total_seconds() / 60.0
        de = ((n["_end"] - o["end"]).total_seconds() / 60.0
              if (n["_end"] and o["end"]) else None)
        if abs(ds) > 15 or (de is not None and abs(de) > 15):
            disagree.append((o, n, how, ds, de))
    return providers, pairs, new_only, old_only, disagree


# main
def main():
    rows, slugs, stats = load_provider_rows()
    attrs = canonical_attrs(rows)

    by_prov = defaultdict(list)
    for r in rows:
        by_prov[r["_provider"]].append(r)

    merged = []
    for prov in sorted(by_prov):
        prows = sorted(by_prov[prov], key=lambda r: (r["_start"], r["_slug"], r["_i"]))
        clusters = cluster_provider(prows)
        clusters.sort(key=lambda c: (min(r["_start"] for r in c),
                                     min((r["_end"] or r["_start"]) for r in c),
                                     sorted(r["incident_id"] for r in c)[0] if c else ""))
        width = max(4, len(str(len(clusters))))
        for k, c in enumerate(clusters, 1):
            merged.append(merge_cluster(c, attrs, "%s-%0*d" % (prov, width, k)))

    merged.sort(key=lambda n: (n["start_utc"], n["provider"], n["dedup_group"]))

    with open(OUT_REGISTRY, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS, extrasaction="ignore")
        w.writeheader()
        for n in merged:
            w.writerow(n)

    dev = [n for n in merged
           if n["own_model"] == "true" and n["dev_relevant"] == "true"
           and n["duration_min"] != ""
           and EXPOSURE_LO <= int(n["duration_min"]) <= EXPOSURE_HI]
    with open(OUT_DEV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS, extrasaction="ignore")
        w.writeheader()
        for n in dev:
            w.writerow(n)

    old_rows = load_old_registry()
    write_coverage(merged, dev, rows, slugs, stats, attrs, old_rows)

    print("rows read        : %d" % len(rows))
    print("incidents merged : %d" % len(merged))
    print("verified         : %d (%.1f%%)" % (
        sum(1 for n in merged if n["verified"] == "true"),
        100.0 * sum(1 for n in merged if n["verified"] == "true") / max(1, len(merged))))
    print("providers        : %d" % len({n["provider"] for n in merged}))
    print("dev 60m-24h      : %d" % len(dev))
    print("wrote %s" % OUT_REGISTRY)
    print("wrote %s" % OUT_DEV)
    print("wrote %s" % OUT_COVERAGE)


def md_esc(s):
    return s.replace("|", "\\|").replace("\n", " ")


_GID_INDEX = {}


def build_gid_index(merged):
    _GID_INDEX.clear()
    for n in merged:
        for r in n["_rows"]:
            _GID_INDEX[(r["_slug"], r["_i"])] = n["dedup_group"]


def gid_of_row(merged, row):
    return _GID_INDEX.get((row["_slug"], row["_i"]))


def write_coverage(merged, dev, rows, slugs, stats, attrs, old_rows):
    build_gid_index(merged)
    L = []
    A = L.append
    provs = sorted({n["provider"] for n in merged})
    by_p = defaultdict(list)
    for n in merged:
        by_p[n["provider"]].append(n)
    dev_by_p = defaultdict(list)
    for n in dev:
        dev_by_p[n["provider"]].append(n)
    tables_by_p = defaultdict(set)
    for r in rows:
        tables_by_p[r["_provider"]].add(r["_slug"])

    n_ver = sum(1 for n in merged if n["verified"] == "true")
    A("# AIDA registry coverage")
    A("")
    A("Window **2025-01-01T00:00:00Z → 2026-10-28T00:00:00Z (UTC)**. "
      "Built by `build_registry.py` from the %d provider tables under `providers/` "
      "(`<slug>.verified.csv` where available, `<slug>.csv` otherwise)." % len(slugs))
    A("")
    A("| | |")
    A("|---|---|")
    A("| source rows read | %d |" % len(rows))
    A("| source rows dropped, outside window | %d |" %
      sum(s["dropped_window"] for s in stats.values()))
    A("| source rows dropped, unparseable start | %d |" %
      sum(s["dropped_bad_ts"] for s in stats.values()))
    A("| merged incidents | %d |" % len(merged))
    A("| verified incidents | %d (%.1f%%) |" % (n_ver, 100.0 * n_ver / max(1, len(merged))))
    A("| providers | %d |" % len(provs))
    A("| model developers | %d |" % len({n["provider"] for n in merged if n["own_model"] == "true"}))
    A("| multi-source incidents (n_sources > 1) | %d |" %
      sum(1 for n in merged if int(n["n_sources"]) > 1))
    A("| incidents with no end stamp (`status=unresolved_end`) | %d |" %
      sum(1 for n in merged if n["end_utc"] == ""))
    A("| dev-relevant 60 min–24 h incidents, model developers | %d |" % len(dev))
    A("| their total exposure | %.1f h |" % (sum(int(n["duration_min"]) for n in dev) / 60.0))
    A("")
    A("Timestamps are truncated to the minute, so `duration_min` is the whole-minute "
      "difference between the printed stamps. Sources for one incident are collapsed into one "
      "row: `source_url` holds every distinct URL joined by `\" | \"`, `n_sources` counts them, "
      "`dedup_group` is the merge key, and `verified` is the OR over the merged rows. Start, "
      "end and cause are taken from the most authoritative source present "
      "(`official_statuspage` / `official_incidentio` > `official_other` > `wayback` > "
      "`isdown` > `statusgator` > `news` > `community`).")
    A("")

    # ---- per-provider table
    A("## Per-provider coverage")
    A("")
    A("`source grade` is the collection assessment recorded in `providers/<slug>.notes.md`; "
      "`n/a` means no assessment. `cause` is the share of incidents with a vendor-stated root cause.")
    A("")
    A("| provider | group | trains models | source rows | incidents | verified | dev 60m–24h | "
      "exposure (h) | first → last incident | source kinds | tables | source grade | cause |")
    A("|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---:|")
    for p in provs:
        ns = by_p[p]
        v = sum(1 for n in ns if n["verified"] == "true")
        d = dev_by_p[p]
        hours = sum(int(n["duration_min"]) for n in d) / 60.0
        kinds = sorted({k for n in ns for k in [n["source_kind"]] if k},
                       key=lambda k: (rank_of(k), k))
        tabs = sorted(tables_by_p[p])
        prim = PRIMARY_TABLE.get(p, p if p in TABLE_META else (tabs[0] if tabs else ""))
        grade = TABLE_META.get(prim, ("?", ""))[0]
        with_cause = sum(1 for n in ns if n["cause"])
        A("| `%s` | %s | %s | %d | %d | %d (%.0f%%) | %d | %.1f | %s → %s | %s | %s | %s | %.0f%% |" % (
            p, attrs[p]["provider_group"], "yes" if attrs[p]["own_model"] else "no",
            sum(1 for r in rows if r["_provider"] == p),
            len(ns), v, 100.0 * v / len(ns), len(d), hours,
            min(n["start_utc"] for n in ns)[:10], max(n["start_utc"] for n in ns)[:10],
            ", ".join(kinds), ", ".join("`%s`" % t for t in tabs), grade,
            100.0 * with_cause / len(ns)))
    A("")
    collapse = []
    for p in provs:
        src = sum(1 for r in rows if r["_provider"] == p)
        inc = len(by_p[p])
        if src - inc >= 20:
            collapse.append((src - inc, p, src, inc))
    collapse.sort(reverse=True)
    A("Largest reductions after deduplication (source rows → incidents): " +
      ", ".join("`%s` %d→%d" % (p, src, inc) for _d, p, src, inc in collapse[:8]) +
      ". On 2026-07-10, Moonshot filed 35 hourly notices with a common resolution time; "
      "these form one 2,083-minute incident with `n_sources=35`.")
    A("")
    A("### Known gaps, per source table")
    A("")
    for slug in slugs:
        grade, gap = TABLE_META.get(slug, ("?", "No notes entry."))
        A("- **`%s`** (grade %s, `%s`): %s" % (slug, grade, stats[slug]["file"], gap))
    A("")

    # ---- no public history
    A("## Vendors with no public incident history")
    A("")
    A("These providers lack public incident histories. Available records come from news, "
      "social media and crowd reports, or concern model deprecations. Missing reports do "
      "not establish uninterrupted availability.")
    A("")
    A("| vendor | table | source rows | incidents | what exists |")
    A("|---|---|---:|---:|---|")
    for name, slug, pref, why in NO_PUBLIC_HISTORY:
        src = [r for r in rows if r["_slug"] == slug
               and (not pref or r["incident_id"].startswith(pref))]
        groups = {g for g in (gid_of_row(merged, r) for r in src) if g}
        A("| %s | `%s` | %d | %d | %s |" % (name, slug, len(src), len(groups), md_esc(why)))
    A("")
    A("Two providers restrict access to historical records:")
    A("")
    A("- `azure-openai`: Microsoft publishes a public PIR only for \"broad issues\"; the "
      "per-incident record lives in tenant-scoped Azure Service Health, and 6 of 19 incidents "
      "are acknowledged only through a webcast signup page.")
    A("- `amazon-bedrock`: AWS has no public closed-event history endpoint at all, so the "
      "table is reconstructed from Wayback captures of the Health Dashboard.")
    A("")
    A("### Vendors this registry does not cover")
    A("")
    for name, why in NOT_COVERED:
        A("- **%s**: %s" % (name, why))
    A("")

    # ---- old registry comparison
    providers_old, pairs, new_only, old_only, disagree = compare_old(merged, old_rows)
    A("## Comparison with the old registry")
    A("")
    A("`../support/incident_registry.csv` covers two vendors (`anthropic`, `openai`) and "
      "%d rows inside this window. Matching is by provider incident id first, then by "
      "interval overlap or a start within 30 minutes." % len(old_rows))
    A("")
    A("| | anthropic | openai | total |")
    A("|---|---:|---:|---:|")

    def cnt(seq, p, key=lambda x: x["provider"]):
        return sum(1 for x in seq if key(x) == p)
    old_a = cnt(old_rows, "anthropic")
    old_o = cnt(old_rows, "openai")
    new_a = len(by_p.get("anthropic", []))
    new_o = len(by_p.get("openai", []))
    A("| old registry rows in window | %d | %d | %d |" % (old_a, old_o, old_a + old_o))
    A("| new registry incidents | %d | %d | %d |" % (new_a, new_o, new_a + new_o))
    A("| matched | %d | %d | %d |" % (
        sum(1 for o, n, h in pairs if o["provider"] == "anthropic"),
        sum(1 for o, n, h in pairs if o["provider"] == "openai"), len(pairs)))
    A("| matched on provider incident id | %d | %d | %d |" % (
        sum(1 for o, n, h in pairs if h == "id" and o["provider"] == "anthropic"),
        sum(1 for o, n, h in pairs if h == "id" and o["provider"] == "openai"),
        sum(1 for o, n, h in pairs if h == "id")))
    A("| matched only on time | %d | %d | %d |" % (
        sum(1 for o, n, h in pairs if h == "time" and o["provider"] == "anthropic"),
        sum(1 for o, n, h in pairs if h == "time" and o["provider"] == "openai"),
        sum(1 for o, n, h in pairs if h == "time")))
    A("| **in new, missing from old** | %d | %d | %d |" % (
        cnt(new_only, "anthropic"), cnt(new_only, "openai"), len(new_only)))
    A("| **in old, absent from new** | %d | %d | %d |" % (
        cnt(old_only, "anthropic"), cnt(old_only, "openai"), len(old_only)))
    A("| matched pairs disagreeing by > 15 min | %d | %d | %d |" % (
        sum(1 for o, n, h, ds, de in disagree if o["provider"] == "anthropic"),
        sum(1 for o, n, h, ds, de in disagree if o["provider"] == "openai"), len(disagree)))
    A("")
    devnew = [n for n in new_only if n["dev_relevant"] == "true"]
    A("Of the %d incidents the old registry missed, %d are dev-relevant and %d fall in the "
      "60 min–24 h exposure band." % (
          len(new_only), len(devnew),
          sum(1 for n in devnew if n["duration_min"] != ""
              and EXPOSURE_LO <= int(n["duration_min"]) <= EXPOSURE_HI)))
    A("")
    A("### Incidents the old registry missed (earliest 25 of %d)" % len(new_only))
    A("")
    A("| provider | incident_id | start | dur (min) | surface | dev | source | title |")
    A("|---|---|---|---:|---|---|---|---|")
    for n in sorted(new_only, key=lambda n: (n["start_utc"], n["provider"]))[:25]:
        A("| `%s` | `%s` | %s | %s | %s | %s | %s | %s |" % (
            n["provider"], n["incident_id"], n["start_utc"], n["duration_min"] or "-",
            n["surface"], n["dev_relevant"], n["source_kind"], md_esc(n["title"][:70])))
    A("")
    synth = [o for o in old_only if o["code"].startswith("reg")
             or o["title"].startswith("registry window")]
    real = [o for o in old_only if o not in synth]
    A("### Incidents only the old registry has (%d)" % len(old_only))
    A("")
    A("%d are synthetic `reg<epoch>` windows titled \"registry window (major/critical)\", "
      "without provider incident IDs. The remaining %d are vendor filings absent from "
      "the new registry." % (len(synth), len(real)))
    A("")
    if real:
        A("| provider | code | start | end | surface | title |")
        A("|---|---|---|---|---|---|")
        for o in sorted(real, key=lambda o: (o["start"], o["provider"]))[:40]:
            A("| `%s` | `%s` | %s | %s | %s | %s |" % (
                o["provider"], o["code"], fmt_ts(o["start"]), fmt_ts(o["end"]),
                o["surface"], md_esc(o["title"][:70])))
        if len(real) > 40:
            A("")
            A("(%d more omitted.)" % (len(real) - 40))
    else:
        A("Every old-registry row backed by a provider incident id matched an incident here.")
    A("")
    A("### Timestamp disagreements greater than 15 minutes (%d)" % len(disagree))
    A("")
    if disagree:
        starts = sorted(ds for _o, _n, _h, ds, _de in disagree)
        med = starts[len(starts) // 2]
        earlier = sum(1 for x in starts if x < 0)
        later = sum(1 for x in starts if x > 0)
        multi = sum(1 for _o, n, _h, _ds, _de in disagree if len(n["_ids"]) > 1)
        A("Δ denotes new minus old, in minutes. Median Δstart is %.0f min: %d starts are "
          "earlier, %d later, and %d pairs differ only in end time. %d of %d pairs match "
          "a multi-ID incident, where merging expands the time window. The remaining %d "
          "compare single filings. New timestamps use the first investigating/identified "
          "update or the vendor's stated impact start; earlier timestamps use the filing date."
          % (med, earlier, later, len(disagree) - earlier - later,
             multi, len(disagree), len(disagree) - multi))
        A("")
        A("| provider | code / incident_id | matched by | old start | new start | Δstart (min) | "
          "Δend (min) | title |")
        A("|---|---|---|---|---|---:|---:|---|")
        for o, n, how, ds, de in sorted(
                disagree, key=lambda x: (-abs(x[3]), x[0]["provider"], x[0]["code"]))[:40]:
            A("| `%s` | `%s` / `%s` | %s | %s | %s | %+.0f | %s | %s |" % (
                o["provider"], o["code"], n["incident_id"], how, fmt_ts(o["start"]),
                n["start_utc"], ds, ("%+.0f" % de) if de is not None else "-",
                md_esc(o["title"][:50])))
        if len(disagree) > 40:
            A("")
            A("(%d more omitted.)" % (len(disagree) - 40))
    else:
        A("None.")
    A("")
    A("### Two other side tables")
    A("")
    for path, label in ((OTHER_PROVIDERS, "../reference/other_providers_outages.csv"),
                        (SEED_INVENTORY, "../support/seed_in_panel_inventory.csv")):
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            side = list(csv.DictReader(fh))
        if "decision" in (side[0] if side else {}):
            inwin = [r for r in side if (parse_ts(r.get("start")) or WIN_START - timedelta(days=1))
                     >= WIN_START]
            by_reason = Counter(r["reason"] for r in inwin)
            A("- **%s**: %d excluded rows, by reason: %s. %d fall inside the study window. "
              "The new registry retains %d zero-duration incidents but excludes them from "
              "the 60-minute development sample." % (
                  label, len(side),
                  ", ".join("%s=%d" % (k, v) for k, v in sorted(by_reason.items())),
                  len(inwin), sum(1 for n in merged if n["duration_min"] == "0")))
        else:
            pcount = Counter(r["provider"].strip().lower() for r in side
                             if (parse_ts(r.get("start")) or WIN_START - timedelta(days=1)) >= WIN_START)
            miss = sorted(p for p in pcount if p not in provs)
            A("- **%s**: %d rows, %d in the study window, across %s. Providers absent from "
              "the new registry: %s. For shared providers, this table adds pre-2025 history; "
              "the new registry uses vendor feeds for the study window." % (
                  label, len(side), sum(pcount.values()),
                  ", ".join("`%s` (%d)" % (k, v) for k, v in sorted(pcount.items())),
                  ", ".join("`%s`" % p for p in miss) if miss else "none"))
    A("")

    # ---- cause distribution
    A("## cause_category distribution, model developers")
    A("")
    own = [n for n in merged if n["own_model"] == "true"]
    owndev = [n for n in own if n["dev_relevant"] == "true"]
    cats = sorted({n["cause_category"] for n in own} | {n["cause_category"] for n in owndev})
    tot_all, tot_dev = len(own), len(owndev)
    with_text = sum(1 for n in own if n["cause"])
    with_cat = sum(1 for n in own if n["cause_category"] != "unknown")
    A("Model developers account for %d of %d incidents, including %d developer-service "
      "disruptions of any duration. Vendors state a cause for %d incidents (%.0f%%); "
      "%d (%.0f%%) have a known cause category. %d stated causes lack enough detail to "
      "classify, while %d categories derive from status updates without a direct cause statement." % (
          tot_all, len(merged), tot_dev, with_text, 100.0 * with_text / max(1, tot_all),
          with_cat, 100.0 * with_cat / max(1, tot_all),
          sum(1 for n in own if n["cause"] and n["cause_category"] == "unknown"),
          sum(1 for n in own if not n["cause"] and n["cause_category"] != "unknown")))
    A("")
    A("| cause_category | all incidents | share | dev-relevant | share | 60m–24h dev |")
    A("|---|---:|---:|---:|---:|---:|")
    ca, cd = Counter(n["cause_category"] for n in own), Counter(n["cause_category"] for n in owndev)
    cx = Counter(n["cause_category"] for n in dev)
    for c in sorted(cats, key=lambda c: (-ca.get(c, 0), c)):
        A("| `%s` | %d | %.1f%% | %d | %.1f%% | %d |" % (
            c, ca.get(c, 0), 100.0 * ca.get(c, 0) / max(1, tot_all),
            cd.get(c, 0), 100.0 * cd.get(c, 0) / max(1, tot_dev), cx.get(c, 0)))
    A("| **total** | **%d** | | **%d** | | **%d** |" % (tot_all, tot_dev, len(dev)))
    A("")
    A("Stated causes of developer-service disruptions at model developers, excluding `unknown`:")
    A("")
    known = [(c, cd[c]) for c in cd if c != "unknown"]
    known.sort(key=lambda kv: (-kv[1], kv[0]))
    denom = sum(v for _, v in known) or 1
    for i, (c, v) in enumerate(known, 1):
        A("%d. `%s`: %d (%.0f%% of stated causes)" % (i, c, v, 100.0 * v / denom))
    A("")
    A("## Per-provider cause_category, model developers")
    A("")
    ownprovs = sorted({n["provider"] for n in own})
    catorder = [c for c, _ in Counter(n["cause_category"] for n in own).most_common()]
    A("| provider | " + " | ".join("`%s`" % c for c in catorder) + " |")
    A("|---|" + "---:|" * len(catorder))
    for p in ownprovs:
        cc = Counter(n["cause_category"] for n in own if n["provider"] == p)
        A("| `%s` | " % p + " | ".join(str(cc.get(c, 0)) for c in catorder) + " |")
    A("")

    with open(OUT_COVERAGE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
