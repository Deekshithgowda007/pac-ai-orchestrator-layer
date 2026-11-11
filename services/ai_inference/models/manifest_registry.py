"""Model manifests, the registry lock, and the matching rule (SDD-SCANSURE-AI-002 T2, Pillar C).

A manifest describes one model version: what it accepts (``intended_use``, in StudyProfile
vocabulary), what it produces, its tier, licence and validation evidence, and how it runs.
The registry is the set of manifests under ``models/manifests/<model_id>/<version>.json``;
``registry.lock.json`` says which of them this deployment has enabled, in priority order.

    load_registry()                       -> Registry (manifests + lock), validated or raises
    select_models(profile, registry, ...) -> list[Match]   or   NoMatch(reason, detail)

Matching (SDD §3.3): a manifest matches a profile iff every ``intended_use`` dimension that is
not ``"any"`` contains the profile's value, and that value is not unknown — **unknown never
matches a constraint**. In authorized-only mode a match additionally needs ``tier == "T1"``
and a current ``license.authorization``. Ties: tier (T1 > T2 > T3), then lock order.

``NoMatch.reason`` is a closed set an operator can act on:
    no_manifest_for_profile   nothing enabled accepts this kind of study
    profile_incomplete        a manifest would accept it but a needed field is unknown (named)
    not_authorized            authorized-only mode and no T1 manifest qualifies
    weights_unavailable       the matching model(s) need weights that are not present
    model_disabled            manifests match but none is in the lock

Manifests are JSON rather than the SDD's YAML: this image has no YAML library and the format
is not worth a dependency. Validation is a hand-rolled schema (``validate_manifest``) for the
same reason; it refuses unknown top-level keys, so a typo cannot silently relax a constraint.
"""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Union

from models import profile_vocab as vocab

MANIFEST_SCHEMA = "scansurex.model_manifest/1"
LOCK_SCHEMA = "scansurex.registry_lock/1"
TIERS = ("T1", "T2", "T3")
OUTPUT_KINDS = frozenset({"classification", "detection", "segmentation", "measurement", "anatomy", "narrative", "heuristic"})
ANY = "any"

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MANIFEST_DIR = HERE / "manifests"
DEFAULT_LOCK_PATH = HERE / "registry.lock.json"

_TOP_KEYS = {"schema", "model_id", "version", "route_name", "display_name", "tier", "intended_use", "license",
             "outputs", "validation", "runtime", "labels", "legacy"}
_INTENDED_USE_KEYS = {"modalities", "protocol_families", "dimensionality", "age_bands", "body_parts", "contrast"}
_INTENDED_USE_VOCAB = {
    "modalities": vocab.MODALITIES, "protocol_families": vocab.PROTOCOL_FAMILIES,
    "dimensionality": vocab.DIMENSIONALITIES, "age_bands": vocab.AGE_BANDS, "body_parts": vocab.BODY_PARTS,
}
_PROFILE_FIELD_FOR = {"modalities": "modality", "protocol_families": "protocol_family",
                      "dimensionality": "dimensionality", "age_bands": "patient_age_band",
                      "body_parts": "body_part", "contrast": "contrast"}


class ManifestError(ValueError):
    """A manifest or lock file that cannot be trusted. Raised at load, never at match time."""


@dataclass(frozen=True)
class Match:
    manifest: Dict[str, Any]
    weights_present: bool
    weights_path: Optional[str]
    lock_position: int

    @property
    def model_id(self) -> str:
        return self.manifest["model_id"]

    @property
    def version(self) -> str:
        return self.manifest["version"]

    @property
    def route_name(self) -> str:
        return self.manifest["route_name"]

    @property
    def tier(self) -> str:
        return self.manifest["tier"]


@dataclass(frozen=True)
class NoMatch:
    reason: str
    detail: str
    candidates: tuple = field(default_factory=tuple)   # route names that were considered

    REASONS = ("no_manifest_for_profile", "profile_incomplete", "not_authorized", "weights_unavailable", "model_disabled")

    def __post_init__(self) -> None:
        if self.reason not in self.REASONS:
            raise ValueError(f"unknown NoMatch reason {self.reason!r}")


@dataclass
class Registry:
    manifests: List[Dict[str, Any]]            # every manifest found, validated
    enabled: List[Dict[str, Any]]              # lock order, each present in `manifests`
    lock_path: Optional[str] = None
    manifest_dir: Optional[str] = None

    def by_route(self, route_name: str) -> Optional[Dict[str, Any]]:
        for manifest in self.manifests:
            if manifest["route_name"] == route_name:
                return manifest
        return None

    def by_id(self, model_id: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        for manifest in self.manifests:
            if manifest["model_id"] == model_id and (version is None or manifest["version"] == version):
                return manifest
        return None


# --------------------------------------------------------------------------- validation

def _require(condition: bool, where: str, message: str) -> None:
    if not condition:
        raise ManifestError(f"{where}: {message}")


def _check_list_or_any(value: Any, allowed: Optional[frozenset], where: str) -> None:
    if value == ANY:
        return
    _require(isinstance(value, list) and value, where, f"must be a non-empty list or \"{ANY}\"")
    _require(all(isinstance(v, str) for v in value), where, "entries must be strings")
    if allowed is not None:
        unknown = sorted(set(value) - allowed)
        _require(not unknown, where, f"not in the StudyProfile vocabulary: {unknown}")
    _require("unknown" not in value, where, '"unknown" may never be an accepted value')


def validate_manifest(manifest: Dict[str, Any], where: str = "<manifest>") -> Dict[str, Any]:
    _require(isinstance(manifest, dict), where, "must be an object")
    extra = sorted(set(manifest) - _TOP_KEYS)
    _require(not extra, where, f"unknown keys {extra}")
    missing = sorted(_TOP_KEYS - {"legacy", "labels", "validation"} - set(manifest))
    _require(not missing, where, f"missing keys {missing}")
    _require(manifest["schema"] == MANIFEST_SCHEMA, where, f"schema must be {MANIFEST_SCHEMA}")
    for key in ("model_id", "version", "route_name", "display_name"):
        _require(isinstance(manifest[key], str) and manifest[key].strip(), where, f"{key} must be a non-empty string")
    _require(manifest["tier"] in TIERS, where, f"tier must be one of {TIERS}")

    use = manifest["intended_use"]
    _require(isinstance(use, dict), where, "intended_use must be an object")
    extra = sorted(set(use) - _INTENDED_USE_KEYS)
    _require(not extra, where, f"intended_use has unknown keys {extra}")
    missing = sorted(_INTENDED_USE_KEYS - set(use))
    _require(not missing, where, f"intended_use is missing {missing} (write \"{ANY}\" to leave a dimension open)")
    for key, allowed in _INTENDED_USE_VOCAB.items():
        _check_list_or_any(use[key], allowed, f"{where}: intended_use.{key}")
    _require(use["contrast"] in (ANY, True, False), where, f'intended_use.contrast must be true, false or "{ANY}"')

    lic = manifest["license"]
    _require(isinstance(lic, dict) and isinstance(lic.get("code"), str) and lic["code"], where, "license.code required")
    auth = lic.get("authorization")
    if auth is not None:
        _require(isinstance(auth, dict), where, "license.authorization must be an object or null")
        for key in ("jurisdiction", "authority", "reference", "valid_until"):
            _require(isinstance(auth.get(key), str) and auth[key], where, f"license.authorization.{key} required")
        try:
            date.fromisoformat(auth["valid_until"])
        except ValueError as exc:
            raise ManifestError(f"{where}: license.authorization.valid_until must be an ISO date") from exc
    if manifest["tier"] == "T1":
        _require(auth is not None, where, "a T1 manifest must carry license.authorization")

    out = manifest["outputs"]
    _require(isinstance(out, dict) and out.get("kind") in OUTPUT_KINDS, where, f"outputs.kind must be one of {sorted(OUTPUT_KINDS)}")
    _require(isinstance(out.get("fields"), list), where, "outputs.fields must be a list")
    for key in ("produces_seg", "produces_sr"):
        _require(isinstance(out.get(key), bool), where, f"outputs.{key} must be a boolean")

    rt = manifest["runtime"]
    _require(isinstance(rt, dict) and isinstance(rt.get("adapter"), str) and rt["adapter"], where, "runtime.adapter required")
    _require(rt.get("weights_env") is None or isinstance(rt["weights_env"], str), where, "runtime.weights_env must be a string or null")
    _require(isinstance(rt.get("weights_required"), bool), where, "runtime.weights_required must be a boolean")
    _require(rt.get("device") in ("cpu", "gpu"), where, 'runtime.device must be "cpu" or "gpu"')
    _require(isinstance(rt.get("timeout_s"), int) and rt["timeout_s"] > 0, where, "runtime.timeout_s must be a positive integer")

    labels = manifest.get("labels") or {}
    _require(isinstance(labels, dict), where, "labels must be an object")
    _require(isinstance(labels.get("screening_only", True), bool), where, "labels.screening_only must be a boolean")
    if manifest["tier"] != "T1":
        _require(labels.get("screening_only", True) is True, where, "only a T1 manifest may claim more than screening")
    return manifest


def validate_lock(lock: Dict[str, Any], where: str = "<lock>") -> Dict[str, Any]:
    _require(isinstance(lock, dict) and lock.get("schema") == LOCK_SCHEMA, where, f"schema must be {LOCK_SCHEMA}")
    enabled = lock.get("enabled")
    _require(isinstance(enabled, list), where, "enabled must be a list (empty means radiologist-only)")
    for i, entry in enumerate(enabled):
        _require(isinstance(entry, dict) and isinstance(entry.get("model_id"), str) and isinstance(entry.get("version"), str),
                 where, f"enabled[{i}] needs model_id and version")
    return lock


# --------------------------------------------------------------------------- loading

def load_manifests(manifest_dir: Union[str, pathlib.Path] = DEFAULT_MANIFEST_DIR) -> List[Dict[str, Any]]:
    root = pathlib.Path(manifest_dir)
    _require(root.is_dir(), str(root), "manifest directory does not exist")
    manifests: List[Dict[str, Any]] = []
    seen_routes: Dict[str, str] = {}
    seen_ids: set = set()
    for path in sorted(root.glob("*/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path}: not valid JSON ({exc})") from exc
        manifest = validate_manifest(data, str(path))
        _require(path.parent.name == manifest["model_id"], str(path), f"folder must be the model_id {manifest['model_id']!r}")
        _require(path.stem == manifest["version"], str(path), f"file must be named <version>.json ({manifest['version']!r})")
        key = (manifest["model_id"], manifest["version"])
        _require(key not in seen_ids, str(path), "duplicate model_id/version")
        seen_ids.add(key)
        route = manifest["route_name"]
        _require(route not in seen_routes or seen_routes[route] == manifest["model_id"], str(path),
                 f"route_name {route!r} already used by {seen_routes.get(route)!r}")
        seen_routes[route] = manifest["model_id"]
        manifests.append(manifest)
    return manifests


def load_lock(lock_path: Union[str, pathlib.Path, None] = None) -> Dict[str, Any]:
    path = pathlib.Path(lock_path or os.getenv("AI_REGISTRY_LOCK_PATH") or DEFAULT_LOCK_PATH)
    _require(path.is_file(), str(path), "registry lock file does not exist")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: not valid JSON ({exc})") from exc
    return validate_lock(data, str(path))


def load_registry(manifest_dir: Union[str, pathlib.Path] = DEFAULT_MANIFEST_DIR,
                  lock_path: Union[str, pathlib.Path, None] = None) -> Registry:
    manifests = load_manifests(manifest_dir)
    lock = load_lock(lock_path)
    enabled: List[Dict[str, Any]] = []
    for entry in lock["enabled"]:
        manifest = next((m for m in manifests if m["model_id"] == entry["model_id"] and m["version"] == entry["version"]), None)
        _require(manifest is not None, str(lock_path or DEFAULT_LOCK_PATH),
                 f"enabled {entry['model_id']}@{entry['version']} has no manifest")
        _require(manifest not in enabled, str(lock_path or DEFAULT_LOCK_PATH), f"{entry['model_id']}@{entry['version']} enabled twice")
        enabled.append(manifest)
    return Registry(manifests=manifests, enabled=enabled,
                    lock_path=str(lock_path or os.getenv("AI_REGISTRY_LOCK_PATH") or DEFAULT_LOCK_PATH),
                    manifest_dir=str(manifest_dir))


# --------------------------------------------------------------------------- weights

def _split_csv_env(name: str) -> List[str]:
    raw = os.getenv(name, "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()] if raw else []


def resolve_weights(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """{weights_path, weights_present, diagnostic_available} exactly as the legacy registry did."""
    rt = manifest["runtime"]
    weights_path: Optional[str] = None
    present = False
    if rt.get("weights_env"):
        candidates = _split_csv_env(rt["weights_env"])
        for candidate in candidates:
            if os.path.exists(candidate):
                weights_path, present = candidate, True
                break
        if not weights_path and candidates:
            weights_path = candidates[0]
    return {
        "weights_path": weights_path,
        "weights_present": present,
        "diagnostic_available": not (rt.get("weights_required") and not present),
    }


# --------------------------------------------------------------------------- matching

def authorized_only_enabled() -> bool:
    return os.getenv("AI_AUTHORIZED_ONLY", "false").strip().lower() in {"1", "true", "yes", "on"}


def _authorization_current(manifest: Dict[str, Any], today: Optional[date] = None) -> bool:
    auth = (manifest.get("license") or {}).get("authorization")
    if not auth:
        return False
    try:
        return date.fromisoformat(auth["valid_until"]) >= (today or date.today())
    except (KeyError, ValueError):
        return False


def _profile_value(profile: Dict[str, Any], dimension: str) -> Any:
    field_name = _PROFILE_FIELD_FOR[dimension]
    value = profile.get(field_name)
    if dimension == "body_parts" and isinstance(value, dict):
        value = value.get("normalized")
    return value


def constraint_outcome(manifest: Dict[str, Any], profile: Dict[str, Any]) -> tuple:
    """("match", None) | ("incomplete", <profile field>) | ("mismatch", <dimension>)."""
    use = manifest["intended_use"]
    for dimension in ("modalities", "protocol_families", "dimensionality", "age_bands", "body_parts"):
        allowed = use[dimension]
        if allowed == ANY:
            continue
        value = _profile_value(profile, dimension)
        if value in vocab.UNKNOWN or value == "MIXED" and dimension == "modalities":
            return "incomplete", _PROFILE_FIELD_FOR[dimension]
        if value not in allowed:
            return "mismatch", dimension
    if use["contrast"] != ANY:
        value = profile.get("contrast")
        if value is None:
            return "incomplete", "contrast"
        if bool(value) != use["contrast"]:
            return "mismatch", "contrast"
    return "match", None


def select_models(profile: Dict[str, Any], registry: Registry, *, authorized_only: Optional[bool] = None,
                  today: Optional[date] = None) -> Union[List[Match], NoMatch]:
    """The models this profile may run on, best first; or why none may."""
    if authorized_only is None:
        authorized_only = authorized_only_enabled()
    if not isinstance(profile, dict) or profile.get("schema") != vocab.PROFILE_SCHEMA:
        return NoMatch("profile_incomplete", "no StudyProfile (schema scansurex.study_profile/1) was supplied")

    # Reasons are judged over the ENABLED manifests first: a disabled catch-all (the heuristic
    # fallback accepts anything) must not turn "the profile is incomplete" into "model disabled".
    enabled: List[tuple] = []
    incomplete_fields: List[str] = []
    for position, manifest in enumerate(registry.enabled):
        outcome, detail = constraint_outcome(manifest, profile)
        if outcome == "match":
            enabled.append((position, manifest))
        elif outcome == "incomplete":
            incomplete_fields.append(f"{manifest['route_name']}:{detail}")
    if not enabled:
        if incomplete_fields:
            return NoMatch("profile_incomplete",
                           "a manifest would accept this study but the profile does not say: " + ", ".join(incomplete_fields),
                           tuple(f.split(":")[0] for f in incomplete_fields))
        disabled_matches = [m["route_name"] for m in registry.manifests
                            if m not in registry.enabled and constraint_outcome(m, profile)[0] == "match"]
        if disabled_matches:
            return NoMatch("model_disabled",
                           "matching manifest(s) are not enabled in the registry lock: " + ", ".join(disabled_matches),
                           tuple(disabled_matches))
        return NoMatch("no_manifest_for_profile",
                       f"no manifest accepts modality={profile.get('modality')} protocol_family={profile.get('protocol_family')} "
                       f"dimensionality={profile.get('dimensionality')}")

    if authorized_only:
        enabled_before = list(enabled)
        enabled = [(i, m) for i, m in enabled if m["tier"] == "T1" and _authorization_current(m, today)]
        if not enabled:
            return NoMatch("not_authorized", "authorized-only mode: no enabled T1 manifest with a current authorization accepts this study",
                           tuple(m["route_name"] for _, m in enabled_before))

    matches: List[Match] = []
    missing_weights: List[str] = []
    for position, manifest in enabled:
        weights = resolve_weights(manifest)
        if manifest["runtime"].get("weights_required") and not weights["weights_present"]:
            missing_weights.append(manifest["route_name"])
            continue
        matches.append(Match(manifest, weights["weights_present"], weights["weights_path"], position))
    if not matches:
        return NoMatch("weights_unavailable", "the matching model(s) need weights that are not present: " + ", ".join(missing_weights),
                       tuple(missing_weights))
    matches.sort(key=lambda m: (TIERS.index(m.tier), m.lock_position))
    return matches


def legacy_entry(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The dict shape ``model_registry.get_model_entry`` has always returned, from a manifest."""
    legacy = manifest.get("legacy") or {}
    rt = manifest["runtime"]
    use = manifest["intended_use"]
    entry = {
        "route_name": manifest["route_name"],
        "modalities": list(use["modalities"]) if use["modalities"] != ANY else [],
        "body_parts": list(legacy.get("body_parts") or []),
        "model_name": manifest["model_id"],
        "model_family": legacy.get("model_family", "unknown"),
        "diagnostic_support": legacy.get("diagnostic_support", "screening-only"),
        "report_type": legacy.get("report_type", "unknown"),
        "diagnostic_available": bool(legacy.get("diagnostic_available", True)),
        "weights_required": bool(rt.get("weights_required")),
        "weights_env": rt.get("weights_env"),
        "weights_path": None,
        "runtime": {"device": rt["device"], "gpu_recommended": bool(legacy.get("gpu_recommended", rt["device"] == "gpu"))},
        "notes": legacy.get("notes", manifest["display_name"]),
        # new, carried alongside so consumers can start reading them
        "model_id": manifest["model_id"],
        "model_version": manifest["version"],
        "tier": manifest["tier"],
        "weights_sha256": rt.get("weights_sha256"),
    }
    weights = resolve_weights(manifest)
    entry["weights_path"], entry["weights_present"] = weights["weights_path"], weights["weights_present"]
    if not weights["diagnostic_available"]:
        entry["diagnostic_available"] = False
    return entry


def explain_candidates(profile: Dict[str, Any], registry: Registry, *, authorized_only: Optional[bool] = None,
                       today: Optional[date] = None) -> List[Dict[str, Any]]:
    """One line per manifest: why it did or did not qualify for this profile. The routing ladder
    records this on every result so a reader sees which models were considered, not just which ran.
    Outcomes: matched / mismatch:<dimension> / incomplete:<field> / model_disabled / not_authorized /
    weights_unavailable."""
    if authorized_only is None:
        authorized_only = authorized_only_enabled()
    rows: List[Dict[str, Any]] = []
    valid_profile = isinstance(profile, dict) and profile.get("schema") == vocab.PROFILE_SCHEMA
    for manifest in registry.manifests:
        row = {"route_name": manifest["route_name"], "model_id": manifest["model_id"], "version": manifest["version"],
               "tier": manifest["tier"], "kind": manifest["outputs"]["kind"]}
        if not valid_profile:
            row.update(outcome="incomplete", detail="no StudyProfile")
        else:
            outcome, detail = constraint_outcome(manifest, profile)
            if outcome == "mismatch":
                row.update(outcome="mismatch", detail=detail)
            elif outcome == "incomplete":
                row.update(outcome="incomplete", detail=detail)
            elif manifest not in registry.enabled:
                row.update(outcome="model_disabled", detail=None)
            elif authorized_only and not (manifest["tier"] == "T1" and _authorization_current(manifest, today)):
                row.update(outcome="not_authorized", detail=None)
            else:
                weights = resolve_weights(manifest)
                if manifest["runtime"].get("weights_required") and not weights["weights_present"]:
                    row.update(outcome="weights_unavailable", detail=manifest["runtime"].get("weights_env"))
                else:
                    row.update(outcome="matched", detail=None)
        rows.append(row)
    return rows
