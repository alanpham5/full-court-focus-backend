                                                  

                                                   
                                                                       
                                                                                   
                                                                   
                                                                                                  
                                                       
                                                                                                       
                                                                         
                                                                                           

                                                                           
                                                                                
                                                                             
                                                                                
                                                                   
                                                                              
                                                                           
                                                                              
        
   
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_APP = Path(__file__).resolve().parents[1]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from env_defaults import load_env_defaults

load_env_defaults()

from pipelines import prospect_tuning as t


def _params_for(name: str | None) -> dict:
    from pipelines.prospects_pipeline import current_tuning
    return t.load_version(name) if name else current_tuning()


def _fmt(params: dict) -> str:
    lines = []
    for k in t.PARAM_KEYS:
        v = params.get(k)
        if k == "weights" and isinstance(v, list):
            v = "[" + ", ".join(f"{x:g}" for x in v) + "]"
        lines.append(f"  {k:22s} {v}")
    return "\n".join(lines)


def cmd_list(_args):
    active = t.active_name()
    versions = t.list_versions()
    if not versions:
        print("(no tuning versions saved)")
        return
    for name in versions:
        mark = " *" if name == active else "  "
        desc = ""
        try:
            import json
            raw = json.loads(t.version_path(name).read_text(encoding="utf-8"))
            desc = raw.get("description", "")
        except Exception:
            pass
        print(f"{mark} {name:28s} {desc}")
    print(f"\n(* = active: {active})")


def cmd_show(args):
    name = args.name or t.active_name()
    if not name:
        sys.exit("no version given and no active version set")
    print(f"=== {name} ===")
    print(_fmt(t.load_version(name)))


def cmd_diff(args):
    a, b = t.load_version(args.a), t.load_version(args.b)
    print(f"=== {args.a}  ->  {args.b} ===")
    any_diff = False
    for k in t.PARAM_KEYS:
        if a.get(k) != b.get(k):
            any_diff = True
            print(f"  {k:22s} {a.get(k)}  ->  {b.get(k)}")
    if not any_diff:
        print("  (identical)")


def cmd_activate(args):
    t.set_active(args.name)
    print(f"active tuning -> {args.name}")


def _coerce(val: str):
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def cmd_snapshot(args):
    base = _params_for(args.from_version)
    for assignment in args.set or []:
        if "=" not in assignment:
            sys.exit(f"--set expects key=value, got '{assignment}'")
        key, raw = assignment.split("=", 1)
        if key not in t.PARAM_KEYS:
            sys.exit(f"unknown param '{key}'. valid: {', '.join(t.PARAM_KEYS)}")
        base[key] = _coerce(raw)
    path = t.save_version(args.name, base, description=args.description,
                          overwrite=args.overwrite)
    print(f"saved {args.name} -> {path}")
    if args.activate:
        t.set_active(args.name)
        print(f"active tuning -> {args.name}")


def cmd_regenerate(args):
    from config import DATA_STATIC_DIR
    from pipelines.prospects_pipeline import (
        ProspectsPipeline, apply_tuning, current_tuning,
    )
    from pipelines.historical_prospects_pipeline import (
        existing_draft_years, process_draft_classes, upload_prospect_outputs_to_gcs,
    )

    do_current = args.scope in ("current", "all")
    do_historical = args.scope in ("historical", "all")
    historical_ok = True

    version = args.version or t.active_name()
    restore = None
    if args.version:
        restore = current_tuning()
        apply_tuning(t.load_version(args.version))
    print(f"regenerating [{args.scope}] with tuning: {version or '(code defaults)'}"
          + ("  (re-scraping)" if args.force_scrape else "  (from cache, no scrape)"))

    try:
        if do_current:
            ProspectsPipeline().recompute_comps_from_cache()

        if do_historical:
            if args.years:
                years = [int(y) for y in args.years.split(",")]
            else:
                years = existing_draft_years()
            if not years:
                print("  no historical draft classes found to recompute")
            else:
                print(f"  historical draft classes: {years}")
                res = process_draft_classes(
                    years, force=args.force_scrape, recompute=not args.force_scrape,
                )
                historical_ok = res["failure"] == 0
                if historical_ok:
                    from analytics.prospect_apfv import normalize_global_prospect_apfv_files
                    stats = normalize_global_prospect_apfv_files(
                        DATA_STATIC_DIR / "prospects.json", DATA_STATIC_DIR / "draft",
                    )
                    print(f"  ✓ historical: {res['success']} classes; APFV normalized {stats}")
                else:
                    print(f"  ⚠ historical: {res['success']} ok, {res['failure']} failed")

                                                                         
                                                                           
                                                                  
        if not args.no_upload:
            n = upload_prospect_outputs_to_gcs(
                include_current=do_current,
                include_historical=do_historical and historical_ok,
            )
            if n:
                print(f"  ✓ uploaded {n} file(s) to GCS")
            elif args.upload:
                print("  (no GCS upload: PLAYER_PROFILES_STORAGE_URI not set to a gs:// URI)")
    finally:
        if restore is not None:
            apply_tuning(restore)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    sp = sub.add_parser("show"); sp.add_argument("name", nargs="?"); sp.set_defaults(func=cmd_show)
    sp = sub.add_parser("diff"); sp.add_argument("a"); sp.add_argument("b"); sp.set_defaults(func=cmd_diff)
    sp = sub.add_parser("activate"); sp.add_argument("name"); sp.set_defaults(func=cmd_activate)

    sp = sub.add_parser("snapshot")
    sp.add_argument("name")
    sp.add_argument("--from", dest="from_version", default=None,
                    help="base version to copy (default: live/active tuning)")
    sp.add_argument("--set", action="append", metavar="key=value",
                    help="override a param, repeatable")
    sp.add_argument("--description", default="")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument("--activate", action="store_true")
    sp.set_defaults(func=cmd_snapshot)

    sp = sub.add_parser("regenerate")
    sp.add_argument("--version", default=None,
                    help="tuning version for this run only (default: active)")
    sp.add_argument("--scope", choices=("current", "historical", "all"), default="all",
                    help="which datasets to recompute (default: all)")
    sp.add_argument("--years", default=None,
                    help="comma-separated historical years (default: all stored classes)")
    sp.add_argument("--force-scrape", action="store_true",
                    help="re-scrape historical raw data instead of recomputing from cache")
    sp.add_argument("--upload", action="store_true",
                    help="explicit GCS upload (also warns if PLAYER_PROFILES_STORAGE_URI is unset)")
    sp.add_argument("--no-upload", action="store_true",
                    help="skip the GCS upload even if PLAYER_PROFILES_STORAGE_URI is set")
    sp.set_defaults(func=cmd_regenerate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
