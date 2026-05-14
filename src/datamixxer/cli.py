from __future__ import annotations

import argparse
from typing import Any

from datamixxer.mix import (
    MixPlan,
    build_mix,
    check_source_access,
    explain_hash,
    hub_check_for_config,
    list_mix_artifacts,
    plan_mix,
    push_mix,
    resolve_mix_artifact,
    validate_config_file,
    write_config_template,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="datamixxer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build deterministic, balanced mixes from Hugging Face dataset splits.",
        epilog="""Typical workflow:
  datamixxer init mix.yaml
  datamixxer validate mix.yaml
  datamixxer plan mix.yaml --explain-hash
  datamixxer build mix.yaml --no-push-to-hub
  datamixxer push <mix-hash> --repo-id owner/dataset-name
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser(
        "init",
        help="Write a starter YAML config",
        description="Write a starter YAML config that uses the recommended `sources` shape.",
    )
    init.add_argument("config", help="Path for the new config file")
    init.add_argument("--force", action="store_true", help="overwrite an existing config file")

    validate = subparsers.add_parser(
        "validate",
        help="Validate a mix config without streaming rows",
        description="Validate required fields, source counts, split settings, output settings, and Hub config.",
    )
    validate.add_argument("config", help="YAML mix config")
    validate.add_argument(
        "--push-to-hub",
        action="store_true",
        help="also require Hub publishing settings to be present",
    )
    validate.add_argument(
        "--check-sources",
        action="store_true",
        help="also verify Hugging Face dataset/config/split access without streaming rows",
    )

    build = subparsers.add_parser(
        "build",
        help="Build a balanced Hugging Face dataset mix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Build a mix from a YAML config.

The build streams configured Hugging Face splits, shuffles deterministically,
deduplicates rows by the configured key, and writes JSONL files plus metadata.
If the same sampling inputs already exist locally, the artifact is reused.
Use --force to rebuild anyway.""",
    )
    build.add_argument("config", help="YAML mix config")
    build.add_argument("--push-to-hub", action="store_true", default=None, help="upload after local build")
    build.add_argument(
        "--no-push-to-hub",
        action="store_false",
        dest="push_to_hub",
        help="build locally even if output.push_to_hub is true",
    )
    build.add_argument("--force", action="store_true", help="rebuild even if the mix hash exists")

    plan = subparsers.add_parser("plan", help="Preview a mix config without streaming rows")
    plan.add_argument("config", help="YAML mix config")
    plan.add_argument(
        "--explain-hash",
        action="store_true",
        help="print the normalized sampling inputs that determine the mix hash",
    )

    list_parser = subparsers.add_parser("list", help="List local dataset mixes")
    list_parser.add_argument("--store-dir", default=None, help="mix store directory")

    show = subparsers.add_parser("show", help="Show local dataset mix details")
    show.add_argument("mix", help="mix hash prefix, artifact id, artifact path, or manifest path")
    show.add_argument("--store-dir", default=None, help="mix store directory")

    push = subparsers.add_parser("push", help="Push a local dataset mix to Hugging Face Hub")
    push.add_argument("mix", help="mix hash prefix, artifact id, artifact path, or manifest path")
    push.add_argument("--store-dir", default=None, help="mix store directory")
    push.add_argument("--repo-id", default=None, help="target Hub dataset repo, for example owner/name")
    push.add_argument("--private", action="store_true", default=None, help="create or update a private repo")
    push.add_argument("--commit-message", default=None, help="Hub commit message")

    hub_check = subparsers.add_parser(
        "hub-check",
        help="Check Hugging Face auth and target repo access without creating repos",
    )
    hub_check.add_argument("config", help="YAML mix config")
    hub_check.add_argument("--repo-id", default=None, help="override output.hub repo")
    hub_check.add_argument("--private", action="store_true", help="check private repo access")

    args = parser.parse_args()
    try:
        if args.command == "init":
            path = write_config_template(args.config, overwrite=args.force)
            print(f"Wrote starter config to {path}")
        elif args.command == "validate":
            config = validate_config_file(args.config, require_hub=args.push_to_hub)
            if args.check_sources:
                source_errors = check_source_access(config)
                if source_errors:
                    raise ValueError("source access check failed: " + "; ".join(source_errors))
                print("Source access OK.")
            print(f"Config is valid: {args.config}")
        elif args.command == "build":
            build_mix(args.config, push=args.push_to_hub, force=args.force)
        elif args.command == "plan":
            mix_plan = plan_mix(args.config)
            print_mix_plan(mix_plan)
            if args.explain_hash:
                print("\nhash inputs:")
                print(explain_hash(mix_plan.config))
        elif args.command == "list":
            print_mix_list(list_mix_artifacts(args.store_dir))
        elif args.command == "show":
            print_mix_details(resolve_mix_artifact(args.mix, args.store_dir).manifest)
        elif args.command == "push":
            artifact = push_mix(
                args.mix,
                store=args.store_dir,
                repo_id=args.repo_id,
                private=args.private,
                commit_message=args.commit_message,
            )
            print(f"Uploaded dataset to {artifact.manifest['hub']['url']}")
        elif args.command == "hub-check":
            check = hub_check_for_config(
                args.config,
                repo_id=args.repo_id,
                private=args.private,
            )
            visibility = "private" if check.private else "public"
            print(f"Authenticated as {check.user}")
            print(f"Repo access OK: {check.repo_type} {check.repo_id} ({visibility})")
            print(f"URL: {check.url}")
        else:
            parser.error(f"unknown command: {args.command}")
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")


def print_mix_list(artifacts: list[Any]) -> None:
    if not artifacts:
        print("No local mixes found.")
        print("Next: run `datamixxer build <config.yaml>` or pass `--store-dir` if mixes live elsewhere.")
        return
    headers = ["hash", "id", "version", "rows", "pushed", "repo", "path"]
    rows = []
    for artifact in artifacts:
        manifest = artifact.manifest
        hub = manifest.get("hub") or {}
        rows.append(
            [
                str(manifest.get("mix_hash", ""))[:12],
                str(manifest.get("artifact_id", "")),
                str(manifest.get("artifact_version", "")),
                str(manifest.get("total_rows", "")),
                "yes" if hub.get("pushed") else "no",
                str(hub.get("repo_id") or manifest.get("hub_repo_id") or ""),
                str(artifact.path),
            ]
        )
    print_table(headers, rows)


def print_mix_plan(plan: MixPlan) -> None:
    config = plan.config
    artifact = plan.artifact
    hub = (artifact.manifest.get("hub") if artifact else {}) or {}
    train_rows = sum(source.train_count for source in plan.sources)
    test_rows = sum(source.test_count for source in plan.sources)
    print(f"hash: {plan.hash_value}")
    print(f"short_hash: {plan.hash_value[:12]}")
    print(f"id: {config.get('id')}")
    print(f"version: {config.get('version') or config.get('artifact_version') or 'v1'}")
    print(f"name: {config.get('name') or ''}")
    print(f"path: {plan.path}")
    print(f"status: {'exists' if artifact else 'new'}")
    print(f"pushed: {'yes' if hub.get('pushed') else 'no'}")
    if hub.get("repo_id") or (config.get("output") or {}).get("hub", {}).get("repo_id"):
        print(f"repo: {hub.get('repo_id') or (config.get('output') or {}).get('hub', {}).get('repo_id')}")
    print(f"\ntrain: {train_rows} rows")
    if test_rows:
        print(f"test: {test_rows} rows")
    print("\nsources:")
    rows = [
        [
            source.name,
            source.dataset_id,
            source.config or "",
            source.split,
            str(source.count),
            str(source.train_count),
            str(source.test_count),
        ]
        for source in plan.sources
    ]
    print_table(["bucket", "dataset", "config", "split", "count", "train", "test"], rows)
    print("\nnext:")
    print("  datamixxer build <config.yaml> --no-push-to-hub")
    print(f"  datamixxer show {plan.hash_value[:12]}")


def print_mix_details(manifest: dict[str, Any]) -> None:
    hub = manifest.get("hub") or {}
    print(f"hash: {manifest.get('mix_hash')}")
    print(f"id: {manifest.get('artifact_id')}")
    print(f"version: {manifest.get('artifact_version')}")
    print(f"name: {manifest.get('name') or ''}")
    print(f"path: {manifest.get('artifact_dir')}")
    print(f"rows: {manifest.get('total_rows')}")
    print(f"pushed: {'yes' if hub.get('pushed') else 'no'}")
    if hub.get("repo_id") or manifest.get("hub_repo_id"):
        print(f"repo: {hub.get('repo_id') or manifest.get('hub_repo_id')}")
    if hub.get("url"):
        print(f"url: {hub['url']}")
    if hub.get("pushed_at"):
        print(f"pushed_at: {hub['pushed_at']}")
    print("\nsplits:")
    for split, info in (manifest.get("splits") or {}).items():
        print(f"  {split}: {info.get('rows')} rows ({info.get('file')})")
    print("\nsources:")
    for source in manifest.get("sources") or []:
        print(
            "  "
            f"{source.get('name')}: {source.get('dataset_id')} "
            f"{source.get('config') or ''} {source.get('split')} "
            f"total={source.get('count')} train={source.get('train_count')} "
            f"test={source.get('test_count')}"
        )


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


if __name__ == "__main__":
    main()
