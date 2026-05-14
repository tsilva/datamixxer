from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datamixxer.mix import (
    MixPlan,
    add_source_to_config,
    build_mix,
    check_source_access,
    explain_hash,
    hub_check_for_config,
    inspect_dataset,
    list_mix_artifacts,
    plan_mix,
    push_mix,
    push_mix_artifact,
    resolve_mix_artifact,
    validate_config_file,
    validate_sample_rows,
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
  datamixxer check mix.yaml
  datamixxer plan mix.yaml
  datamixxer build mix.yaml
  datamixxer publish mix.yaml --repo-id owner/dataset-name
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
    validate.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="stream this many rows per source to validate row-dependent settings such as dedupe fields",
    )

    check = subparsers.add_parser(
        "check",
        help="Validate config and verify Hugging Face source access",
        description="Validate config shape, verify dataset/config/split access, and optionally sample rows.",
    )
    check.add_argument("config", help="YAML mix config")
    check.add_argument(
        "--push-to-hub",
        action="store_true",
        help="also require Hub publishing settings to be present",
    )
    check.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="stream this many rows per source to validate row-dependent settings such as dedupe fields",
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
    show.add_argument("mix", help="mix hash prefix, artifact id, artifact path, manifest path, or config YAML")
    show.add_argument("--store-dir", default=None, help="mix store directory")

    push = subparsers.add_parser("push", help="Push a local dataset mix to Hugging Face Hub")
    add_publish_arguments(push)

    publish = subparsers.add_parser("publish", help="Publish a built dataset mix to Hugging Face Hub")
    add_publish_arguments(publish)

    hub_check = subparsers.add_parser(
        "hub-check",
        help="Check Hugging Face auth and target repo access without creating repos",
    )
    hub_check.add_argument("config", help="YAML mix config")
    hub_check.add_argument("--repo-id", default=None, help="override output.hub repo")
    hub_check.add_argument("--private", action="store_true", help="check private repo access")

    inspect = subparsers.add_parser("inspect", help="Show available configs and splits for a Hugging Face dataset")
    inspect.add_argument("dataset_id", help="Hugging Face dataset id")
    inspect.add_argument("--config", default=None, help="dataset config/subset to inspect")

    add_source = subparsers.add_parser("add-source", help="Append a source bucket to a mix config")
    add_source.add_argument("config", help="YAML mix config to update")
    add_source.add_argument("--name", required=True, help="bucket name")
    add_source.add_argument("--dataset", required=True, help="Hugging Face dataset id")
    add_source.add_argument("--dataset-config", default=None, help="dataset config/subset")
    add_source.add_argument("--split", required=True, help="source split")
    add_source.add_argument("--count", required=True, type=int, help="rows to sample from this source")
    add_source.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="metadata field copied into output rows; repeat for multiple fields",
    )

    args = parser.parse_args()
    try:
        if args.command == "init":
            path = write_config_template(args.config, overwrite=args.force)
            print(f"Wrote starter config to {path}")
            print("Next:")
            print(f"  datamixxer validate {path}")
            print(f"  datamixxer check {path}")
            print(f"  datamixxer plan {path}")
        elif args.command == "validate":
            config = validate_config_file(args.config, require_hub=args.push_to_hub)
            if args.check_sources:
                raise_if_issues("Source access check failed", check_source_access(config))
                print("Source access OK.")
            if args.sample_rows:
                raise_if_issues("Sample row validation failed", validate_sample_rows(config, args.sample_rows))
                print(f"Sample row validation OK ({args.sample_rows} rows per source).")
            print(f"Config is valid: {args.config}")
        elif args.command == "check":
            config = validate_config_file(args.config, require_hub=args.push_to_hub)
            raise_if_issues("Source access check failed", check_source_access(config))
            print("Source access OK.")
            if args.sample_rows:
                raise_if_issues("Sample row validation failed", validate_sample_rows(config, args.sample_rows))
                print(f"Sample row validation OK ({args.sample_rows} rows per source).")
            print(f"Config is valid: {args.config}")
        elif args.command == "build":
            build_mix(args.config, push=args.push_to_hub, force=args.force)
        elif args.command == "plan":
            mix_plan = plan_mix(args.config)
            print_mix_plan(mix_plan, config_path=args.config, show_hash=args.explain_hash)
            if args.explain_hash:
                print("\nhash inputs:")
                print(explain_hash(mix_plan.config))
        elif args.command == "list":
            print_mix_list(list_mix_artifacts(args.store_dir))
        elif args.command == "show":
            print_show(args.mix, args.store_dir)
        elif args.command in ("push", "publish"):
            if args.check:
                check = hub_check_for_config(
                    args.mix,
                    repo_id=args.repo_id,
                    private=bool(args.private),
                )
                print_hub_check(check)
                return
            artifact = publish_mix_reference(args)
            print(f"Uploaded dataset to {artifact.manifest['hub']['url']}")
        elif args.command == "hub-check":
            check = hub_check_for_config(
                args.config,
                repo_id=args.repo_id,
                private=args.private,
            )
            print_hub_check(check)
        elif args.command == "inspect":
            print_dataset_inspection(inspect_dataset(args.dataset_id, args.config))
        elif args.command == "add-source":
            source = add_source_to_config(
                args.config,
                name=args.name,
                dataset_id=args.dataset,
                config=args.dataset_config,
                split=args.split,
                count=args.count,
                metadata=parse_metadata(args.metadata),
            )
            print(f"Added source {source['name']} to {args.config}")
            print(f"Next: datamixxer check {args.config}")
        else:
            parser.error(f"unknown command: {args.command}")
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")


def add_publish_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("mix", help="mix hash prefix, artifact id, artifact path, manifest path, or config YAML")
    parser.add_argument("--store-dir", default=None, help="mix store directory")
    parser.add_argument("--repo-id", default=None, help="target Hub dataset repo, for example owner/name")
    parser.add_argument("--private", action="store_true", default=None, help="create or update a private repo")
    parser.add_argument("--commit-message", default=None, help="Hub commit message")
    parser.add_argument("--check", action="store_true", help="check Hub auth/repo access without uploading")


def raise_if_issues(title: str, issues: list[str]) -> None:
    if issues:
        raise ValueError(title + ":\n" + "\n".join(f"- {issue}" for issue in issues))


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


def print_mix_plan(
    plan: MixPlan,
    *,
    config_path: str | None = None,
    show_hash: bool = False,
) -> None:
    config = plan.config
    artifact = plan.artifact
    hub = (artifact.manifest.get("hub") if artifact else {}) or {}
    train_rows = sum(source.train_count for source in plan.sources)
    test_rows = sum(source.test_count for source in plan.sources)
    version = config.get("version") or config.get("artifact_version") or "v1"
    name = config.get("name") or config.get("id")
    print(f"Mix: {name} {version}")
    print(f"Output: {train_rows} train rows" + (f", {test_rows} test rows" if test_rows else ""))
    print(f"Sources: {len(plan.sources)} buckets")
    print(f"Status: {'built' if artifact else 'ready to build'}")
    print(f"Artifact: {plan.path}")
    print(f"Short hash: {plan.hash_value[:12]}")
    if show_hash:
        print(f"Full hash: {plan.hash_value}")
    print(f"pushed: {'yes' if hub.get('pushed') else 'no'}")
    if hub.get("repo_id") or (config.get("output") or {}).get("hub", {}).get("repo_id"):
        print(f"repo: {hub.get('repo_id') or (config.get('output') or {}).get('hub', {}).get('repo_id')}")
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
    config_label = config_path or "<config.yaml>"
    if artifact:
        print(f"  datamixxer show {plan.hash_value[:12]}")
        print(f"  datamixxer publish {config_label} --repo-id owner/dataset-name")
    else:
        print(f"  datamixxer build {config_label}")


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


def print_show(reference: str, store_dir: str | None) -> None:
    if _looks_like_yaml(reference):
        plan = plan_mix(reference)
        if plan.artifact:
            print_mix_details(plan.artifact.manifest)
        else:
            print_mix_plan(plan, config_path=reference)
        return
    print_mix_details(resolve_mix_artifact(reference, store_dir).manifest)


def publish_mix_reference(args: argparse.Namespace) -> Any:
    if _looks_like_yaml(args.mix):
        plan = plan_mix(args.mix)
        if not plan.artifact:
            raise FileNotFoundError(
                f"No built mix exists for {args.mix}. Run `datamixxer build {args.mix}` first."
            )
        return push_mix_artifact(
            plan.artifact,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message,
        )
    return push_mix(
        args.mix,
        store=args.store_dir,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
    )


def print_hub_check(check: Any) -> None:
    visibility = "private" if check.private else "public"
    print(f"Authenticated as {check.user}")
    print(f"Repo access OK: {check.repo_type} {check.repo_id} ({visibility})")
    print(f"URL: {check.url}")


def print_dataset_inspection(inspection: Any) -> None:
    print(f"Dataset: {inspection.dataset_id}")
    if inspection.configs:
        print("Configs:")
        for config in inspection.configs:
            print(f"- {config}")
    for config, splits in inspection.splits_by_config.items():
        label = config if config is not None else "default"
        print(f"\nSplits for {label}:")
        for split in splits:
            print(f"- {split}")


def parse_metadata(values: list[str]) -> dict[str, str] | None:
    metadata: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metadata must use KEY=VALUE format: {value!r}")
        key, item = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"metadata key cannot be empty: {value!r}")
        metadata[key] = item
    return metadata or None


def _looks_like_yaml(path: str) -> bool:
    return Path(path).suffix.lower() in {".yaml", ".yml"}


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
