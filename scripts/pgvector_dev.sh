#!/usr/bin/env bash
# pgvector_dev.sh - Helper script for pgvector patch development
#
# PURPOSE
#   This script manages a local clone of the pgvector repository kept under
#   <repo-root>/.pgvector_dev_<patch-stem>/.  It is designed to support an
#   iterative patch-development workflow: clone a specific release tag, apply
#   an experimental patch on top, edit the source freely, then write the
#   accumulated diff back out as a new .patch file.
#
# REPOSITORY LAYOUT (after setup)
#   <repo-root>/
#     .pgvector_dev_<patch-stem>/   <- isolated git clone of pgvector
#     scripts/pgvector_dev.sh       <- this script
#     *.patch                       <- patch files to apply / write back to
#
# DIRECTORY NAMING CONVENTION
#   The working directory name encodes which patch is applied, making it
#   possible to maintain multiple patch variants side-by-side.
#
#   Naming rule (applied when --dir is not given to setup):
#     1. Take the patch filename, e.g. pgvector_v0.8.0_hnsw_candidate_pruning_pq.patch
#     2. Strip the .patch extension             -> pgvector_v0.8.0_hnsw_candidate_pruning_pq
#     3. Strip the leading "pgvector_" prefix   -> v0.8.0_hnsw_candidate_pruning_pq
#     4. Prepend ".pgvector_dev_"               -> .pgvector_dev_v0.8.0_hnsw_candidate_pruning_pq
#
#   Example working directories:
#     .pgvector_dev_v0.8.0_hnsw_candidate_pruning_pq/
#     .pgvector_dev_v0.8.0_hnsw_candidate_pruning_simhash/
#     .pgvector_dev_v0.8.0_hnsw_candidate_pruning_turboquant/
#
# TYPICAL WORKFLOW FOR AN AGENT
#   Step 1 - Set up a working directory with the target patch already applied:
#     ./scripts/pgvector_dev.sh setup \
#         --patch pgvector_v0.8.0_hnsw_candidate_pruning_pq.patch
#
#   Step 2 - Inspect or modify source files inside the working directory:
#     ./scripts/pgvector_dev.sh status
#     # edit .pgvector_dev_v0.8.0_hnsw_candidate_pruning_pq/src/*.c as needed
#
#   Step 3 - Write the updated diff back as a patch file:
#     ./scripts/pgvector_dev.sh export \
#         --dir .pgvector_dev_v0.8.0_hnsw_candidate_pruning_pq \
#         pgvector_v0.8.0_hnsw_candidate_pruning_pq.patch
#
#   Step 4 - Reset the working directory to the clean pgvector state:
#     ./scripts/pgvector_dev.sh reset \
#         --dir .pgvector_dev_v0.8.0_hnsw_candidate_pruning_pq
#
# SUBCOMMANDS
#
#   setup --patch <patch-file> [--tag <tag>] [--dir <dir>]
#       REQUIRED: --patch <patch-file>
#           Path to the .patch file to apply.  Relative paths are resolved
#           from the current working directory.
#       OPTIONAL: --tag <tag>   (default: v0.8.0)
#           Git tag of the pgvector release to clone.
#       OPTIONAL: --dir <dir>
#           Override the auto-derived working directory path.
#       BEHAVIOUR:
#           - If the target directory does not yet exist, clones pgvector at
#             the given tag with --depth 1 (shallow clone).
#           - If the directory already exists, the clone step is skipped and
#             the patch is (re-)applied on top of the current state.
#           - Runs `git apply --check` before applying; exits with an error
#             if the patch does not apply cleanly (e.g. already applied,
#             conflicting edits).
#       EXIT CODES: 0 on success, non-zero on any error.
#
#   export [--dir <dir>] [--gz-threshold <bytes>] <output-patch-file>
#       REQUIRED positional: <output-patch-file>
#           Destination path for the generated patch.  Relative paths are
#           resolved from the current working directory.
#       OPTIONAL: --dir <dir>  (default: .pgvector_dev)
#           The working directory to diff.
#       OPTIONAL: --gz-threshold <bytes>  (default: 10485760 = 10 MiB)
#           If the generated patch exceeds this size in bytes, compress it
#           with gzip.  The output file will have ".gz" appended (e.g.
#           foo.patch -> foo.patch.gz).  Set to 0 to always compress, or
#           to a very large value to effectively disable compression.
#       BEHAVIOUR:
#           - Captures ALL uncommitted changes: staged files, unstaged
#             modifications, and untracked new files.
#           - Untracked/unstaged files are temporarily staged with
#             `git add -A` solely to include them in the diff output, then
#             immediately unstaged with `git reset HEAD` - the working tree
#             is never altered.
#           - Output format: unified diff with --patch --stat (compatible
#             with `git apply` / `gunzip | git apply`).
#           - If there are no changes, writes an empty file and exits 0.
#           - If the patch file exceeds --gz-threshold, it is gzip-compressed
#             in place and the final filename is reported.
#       EXIT CODES: 0 on success, non-zero on any error.
#
#   reset [--dir <dir>]
#       OPTIONAL: --dir <dir>  (default: .pgvector_dev)
#       BEHAVIOUR:
#           Runs `git reset --hard HEAD` followed by `git clean -fd` inside
#           the working directory, discarding ALL uncommitted changes and
#           untracked files.  This is destructive and cannot be undone.
#           Use before re-applying a different version of a patch.
#       EXIT CODES: 0 on success, non-zero on any error.
#
#   status [--dir <dir>]
#       OPTIONAL: --dir <dir>  (default: .pgvector_dev)
#       BEHAVIOUR:
#           Prints `git status` of the working directory.  Safe, read-only.
#       EXIT CODES: 0 always.
#
# ERROR HANDLING
#   The script runs with `set -euo pipefail`: any unexpected command failure
#   causes an immediate exit with a non-zero status.  Error messages are
#   written to stderr prefixed with "ERROR:".  Informational messages go to
#   stdout prefixed with "==>".

set -euo pipefail

REPO_URL="https://github.com/pgvector/pgvector.git"
DEFAULT_TAG="v0.8.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------

usage() {
  echo "Usage:"
  echo "  $0 setup  --patch <patch-file> [--tag <tag>] [--dir <dir>]"
  echo "                                Clone pgvector and apply a patch."
  echo "  $0 export [--dir <dir>] [--gz-threshold <bytes>] <output-patch-file>"
  echo "                                Export changes as a patch (auto-gzip if large)."
  echo "  $0 reset  [--dir <dir>]       Restore the working directory to its initial state."
  echo "  $0 status [--dir <dir>]       Show git status of the working directory."
  exit 1
}

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

# Derive a working-directory path from a patch filename.
# Strips the leading "pgvector_" prefix and ".patch" suffix from the
# basename, then prepends ".pgvector_dev_" under ROOT_DIR.
#   Input : pgvector_v0.8.0_hnsw_candidate_pruning_pq.patch
#   Output: <ROOT_DIR>/.pgvector_dev_v0.8.0_hnsw_candidate_pruning_pq
dev_dir_from_patch() {
  local patch_file="$1"
  local stem
  stem="$(basename "$patch_file" .patch)"   # strip .patch extension
  stem="${stem#pgvector_}"                  # strip leading "pgvector_" prefix
  echo "$ROOT_DIR/.pgvector_dev_${stem}"
}

# Parse the optional --dir <path> flag shared by export/reset/status.
# After this call:
#   DEV_DIR      - resolved working directory (absolute or as given)
#   PARSED_ARGS  - remaining positional arguments with --dir consumed
parse_dir_flag() {
  DEV_DIR="$ROOT_DIR/.pgvector_dev"   # fallback default
  local args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dir) DEV_DIR="${2:?--dir requires a path}"; shift 2 ;;
      *)     args+=("$1"); shift ;;
    esac
  done
  PARSED_ARGS=("${args[@]+"${args[@]}"}")
}

# Abort with a clear message if DEV_DIR is not an initialised git repo.
# Call this at the start of every subcommand that requires an existing
# working directory (export, reset, status).
require_dev_dir() {
  [[ -d "$DEV_DIR/.git" ]] || die "$DEV_DIR not found. Run 'setup' first."
}

# -----------------------------------------------------------------------
# Subcommand: setup
# -----------------------------------------------------------------------

cmd_setup() {
  local tag="$DEFAULT_TAG"
  local patch_file=""
  local explicit_dir=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tag)   tag="${2:?--tag requires a version tag}"; shift 2 ;;
      --patch) patch_file="${2:?--patch requires a patch file}"; shift 2 ;;
      --dir)   explicit_dir="${2:?--dir requires a path}"; shift 2 ;;
      *) die "setup: unknown option '$1'" ;;
    esac
  done

  # --patch is mandatory: it drives both the patch application and (by
  # default) the name of the working directory.
  [[ -n "$patch_file" ]] || die "setup: --patch <patch-file> is required."

  # Resolve patch file to an absolute path so it can be referenced even
  # after the working directory changes during git clone.
  [[ "$patch_file" = /* ]] || patch_file="$(pwd)/$patch_file"
  [[ -f "$patch_file" ]] || die "Patch file not found: $patch_file"

  # Determine the working directory: explicit --dir wins; otherwise derive
  # the name from the patch filename so the directory is self-documenting.
  if [[ -n "$explicit_dir" ]]; then
    DEV_DIR="$explicit_dir"
  else
    DEV_DIR="$(dev_dir_from_patch "$patch_file")"
  fi

  if [[ -d "$DEV_DIR/.git" ]]; then
    # Allow idempotent re-runs: skip the clone but still apply the patch.
    info "$DEV_DIR already exists (skipping clone)"
    info "To start fresh, run 'reset' first or use a different --dir."
  else
    info "Cloning pgvector ($tag) into $DEV_DIR ..."
    git clone --depth 1 --branch "$tag" "$REPO_URL" "$DEV_DIR"
    info "Clone complete: $DEV_DIR"
  fi

  # Dry-run first so we get a clear error before mutating the working tree.
  info "Applying patch: $patch_file"
  git -C "$DEV_DIR" apply --check "$patch_file" \
    || die "Patch check failed. The patch may already be applied or have conflicts."
  git -C "$DEV_DIR" apply "$patch_file"
  info "Patch applied."
}

# -----------------------------------------------------------------------
# Subcommand: export
# -----------------------------------------------------------------------

cmd_export() {
  local gz_threshold=10485760  # 10 MiB default
  # Parse --dir and --gz-threshold before falling through to positional args.
  DEV_DIR="$ROOT_DIR/.pgvector_dev"   # fallback default
  local args=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dir)          DEV_DIR="${2:?--dir requires a path}"; shift 2 ;;
      --gz-threshold) gz_threshold="${2:?--gz-threshold requires a byte count}"; shift 2 ;;
      *)              args+=("$1"); shift ;;
    esac
  done
  set -- "${args[@]+"${args[@]}"}"

  local output="${1:-}"
  [[ -n "$output" ]] || die "export: specify an output patch file."
  [[ "$output" = /* ]] || output="$(pwd)/$output"

  require_dev_dir

  # Collect the current state of the working tree.
  local has_staged has_unstaged has_untracked
  has_staged=$(git    -C "$DEV_DIR" diff --cached --name-only)
  has_unstaged=$(git  -C "$DEV_DIR" diff --name-only)
  has_untracked=$(git -C "$DEV_DIR" ls-files --others --exclude-standard)

  if [[ -z "$has_staged" && -z "$has_unstaged" && -z "$has_untracked" ]]; then
    info "No changes found. The patch will be empty."
    echo "" > "$output"
    exit 0
  fi

  # `git diff --cached` only sees staged changes, so temporarily stage
  # everything (unstaged modifications + untracked new files) with
  # `git add -A`.  We undo the staging immediately after capturing the diff
  # so the index is left in the same state as before this command.
  local stash_needed=false
  if [[ -n "$has_untracked" || -n "$has_unstaged" ]]; then
    git -C "$DEV_DIR" add -A
    stash_needed=true
  fi

  info "Exporting diff to: $output"
  git -C "$DEV_DIR" diff --cached --patch --stat > "$output"

  # Restore the index (working tree files are untouched).
  if [[ "$stash_needed" == true ]]; then
    git -C "$DEV_DIR" reset HEAD -- . >/dev/null 2>&1 || true
  fi

  if [[ -s "$output" ]]; then
    # Compress if the patch exceeds the threshold.
    local file_size
    file_size=$(wc -c < "$output" | tr -d ' ')
    if [[ "$file_size" -gt "$gz_threshold" ]]; then
      gzip -f "$output"
      info "Patch compressed: ${output}.gz ($(du -h "${output}.gz" | cut -f1))"
    else
      info "Patch written: $output"
    fi
  else
    info "Diff was empty. File created but has no content."
  fi
}

# -----------------------------------------------------------------------
# Subcommand: reset
# -----------------------------------------------------------------------

cmd_reset() {
  parse_dir_flag "$@"
  require_dev_dir
  # WARNING: destructive - discards all uncommitted changes and untracked files.
  info "Restoring $DEV_DIR to its initial state..."
  git -C "$DEV_DIR" reset --hard HEAD
  git -C "$DEV_DIR" clean -fd
  info "Reset complete."
}

# -----------------------------------------------------------------------
# Subcommand: status
# -----------------------------------------------------------------------

cmd_status() {
  parse_dir_flag "$@"
  require_dev_dir
  git -C "$DEV_DIR" status
}

# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

[[ $# -ge 1 ]] || usage

subcommand="$1"; shift

case "$subcommand" in
  setup)  cmd_setup  "$@" ;;
  export) cmd_export "$@" ;;
  reset)  cmd_reset  "$@" ;;
  status) cmd_status "$@" ;;
  *)      die "Unknown subcommand '$subcommand'" ;;
esac
