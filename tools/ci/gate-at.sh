#!/usr/bin/env bash
# Runs the gate (tools/ci/check-rust.sh) on a fixed revision in a worktree of
# its own, so that a run started later checks exactly that revision while the
# checkout moves on.
#
#   tools/ci/gate-at.sh REV [check-rust.sh options]     e.g. REV --full
#
# The worktree is target/gate-worktree, with its own target directory that
# later runs build on incrementally. The untracked inputs the tests read from
# target/ (flash-next, qwen-mtp, image-ref, qwen35moe; docs/development/TESTING.md) are
# linked into it.
#
# Each test binary that passes is recorded for the revision (in the worktree's
# target/gate-passed/<commit>.tsv, through tools/ci/gate_runner.sh), and a
# later run at the same revision keeps those whose contents are unchanged: a
# run stopped at the end of a window resumes where it stopped. Delete the file
# for a run from the start.
set -euo pipefail
cd "$(dirname "$0")/../.."
rev=${1:?usage: tools/ci/gate-at.sh REV [check-rust.sh options]}
shift
commit=$(git rev-parse --verify "$rev^{commit}")
tree=$PWD/target/gate-worktree

git worktree prune
if [ -e "$tree/.git" ]; then
  git -C "$tree" checkout --quiet --force --detach "$commit"
else
  git worktree add --quiet --detach "$tree" "$commit"
fi
mkdir -p "$tree/target"
for input in flash-next qwen-mtp image-ref qwen35moe; do
  if [ -e "target/$input" ]; then
    ln -sfn "$(readlink -f "target/$input")" "$tree/target/$input"
  fi
done

passed=$tree/target/gate-passed
mkdir -p "$passed"
find "$passed" -type f ! -name "$commit.tsv" -delete
export GUFO_GATE_PASSED=$passed/$commit.tsv

printf 'gate at %s (%s)\n' "$rev" "$commit"
cd "$tree"
exec tools/ci/check-rust.sh "$@"
