#!/bin/sh
# Test runner of the scoped gate (tools/ci/check-rust.sh): Cargo builds every
# test as in the full gate and runs it through this script, which executes
# only the binaries of the packages in GUFO_GATE_PACKAGES. Cargo sets
# CARGO_PKG_NAME for each test it runs.
case " $GUFO_GATE_PACKAGES " in
*" $CARGO_PKG_NAME "*) exec "$@" ;;
esac
