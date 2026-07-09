#!/usr/bin/env bash
# Build, test, and install braai so it can be run from any directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Where to install the binary. ~/.local/bin is used by convention and does
# not require sudo; override with INSTALL_DIR=/some/path ./build.sh.
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
BIN_NAME="braai"

echo "==> Checking formatting"
UNFORMATTED="$(gofmt -l .)"
if [ -n "${UNFORMATTED}" ]; then
	echo "The following files are not gofmt-formatted:"
	echo "${UNFORMATTED}"
	echo "Run: gofmt -w ."
	exit 1
fi

echo "==> Running tests"
go test ./...

echo "==> Vetting"
go vet ./...

echo "==> Building ${BIN_NAME}"
go build -o "${BIN_NAME}" .

echo "==> Installing to ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
mv "${BIN_NAME}" "${INSTALL_DIR}/${BIN_NAME}"
chmod +x "${INSTALL_DIR}/${BIN_NAME}"

echo "==> Done: $(command -v "${BIN_NAME}" || echo "${INSTALL_DIR}/${BIN_NAME}")"

if ! command -v "${BIN_NAME}" >/dev/null 2>&1; then
	echo
	echo "WARNING: ${INSTALL_DIR} is not on your PATH."
	echo "Add this to your shell profile (e.g. ~/.zshrc or ~/.bashrc):"
	echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
fi
