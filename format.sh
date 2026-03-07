#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Quick formatting script for TileGym development
# Formats code and sorts imports using ruff

set -e

RUFF_VERSION="0.14.9"

echo "🔍 Checking ruff installation..."
if ! python3 -m ruff --version 2>/dev/null | grep -q "$RUFF_VERSION"; then
    echo "📦 Installing ruff $RUFF_VERSION..."
    pip install "ruff==$RUFF_VERSION"
fi

echo ""
echo "📝 Adding SPDX headers to files..."
python3 .github/scripts/check_spdx_headers.py --action write

echo ""
echo "📋 Sorting imports..."
python3 -m ruff check --select I --fix --exclude .venv --force-exclude .

echo ""
echo "✨ Formatting code..."
python3 -m ruff format --exclude .venv --force-exclude .

echo ""
echo "✅ Done! SPDX headers added, code is formatted, and imports are sorted."
