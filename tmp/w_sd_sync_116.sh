#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
git rev-parse HEAD
git status --short | head -20
echo "---branch---"
git branch --show-current
