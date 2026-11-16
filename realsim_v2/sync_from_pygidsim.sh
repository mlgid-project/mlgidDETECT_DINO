#!/bin/bash
# Copies the three generator files from the user's pygidSIM working copy into this package (one direction only; nothing in pygidSIM is touched).
# The files stay byte-identical to the source so that a diff shows only the user's own changes; the repo-side logic lives in dataset.py / gen.sh.
S=/home/nicolerch/Documents/mlgid/pygidSIM/realsim
cd "$(dirname "$0")" || exit 1
cp "$S"/pipeline.py "$S"/polar_labels.py "$S"/generate_dataset.py . && sha256sum pipeline.py polar_labels.py generate_dataset.py > SOURCE_SHA256.txt && echo "synced $(date '+%F %T')"
