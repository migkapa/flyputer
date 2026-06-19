#!/usr/bin/env bash
# Download the FlyWire connectome edge list (~852 MB, no login required).
# The neuron annotation TSV (~32 MB) auto-downloads on first run of flysim.py.
set -e

OUT="proofread_connections_783.feather"
URL="https://zenodo.org/records/10676866/files/proofread_connections_783.feather"

if [ -f "$OUT" ]; then
  echo "$OUT already present — nothing to do."
  exit 0
fi

echo "Downloading $OUT (~852 MB)…"
curl -L --fail -o "$OUT" "$URL"
echo "Done. If the filename 404s, open https://zenodo.org/records/10676866 and grab"
echo "the connections .feather manually, then save it here as $OUT."
