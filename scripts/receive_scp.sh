#!/usr/bin/env bash
# Receive DICOMs locally with storescp (DCMTK) and forward to PACS
# Requires dcmtk installed

PORT=${1:-11112}
DEST_AE=${2:-DCM4CHEE}
DEST_HOST=${3:-localhost}
DEST_PORT=${4:-11112}

echo "Starting storescp on port $PORT (will forward to $DEST_HOST:$DEST_PORT)"
storescp -x --fork $PORT
