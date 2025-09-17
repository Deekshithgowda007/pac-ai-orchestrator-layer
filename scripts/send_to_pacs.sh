#!/usr/bin/env bash
set -e
storescu -v -aec ${PACS_AET:-DCM4CHEE} ${PACS_HOST:-localhost} ${PACS_PORT:-11112} "$1"
