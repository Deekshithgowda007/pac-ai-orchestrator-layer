#!/usr/bin/env bash
# Usage: ./send_to_pacs.sh /path/to/file.dcm
FILE="$1"
if [ -z "$FILE" ]; then
  echo "Usage: $0 /path/to/file.dcm"
  exit 1
fi

storescu -v -aec DCM4CHEE localhost 11112 "$FILE"
