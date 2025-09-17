#!/usr/bin/env bash
set -e
storescp -v -od ./incoming ${PORT:-104}
