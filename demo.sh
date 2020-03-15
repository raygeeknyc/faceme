#!/bin/bash
PYTHON=$(which python3)
export GOOGLE_APPLICATION_CREDENTIALS=`ls ../*credentials*.json | head -1`
if [[ -r "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
  echo "CREDS: $GOOGLE_APPLICATION_CREDENTIALS"
else
  echo "No credentials file." >&2
  exit -1
fi
$PYTHON faceme.py resources/*
