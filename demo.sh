#!/bin/bash
export GOOGLE_APPLICATION_CREDENTIALS=`ls ../workgee*json | head -1`
echo "CREDS: $GOOGLE_APPLICATION_CREDENTIALS"
for img in resources/*;do echo $img;python faceme.py $img;echo "waiting...";sleep 10;done
