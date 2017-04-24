#!/bin/bash
for img in resources/*;do echo $img;python faceme.py $img;echo "waiting...";sleep 101;done
