#!/bin/bash

for i in {1..3}
do
    python prepare_pest_run.py --initial_params="v$i" &
done
wait
