#!/bin/bash

for i in {1..5}
do
    python pest_run_analysis.py --point_set="$i" &
done
wait
