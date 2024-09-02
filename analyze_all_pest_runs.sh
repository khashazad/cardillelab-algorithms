#!/bin/bash

for i in {1..4}
do
    python pest_run_analysis.py --point_set="$i" &
done
wait
