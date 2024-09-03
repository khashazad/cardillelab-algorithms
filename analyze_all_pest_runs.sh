#!/bin/bash

for i in {5..8}
do
    python pest_run_analysis.py --point_set="$i" &
done
wait
