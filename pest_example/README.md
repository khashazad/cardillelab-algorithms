to generate the output file for the 3 parameter model from the root eee folder call:

```
python pest.py \
    --input=pest_example/input_file_3_params \
    --output=pest_example/output_file_3_params \
    --points=pest_example/point_file \
    --num_sinusoid_pairs=1 \
    --include_intercept
```

to generate the output file for the 5 parameter model from the root eeek folder call:

```
python pest.py \
    --input=pest_example/input_file_5_params \
    --output=pest_example/output_file_5_params \
    --points=pest_example/point_file \
    --num_sinusoid_pairs=2 \
    --include_intercept
```

