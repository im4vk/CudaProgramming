#!/bin/bash

# Specify the metrics to collect
metrics="lts__t_sector_op_read_hit_rate.pct,sm__ctas_launched,gpu__time_duration.avg,gpu__cycles_active.avg,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__sass_average_branch_targets_threads_uniform.pct,dram__throughput.max.pct_of_peak_sustained_elapsed"

# If the executable is not in the local directory, make sure to specify the absolute path to './nodivergence' and './withdivergence'
#ncu --metrics $metrics ./nodivergence
#ncu --metrics $metrics ./withdivergence
ncu --metrics $metrics ./a