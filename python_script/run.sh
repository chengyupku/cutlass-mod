#!/bin/bash

log_file="execution_log.txt"
output_log_file="output_log.txt"
> "$log_file"
> "$output_log_file"

for index in $(seq 0 81); do
    executable="${index}_gemm_codegen"

    start_time=$(date +%s)
    echo "Executing ($executable)..." >> "$log_file"
    echo "---------------------------------" >> "$output_log_file"
    echo "$executable" >> "$output_log_file"
    ../build/examples/97_gemm_codegen/$executable >> "$output_log_file" &

    pid=$!

    sleep 30

    if ps -p $pid > /dev/null; then
        kill $pid
        echo "$(date): Process $pid ($executable) exceeded 30 seconds and was terminated." >> "$log_file"
    else
        end_time=$(date +%s)
        execution_time=$((end_time - start_time))
        echo "$(date): Process $pid ($executable) completed in $execution_time seconds." >> "$log_file"
    fi
done