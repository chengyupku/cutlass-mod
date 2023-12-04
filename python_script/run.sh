#!/bin/bash

log_file="logs/execution_log.txt"
simulate_multiple=${1}
gmem_slowdonw=$((simulate_multiple + 1))
dsmem_accelerate=${2}

> "$log_file"

# 总共的任务数和 GPU 数量
total_tasks=${3}
num_gpus=7

# 每个 GPU 执行的任务数量
tasks_per_gpu=$(( (total_tasks + num_gpus - 1) / num_gpus ))

# 并行执行任务
for gpu_index in $(seq 1 $((num_gpus))); do
    output_log_file="../logs/log_gmemslowdown_${gmem_slowdonw}x_dsmem${dsmem_accelerate}x_gpu_${gpu_index}.txt"
    > "$output_log_file"
    start_task=$(((gpu_index - 1) * tasks_per_gpu))
    end_task=$(((gpu_index) * tasks_per_gpu - 1))

    # 确保最后一个 GPU 不会超出总任务数
    if [ $end_task -ge $total_tasks ]; then
        end_task=$((total_tasks - 1))
    fi

    (
        for index in $(seq $start_task $end_task); do
            # 设置 CUDA_VISIBLE_DEVICES 以使用特定的 GPU
            export CUDA_VISIBLE_DEVICES=$gpu_index
            executable="${index}_gemm_codegen"

            start_time=$(date +%s)
            echo "Executing ($executable) on GPU $gpu_index..." >> "$log_file"
            echo "---------------------------------" >> "$output_log_file"
            echo "$executable" >> "$output_log_file"
            ../build/examples/97_gemm_codegen/$executable >> "$output_log_file"

            end_time=$(date +%s)
            execution_time=$((end_time - start_time))
            echo "$(date): Process ($executable) completed in $execution_time seconds on GPU $gpu_index." >> "$log_file"
        done
    ) &
done

# 等待所有后台任务完成
wait