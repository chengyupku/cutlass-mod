# profile cutlass
#############################################################
cd ../cutlass
./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*gemm_f16_*_tn_align8 --m=5376 --n=5376 --k=2048 > log.txt 2>&1

./tools/profiler/cutlass_profiler --kernels=cutlass3x_sm90_tensorop_s*gemm_f16_f16_f32_f16_f16_*_1x1x1_0_ntn*  --m=16384 --n=16384 --k=8192 --output=report/cluster1x1x1.csv

# profile cublas
#############################################################
nvcc -arch=sm_90a call_cublas.cu -lcublas -o test_cublas && ./test_cublas

# ncu profile
#############################################################
sudo /usr/local/cuda-12.1/bin/ncu --metrics l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum.per_second ./test > profile.txt
sudo /usr/local/cuda-12.1/bin/ncu --metrics l1tex__m_xbar2l1tex_read_bytes_mem_dshared.sum.per_second,lts__t_sectors_op_read.sum.per_second ./test > profile.txt
sudo /usr/local/cuda-12.1/bin/ncu --metrics l1tex__m_xbar2l1tex_read_bytes.sum.per_second,lts__t_sectors_op_read.sum.per_second,lts__t_sectors_op_read_lookup_hit.sum.per_second ./tes
t > profile.txt
sudo /usr/local/cuda-12.1/bin/ncu --metrics l1tex__m_xbar2l1tex_read_bytes.sum.per_second,lts__t_sectors_op_read.sum.per_second,lts__t_sectors_op_read_lookup_hit.sum.per_second,lts__t_sectors_op_read_lookup_miss.sum.per_second,lts__t_sectors_lookup_hit ./test > profile.txt

sudo /usr/local/cuda-12.1/bin/ncu --metrics l1tex__m_xbar2l1tex_read_bytes.sum.per_second,l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld ./test > profile.txt

smsp__l1tex_tmain_requests
smsp__sass_inst_executed_op_tma
smsp__sass_inst_executed_op_tma_ld
smsp__sass_inst_executed_op_tma_red
smsp__sass_inst_executed_op_tma_st

l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_tma_st
l1tex__m_l1tex2xbar_write_bytes_mem_global_op_tma_st
l1tex__m_l1tex2xbar_write_bytes_pipe_tma

l1tex__m_xbar2l1tex_read_bytes_pipe_tma


# only profile the second invocation of cluster_kernel
sudo /usr/local/cuda-12.1/bin/ncu --kernel-id ::cluster_kernel:2 ./test > profile.txt
# save profile result as 48prof.ncu-rep and can be opened in nsight compute
sudo /usr/local/cuda-12.1/bin/ncu --set full -f -o 48prof ./48_hopper_warp_specialized_gemm

sudo ncu --set full --kernel-id ::device_kernel: -f -o report ./excu


# fix the gpu frequency
sudo nvidia-smi -pm 1
nvidia-smi --query-gpu=clocks.sm --format=csv
sudo nvidia-smi -lgc 1980,1980
nvidia-smi --query-gpu=clocks.sm --format=csv
# unlock
nvidia-smi -rgc
# run on a target GPU
CUDA_VISIBLE_DEVICES=1 ./99_kernel_test