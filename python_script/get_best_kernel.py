import re

def find_max_gflops(filename):
    max_gflops = 0
    max_gflops_name = ""

    with open(filename, 'r') as file:
        content = file.read()
        matches = re.findall(r'(\d+_gemm_codegen)\s+.*?GFLOPS: (\d+)', content, re.DOTALL)

        for name, gflops in matches:
            gflops = float(gflops)
            if gflops > max_gflops:
                max_gflops = gflops
                max_gflops_name = name

    return max_gflops_name, max_gflops

def find_baseline_gflops(filename):
    baseline_gflops = 0

    with open(filename, 'r') as file:
        content = file.read()
        matches = re.findall(r'\b0_gemm_codegen\b.*?GFLOPS: (\d+)', content, re.DOTALL)
        assert(len(matches) == 1)
        for gflops in matches:
            baseline_gflops = float(gflops)

    return baseline_gflops

prefix = "../logs/logs_4x4/"
for gmem_slowdown in range(1, 11):
    filename = prefix + f'log_gmemslowdown_{gmem_slowdown}x_dsmem_1x.txt'
    baseline_gflops = find_baseline_gflops(filename)
    print(f"gmem_slowdown: {gmem_slowdown}x, dsmem_accelerate: 0x, Max_GFLOPS: {baseline_gflops}, Kernel: 0_gemm_codegen")
    for dsmem_acc in range(1, 7):
        filename = prefix + f'log_gmemslowdown_{gmem_slowdown}x_dsmem_{dsmem_acc}x.txt'
        max_gflops_name, max_gflops = find_max_gflops(filename)
        print(f"gmem_slowdown: {gmem_slowdown}x, dsmem_accelerate: {dsmem_acc}x, Max_GFLOPS: {max_gflops}, Kernel: {max_gflops_name}")

# print("Simulate A100 spec")
# for i in range(2, 7):
#     filename = 'output_log_gmem_1_dsmem_acc_{}.txt'.format(i)
#     max_gflops_name, max_gflops = find_max_gflops(filename)
#     print(f"dsmem bw: {i}x, Max GFLOPS: {max_gflops}, Kernel: {max_gflops_name}")