import argparse
import json
import cutlass_code
import glob
import os

valid_cluster_shape = [1, 2, 4, 8, 16]
cluster_shape = [4, 4, 1]
PatternLen = 8
assert (cs in valid_cluster_shape for cs in cluster_shape) and (cluster_shape[0] * cluster_shape[1] * cluster_shape[2] <= 16), "Invalid cluster shape!"
ClusterShape_code = "using ClusterShape = Shape<_{},_{},_{}>; // Shape of the threadblocks in a cluster\n".format(cluster_shape[0], cluster_shape[1], cluster_shape[2])
PatternLen_code = "static constexpr int PatternLen = {};\n".format(PatternLen)
array_shape = "[{}][{}][PatternLen]".format(cluster_shape[0], cluster_shape[1])
shared_storage_code = """
struct SharedStorage
  {{
    struct ScheduleStorage : cute::aligned_struct<128> {{
      int8_t tileOrder{};
      block_iter_id srcA{};
      block_iter_id srcB{};
      block_iter_id dstA{};
      block_iter_id dstB{};
    }} schedules;

    struct TensorStorage : cute::aligned_struct<128> {{
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    }} tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  }};
""".format(array_shape, array_shape, array_shape, array_shape, array_shape)

init_schedule_func_code = """
  CUTLASS_DEVICE void static
  init_schedule(ScheduleStorage& shared_schedules) {
"""
init_schedule_func_code += "\t\tif (threadIdx.x < {}) {{\n".format(cluster_shape[0] * cluster_shape[1] * PatternLen)
index_code = "[(threadIdx.x / {}) % {}][(threadIdx.x / {}) % {}][threadIdx.x % {}]"\
            .format(cluster_shape[1] * PatternLen, cluster_shape[0], PatternLen, cluster_shape[1], PatternLen)
init_schedule_func_code += "\t\t\tshared_schedules.tileOrder{} = tile_order{};\n".format(index_code, index_code)
for arr_s, arr_d in zip(['srcA', 'srcB', 'dstA', 'dstB'], ['src_A', 'src_B', 'dst_A', 'dst_B']):
    for var in ['x', 'y', 'iter']:
        init_schedule_func_code += "\t\t\tshared_schedules.{}{} = {}{};\n".format(arr_s, index_code, arr_d, index_code)

init_schedule_func_code += """    }
  }
"""

def generate_int_array(l):
    return '{' + ', '.join(map(str, l)) + '}'

def generate_dim3_array(l):
    code = '{'
    for sub_list in l:
        code += '{' + ', '.join(map(str, sub_list)) + '},'
    code += '}'
    return code
    
def gen_tile_order_code(l, sA, sB, dA, dB):
    code = ""
    array_shape = "[{}][{}][PatternLen]".format(cluster_shape[0], cluster_shape[1])
    tile_order_code = "__device__ constexpr int8_t tile_order" + array_shape + " = {\n"
    src_A_code = "__device__ constexpr block_iter_id src_A" + array_shape + " = {\n"
    src_B_code = "__device__ constexpr block_iter_id src_B" + array_shape + " = {\n"
    dst_A_code = "__device__ constexpr block_iter_id dst_A" + array_shape + " = {\n"
    dst_B_code = "__device__ constexpr block_iter_id dst_B" + array_shape + " = {\n"
    # code += "if (bid.x == 0) {\n"
    for x, [x_l_group, x_sA_group, x_sB_group, x_dA_group, x_dB_group] in enumerate(zip(l, sA, sB, dA, dB)):
        tile_order_code += "\t\t{\n"
        src_A_code += "\t\t{\n"
        src_B_code += "\t\t{\n"
        dst_A_code += "\t\t{\n"
        dst_B_code += "\t\t{\n"
        for y, [y_l_group, y_sA_group, y_sB_group, y_dA_group, y_dB_group] in enumerate(zip(x_l_group, x_sA_group, x_sB_group, x_dA_group, x_dB_group)):
            # print(y_sA_group)
            order_array = generate_int_array(y_l_group)
            src_A = generate_dim3_array(y_sA_group)
            src_B = generate_dim3_array(y_sB_group)
            dst_A = generate_dim3_array(y_dA_group)
            dst_B = generate_dim3_array(y_dB_group)
            tile_order_code += f"\t    {order_array},\n"
            src_A_code += f"\t    {src_A},\n"
            src_B_code += f"\t    {src_B},\n"
            dst_A_code += f"\t    {dst_A},\n"
            dst_B_code += f"\t    {dst_B},\n"
        tile_order_code += "\t\t},\n"
        src_A_code += "\t\t},\n"
        src_B_code += "\t\t},\n"
        dst_A_code += "\t\t},\n"
        dst_B_code += "\t\t},\n"
    tile_order_code += "\t};\n"
    src_A_code += "\t};\n"
    src_B_code += "\t};\n"
    dst_A_code += "\t};\n"
    dst_B_code += "\t};\n"
    code += tile_order_code + src_A_code + src_B_code + dst_A_code + dst_B_code
    return code

def parse_src(s):
    if s == 'gmem':
        return [-1,-1,-1]
    else:
        s_clean = s.replace('blk(', '').replace(')', '')
        numbers = s_clean.split(',') + [-1]
        return [int(number) for number in numbers]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate_multiple", type=int, default=0)
    parser.add_argument("--dsmem_accelerate", type=int, default=1)
    parser.add_argument("--schedule_path", type=str, default="")
    args = parser.parse_args()

    with open(args.schedule_path, 'r', encoding='utf-8') as file:
        schedule_list = json.load(file)

    pattern = "../examples/97_gemm_codegen/*_gemm_codegen.cu"
    for filename in glob.glob(pattern):
        print(f"Deleting {filename}...")
        os.remove(filename)

    for sid in range(len(schedule_list)):
        schedule = schedule_list[sid]["schedule"]
        # print(schedule)
        schedule_per_blk = [[item['schedule_per_blk'] for item in x_list] for x_list in schedule]
        tileid = [[[item['tileid'] for item in y_list] for y_list in x_list] for x_list in schedule_per_blk]
        # dst_A/B: [x,y,i], the tile will be used by block(x,y) on iteration i + N * PatternLen
        # src_A/B: [x,y,i], the tile comes from block(x,y) on its iteration  i + N * PatternLen
        src_A  = [[[parse_src(item['srcA']) for item in y_list] for y_list in x_list] for x_list in schedule_per_blk]
        src_B  = [[[parse_src(item['srcB']) for item in y_list] for y_list in x_list] for x_list in schedule_per_blk]
        dst_A  = [[[[-1,-1,-1] for _ in y_list] for y_list in x_list] for x_list in src_A]
        dst_B  = [[[[-1,-1,-1] for _ in y_list] for y_list in x_list] for x_list in src_B]

        # print("tileid", tileid)
        # print("src_A", src_A)
        # print("src_B", src_B)
        # print("dst_A", dst_A)
        # print("dst_B", dst_B)

        # Generate sender list dst_A/B[x][y][k]: tile on block[x][y] at iteration k will send to ...
        for dx, [x_list_A, x_list_B] in enumerate(zip(src_A, src_B)):
            for dy, [y_list_A, y_list_B] in enumerate(zip(x_list_A, x_list_B)):
                for k, [item_A, item_B] in enumerate(zip(y_list_A, y_list_B)):
                    xA, yA, _ = item_A
                    xB, yB, _ = item_B
                    if (xA != -1 and yA != -1):
                        src_k = tileid[xA][yA].index(tileid[dx][dy][k])
                        dst_A[xA][yA][src_k] = [dx, dy, k]
                    if (xB != -1 and yB != -1):
                        src_k = tileid[xB][yB].index(tileid[dx][dy][k])
                        dst_B[xB][yB][src_k] = [dx, dy, k]

        for dx, [x_list_A, x_list_B] in enumerate(zip(dst_A, dst_B)):
            for dy, [y_list_A, y_list_B] in enumerate(zip(x_list_A, x_list_B)):
                for k, [item_A, item_B] in enumerate(zip(y_list_A, y_list_B)):
                    xA, yA, dst_Ak = item_A
                    xB, yB, dst_Bk = item_B
                    if (xA != -1 and yA != -1):
                        src_A[xA][yA][dst_Ak][-1] = k
                    if (xB != -1 and yB != -1):
                        src_B[xB][yB][dst_Bk][-1] = k

        code = ""
        code += "#define SIMULATE_MULTIPLE {}\n".format(args.simulate_multiple)
        code += "#define Dsmem_ACC {}\n".format(args.dsmem_accelerate)
        code += cutlass_code.header_0
        code += ClusterShape_code
        code += PatternLen_code
        code += cutlass_code.header_1
        code += gen_tile_order_code(tileid, src_A, src_B, dst_A, dst_B)
        code += cutlass_code.collective_mma_code_0
        code += shared_storage_code
        code += cutlass_code.collective_mma_code_1
        code += init_schedule_func_code
        code += cutlass_code.collective_mma_code_2
        code += cutlass_code.tail
        # print(code)
        # break
        with open('../examples/97_gemm_codegen/{}_gemm_codegen.cu'.format(sid), 'w') as file:
            print(code, file=file)