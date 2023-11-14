import json
import cutlass_code

_collective_load = []
_collective_load.append("""
/// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TMA_LOAD_A,
    class TensorB, class TMA_LOAD_B,
    class KTileIterator
  >
  CUTLASS_DEVICE void
  load(
      MainloopPipeline pipeline, 
      PipelineState smem_pipe_write,
      TensorA const& gA, TMA_LOAD_A& tma_load_a,
      TensorB const& gB, TMA_LOAD_B& tma_load_b,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      int& receiver_ready_phase,
      int& sender_ready_phase,
      int& sender_dsmem_copy_finish_phase,
      int& receiver_dsmem_copy_finish_phase,
      int& mma_wait_phase,
      TensorStorage& shared_tensors
      )
  {
    using namespace cute;
    int warp_idx = canonical_warp_idx();
    int warp_idx_in_warp_group  = warp_idx % 4;
    int lane_predicate = cute::elect_one_sync();
    auto pipeline_params = pipeline.get_params();
""")

_collective_load.append(
"""
    if (warp_idx_in_warp_group == 0 and lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      dim3 bid = cute::block_id_in_cluster();
      // may have bug
      auto block_tma_a = tma_load_a.get_slice(0);
      auto block_tma_b = tma_load_b.get_slice(0);

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      int k_iter = 0;
""")

_collective_load.append(
"""

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count)
      {
        int write_stage = smem_pipe_write.index();

        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_A_barrier = pipeline.producer_get_barrier(smem_pipe_write, 0);
        BarrierType* tma_B_barrier = pipeline.producer_get_barrier(smem_pipe_write, 1);

        int k_tile_iter_AB = tile_order[bid.x][bid.y][k_iter];

        // Check if this stage was sender on iteration (k_iter - K_PIPE_MAX)
        // If true, wait until the copy is done
        // if ((k_iter - K_PIPE_MAX >= 0) && 
        //    (dst_A[bid.x][bid.y][k_iter - K_PIPE_MAX].x != -1 && 
        //     dst_A[bid.x][bid.y][k_iter - K_PIPE_MAX].y != -1)) {
        //  pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, 0);
        //  sender_dsmem_copy_finish_phase ^= 1;
        // }
        // if ((k_iter - K_PIPE_MAX >= 0) && 
        //    (dst_B[bid.x][bid.y][k_iter - K_PIPE_MAX].x != -1 && 
        //     dst_B[bid.x][bid.y][k_iter - K_PIPE_MAX].y != -1)) {
        //  pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, 0);
        //  sender_dsmem_copy_finish_phase ^= 1;
        // }
        if (src_A[bid.x][bid.y][k_iter % PATTERN_LEN].x == -1 || 
            src_A[bid.x][bid.y][k_iter % PATTERN_LEN].y == -1) {
          // TMA load A from gmem to smem
          copy(tma_load_a.with(*tma_A_barrier, mcast_mask_a), tAgA(_,_,_,k_tile_iter_AB), tAsA(_,_,_,write_stage));
        }
        if (src_B[bid.x][bid.y][k_iter % PATTERN_LEN].x == -1 || 
            src_B[bid.x][bid.y][k_iter % PATTERN_LEN].y == -1) {
          // TMA load B from gmem to smem
          copy(tma_load_b.with(*tma_B_barrier, mcast_mask_b), tBgB(_,_,_,k_tile_iter_AB), tBsB(_,_,_,write_stage));
        }

        ++k_tile_iter;
        ++k_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }

    if (warp_idx_in_warp_group == 1 and lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      dim3 bid = cute::block_id_in_cluster();
      auto block_tma_a = tma_load_a.get_slice(0);
      auto block_tma_b = tma_load_b.get_slice(0);

      // Applies the mapping from block_tma_a
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      // assert(k_tile_count % size<0>(ClusterShape{}) == 0 && k_tile_count % size<1>(ClusterShape{}) == 0; "cluster shape not aligned!");
      
      int k_iter = 0;
      int k_dsmem_tile_count = k_tile_count / K_PIPE_MAX;

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_dsmem_tile_count > 0; --k_dsmem_tile_count)
      {
        // wait receiver's arrive
        if (pipeline_params.dsmem_copy_A) {
          // wait sender buffer ready, may have bug (double consumer wait?)
          pipeline.sender_wait_sender_ready(sender_ready_phase, 0);
          // to optimize
          pipeline.sender_wait_receiver_ready(receiver_ready_phase, 0);
        }
        if (pipeline_params.dsmem_copy_B) {
          // wait sender buffer ready, may have bug (double consumer wait?)
          pipeline.sender_wait_sender_ready(sender_ready_phase, 1);
          // to optimize
          pipeline.sender_wait_receiver_ready(receiver_ready_phase, 1);
        }

        receiver_ready_phase ^= 1;
        sender_ready_phase ^= 1;

        //
        // Copy gmem to smem for *k_tile_iter
        //
        uint32_t dst_id = 0;
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        // not support dsmem copy both A and B
        if (pipeline_params.dsmem_copy_A) {
          dst_id = (bid.y + 1) % size<1>(ClusterShape{});
          pipeline.dsmem_copy_prepare(TransactionBytesA, dst_id, 0);
          BarrierType* dsmem_barrier = pipeline.producer_get_dsmem_barrier(0);
          dsmem_copy( ClusterShape{}, 
                      bid,  
                      tAsA(_,_,_,pipeline_params.dsmem_send_stage).data().get(), 
                      tAsA(_,_,_,pipeline_params.dsmem_recv_stage).data().get(), 
                      dsmem_barrier, 
                      TransactionBytesA,
                      0);
        }
        if (pipeline_params.dsmem_copy_B) {
          dst_id = (bid.x + 1) % size<0>(ClusterShape{});
          pipeline.dsmem_copy_prepare(TransactionBytesB, dst_id, 1);
          BarrierType* dsmem_barrier = pipeline.producer_get_dsmem_barrier(1);
          dsmem_copy( ClusterShape{}, 
                      bid,  
                      tBsB(_,_,_,pipeline_params.dsmem_send_stage).data().get(), 
                      tBsB(_,_,_,pipeline_params.dsmem_recv_stage).data().get(), 
                      dsmem_barrier, 
                      TransactionBytesB,
                      1);
        }
        
        ++k_tile_iter;
        ++k_iter;
      }
    }

    // monitor dsmem copy of A
    if (warp_idx_in_warp_group == 2 and lane_predicate and pipeline_params.dsmem_copy_A) {
      int k_dsmem_tile_count = k_tile_count / K_PIPE_MAX;
      dim3 bid = cute::block_id_in_cluster();
      uint32_t src_A_block = bid.y == 0 ?
                              size<1>(ClusterShape{}) - 1 :
                              bid.y - 1;

      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_dsmem_tile_count > 0; --k_dsmem_tile_count)
      {
        pipeline.receiver_wait_dsmem_copy_finish(receiver_dsmem_copy_finish_phase, 0);
        receiver_dsmem_copy_finish_phase ^= 1;
        pipeline.receiver_arrive_dsmem_copy_finish(src_A_block, 0);
      }
    }
    // monitor consumer_wait conditions, to accelerate consumer_wait()
    if (warp_idx_in_warp_group == 3 and lane_predicate) {
      // may have bug
      int mma_stage = 0;
      auto pipeline_params = pipeline.get_params();
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count)
      {
        pipeline.mma_wait(mma_stage, mma_wait_phase);
        // if (PRINT_CONDITION(96) && prof_iter < PROFILE_ITER && mma_stage==pipeline_params.dsmem_recv_stage) {
        //   monitor_wait_done[prof_iter] = get_clock();
        //   mma_wait_dsmem_done[prof_iter] = t0[0];
        //   wait_B[prof_iter] = t2[0] - t1[0];
        //   ++prof_iter;
        // }
        pipeline.arrive_mma(mma_stage);
        mma_stage++;
        if (mma_stage == K_PIPE_MAX) {
          mma_stage = 0;
          mma_wait_phase ^= 1;
        }
      }
    }
  }
""")

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
    # tile_order_code = "int8_t tile_order[size<0>(ClusterShape{})][size<1>(ClusterShape{})][PATTERN_LEN] = {\n"
    # src_A_code = "block_id src_A[size<0>(ClusterShape{})][size<1>(ClusterShape{})][PATTERN_LEN][2] = {\n"
    # src_B_code = "block_id src_B[size<0>(ClusterShape{})][size<1>(ClusterShape{})][PATTERN_LEN][2] = {\n"
    # dst_A_code = "block_id dst_A[size<0>(ClusterShape{})][size<1>(ClusterShape{})][PATTERN_LEN][2] = {\n"
    # dst_B_code = "block_id dst_B[size<0>(ClusterShape{})][size<1>(ClusterShape{})][PATTERN_LEN][2] = {\n"
    tile_order_code = "\tint8_t tile_order[2][4][PATTERN_LEN] = {\n"
    src_A_code = "\tblock_iter_id src_A[2][4][PATTERN_LEN] = {\n"
    src_B_code = "\tblock_iter_id src_B[2][4][PATTERN_LEN] = {\n"
    dst_A_code = "\tblock_iter_id dst_A[2][4][PATTERN_LEN] = {\n"
    dst_B_code = "\tblock_iter_id dst_B[2][4][PATTERN_LEN] = {\n"
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
            # if y == 0:
            #     code += f"  if (bid.y == {y}) {{\n"
            # else:
            #     code += f"  else if (bid.y == {y}) {{\n"
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
        # code += "  }\n"
        # if x != len(l) - 1:
        #     code += f"}}\nelse if (bid.x == {x+1}) {{\n"
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
    with open('schedule.json', 'r', encoding='utf-8') as file:
        schedule_list = json.load(file)
    schedule = schedule_list[0]["schedule"]
    # print(schedule)
    schedule_per_blk = [[item['schedule_per_blk'] for item in x_list] for x_list in schedule]
    tileid = [[[item['tileid'] for item in y_list] for y_list in x_list] for x_list in schedule_per_blk]
    # dst_A/B: [x,y,i], the tile will be used by block(x,y) on iteration i + N * PATTERN_LEN
    # src_A/B: [x,y,i], the tile comes from block(x,y) on its iteration  i + N * PATTERN_LEN
    src_A  = [[[parse_src(item['srcA']) for item in y_list] for y_list in x_list] for x_list in schedule_per_blk]
    src_B  = [[[parse_src(item['srcB']) for item in y_list] for y_list in x_list] for x_list in schedule_per_blk]
    dst_A  = [[[[-1,-1,-1] for _ in y_list] for y_list in x_list] for x_list in src_A]
    dst_B  = [[[[-1,-1,-1] for _ in y_list] for y_list in x_list] for x_list in src_B]

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

    # print(dst_A)
    # print(dst_B)
    # print(tileid)
    # print(src_A)
    # print(src_B)

    code = ""
    struct_def_code = """
struct block_iter_id {
  int8_t x, y, iter;
};
"""

    code += gen_tile_order_code(tileid, src_A, src_B, dst_A, dst_B)
    # print(cutlass_code.header)
    # print(block_id_def_code)
    # print(cutlass_code.collective_header)
    print(code)
    # print(cutlass_code.collective_middle)
    # print(_collective_load[0])
    # print(_collective_load[1])
    # print(_collective_load[2])
    # print(cutlass_code.collective_load_tail)
    # print(cutlass_code.collective_mma)
    # print(cutlass_code.collective_mma_tail)
    # print(cutlass_code.collective_tail)
    # print(cutlass_code.tail)