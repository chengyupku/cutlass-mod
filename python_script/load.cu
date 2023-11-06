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

    if (warp_idx_in_warp_group == 0 and lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A and B
      //

      dim3 cluster_local_block_id = cute::block_id_in_cluster();
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
      int order[4] = {3, 1, 2, 0};

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

        int k_tile_iter_AB;
        if (cluster_local_block_id.y == 0) {
          k_tile_iter_AB = k_iter;
        }
        else {
          k_tile_iter_AB = (k_iter / 4) * 4 + order[k_iter % 4];
        }

        // to optimize
        if (write_stage == pipeline_params.dsmem_send_stage)
        {
          if (pipeline_params.dsmem_copy_A) {
            pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, 0);
          }
          if (pipeline_params.dsmem_copy_B) {
            pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, 1);
          }
          sender_dsmem_copy_finish_phase ^= 1;
        }
        if (write_stage != pipeline_params.dsmem_recv_stage || (not pipeline_params.dsmem_copy_A)) {
          // TMA load A from gmem to smem
          copy(tma_load_a.with(*tma_A_barrier, mcast_mask_a), tAgA(_,_,_,k_tile_iter_AB), tAsA(_,_,_,write_stage));
        }
        if (write_stage != pipeline_params.dsmem_recv_stage || (not pipeline_params.dsmem_copy_B)) {
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

      dim3 cluster_local_block_id = cute::block_id_in_cluster();
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
          dst_id = (cluster_local_block_id.y + 1) % size<1>(ClusterShape{});
          pipeline.dsmem_copy_prepare(TransactionBytesA, dst_id, 0);
          BarrierType* dsmem_barrier = pipeline.producer_get_dsmem_barrier(0);
          dsmem_copy( ClusterShape{}, 
                      cluster_local_block_id,  
                      tAsA(_,_,_,pipeline_params.dsmem_send_stage).data().get(), 
                      tAsA(_,_,_,pipeline_params.dsmem_recv_stage).data().get(), 
                      dsmem_barrier, 
                      TransactionBytesA,
                      0);
        }
        if (pipeline_params.dsmem_copy_B) {
          dst_id = (cluster_local_block_id.x + 1) % size<0>(ClusterShape{});
          pipeline.dsmem_copy_prepare(TransactionBytesB, dst_id, 1);
          BarrierType* dsmem_barrier = pipeline.producer_get_dsmem_barrier(1);
          dsmem_copy( ClusterShape{}, 
                      cluster_local_block_id,  
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
      dim3 cluster_local_block_id = cute::block_id_in_cluster();
      uint32_t src_A_block = cluster_local_block_id.y == 0 ?
                              size<1>(ClusterShape{}) - 1 :
                              cluster_local_block_id.y - 1;

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
        pipeline.arrive_mma(mma_stage);
        mma_stage++;
        if (mma_stage == K_PIPE_MAX) {
          mma_stage = 0;
          mma_wait_phase ^= 1;
        }
      }
    }
  }