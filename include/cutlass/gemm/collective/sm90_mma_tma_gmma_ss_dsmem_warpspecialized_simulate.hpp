/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cutlass/pipeline/pipeline.hpp"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <
  int Stages,
  class ClusterShape,
  class KernelSchedule,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm90TmaGmmaWarpSpecializedDSMEMSimulate<Stages, ClusterShape, KernelSchedule>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedDSMEMSimulate<Stages, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass::PipelineTmaDsmemAsync<
                             DispatchPolicy::Stages,
                             typename DispatchPolicy::ClusterShape>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      Step<_2,_1,_3>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      Step<_2,_1,_3>{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using InternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using InternalElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;

  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<128> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(static_cast<InternalElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,0)));  // force no mcast
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(static_cast<InternalElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,0))); // force no mcast
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    void* workspace; 
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    // (void) workspace;

    // Optionally append _1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    auto ptr_A = reinterpret_cast<InternalElementA const*>(args.ptr_A);
    auto ptr_B = reinterpret_cast<InternalElementB const*>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));
    typename Params::TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{})); // force no mcast
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{})); // force no mcast
    return {
      tma_load_a,
      tma_load_b,
      workspace
    };
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytes =
        (size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof(ElementA)))+
        (size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof(ElementB)));

  static constexpr uint32_t TransactionBytesA = size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof(ElementA));
  static constexpr uint32_t TransactionBytesB = size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof(ElementB));
  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params)
  {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TMA_LOAD_A,
    class TensorB, class TMA_LOAD_B,
    class KTileIterator,
    class TensorDummyA, class TensorDummyB
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
      TensorStorage& shared_tensors,
      TensorDummyA const& dgtA,
      TensorDummyB const& dgtB
      #if PROFILE
      , uint64_t* issue_copy_B
      , uint64_t* copy_B_done
      , uint64_t* issue_dsmem
      #endif
      )
  {
    using namespace cute;
    int warp_idx = canonical_warp_idx();
    int warp_idx_in_warp_group  = warp_idx % 4;
    int lane_predicate = cute::elect_one_sync();
    auto pipeline_params = pipeline.get_params();
    #if PROFILE
    int prof_iter = 0;
    #endif

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
      #if 0
      static int iter = 0;
      if (PRINT_CONDITION(0) && iter==0) {
        iter += 1;
        // print("block_tma_a: "); print(block_tma_a.layout()); print("\n");
        // print("block_tma_b: "); print(block_tma_b.layout()); print("\n");
        // print("tma_load_a: "); print(tma_load_a); print("\n");
        // print("tma_load_b: "); print(tma_load_b); print("\n");
        print("tAgA: "); print(tAgA.layout()); print("\n");
        print("tAsA: "); print(tAsA.layout()); print("\n");
        // print("tBgB: "); print(tBgB(0)); print("\n");
        // print("tBsB: "); print(tBsB(0)); print("\n");
      }
      #endif
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
        uint32_t dummy_transaction_bytes_A = size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof(ElementA));
        uint32_t dummy_transaction_bytes_B = size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof(ElementB));
        uint32_t A_transaction_bytes = (size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof(ElementA))) 
                                        + dummy_transaction_bytes_A * SIMULATE_MULTIPLE;
        uint32_t B_transaction_bytes = (size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof(ElementB))) 
                                        + dummy_transaction_bytes_B * SIMULATE_MULTIPLE;
        
        pipeline.producer_acquire(smem_pipe_write, A_transaction_bytes, B_transaction_bytes);

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
          // Dummy copy from gmem to smem to simulate low gmem bandwidth hardware spec
          for (int i=0; i<SIMULATE_MULTIPLE; i++) {
            gmem2cta_copy_kernel(dgtA(i,_).data().get(), tAsA(_,_,_,write_stage).data().get(), tma_A_barrier, dummy_transaction_bytes_A);
          }
          copy(tma_load_a.with(*tma_A_barrier, mcast_mask_a), tAgA(_,_,_,k_tile_iter_AB), tAsA(_,_,_,write_stage));
        }
        if (write_stage != pipeline_params.dsmem_recv_stage || (not pipeline_params.dsmem_copy_B)) {
          #if PROFILE
          if (PRINT_CONDITION(0) && prof_iter < PROFILE_ITER && write_stage==pipeline_params.dsmem_recv_stage) {
            issue_copy_B[prof_iter] = get_clock();
            prof_iter++;
          }
          #endif
          // TMA load B from gmem to smem
          for (int i=0; i<SIMULATE_MULTIPLE; i++) {
            gmem2cta_copy_kernel(dgtB(i,_).data().get(), tBsB(_,_,_,write_stage).data().get(), tma_B_barrier, dummy_transaction_bytes_B);
          }
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
          #if PROFILE
          if (PRINT_CONDITION(32) && prof_iter < PROFILE_ITER) {
            issue_dsmem[prof_iter] = get_clock();
            prof_iter++;
          }
          #endif
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
        #if PROFILE
        if (PRINT_CONDITION(96) && prof_iter < PROFILE_ITER && mma_stage==pipeline_params.dsmem_recv_stage) {
          copy_B_done[prof_iter] = get_clock();
          prof_iter++;
        }
        #endif
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

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(
      MainloopPipeline pipeline, 
      PipelineState smem_pipe_write,
      int& receiver_ready_phase)
  {
    int warp_idx = canonical_warp_idx();
    int warp_idx_in_warp_group = warp_idx % 4;
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (warp_idx_in_warp_group == 1 and lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all 
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was 
       * still inverted from make_producer_start_state
       */
      // print("load tail\n");
      pipeline.producer_tail(smem_pipe_write, receiver_ready_phase);
      // pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params
      #if PROFILE
      , uint64_t* consumer_wait_latency
      , uint64_t* on_consumer_wait
      , uint64_t* mma_release
      #endif
      )
  {
    using namespace cute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    #if PROFILE
    uint64_t t0, t1;
    uint32_t prof_iter = 0;
    uint32_t prof_iter_stage = 0;
    #endif

    #if 0
    static int iter = 0;
    if (PRINT_CONDITION(0) && iter==0) {
      iter += 1;
      print("SmemLayoutA: "); print(SmemLayoutA{}); print("\n");
      print("SmemLayoutB: "); print(SmemLayoutB{}); print("\n");
      print("SmemLayoutAtomA: "); print(SmemLayoutAtomA{}); print("\n");
      print("SmemLayoutAtomB: "); print(SmemLayoutAtomB{}); print("\n");
      print("tCrA: "); print(tCrA(0)); print("\n");
      print("tCrB: "); print(tCrB.layout()); print("\n");
      print("tCsA: "); print(tCsA.layout()); print(" "); print(sizeof(tCsA(0,0,0,0))); print("\n");
      print("sA: "); print(sA.layout()); print(" "); print(sizeof(sA(0,0,0))); print("\n");
      print("GmmaDescriptor: "); print(sizeof(GmmaDescriptor)); print("\n");
      print("TileShape        : "); print(TileShape{}); print("\n");
      print("sA_layout        : "); print(sA.layout()); print("\n");
      print("sB_layout        : "); print(sB.layout()); print("\n");
      print("tCsA_layout      : "); print(tCsA.layout()); print("\n");
      print("tCsB_layout      : "); print(tCsB.layout()); print("\n");
    }
    #endif

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                         // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;
    auto pipeline_params = pipeline.get_params();

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);

    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    dim3 cluster_local_block_id = cute::block_id_in_cluster();
    uint32_t src_A_block = cluster_local_block_id.y == 0 ?
                            size<1>(ClusterShape{}) - 1 :
                            cluster_local_block_id.y - 1;
    uint32_t src_B_block = cluster_local_block_id.x == 0 ?
                            size<0>(ClusterShape{}) - 1 :
                            cluster_local_block_id.x - 1; 

    warpgroup_fence_operand(accum);
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count; k_tile_prologue > 0; --k_tile_prologue)
    {
      #if PROFILE
      if ((PRINT_CONDITION(128) || PRINT_CONDITION(256)) && prof_iter < PROFILE_ITER) {
        t0 = get_clock();
      }
      #endif
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      pipeline.consumer_wait(smem_pipe_read);
      #if PROFILE
      if ((PRINT_CONDITION(128) || PRINT_CONDITION(256)) && prof_iter < PROFILE_ITER) {
        t1 = get_clock();
        consumer_wait_latency[prof_iter] = t1 - t0;
        prof_iter++;
      }
      #endif

      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }

      warpgroup_commit_batch();

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      #if PROFILE
      if ((PRINT_CONDITION(128) || PRINT_CONDITION(256)) && prof_iter_stage < PROFILE_ITER) {
        t0 = get_clock();
        if (smem_pipe_read.index() == pipeline_params.dsmem_recv_stage) {
          on_consumer_wait[prof_iter_stage] = t0;
        }
      }
      #endif
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      pipeline.consumer_wait(smem_pipe_read);
      #if PROFILE
      if ((PRINT_CONDITION(128) || PRINT_CONDITION(256)) && prof_iter < PROFILE_ITER) {
        t1 = get_clock();
        consumer_wait_latency[prof_iter] = t1 - t0;
        // prof_iter++;
      }
      #endif

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      if (smem_pipe_release.index() == pipeline_params.dsmem_recv_stage) {
        // reset & arrive on full_barrier after mma
        // but the ending may have bug
        if (threadIdx.x == 128 || threadIdx.x == 256) {
          // may have bug, 2 consumer warpgroup should both arrive
          if (pipeline_params.dsmem_copy_A) {
            pipeline.receiver_arrive_sender(src_A_block, 0);
          }
          if (pipeline_params.dsmem_copy_B) {
            pipeline.receiver_arrive_sender(src_B_block, 1);
          }
        }
      }
      // UNLOCK smem_pipe_release, done _computing_ on it
      pipeline.consumer_release(smem_pipe_release);
      #if PROFILE
      if ((PRINT_CONDITION(128) || PRINT_CONDITION(256)) && prof_iter_stage < PROFILE_ITER) {
        if (smem_pipe_release.index() == pipeline_params.dsmem_recv_stage) {
          mma_release[prof_iter_stage] = get_clock();
          prof_iter_stage++;
        }
        prof_iter++;
      }
      #endif

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    warpgroup_fence_operand(accum);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail( MainloopPipeline pipeline, 
            PipelineState smem_pipe_release, 
            int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;
    dim3 cluster_local_block_id = cute::block_id_in_cluster();
    auto pipeline_params = pipeline.get_params();
    uint32_t src_A_block = cluster_local_block_id.y == 0 ?
                      size<1>(ClusterShape{}) - 1 :
                      cluster_local_block_id.y - 1;
    uint32_t src_B_block = cluster_local_block_id.x == 0 ?
                      size<0>(ClusterShape{}) - 1 :
                      cluster_local_block_id.x - 1; 

    smem_pipe_release.advance(k_tile_count);
    
    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);     // UNLOCK smem_pipe_release, done _computing_ on it
      if (smem_pipe_release.index() == pipeline_params.dsmem_recv_stage) {
        if (threadIdx.x == 128 || threadIdx.x == 256) {
          // may have bug
          if (pipeline_params.dsmem_copy_A) {
            pipeline.receiver_arrive_sender(src_A_block, 0);
          }
          if (pipeline_params.dsmem_copy_B) {
            pipeline.receiver_arrive_sender(src_B_block, 1);
          }
        }
      }
      ++smem_pipe_release;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
