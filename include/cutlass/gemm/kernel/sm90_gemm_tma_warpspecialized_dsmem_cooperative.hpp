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
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class GridSwizzle_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  GridSwizzle_,
  cute::enable_if_t<
    (cute::is_base_of_v<KernelTmaWarpSpecializedCooperativeDSMEM, typename CollectiveMainloop_::DispatchPolicy::Schedule>)>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  using GridSwizzle = GridSwizzle_;
  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
    "Mainloop and epilogue do not agree on accumulator value type.");

  using PersistentTileSchedulerParams = typename detail::PersistentTileSchedulerSm90::Params;
  static_assert(ArchTag::kMinComputeCapability >= 90);

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{}) + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  /// Register requirement for Load and Math WGs
  static constexpr uint32_t LoadRegisterRequirement = 40;
  static constexpr uint32_t MmaRegisterRequirement = 232;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
    } pipelines;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    KernelHardwareInfo hw_info;
    PersistentTileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    (void) workspace;
    auto problem_shape = args.problem_shape;
    if constexpr (detail::IF_SWAP_AB<CollectiveMainloop>::value) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, Int<1>{});

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
      {args.hw_info.device_id, sm_count},
      detail::PersistentTileSchedulerSm90::to_underlying_arguments(problem_shape_MNKL, TileShape{}, ClusterShape{})
    };
  }

  CUTLASS_HOST_DEVICE static
  bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Size don't meet the requirements.\n");
      return implementable;
    }
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    auto M = get<0>(args.problem_shape);
    auto N = get<1>(args.problem_shape);
    auto K = get<2>(args.problem_shape);
    // Contiguous dimension for the TMA tensor should be 128b aligned
    implementable = std::is_same_v<gemm::detail::StrideToLayoutTagA_t<StrideA>, layout::RowMajor> ?
                        K % min_tma_aligned_elements == 0 : M % min_tma_aligned_elements == 0;
    implementable = implementable && (std::is_same_v<gemm::detail::StrideToLayoutTagB_t<StrideB>, layout::RowMajor> ?
                        N % min_tma_aligned_elements == 0 : K % min_tma_aligned_elements == 0);
    implementable = implementable && (!cutlass::epilogue::collective::detail::IF_EPILOGUE_USES_TMA<CollectiveEpilogue>::value ||
                        (cutlass::epilogue::collective::detail::IF_EPILOGUE_USES_TMA<CollectiveEpilogue>::value &&
                        std::is_same_v<gemm::detail::StrideToLayoutTagC_t<StrideC>, layout::RowMajor> ?
                        N % min_tma_aligned_elements == 0 : M % min_tma_aligned_elements == 0));
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
      return implementable;
    }

    constexpr bool is_beta_supported =
      CollectiveEpilogue::ThreadEpilogueOp::kScale == cutlass::epilogue::thread::ScaleType::Default;
    implementable = is_beta_supported || (args.epilogue.thread.beta == 0 && args.epilogue.thread.beta_ptr == nullptr);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Scaling params don't meet ThreadEpilogueOp requirements.\n");
      return implementable;
    }

    return implementable;
  }

  static
  int
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    return detail::PersistentTileSchedulerSm90::get_grid_shape(params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

    // Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
    #if ! defined(__CUDA_ARCH_FEAT_SM90_ALL)
      if constexpr(size<0>(typename TiledMma::AtomShape_MNK{}) == 64) {
        printf("ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. Aborting.\n");
        return;
      }
    #endif

    // Preconditions
    static_assert(size(TiledMma{}) == 256, "Cooperative kernel must have TiledMMA operating using 256 threads.");
    static_assert(size<0>(TileShape{}) >= 128,
        "Cooperative kernel requires Tile Size to be greater than or equal to 128 along the M-dimension.");

    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    /* In the Cooperative kernel, Consumer0 and Consumer1 collaborate on the same tile */
    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int warp_idx   = canonical_warp_idx();
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    int mma_thread_idx = thread_idx % size(TiledMma{});
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    int lane_predicate = cute::elect_one_sync();
    auto stage_num = DispatchPolicy::Stages;

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = size(TiledMma{});
    mainloop_pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
    mainloop_pipeline_params.use_dsmem_copy = 1;
    mainloop_pipeline_params.dsmem_copy_A = 1;
    mainloop_pipeline_params.dsmem_copy_B = 0;
    mainloop_pipeline_params.A_transaction_bytes = CollectiveMainloop::TransactionBytesA;
    mainloop_pipeline_params.B_transaction_bytes = CollectiveMainloop::TransactionBytesB;
    mainloop_pipeline_params.dsmem_send_stage = 0;
    mainloop_pipeline_params.dsmem_recv_stage = stage_num - 1;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params);

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = 1; // 1 thread issues TMA load
    epi_load_pipeline_params.consumer_arv_count = size(TiledMma{});
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();
    int receiver_ready_phase = 1;
    int sender_ready_phase = 0;
    int sender_dsmem_copy_finish_phase = 1;
    int receiver_dsmem_copy_finish_phase = 0;
    int mma_wait_phase = 0;

    auto cluster_wait_fn = [&] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      }
      else {
        __syncthreads();
        return [] () {}; // do nothing
      }
    } ();

    // Separate out problem shape for convenience
    // Optionally append _1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = params.mainloop.tma_load_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
    Tensor mB_nkl = params.mainloop.tma_load_b.get_tma_tensor(make_shape(N,K,L));                            // (n,k,l)

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    TiledMma tiled_mma;
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, blk_shape, make_coord(_,_,_), Step<_1, X,_1>{});          // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, blk_shape, make_coord(_,_,_), Step< X,_1,_1>{});          // (BLK_N,BLK_K,n,k,l)

    // Get pipeline stage increments from tensor shapes
    auto k_tile_count = size<3>(gA_mkl);
    auto c_tile_count = CollectiveEpilogue::get_load_pipe_increment(blk_shape);
    auto d_tile_count = CollectiveEpilogue::get_store_pipe_increment(blk_shape);

    detail::PersistentTileSchedulerSm90 scheduler;
    auto work_tile_info = scheduler.get_current_work(params.scheduler);

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue{params.epilogue};

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    #if PROFILE
    uint64_t consumer_wait_latency[PROFILE_ITER];
    uint64_t issue_copy_B[PROFILE_ITER];
    uint64_t copy_B_done[PROFILE_ITER];
    uint64_t on_consumer_wait[PROFILE_ITER];
    uint64_t mma_release[PROFILE_ITER];
    uint64_t issue_dsmem[PROFILE_ITER];
    #endif

    // int iter = 0;
    if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      while (work_tile_info.is_valid_tile) {
        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

        // Slice with our work tile coordinates to construct mainloop tensor views
        Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                   // (BLK_M,BLK_K,k)
        Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                   // (BLK_N,BLK_K,k)

        auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));

        collective_mainloop.load(
          mainloop_pipeline,
          mainloop_pipe_producer_state,
          gA, params.mainloop.tma_load_a,
          gB, params.mainloop.tma_load_b,
          k_tile_iter, k_tile_count,
          thread_idx,
          receiver_ready_phase,
          sender_ready_phase,
          sender_dsmem_copy_finish_phase,
          receiver_dsmem_copy_finish_phase,
          mma_wait_phase,
          shared_storage.tensors.mainloop
          #if PROFILE
          , issue_copy_B
          , copy_B_done
          , issue_dsmem
          #endif
        );
        // Update starting pipeline state for the next tile
        mainloop_pipe_producer_state.advance(k_tile_count);

        if (collective_epilogue.is_source_needed()) {
          collective_epilogue.load(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            tiled_mma,
            warp_group_thread_idx,
            shared_storage.tensors.epilogue
          );
          // Update starting pipeline state for the next tile
          epi_load_pipe_producer_state.advance(c_tile_count);
        }

        // Get next work tile
        scheduler.advance_to_next_work();
        work_tile_info = scheduler.get_current_work(params.scheduler);
        #if 0
        if (PRINT_CONDITION(0)) {
          print("work_tile_info      : "); 
          print("%d, ", work_tile_info.M_idx);
          print("%d, ", work_tile_info.N_idx);
          print("%d, ", work_tile_info.L_idx);
          print("%u",   work_tile_info.is_valid_tile);
          print("\n");
        }
        #endif
      } // Scheduler work fetch loop
      // Make sure all Consumer Warp Groups have been waited upon
      collective_mainloop.load_tail(mainloop_pipeline, 
                                    mainloop_pipe_producer_state, 
                                    receiver_ready_phase);
      if (collective_epilogue.is_source_needed()) {
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      }

      #if PROFILE
      if (PRINT_CONDITION(0)) {
        for (int i=0; i<PROFILE_ITER; i++) {
          printf("[issue_copy_B]:%lu\n", issue_copy_B[i]);
        }
      }
      if (PRINT_CONDITION(32)) {
        for (int i=0; i<PROFILE_ITER; i++) {
          printf("[issue_dsmem]:%lu\n", issue_dsmem[i]);
        }
      }
      if (PRINT_CONDITION(96)) {
        for (int i=0; i<PROFILE_ITER; i++) {
          printf("[copy_B_done]:%lu\n", copy_B_done[i]);
        }
      }
      #endif

    } // Producer Warp Group End

    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      while (work_tile_info.is_valid_tile) {
        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

        // Allocate the the accumulators for the (M,N) blk_shape
        Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));               // (MMA,MMA_M,MMA_N)

        collective_mainloop.mma(
          mainloop_pipeline,
          mainloop_pipe_consumer_state,
          accumulators,
          k_tile_count,
          mma_thread_idx,
          shared_storage.tensors.mainloop,
          params.mainloop
          #if PROFILE
          , consumer_wait_latency
          , on_consumer_wait
          , mma_release
          #endif
        );

        // Make sure the math instructions are done and free buffers before entering the epilogue
        collective_mainloop.mma_tail(
          mainloop_pipeline,
          mainloop_pipe_consumer_state,
          k_tile_count
        );
        // Update starting mainloop pipeline state for the next tile
        mainloop_pipe_consumer_state.advance(k_tile_count);

        // Epilogue and write to gD
        collective_epilogue.store(
          epi_load_pipeline,
          epi_load_pipe_consumer_state,
          epi_store_pipeline,
          epi_store_pipe_producer_state,
          problem_shape_MNKL,
          blk_shape,
          blk_coord,
          accumulators,
          tiled_mma,
          mma_thread_idx,
          shared_storage.tensors.epilogue
        );
        // Update starting load/store pipeline states for the next tile
        epi_load_pipe_consumer_state.advance(c_tile_count);
        epi_store_pipe_producer_state.advance(d_tile_count);

        // Get next work tile
        scheduler.advance_to_next_work();
        work_tile_info = scheduler.get_current_work(params.scheduler);
      } // Scheduler work fetch loop

      #if PROFILE
      if (PRINT_CONDITION(128)) {
        float sum = 0;
        for (int i=0; i<PROFILE_ITER; i++) {
          printf("[consumer_wait_latency_0]:%lu\n", consumer_wait_latency[i]);
          printf("[on_consumer_wait_0]:%lu\n", on_consumer_wait[i]);
          printf("[mma_release_0]:%lu\n", mma_release[i]);
          sum += consumer_wait_latency[i];
        }
        printf("[consumer_wait_latency_0 MEAN]:%f\n", float(sum/PROFILE_ITER));
      }
      if (PRINT_CONDITION(256)) {
        float sum = 0;
        for (int i=0; i<PROFILE_ITER; i++) {
          printf("[consumer_wait_latency_1]:%lu\n", consumer_wait_latency[i]);
          printf("[on_consumer_wait_1]:%lu\n", on_consumer_wait[i]);
          printf("[mma_release_1]:%lu\n", mma_release[i]);
          sum += consumer_wait_latency[i];
        }
        printf("[consumer_wait_latency_1 MEAN]:%f\n", float(sum/PROFILE_ITER));
      }
      #endif

    } // Consumer Warp Groups End
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
