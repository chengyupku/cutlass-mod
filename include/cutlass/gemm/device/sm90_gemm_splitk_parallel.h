/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*!
  \file
  \brief The universal GEMM accommodates serial reductions, parallel reductions, batched strided, and
    batched array variants.
*/

#pragma once

// common
#include "cutlass/cutlass.h"
#include "cutlass/trace.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"

// 2.x
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/reduction/kernel/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"

// 3.x
#include "cutlass/gemm/kernel/gemm_universal.hpp"


////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::device {

////////////////////////////////////////////////////////////////////////////////

/*!
  GemmUniversalAdapter is a stateful, reusable GEMM handle built around a kernel
  of type cutlass::gemm::kernel::Gemm or cutlass::gemm::kernel::GemmUniversal.

  It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
  to create it from the host facing arguments. For power users, new static methods
  are exposed in 3.x APIs that bypass the stateful methods or args->params lowering.

  It supports kernel types that implement both the 2.x and 3.0 APIs,
  however, this is done by specializing the implementation of GemmUniversalAdapter
  on the two kernel API types, and thus, GemmUniversalAdapter's behaviour might
  differ between the two specializations.
*/

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 3.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <
  class ElementA_,
  class LayoutA_,
  int AlignmentA,
  class ElementB_,
  class LayoutB_,
  int AlignmentB,
  class ElementC_,
  class LayoutC_,
  int AlignmentC,
  class ElementAccumulator_,
  class ArchTag_,
  class OperatorClass_,
  class TileShape_,
  class ClusterShape_,
  class StageCountType_,
  class KernelSchedule_
>
class Sm90GemmSplitKParallel
{
public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  // may have bug
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  using ElementAccumulator = ElementAccumulator_;
  using ArchTag = ArchTag_;
  using OperatorClass = OperatorClass_;
  using TileShape = TileShape_;
  using ClusterShape = ClusterShape_;
  using StageCountType = StageCountType_;
  using KernelSchedule = KernelSchedule_;

  using CollectiveEpilogueBuilder = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementC, LayoutC, AlignmentC,
      cutlass::epilogue::TmaWarpSpecializedCooperativeSplitK
    >;
  using CollectiveEpilogue = typename CollectiveEpilogueBuilder::CollectiveOp;
  using EpilogueOutputOp = typename CollectiveEpilogueBuilder::ThreadOp;

  using CollectiveMainloopBuilder = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
        sizeof(typename CollectiveEpilogue::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeSplitK
    >;
  using CollectiveMainloop = typename CollectiveMainloopBuilder::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int>, // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  /// Reduction kernel
  using ReductionOp = cutlass::reduction::thread::ReduceAdd<
    ElementAccumulator, typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;
  using ReductionKernel = cutlass::reduction::kernel::ReduceSplitK<
    cutlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
    EpilogueOutputOp,
    ReductionOp
  >;

  /// Argument structure: User API
  using GemmArguments = typename GemmKernel::Arguments;
  /// Argument structure: Kernel API
  // using Params = typename GemmKernel::Params;

  struct ReduceArguments {
    int problem_shape_m;
    int problem_shape_n;
    int split_k_slices;
    ElementC const* ptr_C;
    ElementD const* ptr_D;
    typename CollectiveEpilogue::StrideC dC;
    typename CollectiveEpilogue::StrideD dD;
    typename EpilogueOutputOp::Params epilogue;
  };

  struct Arguments {
    GemmArguments gemm_args;
    ReduceArguments reduce_args;
  };

private:

  /// Kernel API parameters object
  typename GemmKernel::Params gemm_params_;
  typename ReductionKernel::Params reduce_params_;

public:

  /// Determines whether the GEMM can execute the given problem.
  static Status
  can_implement(Arguments const& args) {
    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;

    workspace_bytes +=  args.reduce_args.problem_shape_m
                      * args.reduce_args.problem_shape_n
                      * args.reduce_args.split_k_slices
                      * sizeof(ElementAccumulator);

    CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);
    // printf("  workspace_bytes: %d\n", workspace_bytes);

    return workspace_bytes;
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(GemmArguments const& args, void* workspace = nullptr) {
    auto tmp_params = GemmKernel::to_underlying_arguments(args, workspace);
    return GemmKernel::get_grid_shape(tmp_params);
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(typename GemmKernel::Params const& params) {
    return GemmKernel::get_grid_shape(params);
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    CUTLASS_TRACE_HOST("GemmUniversal::maximum_active_blocks()");
    int max_active_blocks = -1;
    int smem_size = GemmKernel::SharedStorageSize;

    // first, account for dynamic smem capacity if needed
    cudaError_t result;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      result = cudaFuncSetAttribute(
          device_kernel<GemmKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        CUTLASS_TRACE_HOST(
          "  cudaFuncSetAttribute() returned error: "
          << cudaGetErrorString(result));
        return -1;
      }
    }

    // query occupancy after setting smem size
    result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        device_kernel<GemmKernel>,
        GemmKernel::MaxThreadsPerBlock,
        smem_size);

    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      CUTLASS_TRACE_HOST(
        "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error: "
        << cudaGetErrorString(result));
      return -1;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

  /// Initializes GEMM state from arguments.
  Status
  gemm_initialize(GemmArguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("GemmUniversal::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    size_t workspace_bytes = GemmKernel::get_workspace_size(args);
    CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

    if (workspace_bytes) {
      if (!workspace) {
        CUTLASS_TRACE_HOST("  error: device workspace must not be null");
        return Status::kErrorWorkspaceNull;
      }

      if (args.mode == GemmUniversalMode::kGemm) {
        CUTLASS_TRACE_HOST("  clearing device workspace");
        cudaError_t result = cudaMemsetAsync(workspace, 0, workspace_bytes, stream);
        if (cudaSuccess != result) {
          result = cudaGetLastError(); // to clear the error bit
          CUTLASS_TRACE_HOST("  cudaMemsetAsync() returned error " << cudaGetErrorString(result));
          return Status::kErrorInternal;
        }
      }
    }

    // Initialize the Params structure
    gemm_params_ = GemmKernel::to_underlying_arguments(args, workspace);

    // account for dynamic smem capacity if needed
    int smem_size = GemmKernel::SharedStorageSize;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      cudaError_t result = cudaFuncSetAttribute(
          device_kernel<GemmKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
        return Status::kErrorInternal;
      }
    }
    return Status::kSuccess;
  }

  /// Initializes Reduce state from arguments.
  Status
  reduce_initialize(ReduceArguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    int64_t partition_stride = int64_t(args.problem_shape_m) * int64_t(args.problem_shape_n);
    TensorRef<ElementAccumulator, layout::RowMajor> ref_workspace(
      static_cast<ElementAccumulator *>(workspace), 
      args.problem_shape_n);
    cutlass::TensorRef ref_C(args.ptr_C, LayoutC::packed({args.problem_shape_m, args.problem_shape_n}));
    cutlass::TensorRef ref_D(args.ptr_D, LayoutD::packed({args.problem_shape_m, args.problem_shape_n}));
    reduce_params_ = typename ReductionKernel::Params(
      MatrixCoord(args.problem_shape_m, args.problem_shape_n),
      args.split_k_slices,
      partition_stride,
      ref_workspace,
      ref_D.non_const_ref(),
      ref_C.non_const_ref(),
      args.epilogue
    );
    return Status::kSuccess;
  }

  Status
  initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    gemm_initialize(args.gemm_args, workspace, stream);
    reduce_initialize(args.reduce_args, workspace, stream);
    return Status::kSuccess;
  }

  /// Update API is preserved in 3.0, but does not guarantee a lightweight update of params.
  // Status
  // update(Arguments const& args, void* workspace = nullptr) {
  //   CUTLASS_TRACE_HOST("GemmUniversal()::update() - workspace: " << workspace);

  //   size_t workspace_bytes = get_workspace_size(args);
  //   if (workspace_bytes > 0 && nullptr == workspace) {
  //     return Status::kErrorWorkspaceNull;
  //   }

  //   params_ = GemmKernel::to_underlying_arguments(args, workspace);
  //   return Status::kSuccess;
  // }

  /// Primary run() entry point API that is static allowing users to create and manage their own params.
  /// Supplied params struct must be construct by calling GemmKernel::to_underling_arguments()
  static Status
  run(typename GemmKernel::Params& gemm_params, 
      typename ReductionKernel::Params& reduce_params ,
      cudaStream_t stream = nullptr) {
    CUTLASS_TRACE_HOST("GemmUniversal::run()");
    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = get_grid_shape(gemm_params);

    // configure smem size and carveout
    int smem_size = GemmKernel::SharedStorageSize;

    Status launch_result;
    // Use extended launch API only for mainloops that use it
    if constexpr(GemmKernel::ArchTag::kMinComputeCapability >= 90) {
      dim3 cluster(cute::size<0>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
                   cute::size<1>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
                   cute::size<2>(typename GemmKernel::DispatchPolicy::ClusterShape{}));
      void const* kernel = (void const*) device_kernel<GemmKernel>;
      void* kernel_params[] = {&gemm_params};
      launch_result = ClusterLauncher::launch(grid, cluster, block, smem_size, stream, kernel, kernel_params);
    }
    else {
      launch_result = Status::kSuccess;
      device_kernel<GemmKernel><<<grid, block, smem_size, stream>>>(gemm_params);
    }

    // block = ReductionKernel::block_shape();
    // grid = ReductionKernel::grid_shape(reduce_params.problem_size);
    // Kernel<ReductionKernel><<< grid, block, 0, stream >>>(reduce_params);

    cudaError_t result = cudaGetLastError();
    if (cudaSuccess == result && Status::kSuccess == launch_result) {
      return Status::kSuccess;
    }
    else {
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }
  }

  //
  // Non-static launch overloads that first create and set the internal params struct of this kernel handle.
  //

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);
    if (Status::kSuccess == status) {
      status = run(gemm_params_, reduce_params_, stream);
    }
    return status;
  }

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  operator()(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    return run(args, workspace, stream);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  run(cudaStream_t stream = nullptr) {
    return run(gemm_params_, reduce_params_, stream);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  operator()(cudaStream_t stream = nullptr) {
    return run(gemm_params_, reduce_params_, stream);
  }
};

} // namespace cutlass::gemm::device

////////////////////////////////////////////////////////////////////////////////
