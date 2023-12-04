#define SIMULATE_MULTIPLE 13
#define Dsmem_ACC 6

#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include "codegen_include/async_pipeline.hpp"
#include "codegen_include/gemm_kernel_simulate_gmem.hpp"

#include "helper.h"

#define TEST_CORRECTNESS 0
#define TEST_ROUND 10

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::half_t;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::ColumnMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::half_t;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::RowMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = cutlass::half_t;                                // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;       // Kernel to launch based on the default setting in the Collective Builder 
using ClusterShape = Shape<_2,_4,_1>; // Shape of the threadblocks in a cluster
static constexpr int PatternLen = 8;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecializedCooperative
  >::CollectiveOp;

static_assert(is_static<TileShape>::value);
static_assert(is_static<ClusterShape>::value);
static_assert(cutlass::gemm::collective::detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, cutlass::gemm::collective::detail::tma_alignment_bytes>(),
            "Should meet TMA alignment requirement\n");

// For fp32 types, map to tf32 MMA value type
using MmaElementA = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
using MmaElementB = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

static constexpr cute::GMMA::Major GmmaMajorA = cutlass::gemm::collective::detail::gmma_ss_tag_to_major_A<MmaElementA, LayoutA>();
static constexpr cute::GMMA::Major GmmaMajorB = cutlass::gemm::collective::detail::gmma_ss_tag_to_major_B<MmaElementB, LayoutB>();

using AtomLayoutMNK =
    Layout<Shape<_2,_1,_1>>;

using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
    MmaElementA, MmaElementB, ElementAccumulator, TileShape, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

// force not to use TMA multicast 
using GmemTiledCopyA = decltype(cute::SM90_TMA_LOAD{});
using GmemTiledCopyB = decltype(cute::SM90_TMA_LOAD{});

using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GmmaMajorA, MmaElementA, decltype(cute::get<0>(TileShape{})), decltype(cute::get<2>(TileShape{}))>());
using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GmmaMajorB, MmaElementB, decltype(cute::get<1>(TileShape{})), decltype(cute::get<2>(TileShape{}))>());

static constexpr int PipelineStages = cutlass::gemm::collective::detail::compute_stage_count_or_override<cutlass::gemm::collective::detail::sm90_smem_capacity_bytes,
    MmaElementA, MmaElementB, TileShape>(cutlass::gemm::collective::StageCountAutoCarveout<
    sizeof(typename CollectiveEpilogue::SharedStorage)>{});

using DispatchPolicy = cutlass::gemm::MainloopSm90TmaGmmaWarpSpecializedDSMEM<
    PipelineStages, ClusterShape, cutlass::gemm::KernelTmaWarpSpecializedCooperativeDSMEM>;

using SmemCopyAtomA = void; 
using SmemCopyAtomB = void;

struct block_iter_id {
  int8_t x, y, iter;
};
__device__ constexpr int8_t tile_order[2][4][PatternLen] = {
		{
	    {0, 1, 4, 5, 6, 7, 2, 3},
	    {1, 0, 4, 5, 6, 7, 3, 2},
	    {0, 1, 4, 5, 6, 7, 2, 3},
	    {1, 0, 4, 5, 6, 7, 3, 2},
		},
		{
	    {1, 0, 4, 5, 6, 7, 3, 2},
	    {0, 1, 4, 5, 6, 7, 2, 3},
	    {1, 0, 4, 5, 6, 7, 3, 2},
	    {0, 1, 4, 5, 6, 7, 2, 3},
		},
	};
__device__ constexpr block_iter_id src_A[2][4][PatternLen] = {
		{
	    {{-1, -1, -1},{0, 1, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 1, 6},},
	    {{-1, -1, -1},{0, 0, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 0, 6},},
	    {{-1, -1, -1},{0, 3, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 3, 6},},
	    {{-1, -1, -1},{0, 2, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 2, 6},},
		},
		{
	    {{-1, -1, -1},{1, 1, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 1, 6},},
	    {{-1, -1, -1},{1, 0, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 0, 6},},
	    {{-1, -1, -1},{1, 3, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 3, 6},},
	    {{-1, -1, -1},{1, 2, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 2, 6},},
		},
	};
__device__ constexpr block_iter_id src_B[2][4][PatternLen] = {
		{
	    {{-1, -1, -1},{1, 0, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 0, 6},},
	    {{-1, -1, -1},{1, 1, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 1, 6},},
	    {{-1, -1, -1},{1, 2, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 2, 6},},
	    {{-1, -1, -1},{1, 3, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 3, 6},},
		},
		{
	    {{-1, -1, -1},{0, 0, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 0, 6},},
	    {{-1, -1, -1},{0, 1, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 1, 6},},
	    {{-1, -1, -1},{0, 2, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 2, 6},},
	    {{-1, -1, -1},{0, 3, 0},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 3, 6},},
		},
	};
__device__ constexpr block_iter_id dst_A[2][4][PatternLen] = {
		{
	    {{0, 1, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 1, 7},{-1, -1, -1},},
	    {{0, 0, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 0, 7},{-1, -1, -1},},
	    {{0, 3, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 3, 7},{-1, -1, -1},},
	    {{0, 2, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 2, 7},{-1, -1, -1},},
		},
		{
	    {{1, 1, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 1, 7},{-1, -1, -1},},
	    {{1, 0, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 0, 7},{-1, -1, -1},},
	    {{1, 3, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 3, 7},{-1, -1, -1},},
	    {{1, 2, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 2, 7},{-1, -1, -1},},
		},
	};
__device__ constexpr block_iter_id dst_B[2][4][PatternLen] = {
		{
	    {{1, 0, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 0, 7},{-1, -1, -1},},
	    {{1, 1, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 1, 7},{-1, -1, -1},},
	    {{1, 2, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 2, 7},{-1, -1, -1},},
	    {{1, 3, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{1, 3, 7},{-1, -1, -1},},
		},
		{
	    {{0, 0, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 0, 7},{-1, -1, -1},},
	    {{0, 1, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 1, 7},{-1, -1, -1},},
	    {{0, 2, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 2, 7},{-1, -1, -1},},
	    {{0, 3, 1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{-1, -1, -1},{0, 3, 7},{-1, -1, -1},},
		},
	};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM collective code
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <
  int Stages,
  int PatternLen_,
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
struct CollectiveMmaGen
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedDSMEM<Stages, ClusterShape, KernelSchedule>;
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

  static const int PatternLen = PatternLen_;
  
  using MainloopPipeline = cutlass::PipelineTmaDsmemAsyncGen<
                             DispatchPolicy::Stages,
                             PatternLen,
                             typename DispatchPolicy::ClusterShape>;
  using PhysicalPipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;
  using LogicalPipelineState = cutlass::PipelineState<PatternLen>;
  using SepPipelineState = cutlass::SeparatePipelineState<PatternLen>;

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
    struct ScheduleStorage : cute::aligned_struct<128> {
      int8_t tileOrder[2][4][PatternLen];
      block_iter_id srcA[2][4][PatternLen];
      block_iter_id srcB[2][4][PatternLen];
      block_iter_id dstA[2][4][PatternLen];
      block_iter_id dstB[2][4][PatternLen];
    } schedules;

    struct TensorStorage : cute::aligned_struct<128> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;
  using ScheduleStorage = typename SharedStorage::ScheduleStorage;

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
    (void) workspace;

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

  CUTLASS_DEVICE void static
  init_schedule(ScheduleStorage& shared_schedules) {
    if (threadIdx.x < 64) {
      shared_schedules.tileOrder[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8]  = tile_order[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8];
      shared_schedules.srcA[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].x     = src_A[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].x;
      shared_schedules.srcA[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].y     = src_A[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].y;
      shared_schedules.srcA[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].iter  = src_A[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].iter;
      shared_schedules.srcB[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].x     = src_B[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].x;
      shared_schedules.srcB[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].y     = src_B[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].y;
      shared_schedules.srcB[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].iter  = src_B[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].iter;
      shared_schedules.dstA[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].x     = dst_A[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].x;
      shared_schedules.dstA[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].y     = dst_A[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].y;
      shared_schedules.dstA[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].iter  = dst_A[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].iter;
      shared_schedules.dstB[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].x     = dst_B[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].x;
      shared_schedules.dstB[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].y     = dst_B[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].y;
      shared_schedules.dstB[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].iter  = dst_B[(threadIdx.x / 32) % 2][(threadIdx.x / 8) % 4][threadIdx.x % 8].iter;
    }
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
      PhysicalPipelineState smem_pipe_write_physical,
      LogicalPipelineState smem_pipe_write_logical,
      PhysicalPipelineState smem_pipe_dsmem_send,
      TensorA const& gA, TMA_LOAD_A& tma_load_a,
      TensorB const& gB, TMA_LOAD_B& tma_load_b,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      SepPipelineState receiver_ready_state_A,
      SepPipelineState receiver_ready_state_B,
      int& sender_ready_phase,
      int& sender_dsmem_copy_finish_phase,
      int& receiver_dsmem_copy_finish_phase,
      int& mma_wait_phase,
      TensorStorage& shared_tensors,
      ScheduleStorage& shared_schedules,
      TensorDummyA const& dgtA,
      TensorDummyB const& dgtB
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

      // Previous tile loop: last iteration logical pipeline left ((PatternLen - (k_tile_count % PatternLen)) % PatternLen) steps
      int prev_left_steps = (PatternLen - (k_tile_count % PatternLen)) % PatternLen;


      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count)
      {
        uint32_t dummy_transaction_bytes_A = size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof(ElementA));
        uint32_t dummy_transaction_bytes_B = size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof(ElementB));
        int write_stage = smem_pipe_write_physical.index();
        block_iter_id src_id_A = shared_schedules.srcA[bid.x][bid.y][k_iter % PatternLen];
        block_iter_id src_id_B = shared_schedules.srcB[bid.x][bid.y][k_iter % PatternLen];
        // LOCK smem_pipe_write_physical for _writing_
        pipeline.wait_empty(smem_pipe_write_physical, eA);
        pipeline.wait_empty(smem_pipe_write_physical, eB);

        if (src_id_A.x == -1 || src_id_A.y == -1) {
          pipeline.copy_prepare(smem_pipe_write_logical, eA, (size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof(ElementA))) + dummy_transaction_bytes_A * SIMULATE_MULTIPLE);
        }
        if (src_id_B.x == -1 || src_id_B.y == -1) {
          pipeline.copy_prepare(smem_pipe_write_logical, eB, (size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof(ElementB))) + dummy_transaction_bytes_B * SIMULATE_MULTIPLE);
        }

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_A_barrier = pipeline.producer_get_barrier(smem_pipe_write_logical, eA);
        BarrierType* tma_B_barrier = pipeline.producer_get_barrier(smem_pipe_write_logical, eB);

        int k_tile_iter_AB = (k_iter / PatternLen) * PatternLen + shared_schedules.tileOrder[bid.x][bid.y][k_iter % PatternLen];

        // Check if this stage was sender on iteration (k_iter - K_PIPE_MAX)
        // If true, wait until the copy is done
        if (src_id_A.x == -1 || src_id_A.y == -1) {
          if (( shared_schedules.dstA[bid.x][bid.y][((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen].x != -1 && 
                shared_schedules.dstA[bid.x][bid.y][((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen].y != -1)) {
            pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eA, k_iter % PatternLen);
          }
          for (int i=0; i<SIMULATE_MULTIPLE; i++) {
            gmem2cta_copy_kernel(dgtA(k_iter,i,_).data().get(), tAsA(_,_,_,write_stage).data().get(), tma_A_barrier, dummy_transaction_bytes_A);
          }
          // TMA load A from gmem to smem
          copy(tma_load_a.with(*tma_A_barrier, mcast_mask_a), tAgA(_,_,_,k_tile_iter_AB), tAsA(_,_,_,write_stage));
        }
        if (src_id_B.x == -1 || src_id_B.y == -1) {
          if (( shared_schedules.dstB[bid.x][bid.y][((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen].x != -1 && 
                shared_schedules.dstB[bid.x][bid.y][((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen].y != -1)) {
            pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eB, k_iter % PatternLen);
          }
          for (int i=0; i<SIMULATE_MULTIPLE; i++) {
            gmem2cta_copy_kernel(dgtB(k_iter,i,_).data().get(), tBsB(_,_,_,write_stage).data().get(), tma_B_barrier, dummy_transaction_bytes_B);
          }
          // TMA load B from gmem to smem
          copy(tma_load_b.with(*tma_B_barrier, mcast_mask_b), tBgB(_,_,_,k_tile_iter_AB), tBsB(_,_,_,write_stage));
        }

        ++k_iter;

        // Advance smem_pipe_write_physical
        ++smem_pipe_write_physical;
        ++smem_pipe_write_logical;

        if (((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen == 0) {
          sender_dsmem_copy_finish_phase ^= 1;
        }
      }

      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 0; i < prev_left_steps; i++)
      {
        block_iter_id src_id_A = shared_schedules.srcA[bid.x][bid.y][k_iter % PatternLen];
        block_iter_id src_id_B = shared_schedules.srcB[bid.x][bid.y][k_iter % PatternLen];
        pipeline.copy_prepare(smem_pipe_write_logical, eA, 0, true);
        pipeline.copy_prepare(smem_pipe_write_logical, eB, 0, true);

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_A_barrier = pipeline.producer_get_barrier(smem_pipe_write_logical, eA);
        BarrierType* tma_B_barrier = pipeline.producer_get_barrier(smem_pipe_write_logical, eB);

        if (src_id_A.x == -1 || src_id_A.y == -1) {
          if (( shared_schedules.dstA[bid.x][bid.y][((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen].x != -1 && 
                shared_schedules.dstA[bid.x][bid.y][((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen].y != -1)) {
            pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eA, k_iter % PatternLen);
          }
        }
        if (src_id_B.x == -1 || src_id_B.y == -1) {
          if (( shared_schedules.dstB[bid.x][bid.y][((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen].x != -1 && 
                shared_schedules.dstB[bid.x][bid.y][((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen].y != -1)) {
            pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eB, k_iter % PatternLen);
          }
        }

        ++k_iter;
        ++smem_pipe_write_physical;
        ++smem_pipe_write_logical;

        if (((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen == 0) {
          sender_dsmem_copy_finish_phase ^= 1;
        }
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
      int prev_left_steps = (PatternLen - (k_tile_count % PatternLen)) % PatternLen;

      int sep_stage_A = 0;
      int sep_stage_B = 0;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i=0; i<PatternLen; i++) {
        sep_stage_A = i;
        if (shared_schedules.dstA[bid.x][bid.y][i].iter >= K_PIPE_MAX) {
          break;
        }
      }
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i=0; i<PatternLen; i++) {
        sep_stage_B = i;
        if (shared_schedules.dstB[bid.x][bid.y][i].iter >= K_PIPE_MAX) {
          break;
        }
      }
      receiver_ready_state_A.set_sep_stage(sep_stage_A);
      receiver_ready_state_B.set_sep_stage(sep_stage_B);

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count)
      {
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        block_iter_id dst_id_A = shared_schedules.dstA[bid.x][bid.y][k_iter % PatternLen];
        block_iter_id dst_id_B = shared_schedules.dstB[bid.x][bid.y][k_iter % PatternLen];
        int dst_A_stage = (dst_id_A.iter - (k_iter % PatternLen) + smem_pipe_dsmem_send.index()) % K_PIPE_MAX;
        int dst_B_stage = (dst_id_B.iter - (k_iter % PatternLen) + smem_pipe_dsmem_send.index()) % K_PIPE_MAX;

        // wait receiver's arrive
        if (dst_id_A.x != -1 && dst_id_A.y != -1) {
          // wait sender buffer ready, may have bug (double consumer wait?)
          pipeline.sender_wait_sender_ready(sender_ready_phase, eA, k_iter % PatternLen);
          pipeline.sender_wait_receiver_ready(receiver_ready_state_A, eA);
          if (shared_schedules.dstA[dst_id_A.x][dst_id_A.y][((dst_id_A.iter - K_PIPE_MAX) + PatternLen) % PatternLen].x != -1 && 
              shared_schedules.dstA[dst_id_A.x][dst_id_A.y][((dst_id_A.iter - K_PIPE_MAX) + PatternLen) % PatternLen].y != -1) {
            pipeline.sync_wait(sender_ready_phase, eA, k_iter % PatternLen);
          }
          uint32_t block_id = dst_id_A.x + dst_id_A.y * size<0>(ClusterShape{});
          pipeline.dsmem_copy_prepare(((TransactionBytesA / Dsmem_ACC) / 128) * 128, block_id, eA, dst_id_A.iter % PatternLen);
          BarrierType* tma_A_barrier = pipeline.producer_get_barrier_by_stage(dst_id_A.iter % PatternLen, eA);
          dsmem_copy_func(block_id,
                          tAsA(_,_,_,smem_pipe_dsmem_send.index()).data().get(), 
                          tAsA(_,_,_,dst_A_stage).data().get(), 
                          tma_A_barrier, 
                          ((TransactionBytesA / Dsmem_ACC) / 128) * 128
                          );
        }
        if (dst_id_B.x != -1 && dst_id_B.y != -1) {
          // wait sender buffer ready, may have bug (double consumer wait?)
          pipeline.sender_wait_sender_ready(sender_ready_phase, eB, k_iter % PatternLen);
          pipeline.sender_wait_receiver_ready(receiver_ready_state_B, eB);
          if (shared_schedules.dstB[dst_id_B.x][dst_id_B.y][((dst_id_B.iter - K_PIPE_MAX) + PatternLen) % PatternLen].x != -1 && 
              shared_schedules.dstB[dst_id_B.x][dst_id_B.y][((dst_id_B.iter - K_PIPE_MAX) + PatternLen) % PatternLen].y != -1) {
            pipeline.sync_wait(sender_ready_phase, eB, k_iter % PatternLen);
          }
          uint32_t block_id = dst_id_B.x + dst_id_B.y * size<0>(ClusterShape{});
          pipeline.dsmem_copy_prepare(((TransactionBytesB / Dsmem_ACC) / 128) * 128, block_id, eB, dst_id_B.iter % PatternLen);
          BarrierType* tma_B_barrier = pipeline.producer_get_barrier_by_stage(dst_id_B.iter % PatternLen, eB);
          dsmem_copy_func(block_id,
                          tBsB(_,_,_,smem_pipe_dsmem_send.index()).data().get(), 
                          tBsB(_,_,_,dst_B_stage).data().get(), 
                          tma_B_barrier, 
                          ((TransactionBytesB / Dsmem_ACC) / 128) * 128
                          );
        }
        ++k_iter;
        ++receiver_ready_state_A;
        ++receiver_ready_state_B;
        ++smem_pipe_dsmem_send;
        if (k_iter % PatternLen == 0) {
          sender_ready_phase ^= 1;
        }
      }

      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 0; i < prev_left_steps; i++)
      {
        block_iter_id dst_id_A = shared_schedules.dstA[bid.x][bid.y][k_iter % PatternLen];
        block_iter_id dst_id_B = shared_schedules.dstB[bid.x][bid.y][k_iter % PatternLen];

        // wait receiver's arrive
        if (dst_id_A.x != -1 && dst_id_A.y != -1) {
          // wait sender buffer ready, may have bug (double consumer wait?)
          pipeline.sender_wait_sender_ready(sender_ready_phase, eA, k_iter % PatternLen);
          pipeline.sender_wait_receiver_ready(receiver_ready_state_A, eA);
          if (shared_schedules.dstA[dst_id_A.x][dst_id_A.y][((dst_id_A.iter - K_PIPE_MAX) + PatternLen) % PatternLen].x != -1 && 
              shared_schedules.dstA[dst_id_A.x][dst_id_A.y][((dst_id_A.iter - K_PIPE_MAX) + PatternLen) % PatternLen].y != -1) {
            pipeline.sync_wait(sender_ready_phase, eA, k_iter % PatternLen);
          }
        }
        if (dst_id_B.x != -1 && dst_id_B.y != -1) {
          // wait sender buffer ready, may have bug (double consumer wait?)
          pipeline.sender_wait_sender_ready(sender_ready_phase, eB, k_iter % PatternLen);
          pipeline.sender_wait_receiver_ready(receiver_ready_state_B, eB);
          if (shared_schedules.dstB[dst_id_B.x][dst_id_B.y][((dst_id_B.iter - K_PIPE_MAX) + PatternLen) % PatternLen].x != -1 && 
              shared_schedules.dstB[dst_id_B.x][dst_id_B.y][((dst_id_B.iter - K_PIPE_MAX) + PatternLen) % PatternLen].y != -1) {
            pipeline.sync_wait(sender_ready_phase, eB, k_iter % PatternLen);
          }
        }

        ++k_iter;
        ++receiver_ready_state_A;
        ++receiver_ready_state_B;
        if (k_iter % PatternLen == 0) {
          sender_ready_phase ^= 1;
        }
      }
    }

    // monitor dsmem copy of A
    if (warp_idx_in_warp_group == 2 and lane_predicate) {
      dim3 bid = cute::block_id_in_cluster();
      int k_iter = 0;
      int prev_left_steps = (PatternLen - (k_tile_count % PatternLen)) % PatternLen;
      k_tile_count +=  prev_left_steps;
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        block_iter_id src_id_A = shared_schedules.srcA[bid.x][bid.y][k_iter % PatternLen];
        block_iter_id src_id_B = shared_schedules.srcB[bid.x][bid.y][k_iter % PatternLen];
        // Copy on this iteration is from dsmem
        if (src_id_A.x != -1 && src_id_A.y != -1) {
          uint32_t block_id = src_id_A.x + src_id_A.y * size<0>(ClusterShape{});
          pipeline.receiver_wait_dsmem_copy_finish(receiver_dsmem_copy_finish_phase, eA, k_iter % PatternLen);
          pipeline.receiver_arrive_dsmem_copy_finish(block_id, eA, (src_id_A.iter + K_PIPE_MAX) % PatternLen);
        }
        if (src_id_B.x != -1 && src_id_B.y != -1) {
          uint32_t block_id = src_id_B.x + src_id_B.y * size<0>(ClusterShape{});
          pipeline.receiver_wait_dsmem_copy_finish(receiver_dsmem_copy_finish_phase, eB, k_iter % PatternLen);
          pipeline.receiver_arrive_dsmem_copy_finish(block_id, eB, (src_id_B.iter + K_PIPE_MAX) % PatternLen);
        }
        ++k_iter;
        if (k_iter % PatternLen == 0) {
          receiver_dsmem_copy_finish_phase ^= 1;
        }
      }
    }

    if (warp_idx_in_warp_group == 3 and lane_predicate) {
      dim3 bid = cute::block_id_in_cluster();
      int k_iter = 0;
      int prev_left_steps = (PatternLen - (k_tile_count % PatternLen)) % PatternLen;
      k_tile_count +=  prev_left_steps;
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        if (( shared_schedules.dstA[bid.x][bid.y][((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen].x != -1 && 
              shared_schedules.dstA[bid.x][bid.y][((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen].y != -1)) {
          pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eA, k_iter % PatternLen);
          block_iter_id src_id_A = shared_schedules.srcA[bid.x][bid.y][k_iter % PatternLen];
          if (src_id_A.x != -1 && src_id_A.y != -1) {
            uint32_t block_id = src_id_A.x + src_id_A.y * size<0>(ClusterShape{});
            pipeline.sync_arrive(block_id, eA, src_id_A.iter);
          }
        }
        if (( shared_schedules.dstB[bid.x][bid.y][((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen].x != -1 && 
              shared_schedules.dstB[bid.x][bid.y][((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen].y != -1)) {
          pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eB, k_iter % PatternLen);
          block_iter_id src_id_B = shared_schedules.srcB[bid.x][bid.y][k_iter % PatternLen];
          if (src_id_B.x != -1 && src_id_B.y != -1) {
            uint32_t block_id = src_id_B.x + src_id_B.y * size<0>(ClusterShape{});
            pipeline.sync_arrive(block_id, eB, src_id_B.iter);
          }
        }
        ++k_iter;
        if (((k_iter - K_PIPE_MAX) % PatternLen + PatternLen) % PatternLen == 0) {
          sender_dsmem_copy_finish_phase ^= 1;
        }
      }
    }
  }


/// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline)
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
      pipeline.wait_mma_finish(0);
    }
  }


/// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PhysicalPipelineState smem_pipe_read_physical,
      LogicalPipelineState smem_pipe_read_logical,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      ScheduleStorage& shared_schedules,
      Params const& mainloop_params
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
    PhysicalPipelineState smem_pipe_release = smem_pipe_read_physical;
    auto pipeline_params = pipeline.get_params();

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);

    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    dim3 bid = cute::block_id_in_cluster();

    warpgroup_fence_operand(accum);
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count; k_tile_prologue > 0; --k_tile_prologue)
    {
      // WAIT on smem_pipe_read_physical until its data are available (phase bit flips from rdPhaseBit value)
      pipeline.consumer_wait(smem_pipe_read_logical);

      int read_stage = smem_pipe_read_physical.index();
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }

      warpgroup_commit_batch();

      ++smem_pipe_read_physical;
      ++smem_pipe_read_logical;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;
    int k_release_iter = 0;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // WAIT on smem_pipe_read_physical until its data are available (phase bit flips from rdPhaseBit value)
      pipeline.consumer_wait(smem_pipe_read_logical);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read_physical.index();
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

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write_physical is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      // Use dsmem as input when using this stage the next time
      if (threadIdx.x == 128 || threadIdx.x == 256) {
        block_iter_id src_id_A = shared_schedules.srcA[bid.x][bid.y][(k_release_iter + K_PIPE_MAX) % PatternLen];
        block_iter_id src_id_B = shared_schedules.srcB[bid.x][bid.y][(k_release_iter + K_PIPE_MAX) % PatternLen];
        if (src_id_A.x != -1 && src_id_A.y != -1) {
          uint32_t block_id = src_id_A.x + src_id_A.y * size<0>(ClusterShape{});
          pipeline.receiver_arrive_sender(block_id, eA, src_id_A.iter);
        }
        if (src_id_B.x != -1 && src_id_B.y != -1) {
          uint32_t block_id = src_id_B.x + src_id_B.y * size<0>(ClusterShape{});
          pipeline.receiver_arrive_sender(block_id, eB, src_id_B.iter);
        }
      }
      pipeline.consumer_release(smem_pipe_release, eA);
      pipeline.consumer_release(smem_pipe_release, eB);

      // Advance smem_pipe_read_physical and smem_pipe_release
      ++smem_pipe_read_physical;
      ++smem_pipe_read_logical;
      ++smem_pipe_release;
      ++k_release_iter;
    }

    warpgroup_fence_operand(accum);
  }


/// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail( MainloopPipeline pipeline, 
            PhysicalPipelineState smem_pipe_release, 
            LogicalPipelineState smem_pipe_read_logical,
            ScheduleStorage& shared_schedules,
            int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    int prev_left_steps = (PatternLen - (k_tile_count % PatternLen)) % PatternLen;
    smem_pipe_read_logical.advance(k_tile_count);
    k_tile_count -= prologue_mma_count;
    int k_iter = k_tile_count;
    dim3 bid = cute::block_id_in_cluster();
    auto pipeline_params = pipeline.get_params();

    smem_pipe_release.advance(k_tile_count);
    
    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      block_iter_id src_id_A = shared_schedules.srcA[bid.x][bid.y][(k_iter + K_PIPE_MAX) % PatternLen];
      block_iter_id src_id_B = shared_schedules.srcB[bid.x][bid.y][(k_iter + K_PIPE_MAX) % PatternLen];
      // pipeline.consumer_release(smem_pipe_release);     // UNLOCK smem_pipe_release, done _computing_ on it
      if (threadIdx.x == 128 || threadIdx.x == 256) {
        if (src_id_A.x != -1 && src_id_A.y != -1) {
          uint32_t block_id = src_id_A.x + src_id_A.y * size<0>(ClusterShape{});
          pipeline.receiver_arrive_sender(block_id, eA, src_id_A.iter);
        }
        if (src_id_B.x != -1 && src_id_B.y != -1) {
          uint32_t block_id = src_id_B.x + src_id_B.y * size<0>(ClusterShape{});
          pipeline.receiver_arrive_sender(block_id, eB, src_id_B.iter);
        }
      }
      pipeline.consumer_release(smem_pipe_release, eA);
      pipeline.consumer_release(smem_pipe_release, eB);
      ++smem_pipe_release;
      ++k_iter;
    }

    for (int i = 0; i < prev_left_steps; ++i) {
      pipeline.consumer_wait(smem_pipe_read_logical);
      block_iter_id src_id_A = shared_schedules.srcA[bid.x][bid.y][(k_iter + K_PIPE_MAX) % PatternLen];
      block_iter_id src_id_B = shared_schedules.srcB[bid.x][bid.y][(k_iter + K_PIPE_MAX) % PatternLen];
      if (threadIdx.x == 128 || threadIdx.x == 256) {
        if (src_id_A.x != -1 && src_id_A.y != -1) {
          uint32_t block_id = src_id_A.x + src_id_A.y * size<0>(ClusterShape{});
          pipeline.receiver_arrive_sender(block_id, eA, src_id_A.iter);
        }
        if (src_id_B.x != -1 && src_id_B.y != -1) {
          uint32_t block_id = src_id_B.x + src_id_B.y * size<0>(ClusterShape{});
          pipeline.receiver_arrive_sender(block_id, eB, src_id_B.iter);
        }
      }
      ++k_iter;
      ++smem_pipe_read_logical;
    }
  }

  CUTLASS_DEVICE void
  mma_finish(MainloopPipeline pipeline) {
    if (threadIdx.x == 128 || threadIdx.x == 256) {
      pipeline.mma_finish();
    }
  }


};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMmaGen<
      PipelineStages,
      PatternLen,
      ClusterShape,
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeDSMEM,
      TileShape,
      ElementA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;

using GemmKernel = cutlass::gemm::kernel::GemmUniversalGenSimGmem<
    Shape<int,int,int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename cutlass::half_t> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;

  Options():
    help(false),
    m(16384), n(16384), k(16384),
    alpha(1.f), beta(0.f),
    iterations(10)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "48_hopper_warp_specialized_gemm\n\n"
      << "  Hopper FP32 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "48_hopper_warp_specialized_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  } else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  } else {
    scope_max = 8;
    scope_min = -8;
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, Int<1>{}));
  stride_B = make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, Int<1>{}));
  stride_C = make_cute_packed_stride(StrideC{}, cute::make_shape(options.m, options.n, Int<1>{}));
  stride_D = make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, Int<1>{}));

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_C.reset(options.m * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D}
  };

  return arguments;
}

bool verify(const Options &options) {
  cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
  cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.n, options.k}));
  cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
  cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));

  //
  // Compute reference output
  //

  // Create instantiation for device reference gemm kernel
  DeviceGemmReference gemm_reference;

  // Launch device reference gemm kernel
  gemm_reference(
    {options.m, options.n, options.k},
    ElementAccumulator(options.alpha),
    ref_A,
    ref_B,
    ElementAccumulator(options.beta),
    ref_C,
    ref_D);

  // Wait for kernel to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

  return passed;
  // return true;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options, int& passed, int& failed)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  // std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    failed++;
  }
  else {
    passed++;
  }

  // Run profiling loop
#if !TEST_CORRECTNESS
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }
#endif

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////



int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  cudaGetDevice(&current_device_id);
  cudaGetDeviceProperties(&props, current_device_id);
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major < 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //
  int passed = 0;
  int failed = 0;
#if TEST_CORRECTNESS
  #if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    for (int i=0; i<TEST_ROUND; i++) {
      run<Gemm>(options, passed, failed);
    }
    std::cout << "Passed:" << passed << ", failed:" << failed << std::endl;
  #endif
#else
  #if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    run<Gemm>(options, passed, failed);
  #endif
#endif
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

