#include "cutlass/pipeline/sm90_pipeline.hpp"

#define eA 0
#define eB 1

namespace cutlass {

using namespace cute;

// Circular Buffer Index + Associated Phase
// Assumes only one operation possible - i.e., ++
template<uint32_t Stages_>
struct SeparatePipelineState {

  static constexpr uint32_t Stages = Stages_;

private:
  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t phase_count_ = 0;
  // When get to sep_stage_, flip the phase
  uint32_t sep_stage_ = 0;

public:
  CUTLASS_DEVICE
  SeparatePipelineState(): index_{}, phase_{}, phase_count_{}, sep_stage_{} {}

  CUTLASS_DEVICE
  SeparatePipelineState(int index, uint32_t phase, uint32_t phase_count, uint32_t sep_stage=0)
    : index_(index)
    , phase_(phase)
    , phase_count_(phase_count)
    , sep_stage_(sep_stage) {}

  CUTLASS_DEVICE
  void set_sep_stage(uint32_t sep_stage) {
    sep_stage_ = sep_stage;
  }

  CUTLASS_DEVICE
  int index() const {
    return index_;
  }

  CUTLASS_DEVICE
  uint32_t phase() const {
    return phase_;
  }

  CUTLASS_DEVICE
  uint32_t phase_count() const {
    return phase_count_;
  }

  CUTLASS_DEVICE
  uint32_t sep_stage() const {
    return sep_stage_;
  }

  CUTLASS_DEVICE
  void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      if (index_ == Stages) {
        index_ = 0;
      }
      if (index_ == sep_stage_) {
        phase_ ^= 1;
        ++phase_count_;
      }
    }
  }

  CUTLASS_DEVICE
  SeparatePipelineState& operator=(const SeparatePipelineState& other) {
    index_ = other.index();
    phase_ = other.phase();
    phase_count_ = other.phase_count();
    sep_stage_ = other.sep_stage();
    return *this;
  }

  CUTLASS_DEVICE
  SeparatePipelineState advance(uint32_t num_iterations) {
    if constexpr (Stages > 0) {
      // Number of iterations cross over the stage boundary => flipped phase
      if ((num_iterations < Stages) && (index_ + num_iterations) % Stages >= sep_stage_) {
        phase_ ^= 1;
      }
      // How many times number of iterations cross over the stage boundary and
      // end up on a odd number => flipped phase
      if ((num_iterations >= Stages) && (((index_ + num_iterations - sep_stage_) / Stages) % 2) == 1) {
        phase_ ^= 1;
      }
      phase_count_ += (index_ + num_iterations - sep_stage_) / Stages;
      index_ = (index_ + num_iterations) % Stages;
    }
    return *this;
  }

  CUTLASS_DEVICE
  static SeparatePipelineState make_pipeline_state(SeparatePipelineState start_state, uint32_t num_iterations) {
    return start_state.advance(num_iterations);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMA load wiat DSMEM Copy (producer) Async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////
// Assumptions : Constructor is visible Cluster-wide (as it needs a Cluster-Sync)
// We have exactly one thread elected in the Producer as the "leader"
// Currently, it is optional to elect a leader for the Consumers
template <
  int Stages_,
  int PatternLen_,
  class ClusterShape_
>
class PipelineTmaDsmemAsyncGen {
public :
  using ClusterShape = ClusterShape_;
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using SignalBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;
  static constexpr uint32_t PatternLen = PatternLen_;

  struct SharedStorage {
    FullBarrier full_barrier_[2][PatternLen];
    EmptyBarrier empty_barrier_[2][Stages];
    SignalBarrier can_send_barrier_[2][PatternLen];
    SignalBarrier copy_finish_barrier_[2][PatternLen];
    SignalBarrier mma_finish_barrier_[1];
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    uint32_t transaction_bytes = 0;
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t is_leader = 0;
    uint32_t num_consumers = 0;
    uint32_t A_transaction_bytes = 0;
    uint32_t B_transaction_bytes = 0;
  };

  // Constructor
  CUTLASS_DEVICE
  PipelineTmaDsmemAsyncGen(SharedStorage& storage, Params params)
      : params_(params)
      , full_barrier_ptr_(storage.full_barrier_)
      , empty_barrier_ptr_(storage.empty_barrier_) 
      , can_send_barrier_ptr_(storage.can_send_barrier_)
      , copy_finish_barrier_ptr_(storage.copy_finish_barrier_)
      , mma_finish_barrier_ptr(storage.mma_finish_barrier_)
    {

    int warp_idx = canonical_warp_idx();
    int lane_predicate = cute::elect_one_sync();
    auto cluster_shape = ClusterShape{};

    if (warp_idx == 0 && lane_predicate == 1) {
      // Barrier FULL init
      for (int i = 0; i < PatternLen; ++i) {
        full_barrier_ptr_[eA][i].init(1);
        full_barrier_ptr_[eB][i].init(1);
      }
      // Barrier EMPTY init
      uint32_t const num_consumer_warpgroups_per_cluster = params_.num_consumers / NumThreadsPerWarpGroup;
      uint32_t const multicast_consumer_arrival_count = (cute::size<0>(cluster_shape) + cute::size<1>(cluster_shape) - 1) *
          num_consumer_warpgroups_per_cluster;
      for (int i = 0; i < Stages; ++i) {
        empty_barrier_ptr_[eA][i].init(num_consumer_warpgroups_per_cluster);
        empty_barrier_ptr_[eB][i].init(num_consumer_warpgroups_per_cluster);
      }
      // Barrier SIGNAL init
      for (int i = 0; i < PatternLen; ++i) {
        copy_finish_barrier_ptr_[eA][i].init(1);
        copy_finish_barrier_ptr_[eB][i].init(1);
      }
      for (int i = 0; i < PatternLen; ++i) {
        can_send_barrier_ptr_[eA][i].init(2);
        can_send_barrier_ptr_[eB][i].init(2);
      }
      mma_finish_barrier_ptr[0].init(num_consumer_warpgroups_per_cluster * cute::size<0>(cluster_shape) * cute::size<1>(cluster_shape));
    }

    // Logic to optimally schedule Empty Arrives
    // Goal : To divide SYNCS Empty Arrival duty equally amongst the Warp-Group (128 threads)
    dim3 block_id = cute::block_id_in_cluster();
    auto cluster_size = cute::size(cluster_shape);
    static constexpr int MaxClusterSize = 16;
    static_assert(cluster_size <= MaxClusterSize, "ERROR : Cluster size too large !" );

    // STEP 1 : Use Cute Layout function to generate an optimal dst block-id (0-15)
    if (params_.num_consumers % NumThreadsPerWarpGroup == 0) {
      int thread_idx = threadIdx.x % NumThreadsPerWarpGroup;
      is_signalling_thread_ = (thread_idx % (NumThreadsPerWarpGroup / MaxClusterSize)) == 0;
      auto layout = cute::composition(Swizzle<2,0,-2>{},
                                      Layout<Shape<_4,_4>,Stride<_4,_1>>{});
      uint32_t thread_row = warp_idx % 4;
      uint32_t thread_col = (thread_idx / 8) % 4;
      dst_blockid_ = layout(thread_row, thread_col);
    }
    else if (params_.num_consumers == 32) {
      int thread_idx = threadIdx.x % 32;
      is_signalling_thread_ = (thread_idx % (32 / MaxClusterSize)) == 0;
      auto layout = Layout<Shape<_4,_4>,Stride<_4, _1>>{};
      uint32_t thread_row = thread_idx / 8;
      uint32_t thread_col = (thread_idx % 8) / 2;
      dst_blockid_ = layout(thread_row, thread_col);
    }
    else {
      is_signalling_thread_ = 0;
      #ifndef NDEBUG
        asm volatile ("brkpt;\n" ::);
      #endif
    }

    // STEP 2: Find if this dst block-id needs an arrival for this problem
    is_signalling_thread_ &= dst_blockid_ < cluster_size;
    is_signalling_thread_ &= is_same_row_or_col(dst_blockid_, block_id, cluster_shape);

    cutlass::arch::fence_barrier_init();
  }

  CUTLASS_DEVICE
  bool is_same_row_or_col(int dst_block_id, dim3 block_id, ClusterShape cluster_shape) {
    return ((dst_block_id % cute::size<0>(cluster_shape)) == block_id.x ||
            (dst_block_id / cute::size<0>(cluster_shape)) == block_id.y);
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait. 
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the barrier to flip.
  // It opportunistically waits for an implementation-dependent timeout.
  // Whether or not the barrier has flipped yet, the try function will return a token.
  // If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.

  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState<Stages> state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void wait_empty(PipelineState<Stages> state, uint32_t var) {
    empty_barrier_ptr_[var][state.index()].wait(state.phase());
  }

  CUTLASS_DEVICE
  void copy_prepare(PipelineState<PatternLen> state, uint32_t var, uint32_t transaction_bytes = 0) {
    if (params_.is_leader) {
      if (transaction_bytes == 0) {
        if (var == eA) {
          transaction_bytes = params_.A_transaction_bytes;
        }
        else if (var == eB) {
          transaction_bytes = params_.B_transaction_bytes;
        }
      }
      full_barrier_ptr_[var][state.index()].arrive_and_reset_bytes(transaction_bytes);
    }
  }

// sender_wait_receiver_ready()     : receiver_arrive_sender()
// sender_wait_dsmem_copy_finish()  : receiver_arrive_dsmem_copy_finish()


  CUTLASS_DEVICE
  void receiver_arrive_sender(uint32_t dst_block_id, uint32_t var, uint32_t stage) {
    can_send_barrier_ptr_[var][stage].arrive(dst_block_id);
  }
  
  CUTLASS_DEVICE
  void sender_wait_receiver_ready(SeparatePipelineState<PatternLen> state, uint32_t var) {
    can_send_barrier_ptr_[var][state.index()].wait(state.phase());
  }

  CUTLASS_DEVICE
  void sender_wait_dsmem_copy_finish(uint32_t phase, uint32_t var, uint32_t stage) {
    copy_finish_barrier_ptr_[var][stage].wait(phase);
  }

  CUTLASS_DEVICE
  void sender_wait_sender_ready(uint32_t phase, uint32_t var, uint32_t stage) {
    uint32_t done;
    done = full_barrier_ptr_[var][stage].test_wait(phase);
    if (not done) {
      full_barrier_ptr_[var][stage].wait(phase);
    }
  }
  
  CUTLASS_DEVICE
  void dsmem_copy_prepare(uint32_t transaction_bytes, uint32_t cta_id, uint32_t var, uint32_t stage) {
    full_barrier_ptr_[var][stage].arrive_and_reset_bytes(transaction_bytes, cta_id);
  }
  
  CUTLASS_DEVICE
  void receiver_wait_dsmem_copy_finish(uint32_t phase, uint32_t var, uint32_t stage) {
    // phase may have bug
    uint32_t done;
    done = full_barrier_ptr_[var][stage].test_wait(phase);
    if (not done) {
      full_barrier_ptr_[var][stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void receiver_arrive_dsmem_copy_finish(uint32_t dst_block_id, uint32_t var, uint32_t stage) {
    copy_finish_barrier_ptr_[var][stage].arrive(dst_block_id);
  }

  CUTLASS_DEVICE
  void mma_finish() {
    for (int x = 0; x < cute::size<0>(ClusterShape{}); x++) {
      for (int y = 0; y < cute::size<1>(ClusterShape{}); y++) {
        uint32_t block_id = x + y * cute::size<0>(ClusterShape{});
        mma_finish_barrier_ptr[0].arrive(block_id);
      }
    }
  }

  CUTLASS_DEVICE
  void wait_mma_finish(uint32_t phase) {
    uint32_t done;
    done = mma_finish_barrier_ptr[0].test_wait(phase);
    if (not done) {
      mma_finish_barrier_ptr[0].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState<Stages> state, uint32_t bytes) {
    producer_commit(state.index(), bytes);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier_by_stage(uint32_t stage, uint32_t var) {
    return producer_get_barrier(stage, var);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState<PatternLen> state, uint32_t var) {
    return producer_get_barrier(state.index(), var);
  }


  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState<Stages> state, uint32_t var, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), var, skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState<PatternLen> state) {
    consumer_wait(state.index(), state.phase());
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState<Stages> state, uint32_t var) {
    consumer_release(state.index(), var);
  }

  CUTLASS_DEVICE
  Params get_params() {
    return params_;
  }

private :
  uint32_t dst_blockid_ = 0;
  uint32_t is_signalling_thread_ = 0;
  FullBarrier (*full_barrier_ptr_)[PatternLen] = nullptr;
  EmptyBarrier (*empty_barrier_ptr_)[Stages] = nullptr;
  SignalBarrier (*can_send_barrier_ptr_)[PatternLen] = nullptr;
  SignalBarrier (*copy_finish_barrier_ptr_)[PatternLen] = nullptr;
  SignalBarrier (*mma_finish_barrier_ptr) = nullptr;
  Params params_;

  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  // Wait for producer to commit transactions (done by TMA)
  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    uint32_t done;
    done = full_barrier_ptr_[eA][stage].test_wait(phase);
    if (not done) {
      full_barrier_ptr_[eA][stage].wait(phase);
    }
    done = full_barrier_ptr_[eB][stage].test_wait(phase);
    if (not done) {
      full_barrier_ptr_[eB][stage].wait(phase);
    }
  }

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t var, uint32_t skip = false) {
    if (threadIdx.x==128 || threadIdx.x==256) {
      empty_barrier_ptr_[var][stage].arrive();
    }
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Producer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage, uint32_t var) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[var][stage]);
  }
};

}  // end namespace cutlass