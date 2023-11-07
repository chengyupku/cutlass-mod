# include "cutlass/pipeline/sm90_pipeline.hpp"


namespace cutlass {

using namespace cute;

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

  struct SharedStorage {
    FullBarrier full_barrier_[2][Stages];
    EmptyBarrier empty_barrier_[Stages];
    SignalBarrier can_send_barrier_[2];
    SignalBarrier copy_finish_barrier_[2];
    FullBarrier dsmem_barrier_[2];
    SignalBarrier mma_wait_barrier_[Stages];
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  enum class Mat {
    A,
    B
  };

  struct Params {
    uint32_t transaction_bytes = 0;
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t is_leader = 0;
    uint32_t num_consumers = 0;
    uint32_t use_dsmem_copy = 0;
    uint32_t dsmem_copy_A = 0;
    uint32_t dsmem_copy_B = 0;
    uint32_t dsmem_send_stage = 0;
    uint32_t dsmem_recv_stage = 0;
    uint32_t A_transaction_bytes = 0;
    uint32_t B_transaction_bytes = 0;
  };

  // Constructor
  CUTLASS_DEVICE
  PipelineTmaDsmemAsyncGen(SharedStorage& storage, Params params)
      : params_(params)
      , full_barrier_ptr_(storage.full_barrier_)
      , empty_barrier_ptr_(&storage.empty_barrier_[0]) 
      , can_send_barrier_ptr_(&storage.can_send_barrier_[0])
      , copy_finish_barrier_ptr_(&storage.copy_finish_barrier_[0])
      , dsmem_barrier_ptr_(&storage.dsmem_barrier_[0])
      , mma_wait_barrier_ptr(&storage.mma_wait_barrier_[0]) {

    int warp_idx = canonical_warp_idx();
    int lane_predicate = cute::elect_one_sync();
    auto cluster_shape = ClusterShape{};

    if (warp_idx == 0 && lane_predicate == 1) {
      // Barrier FULL init
      for (int i = 0; i < Stages; ++i) {
        full_barrier_ptr_[0][i].init(1);
        full_barrier_ptr_[1][i].init(1);
      }
      // Barrier EMPTY init
      uint32_t const num_consumer_warpgroups_per_cluster = params_.num_consumers / NumThreadsPerWarpGroup;
      uint32_t const multicast_consumer_arrival_count = (cute::size<0>(cluster_shape) + cute::size<1>(cluster_shape) - 1) *
          num_consumer_warpgroups_per_cluster;
      for (int i = 0; i < Stages; ++i) {
        empty_barrier_ptr_[i].init(num_consumer_warpgroups_per_cluster);
        mma_wait_barrier_ptr[i].init(1);
      }
      // Barrier SIGNAL init
      can_send_barrier_ptr_[0].init(2);
      can_send_barrier_ptr_[1].init(2);
      copy_finish_barrier_ptr_[0].init(1);
      copy_finish_barrier_ptr_[1].init(1);
      dsmem_barrier_ptr_[0].init(1);
      dsmem_barrier_ptr_[1].init(1);
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
  void producer_acquire(PipelineState<Stages> state, 
                        uint32_t A_transaction_bytes=0, 
                        uint32_t B_transaction_bytes=0, 
                        ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token, A_transaction_bytes, B_transaction_bytes);
  }

// sender_wait_receiver_ready()     : receiver_arrive_sender()
// sender_wait_dsmem_copy_finish()  : receiver_arrive_dsmem_copy_finish()


  CUTLASS_DEVICE
  void receiver_arrive_sender(uint32_t dst_block_id, uint32_t var) {
    can_send_barrier_ptr_[var].arrive(dst_block_id);
  }
  
  CUTLASS_DEVICE
  void sender_wait_receiver_ready(uint32_t phase, uint32_t var) {
    can_send_barrier_ptr_[var].wait(phase);
  }

  CUTLASS_DEVICE
  void sender_wait_dsmem_copy_finish(uint32_t phase, uint32_t var) {
    copy_finish_barrier_ptr_[var].wait(phase);
  }

  CUTLASS_DEVICE
  void sender_wait_sender_ready(uint32_t phase, uint32_t var) {
    uint32_t done;
    uint32_t stage = params_.dsmem_send_stage;
    done = full_barrier_ptr_[var][stage].test_wait(phase);
    if (not done) {
      full_barrier_ptr_[var][stage].wait(phase);
    }
  }
  
  CUTLASS_DEVICE
  void dsmem_copy_prepare(uint32_t transaction_bytes, uint32_t cta_id, uint32_t var) {
    dsmem_barrier_ptr_[var].arrive_and_reset_bytes(transaction_bytes, cta_id);
  }
  
  CUTLASS_DEVICE
  void receiver_wait_dsmem_copy_finish(uint32_t phase, uint32_t var) {
    // phase may have bug
    uint32_t done = dsmem_barrier_ptr_[var].test_wait(phase);
    if (not done) {
      dsmem_barrier_ptr_[var].wait(phase);
    } 
  }

  CUTLASS_DEVICE
  void receiver_arrive_dsmem_copy_finish(uint32_t dst_block_id, uint32_t var) {
      copy_finish_barrier_ptr_[var].arrive(dst_block_id);
  }

  CUTLASS_DEVICE
  void mma_wait(uint32_t stage, uint32_t phase) {
    if (stage == params_.dsmem_recv_stage) {
      // phase may have bug
      if (params_.dsmem_copy_A) {
        uint32_t done = dsmem_barrier_ptr_[0].test_wait(phase);
        if (not done) {
          dsmem_barrier_ptr_[0].wait(phase);
        }
      }
      if (params_.dsmem_copy_B) {
        uint32_t done = dsmem_barrier_ptr_[1].test_wait(phase);
        if (not done) {
          dsmem_barrier_ptr_[1].wait(phase);
        }
      }
    }
    if (stage != params_.dsmem_recv_stage || not params_.dsmem_copy_A) {
      uint32_t done = full_barrier_ptr_[0][stage].test_wait(phase);
      if (not done) {
        full_barrier_ptr_[0][stage].wait(phase);
      }
    }
    if (stage != params_.dsmem_recv_stage || not params_.dsmem_copy_B) {
      uint32_t done = full_barrier_ptr_[1][stage].test_wait(phase);
      if (not done) {
        full_barrier_ptr_[1][stage].wait(phase);
      }
    }
  }

  CUTLASS_DEVICE
  void arrive_mma(uint32_t stage) {
    mma_wait_barrier_ptr[stage].arrive();
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState<Stages> state, uint32_t bytes) {
    producer_commit(state.index(), bytes);
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState<Stages> state) {
    for (int count = 0; count < Stages; ++count) {
      producer_acquire(state);  
      ++state;
    }
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState<Stages> state, int& receiver_ready_phase) {
    for (int count = 0; count < Stages; ++count) {
      // may have bug
      producer_acquire(state);  
      if (params_.use_dsmem_copy && state.index() == params_.dsmem_recv_stage) {
        if (params_.dsmem_copy_A) {
          sender_wait_receiver_ready(receiver_ready_phase, 0);
        }
        if (params_.dsmem_copy_B) {
          sender_wait_receiver_ready(receiver_ready_phase, 1);
        }
        receiver_ready_phase ^= 1;
      }
      ++state;
    }
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier_by_stage(uint32_t stage, uint32_t var) {
    return producer_get_barrier(stage, var);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState<Stages> state, uint32_t var) {
    return producer_get_barrier(state.index(), var);
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_dsmem_barrier(uint32_t var) {
    return producer_get_dsmem_barrier_detail(var);
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState<Stages> state, uint32_t var, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), var, skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState<Stages> state) {
    consumer_wait(state.index(), state.phase());
  }

  // CUTLASS_DEVICE
  // void consumer_wait_by_stage(uint32_t stage, uint32_t phase) {
  //   uint32_t done = full_barrier_ptr_[stage].test_wait(phase);
  //   if (not done) {
  //     full_barrier_ptr_[stage].wait(phase);
  //   }
  // }

  CUTLASS_DEVICE
  void consumer_release(PipelineState<Stages> state) {
    consumer_release(state.index());
  }

  CUTLASS_DEVICE
  Params get_params() {
    return params_;
  }

private :
  uint32_t dst_blockid_ = 0;
  uint32_t is_signalling_thread_ = 0;
  FullBarrier (*full_barrier_ptr_)[4] = nullptr;
  EmptyBarrier *empty_barrier_ptr_ = nullptr;
  SignalBarrier *can_send_barrier_ptr_ = nullptr;
  SignalBarrier *copy_finish_barrier_ptr_ = nullptr;
  FullBarrier *dsmem_barrier_ptr_ = nullptr;
  SignalBarrier *mma_wait_barrier_ptr = nullptr;
  Params params_;

  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token, 
                        uint32_t A_transaction_bytes=0, uint32_t B_transaction_bytes=0) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }

    if (params_.is_leader) {
      if (A_transaction_bytes == 0) {
        A_transaction_bytes = params_.A_transaction_bytes;
      }
      if (B_transaction_bytes == 0) {
        B_transaction_bytes = params_.B_transaction_bytes;
      }
      if (stage != params_.dsmem_recv_stage || not params_.dsmem_copy_A) {
        full_barrier_ptr_[0][stage].arrive_and_reset_bytes(A_transaction_bytes);
      }
      if (stage != params_.dsmem_recv_stage || not params_.dsmem_copy_B) {
        full_barrier_ptr_[1][stage].arrive_and_reset_bytes(B_transaction_bytes);      
      }
    }
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Consumer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }

    // Most likely you have elected more than one leader
    if (params_.is_leader && (threadIdx.x % 32 != 0)) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }

  // NOP for TMA based mainloop
  CUTLASS_DEVICE
  void producer_commit(uint32_t stage, uint32_t bytes) {
    // Below code is used only for unit-testing (in the absence of TMA commit)
    #if CUTLASS_UNIT_TEST_PIPELINE
      if (params_.is_leader) {
        // STEP 1 : Commit to self
        full_barrier_ptr_[stage].commit(bytes);

        // STEP 2 : Commit to other blocks in our cluster
        auto cluster_shape = ClusterShape{};
        Layout block_layout_in_cluster = make_layout(cluster_shape);
        dim3 local_block_id = cute::block_id_in_cluster();

        CUTLASS_PRAGMA_UNROLL
        for(int n = 0; n < size<1>(block_layout_in_cluster); ++n) {
          uint32_t dst_block_id = block_layout_in_cluster(local_block_id.x,n,Int<0>{});
          full_barrier_ptr_[stage].commit(dst_block_id, bytes, n!=local_block_id.y);
        }

        CUTLASS_PRAGMA_UNROLL
        for(int m = 0; m < size<0>(block_layout_in_cluster); ++m) {
          uint32_t dst_block_id = block_layout_in_cluster(m,local_block_id.y,Int<0>{});
          full_barrier_ptr_[stage].commit(dst_block_id, bytes, m!=local_block_id.x);
        }
      }
    #endif
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t var, uint32_t skip_wait) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = full_barrier_ptr_[var][stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  // Wait for producer to commit transactions (done by TMA)
  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    if (stage == params_.dsmem_recv_stage) {
      // phase may have bug
      if (params_.dsmem_copy_A) {
        uint32_t done = dsmem_barrier_ptr_[0].test_wait(phase);
        if (not done) {
          dsmem_barrier_ptr_[0].wait(phase);
        }
      }
      if (params_.dsmem_copy_B) {
        uint32_t done = dsmem_barrier_ptr_[1].test_wait(phase);
        if (not done) {
          dsmem_barrier_ptr_[1].wait(phase);
        }
      }
      if (not params_.dsmem_copy_A) {
        uint32_t done = full_barrier_ptr_[0][stage].test_wait(phase);
        if (not done) {
          full_barrier_ptr_[0][stage].wait(phase);
        }
      }
      if (not params_.dsmem_copy_B) {
        uint32_t done = full_barrier_ptr_[1][stage].test_wait(phase);
        if (not done) {
          full_barrier_ptr_[1][stage].wait(phase);
        }
      }
    }
    else {
      uint32_t done = mma_wait_barrier_ptr[stage].test_wait(phase);
      if (not done) {
        mma_wait_barrier_ptr[stage].wait(phase);
      }
    }
  }

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    if (threadIdx.x==128 || threadIdx.x==256) {
      empty_barrier_ptr_[stage].arrive();
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

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_dsmem_barrier_detail(uint32_t var) {
    return reinterpret_cast<ProducerBarrierType*>(&dsmem_barrier_ptr_[var]);
  }
};

}  // end namespace cutlass