// ============================================================================
//  aggregator.cpp — Blazing-fast parallel tick aggregator
//
//  Architecture:
//    - Persistent thread pool spawned once in on_init() (zero thread-creation
//      cost in the timed run()).
//    - Per-thread partial tables aligned to 64-byte cache-line boundaries
//      (eliminates false sharing entirely).
//    - Workers pinned to distinct cores via sched_setaffinity (eliminates
//      scheduler migration jitter).
//    - Software prefetch at distance 12 promotes partial[] from L2→L1
//      just-in-time. The 64 KiB partial table fits in L2 (~1-2 MiB),
//      so no FAR/DRAM prefetch needed — only L2→L1 promotion matters.
//    - 2x loop unrolling: processes two ticks per iteration so the OOO
//      engine can overlap independent cache misses from different symbols.
//    - No explicit tick-stream prefetch — HW prefetcher handles sequential
//      32-byte stride perfectly.
//    - Branchless min/max using conditional moves in the inner loop.
//    - Workers initialize (first-touch) their own partial tables for
//      NUMA-local page placement.
//    - Merge uses count-guards to handle the empty-partition edge case
//      correctly (AGG_SPEC.md §7).
//    - Zero heap allocation in run().
//
//  Single translation unit — everything lives here.
// ============================================================================

#include "aggregate.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstring>
#include <thread>

#include <pthread.h>
#include <sched.h>

namespace {

// ---- Compile-time constants -------------------------------------------------
// All constexpr — the compiler can fold these into immediate operands,
// enabling shl-by-6, known-trip-count autovectorization, etc.
constexpr int CACHE_LINE = 64;
constexpr unsigned NUM_WORKERS = 4;         // 4 vCPUs on c7i.xlarge
constexpr std::uint32_t NUM_SYMBOLS = 1024; // frozen by AGG_SPEC.md §3

// Prefetch distance: promotes partial[] entries from L2 into L1 ~12 ticks
// ahead. The 64 KiB partial table fits entirely in L2 (1-2 MiB on
// Ice Lake / Sapphire Rapids), so it stays warm there after initial touch.
// We only need L2→L1 promotion (~12 cycles), not DRAM warmup.
// Distance 12 × ~5 cycles/iteration ÷ 2 (unrolled) ≈ 30 cycles headroom.
// The tick stream is sequential — the HW prefetcher handles it perfectly.
constexpr std::size_t PREFETCH_AHEAD = 12;

// ---- Pin calling thread to a specific core ----------------------------------
inline void pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

// ---- Per-thread partial table, cache-line aligned ---------------------------
// Internal 64-byte padded aggregator to avoid cache-line straddling and
// eliminate the 40-byte multiplication overhead (2x LEA instructions) in the
// hot loop address calculation. 64-byte stride = single shl instruction.
struct alignas(CACHE_LINE) InternalAgg {
    std::uint64_t count;      // offset  0
    std::int64_t  sum_price;  // offset  8
    std::uint64_t sum_qty;    // offset 16
    std::int64_t  min_price;  // offset 24
    std::int64_t  max_price;  // offset 32
    char _pad[24];            // offset 40, pads to 64
};
// Compile-time proof that sizeof == 64 (enables shl-by-6 codegen)
static_assert(sizeof(InternalAgg) == CACHE_LINE,
              "InternalAgg must be exactly one cache line");
static_assert(alignof(InternalAgg) == CACHE_LINE,
              "InternalAgg must be cache-line aligned");

// Each worker gets its own PaddedPartial so no two workers' hot data shares
// a cache line. NUM_SYMBOLS rows × 64 bytes == 64 KiB per worker.
struct alignas(CACHE_LINE) PaddedPartial {
    InternalAgg rows[NUM_SYMBOLS]; // compile-time constant from spec
};
static_assert(sizeof(PaddedPartial) == NUM_SYMBOLS * CACHE_LINE,
              "PaddedPartial must be exactly NUM_SYMBOLS cache lines");

// ---- Thread pool synchronization state (per worker) -------------------------
// Each worker spins on its own atomic flag on its own cache line. The main
// thread publishes work parameters, then sets the flag. After the worker
// finishes, it clears the flag. This avoids condition_variable overhead
// and keeps synchronization purely in userspace with spin-wait.
struct alignas(CACHE_LINE) WorkerCtl {
    // Phase protocol:
    //   0 = idle (worker is spinning, waiting for work)
    //   1 = work available (main thread posted a task)
    //   2 = work done (worker finished, main thread can read partial)
    //   3 = shutdown (worker should exit)
    std::atomic<int> phase{0};

    // Work parameters (written by main before setting phase=1, read by worker
    // after observing phase==1 — the acquire/release on phase orders these).
    const csot::AggTick* ticks;
    std::size_t begin;
    std::size_t end;
};

// ---- Compile-time helper: init partial table blazingly fast -----------------
// Uses memset to zero 64 KiB in one shot (compiles to rep stosq or AVX-512
// vmovdqa64), then fixes up only the 2 sentinel fields per row.
// This is ~60% fewer stores than the naive 5-field-per-element loop.
__attribute__((always_inline))
static inline void init_partial(InternalAgg* __restrict__ rows,
                                [[maybe_unused]] std::uint32_t ns) {
    // Zero everything: count=0, sum_price=0, sum_qty=0, min=0, max=0, pad=0
    std::memset(rows, 0, NUM_SYMBOLS * sizeof(InternalAgg));
    // Fix up sentinels for branchless min/max
    for (std::uint32_t i = 0; i < NUM_SYMBOLS; ++i) {
        rows[i].min_price = INT64_MAX;
        rows[i].max_price = INT64_MIN;
    }
}

// ---- The inner reduction loop -----------------------------------------------
// Reduces ticks[begin..end) into the given partial table. This is the hot path.
//
// Compile-time optimizations:
//   - always_inline: forces inlining into caller, eliminating call overhead
//     and giving the compiler full register allocation across the boundary.
//   - static_assert on InternalAgg size ensures shl-by-6 codegen.
//   - constexpr PREFETCH_AHEAD folds into immediate addressing.
//
// Runtime optimizations:
//   1. Single-distance prefetch (12 ahead): promotes partial[] from L2→L1.
//   2. 2x loop unrolling: OOO engine overlaps independent cache misses.
//   3. No tick-stream prefetch — HW prefetcher handles 32-byte stride.
//   4. __builtin_expect hints the common prefetch-guard path.
//   5. 100% branchless min/max via conditional moves.
__attribute__((hot, always_inline))
static inline void reduce_chunk(const csot::AggTick* __restrict__ ticks,
                                std::size_t begin, std::size_t end,
                                InternalAgg* __restrict__ partial) {
    const csot::AggTick* __restrict__ t = ticks + begin;
    const std::size_t count = end - begin;

    // ---- Main loop: 2x unrolled with dual-distance prefetch pipeline ----
    std::size_t i = 0;
    const std::size_t count2 = count & ~std::size_t(1); // round down to even

    for (; i < count2; i += 2) {
        // Prefetch: promote partial[] entries from L2 → L1 ~12 ticks ahead.
        // Uses rw=1 (write intent → PREFETCHW → Modified state) and
        // locality=3 (bring into L1). The tick data at 12×32=384 bytes
        // ahead is reliably in L1 from the HW prefetcher, so reading
        // symbol_id is free.
        if (__builtin_expect(i + PREFETCH_AHEAD + 1 < count, 1)) {
            __builtin_prefetch(&partial[t[i + PREFETCH_AHEAD].symbol_id], 1, 3);
            __builtin_prefetch(&partial[t[i + PREFETCH_AHEAD + 1].symbol_id], 1, 3);
        }

        // Load both ticks' data upfront (sequential — HW prefetcher has this)
        const std::uint32_t s0  = t[i].symbol_id;
        const std::int64_t  px0 = t[i].price;
        const std::uint32_t q0  = t[i].qty;

        const std::uint32_t s1  = t[i + 1].symbol_id;
        const std::int64_t  px1 = t[i + 1].price;
        const std::uint32_t q1  = t[i + 1].qty;

        // Scatter-update tick 0
        {
            InternalAgg& __restrict__ r = partial[s0];
            r.min_price = px0 < r.min_price ? px0 : r.min_price;
            r.max_price = px0 > r.max_price ? px0 : r.max_price;
            r.count     += 1;
            r.sum_price += px0;
            r.sum_qty   += q0;
        }

        // Scatter-update tick 1 (independent memory op when s1 != s0 —
        // the OOO engine overlaps the cache miss with tick 0's stores)
        {
            InternalAgg& __restrict__ r = partial[s1];
            r.min_price = px1 < r.min_price ? px1 : r.min_price;
            r.max_price = px1 > r.max_price ? px1 : r.max_price;
            r.count     += 1;
            r.sum_price += px1;
            r.sum_qty   += q1;
        }
    }

    // Handle the odd remainder (at most 1 tick)
    if (i < count) {
        const std::uint32_t s  = t[i].symbol_id;
        const std::int64_t  px = t[i].price;
        const std::uint32_t q  = t[i].qty;
        InternalAgg& __restrict__ r = partial[s];
        r.min_price = px < r.min_price ? px : r.min_price;
        r.max_price = px > r.max_price ? px : r.max_price;
        r.count     += 1;
        r.sum_price += px;
        r.sum_qty   += q;
    }
}

// ---- Worker thread function -------------------------------------------------
// __attribute__((flatten)): inline ALL callees (reduce_chunk, init_partial,
// pin_to_core) so the compiler sees the entire hot path as one function,
// enabling cross-function register allocation and scheduling.
__attribute__((flatten))
static void worker_main(int core_id, WorkerCtl* ctl, PaddedPartial* partial,
                        [[maybe_unused]] std::uint32_t num_symbols) {
    pin_to_core(core_id);

    for (;;) {
        // Spin-wait with __builtin_ia32_pause() — stays in userspace.
        // std::this_thread::yield() is a syscall (sched_yield) that causes
        // a context switch, adding ~1-10µs of latency per call.
        int p;
        while ((p = ctl->phase.load(std::memory_order_acquire)) == 0 ||
               p == 2) {
            __builtin_ia32_pause();
        }

        if (p == 3) return; // shutdown

        const csot::AggTick* ticks = ctl->ticks;
        const std::size_t begin = ctl->begin;
        const std::size_t end = ctl->end;

        // Vectorized init: memset + sentinel fixup (compile-time constant size)
        init_partial(partial->rows, NUM_SYMBOLS);

        // Inlined reduce (always_inline ensures no call overhead)
        reduce_chunk(ticks, begin, end, partial->rows);

        ctl->phase.store(2, std::memory_order_release);
    }
}

// ---- The aggregator ---------------------------------------------------------
class FastAggregator final : public csot::Aggregator {
    std::uint32_t num_symbols_ = 0;

    // Per-worker partial tables. Heap-allocated array of cache-line-aligned
    // structs. Allocated once in on_init, reused every run().
    PaddedPartial* partials_ = nullptr;

    // Per-worker control blocks (spin flags + work params).
    WorkerCtl* ctls_ = nullptr;

    // Worker threads (persistent pool). We only spawn (N-1) background workers,
    // as the main thread will process chunk 0.
    std::thread* workers_ = nullptr;

    unsigned num_total_workers_ = 0;
    unsigned num_bg_workers_ = 0;

public:
    ~FastAggregator() override {
        // Signal all bg workers to shut down
        for (unsigned k = 0; k < num_bg_workers_; ++k) {
            // Wait for any in-progress work to finish
            while (ctls_[k].phase.load(std::memory_order_acquire) == 1) {
                __builtin_ia32_pause();
            }
            ctls_[k].phase.store(3, std::memory_order_release);
        }
        // Join all worker threads
        for (unsigned k = 0; k < num_bg_workers_; ++k) {
            if (workers_[k].joinable()) {
                workers_[k].join();
            }
        }
        delete[] workers_;
        delete[] ctls_;
        delete[] partials_;
    }

    void on_init(std::uint32_t num_symbols) override {
        num_symbols_ = num_symbols;

        // Determine total worker count based on the CPU affinity mask.
        // This makes it respect `taskset` bounds correctly, instead of
        // blindly using the system-wide hardware_concurrency().
        unsigned hw = 1;
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
            hw = CPU_COUNT(&cpuset);
        }

        num_total_workers_ = std::min(hw, NUM_WORKERS);
        num_bg_workers_ = num_total_workers_ > 1 ? num_total_workers_ - 1 : 0;

        // Allocate cache-line-aligned partial tables (one per total worker).
        // Using aligned new to guarantee the alignas(64) is respected.
        partials_ = new PaddedPartial[num_total_workers_];

        // Allocate control blocks (one per background worker).
        if (num_bg_workers_ > 0) {
            ctls_ = new WorkerCtl[num_bg_workers_];
            workers_ = new std::thread[num_bg_workers_];

            // Spawn persistent worker threads (cold path — done once).
            unsigned worker_idx = 0;
            for (int core = 0; core < CPU_SETSIZE && worker_idx < num_bg_workers_; ++core) {
                // Only spawn workers on cores we are allowed to use
                if (CPU_ISSET(core, &cpuset)) {
                    // Skip the first allowed core, as we reserve that for the main thread
                    if (worker_idx == 0 && CPU_COUNT(&cpuset) > num_bg_workers_) {
                        // Wait, easier approach: just use an incrementing logical index
                    }
                }
            }
            
            for (unsigned k = 0; k < num_bg_workers_; ++k) {
                ctls_[k].phase.store(0, std::memory_order_relaxed);
                // We'll just pass a logical worker ID (k+1). The worker will
                // pin itself. To be perfectly accurate with taskset, we should
                // map this to the allowed cores in cpuset.
                int target_core = -1;
                unsigned seen = 0;
                for (int c = 0; c < CPU_SETSIZE; ++c) {
                    if (CPU_ISSET(c, &cpuset)) {
                        if (seen == k + 1) { target_core = c; break; }
                        seen++;
                    }
                }
                if (target_core == -1) target_core = k + 1; // fallback

                workers_[k] = std::thread(worker_main, target_core,
                                          &ctls_[k], &partials_[k + 1], num_symbols_);
            }
        }
    }

    void run(const csot::AggTick* ticks, std::size_t n,
             csot::SymbolAgg* out) override {
        // Pin main thread to the first allowed core to keep it out of the bg workers' way
        // and keep its cache warm.
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (CPU_ISSET(c, &cpuset)) {
                    pin_to_core(c);
                    break;
                }
            }
        }

        // ---- Dispatch work to bg workers (zero allocation) ------------------
        for (unsigned k = 0; k < num_bg_workers_; ++k) {
            unsigned part = k + 1;
            const std::size_t lo = n *  part       / num_total_workers_;
            const std::size_t hi = n * (part + 1u) / num_total_workers_;

            ctls_[k].ticks = ticks;
            ctls_[k].begin = lo;
            ctls_[k].end   = hi;

            // Release: ensures ticks/begin/end are visible before phase==1
            ctls_[k].phase.store(1, std::memory_order_release);
        }

        // ---- Main thread does chunk 0 ---------------------------------------
        const std::size_t lo0 = 0;
        const std::size_t hi0 = n / num_total_workers_;
        
        // Vectorized init (compile-time constant size → autovectorized)
        init_partial(partials_[0].rows, NUM_SYMBOLS);
        
        reduce_chunk(ticks, lo0, hi0, partials_[0].rows);

        // ---- Wait for all bg workers to finish ------------------------------
        for (unsigned k = 0; k < num_bg_workers_; ++k) {
            while (ctls_[k].phase.load(std::memory_order_acquire) != 2) {
                __builtin_ia32_pause();
            }
        }

        // ---- Merge partials into out ----------------------------------------
        // Uses compile-time-known NUM_SYMBOLS for autovectorization.
        // Branchless base copy: ternary compiles to cmov, no branch mispredict.
        const InternalAgg* __restrict__ p0 = partials_[0].rows;
        for (std::uint32_t s = 0; s < NUM_SYMBOLS; ++s) {
            const bool has = p0[s].count != 0;
            out[s].count     = p0[s].count;
            out[s].sum_price = p0[s].sum_price;
            out[s].sum_qty   = p0[s].sum_qty;
            // Branchless: if count==0, min/max stay 0 (canonical empty row)
            out[s].min_price = has ? p0[s].min_price : 0;
            out[s].max_price = has ? p0[s].max_price : 0;
        }

        // Merge remaining partials — compile-time-known NUM_SYMBOLS loop
        // bound enables autovectorization. Branchless merge avoids
        // branch mispredictions on the per-symbol if/else.
        for (unsigned k = 1; k < num_total_workers_; ++k) {
            const InternalAgg* __restrict__ p = partials_[k].rows;
            for (std::uint32_t s = 0; s < NUM_SYMBOLS; ++s) {
                if (p[s].count == 0) continue; // skip empty partition — fast

                csot::SymbolAgg& __restrict__ r = out[s];
                // Branchless merge: always add sums; min/max use ternary→cmov.
                // When r.count==0 (first non-empty), the adds produce correct
                // values since r was zeroed, and min/max overwrite correctly
                // because p[s].min <= p[s].max is guaranteed.
                const bool was_empty = (r.count == 0);
                r.count     += p[s].count;
                r.sum_price += p[s].sum_price;
                r.sum_qty   += p[s].sum_qty;
                r.min_price = (was_empty || p[s].min_price < r.min_price)
                              ? p[s].min_price : r.min_price;
                r.max_price = (was_empty || p[s].max_price > r.max_price)
                              ? p[s].max_price : r.max_price;
            }
        }

        // ---- Reset worker phases for next run() call ------------------------
        for (unsigned k = 0; k < num_bg_workers_; ++k) {
            ctls_[k].phase.store(0, std::memory_order_release);
        }
    }
};

}  // namespace

extern "C" csot::Aggregator* create_aggregator() {
    return new FastAggregator();
}