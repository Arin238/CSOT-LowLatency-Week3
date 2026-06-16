// ============================================================================
//  aggregator.cpp — Blazing-fast parallel tick aggregator
//
//  Architecture:
//    - Persistent thread pool spawned once in on_init() (zero thread-creation
//      cost in the timed run()).
//    - Per-thread partial tables aligned to 64-byte cache-line boundaries
//      (eliminates false sharing entirely).
//    - Workers pinned to distinct allowed cores (no migrations).
//    - Branchless min/max using conditional moves in the inner reduction loop.
//    - Branchless merge phase — no per-symbol branches when combining partials.
//    - Workers initialize (first-touch) their own partial tables for
//      NUMA-local page placement.
//    - InternalAgg layout ensures min_price/max_price at 16-byte boundaries
//      for aligned vector stores.
//    - Merge uses count-guards to handle the empty-partition edge case
//      correctly (AGG_SPEC.md §7).
//    - Zero heap allocation in run().
//
//  Single translation unit — everything lives here.
// ============================================================================

#include "aggregate.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>

#include <pthread.h>
#include <sched.h>

namespace {

// ---- Constants --------------------------------------------------------------
constexpr int CACHE_LINE = 64;

// We target 4 vCPUs on the judge (c7i.xlarge). Using exactly 4 workers
// avoids oversubscription and matches the hardware.
constexpr unsigned NUM_WORKERS = 4;

// ---- Pin calling thread to a specific core (returns true on success) -------
inline bool pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

// ---- Per-thread partial table, cache-line aligned ---------------------------
// Internal 64-byte padded aggregator with min/max fields at a 16-byte boundary
// to allow the compiler to use aligned 16-byte stores (movdqa instead of movdqu).
struct alignas(CACHE_LINE) InternalAgg {
    std::uint64_t count = 0;
    std::int64_t  sum_price = 0;
    std::uint64_t sum_qty = 0;
    std::uint64_t _pad1 = 0;          // force min_price to offset 32 (16-byte aligned)
    std::int64_t  min_price = 0;
    std::int64_t  max_price = 0;
    char _pad2[16] = {0};             // total struct size = 64 bytes
};

// Each worker gets its own PaddedPartial so no two workers' hot data shares
// a cache line. sizeof(InternalAgg) == 64, so 1024 rows == 64 KiB per worker.
// The alignas(64) ensures the start of each table is on its own cache line.
struct alignas(CACHE_LINE) PaddedPartial {
    InternalAgg rows[1024];  // sized at compile time for the spec constant
};

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

// ---- The inner reduction loop -----------------------------------------------
// Reduces ticks[begin..end) into the given partial table. This is the hot path.
// All software prefetching has been removed — the hardware prefetcher handles
// the sequential tick stream flawlessly, and the random partial-table accesses
// are too irregular for software prefetch to help without adding overhead.
__attribute__((hot))
static void reduce_chunk(const csot::AggTick* __restrict__ ticks,
                         std::size_t begin, std::size_t end,
                         InternalAgg* __restrict__ partial) {
    const csot::AggTick* __restrict__ t = ticks + begin;
    const std::size_t count = end - begin;

    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t s  = t[i].symbol_id;
        const std::int64_t  px = t[i].price;
        const std::uint32_t q  = t[i].qty;

        InternalAgg& __restrict__ r = partial[s];

        // 100% branchless logic.
        // min_price initialized to INT64_MAX and max_price to INT64_MIN before
        // the loop, so the first tick for each symbol will always overwrite them.
        r.min_price = px < r.min_price ? px : r.min_price;
        r.max_price = px > r.max_price ? px : r.max_price;
        r.count     += 1;
        r.sum_price += px;
        r.sum_qty   += q;
    }
}

// ---- Worker thread function -------------------------------------------------
// Runs for the lifetime of the aggregator. Spins on its control flag, processes
// work when signaled, and exits on shutdown.
static void worker_main(int core_id, WorkerCtl* ctl, PaddedPartial* partial,
                        std::uint32_t num_symbols) {
    // Pin to our assigned core immediately and never migrate.
    pin_to_core(core_id);

    for (;;) {
        // Spin-wait for work or shutdown signal (userspace only, no migrations)
        int p;
        while ((p = ctl->phase.load(std::memory_order_acquire)) == 0 ||
               p == 2) {
            // Pure userspace spin-loop hint; avoids scheduler interaction
            __builtin_ia32_pause();
        }

        if (p == 3) {
            // Shutdown
            return;
        }

        // p == 1: work available. Read the parameters (ordered by acquire above).
        const csot::AggTick* ticks = ctl->ticks;
        const std::size_t begin = ctl->begin;
        const std::size_t end = ctl->end;

        // Initialize our partial table (first-touch: the worker does this so pages
        // are placed local on NUMA). We initialize min/max to limits so the
        // hot loop can be 100% branchless.
        for (std::uint32_t i = 0; i < num_symbols; ++i) {
            partial->rows[i].count = 0;
            partial->rows[i].sum_price = 0;
            partial->rows[i].sum_qty = 0;
            partial->rows[i].min_price = INT64_MAX;
            partial->rows[i].max_price = INT64_MIN;
        }

        // Reduce our chunk
        reduce_chunk(ticks, begin, end, partial->rows);

        // Signal done
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

        // ---- Build list of allowed cores from our affinity mask ----------------
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (sched_getaffinity(0, sizeof(cpuset), &cpuset) != 0) {
            // Fallback: allow all cores (won't pin but won't crash)
            for (int i = 0; i < CPU_SETSIZE; ++i) CPU_SET(i, &cpuset);
        }
        int allowed_cores[CPU_SETSIZE];
        int num_allowed = 0;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &cpuset)) {
                allowed_cores[num_allowed++] = c;
            }
        }

        // ---- Determine worker count --------------------------------------------
        unsigned hw = static_cast<unsigned>(num_allowed);
        if (hw == 0) hw = 1;   // safety, will use core 0

        num_total_workers_ = std::min(hw, NUM_WORKERS);
        num_bg_workers_ = num_total_workers_ > 1 ? num_total_workers_ - 1 : 0;

        // ---- Allocate partial tables and control blocks -----------------------
        partials_ = new PaddedPartial[num_total_workers_];
        if (num_bg_workers_ > 0) {
            ctls_ = new WorkerCtl[num_bg_workers_];
            workers_ = new std::thread[num_bg_workers_];

            for (unsigned k = 0; k < num_bg_workers_; ++k) {
                ctls_[k].phase.store(0, std::memory_order_relaxed);

                // Assign each background worker a distinct allowed core,
                // wrapping around if there are more workers than cores.
                // We reserve allowed_cores[0] for the main thread.
                int core_idx = (k + 1) % num_allowed;
                int target_core = allowed_cores[core_idx];

                workers_[k] = std::thread(worker_main, target_core,
                                          &ctls_[k], &partials_[k + 1], num_symbols_);
            }
        }
    }

    void run(const csot::AggTick* ticks, std::size_t n,
             csot::SymbolAgg* out) override {
        // ---- Pin main thread to the first allowed core ------------------------
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

        // Initialize partial 0 (first-touch local to core 0)
        for (std::uint32_t i = 0; i < num_symbols_; ++i) {
            partials_[0].rows[i].count = 0;
            partials_[0].rows[i].sum_price = 0;
            partials_[0].rows[i].sum_qty = 0;
            partials_[0].rows[i].min_price = INT64_MAX;
            partials_[0].rows[i].max_price = INT64_MIN;
        }

        reduce_chunk(ticks, lo0, hi0, partials_[0].rows);

        // ---- Wait for all bg workers to finish ------------------------------
        for (unsigned k = 0; k < num_bg_workers_; ++k) {
            while (ctls_[k].phase.load(std::memory_order_acquire) != 2) {
                __builtin_ia32_pause();
            }
        }

        // ---- Merge partials into out (branchless) ---------------------------
        const std::uint32_t ns = num_symbols_;

        // Unpack first partial as base into `out`.
        // We explicitly zero rows that have no data to match the python reference.
        for (std::uint32_t s = 0; s < ns; ++s) {
            if (partials_[0].rows[s].count == 0) {
                out[s].count     = 0;
                out[s].sum_price = 0;
                out[s].sum_qty   = 0;
                out[s].min_price = 0;
                out[s].max_price = 0;
            } else {
                out[s].count     = partials_[0].rows[s].count;
                out[s].sum_price = partials_[0].rows[s].sum_price;
                out[s].sum_qty   = partials_[0].rows[s].sum_qty;
                out[s].min_price = partials_[0].rows[s].min_price;
                out[s].max_price = partials_[0].rows[s].max_price;
            }
        }

        // Merge remaining partials — completely branchless.
        for (unsigned k = 1; k < num_total_workers_; ++k) {
            const InternalAgg* __restrict__ p = partials_[k].rows;
            for (std::uint32_t s = 0; s < ns; ++s) {
                std::uint64_t pcount = p[s].count;
                // Mask: all‑ones if pcount != 0, else zero. No branch.
                std::uint64_t mask = (pcount != 0) ? (~std::uint64_t{0}) : 0;

                csot::SymbolAgg& __restrict__ r = out[s];

                // Load current r values; if r is empty we'll treat it as identity.
                std::uint64_t rcount = r.count;
                std::int64_t  rmin   = r.min_price;
                std::int64_t  rmax   = r.max_price;

                // If r is empty, force the selection of p's min/max.
                std::uint64_t rcount_zero = (rcount == 0);
                std::int64_t  sel_min = rcount_zero ? p[s].min_price : rmin;
                std::int64_t  sel_max = rcount_zero ? p[s].max_price : rmax;

                // Branchless min/max for the combined values.
                std::int64_t new_min = p[s].min_price < sel_min ? p[s].min_price : sel_min;
                std::int64_t new_max = p[s].max_price > sel_max ? p[s].max_price : sel_max;

                // Apply mask: if mask==0 (pcount==0) we keep r unchanged.
                r.count     = (mask & (rcount + pcount))               | (~mask & rcount);
                r.sum_price = (mask & (r.sum_price + p[s].sum_price))  | (~mask & r.sum_price);
                r.sum_qty   = (mask & (r.sum_qty   + p[s].sum_qty))    | (~mask & r.sum_qty);
                r.min_price = (mask & new_min)                         | (~mask & rmin);
                r.max_price = (mask & new_max)                         | (~mask & rmax);
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