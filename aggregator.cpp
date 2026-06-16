// ============================================================================
//  aggregator.cpp — Blazing-fast parallel tick aggregator
//  With software write-combining cache to break the memory wall.
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

constexpr int CACHE_LINE = 64;
constexpr unsigned NUM_WORKERS = 4;

// --- Staging cache size ---
constexpr unsigned STAGING_SETS = 64;          // direct-mapped
constexpr unsigned STAGING_MASK = STAGING_SETS - 1;

inline bool pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

struct alignas(CACHE_LINE) InternalAgg {
    std::uint64_t count = 0;
    std::int64_t  sum_price = 0;
    std::uint64_t sum_qty = 0;
    std::uint64_t _pad1 = 0;
    std::int64_t  min_price = 0;
    std::int64_t  max_price = 0;
    char _pad2[16] = {0};
};

struct alignas(CACHE_LINE) PaddedPartial {
    InternalAgg rows[1024];
};

struct alignas(CACHE_LINE) WorkerCtl {
    std::atomic<int> phase{0};
    const csot::AggTick* ticks;
    std::size_t begin;
    std::size_t end;
};

struct alignas(64) StagingCache {
    std::uint32_t tags[STAGING_SETS];
    InternalAgg   data[STAGING_SETS];
};

// Merge a cache entry (which contains only deltas for count/sum) into
// the backing partial row. Min/max are already the full extremes.
inline void merge_into_partial(InternalAgg& partial_row,
                               const InternalAgg& cache_entry) {
    partial_row.count     += cache_entry.count;
    partial_row.sum_price += cache_entry.sum_price;
    partial_row.sum_qty   += cache_entry.sum_qty;
    if (cache_entry.min_price < partial_row.min_price)
        partial_row.min_price = cache_entry.min_price;
    if (cache_entry.max_price > partial_row.max_price)
        partial_row.max_price = cache_entry.max_price;
}

// ---- Reduction loop with staging cache (corrected) --------------------------
__attribute__((hot))
static void reduce_chunk(const csot::AggTick* __restrict__ ticks,
                         std::size_t begin, std::size_t end,
                         InternalAgg* __restrict__ partial) {
    StagingCache cache;
    for (unsigned i = 0; i < STAGING_SETS; ++i) cache.tags[i] = 1024;

    const csot::AggTick* __restrict__ t = ticks + begin;
    const std::size_t count = end - begin;

    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t s  = t[i].symbol_id;
        std::int64_t  px = t[i].price;
        std::uint32_t q  = t[i].qty;

        unsigned cache_idx = s & STAGING_MASK;
        std::uint32_t& tag = cache.tags[cache_idx];
        InternalAgg& entry = cache.data[cache_idx];

        if (tag != s) {
            // Evict old line
            if (tag != 1024) merge_into_partial(partial[tag], entry);
            // Load new symbol – keep min/max, zero deltas
            entry.min_price = partial[s].min_price;
            entry.max_price = partial[s].max_price;
            entry.count     = 0;
            entry.sum_price = 0;
            entry.sum_qty   = 0;
            tag = s;
        }

        // Branchless update
        entry.min_price = px < entry.min_price ? px : entry.min_price;
        entry.max_price = px > entry.max_price ? px : entry.max_price;
        entry.count     += 1;
        entry.sum_price += px;
        entry.sum_qty   += q;
    }

    // Flush cache
    for (unsigned i = 0; i < STAGING_SETS; ++i) {
        if (cache.tags[i] != 1024) merge_into_partial(partial[cache.tags[i]], cache.data[i]);
    }
}

// ---- Worker thread ----------------------------------------------------------
static void worker_main(int core_id, WorkerCtl* ctl, PaddedPartial* partial,
                        std::uint32_t num_symbols) {
    pin_to_core(core_id);
    for (;;) {
        int p;
        while ((p = ctl->phase.load(std::memory_order_acquire)) == 0 ||
               p == 2) __builtin_ia32_pause();
        if (p == 3) return;

        const csot::AggTick* ticks = ctl->ticks;
        std::size_t begin = ctl->begin;
        std::size_t end   = ctl->end;

        for (std::uint32_t i = 0; i < num_symbols; ++i) {
            partial->rows[i].count = 0;
            partial->rows[i].sum_price = 0;
            partial->rows[i].sum_qty = 0;
            partial->rows[i].min_price = INT64_MAX;
            partial->rows[i].max_price = INT64_MIN;
        }

        reduce_chunk(ticks, begin, end, partial->rows);
        ctl->phase.store(2, std::memory_order_release);
    }
}

// ---- FastAggregator class ---------------------------------------------------
class FastAggregator final : public csot::Aggregator {
    std::uint32_t num_symbols_ = 0;
    PaddedPartial* partials_ = nullptr;
    WorkerCtl* ctls_ = nullptr;
    std::thread* workers_ = nullptr;
    unsigned num_total_workers_ = 0;
    unsigned num_bg_workers_ = 0;

public:
    ~FastAggregator() override {
        for (unsigned k = 0; k < num_bg_workers_; ++k) {
            while (ctls_[k].phase.load(std::memory_order_acquire) == 1)
                __builtin_ia32_pause();
            ctls_[k].phase.store(3, std::memory_order_release);
        }
        for (unsigned k = 0; k < num_bg_workers_; ++k)
            if (workers_[k].joinable()) workers_[k].join();
        delete[] workers_;
        delete[] ctls_;
        delete[] partials_;
    }

    void on_init(std::uint32_t num_symbols) override {
        num_symbols_ = num_symbols;

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (sched_getaffinity(0, sizeof(cpuset), &cpuset) != 0) {
            for (int i = 0; i < CPU_SETSIZE; ++i) CPU_SET(i, &cpuset);
        }
        int allowed_cores[CPU_SETSIZE];
        int num_allowed = 0;
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &cpuset)) allowed_cores[num_allowed++] = c;

        unsigned hw = static_cast<unsigned>(num_allowed);
        if (hw == 0) hw = 1;
        num_total_workers_ = std::min(hw, NUM_WORKERS);
        num_bg_workers_ = num_total_workers_ > 1 ? num_total_workers_ - 1 : 0;

        partials_ = new PaddedPartial[num_total_workers_];
        if (num_bg_workers_ > 0) {
            ctls_ = new WorkerCtl[num_bg_workers_];
            workers_ = new std::thread[num_bg_workers_];
            for (unsigned k = 0; k < num_bg_workers_; ++k) {
                ctls_[k].phase.store(0, std::memory_order_relaxed);
                int core_idx = (k + 1) % num_allowed;
                int target_core = allowed_cores[core_idx];
                workers_[k] = std::thread(worker_main, target_core,
                                          &ctls_[k], &partials_[k + 1], num_symbols_);
            }
        }
    }

    void run(const csot::AggTick* ticks, std::size_t n,
             csot::SymbolAgg* out) override {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
            for (int c = 0; c < CPU_SETSIZE; ++c)
                if (CPU_ISSET(c, &cpuset)) { pin_to_core(c); break; }
        }

        for (unsigned k = 0; k < num_bg_workers_; ++k) {
            unsigned part = k + 1;
            ctls_[k].ticks = ticks;
            ctls_[k].begin = n *  part       / num_total_workers_;
            ctls_[k].end   = n * (part + 1u) / num_total_workers_;
            ctls_[k].phase.store(1, std::memory_order_release);
        }

        const std::size_t lo0 = 0;
        const std::size_t hi0 = n / num_total_workers_;
        for (std::uint32_t i = 0; i < num_symbols_; ++i) {
            partials_[0].rows[i].count = 0;
            partials_[0].rows[i].sum_price = 0;
            partials_[0].rows[i].sum_qty = 0;
            partials_[0].rows[i].min_price = INT64_MAX;
            partials_[0].rows[i].max_price = INT64_MIN;
        }
        reduce_chunk(ticks, lo0, hi0, partials_[0].rows);

        for (unsigned k = 0; k < num_bg_workers_; ++k)
            while (ctls_[k].phase.load(std::memory_order_acquire) != 2)
                __builtin_ia32_pause();

        // Branchless merge (unchanged)
        const std::uint32_t ns = num_symbols_;
        for (std::uint32_t s = 0; s < ns; ++s) {
            if (partials_[0].rows[s].count == 0) {
                out[s].count = out[s].sum_price = out[s].sum_qty = 0;
                out[s].min_price = out[s].max_price = 0;
            } else {
                out[s].count     = partials_[0].rows[s].count;
                out[s].sum_price = partials_[0].rows[s].sum_price;
                out[s].sum_qty   = partials_[0].rows[s].sum_qty;
                out[s].min_price = partials_[0].rows[s].min_price;
                out[s].max_price = partials_[0].rows[s].max_price;
            }
        }
        for (unsigned k = 1; k < num_total_workers_; ++k) {
            const InternalAgg* __restrict__ p = partials_[k].rows;
            for (std::uint32_t s = 0; s < ns; ++s) {
                std::uint64_t pcount = p[s].count;
                std::uint64_t mask = (pcount != 0) ? (~std::uint64_t{0}) : 0;
                csot::SymbolAgg& __restrict__ r = out[s];
                std::uint64_t rcount = r.count;
                std::int64_t  rmin   = r.min_price;
                std::int64_t  rmax   = r.max_price;
                std::uint64_t rcount_zero = (rcount == 0);
                std::int64_t sel_min = rcount_zero ? p[s].min_price : rmin;
                std::int64_t sel_max = rcount_zero ? p[s].max_price : rmax;
                std::int64_t new_min = p[s].min_price < sel_min ? p[s].min_price : sel_min;
                std::int64_t new_max = p[s].max_price > sel_max ? p[s].max_price : sel_max;
                r.count     = (mask & (rcount + pcount))               | (~mask & rcount);
                r.sum_price = (mask & (r.sum_price + p[s].sum_price))  | (~mask & r.sum_price);
                r.sum_qty   = (mask & (r.sum_qty   + p[s].sum_qty))    | (~mask & r.sum_qty);
                r.min_price = (mask & new_min)                         | (~mask & rmin);
                r.max_price = (mask & new_max)                         | (~mask & rmax);
            }
        }

        for (unsigned k = 0; k < num_bg_workers_; ++k)
            ctls_[k].phase.store(0, std::memory_order_release);
    }
};

}  // namespace

extern "C" csot::Aggregator* create_aggregator() {
    return new FastAggregator();
}