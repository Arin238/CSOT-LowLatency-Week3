#!/bin/bash
# run.sh — One-stop build, test, benchmark, and profiling script for the aggregator
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

STREAM_LARGE="data/large.ticks"
STREAM_TINY="data/tiny.ticks"
BINARY="./build/agg_runner"

usage() {
    echo -e "${BOLD}Usage:${NC} ./run.sh <command>"
    echo ""
    echo -e "${BOLD}  Build & Test${NC}"
    echo -e "  ${CYAN}build${NC}         Build in Release mode"
    echo -e "  ${CYAN}judge${NC}         Build with judge flags (-march=x86-64-v2)"
    echo -e "  ${CYAN}test${NC}          Build + diff against tiny.agg.json"
    echo -e "  ${CYAN}gen${NC}           Generate large.ticks (5M ticks, seed 42)"
    echo -e "  ${CYAN}clean${NC}         Remove all build dirs"
    echo ""
    echo -e "${BOLD}  Sanitizers${NC}"
    echo -e "  ${CYAN}tsan${NC}          ThreadSanitizer build + run (catches data races)"
    echo -e "  ${CYAN}asan${NC}          AddressSanitizer + UBSan build + run"
    echo ""
    echo -e "${BOLD}  Benchmarking${NC}"
    echo -e "  ${CYAN}bench${NC}         Build + run 3x on large.ticks (throughput)"
    echo -e "  ${CYAN}repeat${NC}        Run 10x, verify determinism (checksums)"
    echo -e "  ${CYAN}scaling${NC}       Sweep 1→$(nproc) threads, show throughput per thread count"
    echo ""
    echo -e "${BOLD}  Perf & Profiling${NC}"
    echo -e "  ${CYAN}perf-stat${NC}     perf stat: cycles, instructions, IPC, cache misses, etc."
    echo -e "  ${CYAN}perf-cache${NC}    perf stat: detailed cache & false-sharing events"
    echo -e "  ${CYAN}perf-record${NC}   perf record + report (CPU profile with call graph)"
    echo -e "  ${CYAN}perf-annotate${NC} perf record + annotate hottest function (asm view)"
    echo -e "  ${CYAN}perf-faults${NC}   perf stat: page faults (verify zero-alloc hot path)"
    echo -e "  ${CYAN}perf-ctx${NC}      perf stat: context switches & migrations"
    echo -e "  ${CYAN}perf-all${NC}      Run all perf analyses sequentially"
    echo ""
    echo -e "${BOLD}  System Info${NC}"
    echo -e "  ${CYAN}hwinfo${NC}        CPU topology, caches, NUMA, frequency governor"
    echo ""
    echo -e "${BOLD}  Meta${NC}"
    echo -e "  ${CYAN}all${NC}           test + tsan + repeat + bench + perf-stat"
    echo ""
}

# ---- Build helpers -----------------------------------------------------------

build_release() {
    echo -e "${YELLOW}▸ Building Release...${NC}"
    cmake -B build -DCSOT_AGG_SRC=aggregator.cpp -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -8
    cmake --build build -j$(nproc) 2>&1
    echo -e "${GREEN}✓ Release build done${NC}"
}

build_judge() {
    echo -e "${YELLOW}▸ Building Judge mode...${NC}"
    cmake -B build-judge -DCSOT_JUDGE_BUILD=ON -DCSOT_AGG_SRC=aggregator.cpp -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -8
    cmake --build build-judge -j$(nproc) 2>&1
    echo -e "${GREEN}✓ Judge build done${NC}"
}

build_tsan() {
    echo -e "${YELLOW}▸ Building TSan...${NC}"
    cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DENABLE_TSAN=ON -DCSOT_AGG_SRC=aggregator.cpp 2>&1 | tail -8
    cmake --build build-tsan -j$(nproc) 2>&1
    echo -e "${GREEN}✓ TSan build done${NC}"
}

build_asan() {
    echo -e "${YELLOW}▸ Building ASan...${NC}"
    cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON -DCSOT_AGG_SRC=aggregator.cpp 2>&1 | tail -8
    cmake --build build-asan -j$(nproc) 2>&1
    echo -e "${GREEN}✓ ASan build done${NC}"
}

ensure_release() {
    if [ ! -f "$BINARY" ]; then
        build_release
    fi
}

gen_large() {
    if [ ! -f "$STREAM_LARGE" ]; then
        echo -e "${YELLOW}▸ Generating large.ticks (5M ticks, seed 42)...${NC}"
        python3 data/gen_ticks.py --accesses 5000000 --seed 42 --out "$STREAM_LARGE"
        echo -e "${GREEN}✓ large.ticks generated${NC}"
    else
        echo -e "${CYAN}  large.ticks already exists, skipping gen${NC}"
    fi
}

ensure_large() {
    ensure_release
    gen_large
}

# ---- Test & Correctness ------------------------------------------------------

cmd_test() {
    build_release
    echo ""
    echo -e "${YELLOW}▸ Testing against tiny.agg.json...${NC}"
    DIFF=$(diff <($BINARY $STREAM_TINY 2>/dev/null) data/tiny.agg.json || true)
    if [ -z "$DIFF" ]; then
        echo -e "${GREEN}✓ PASS — output matches tiny.agg.json exactly${NC}"
    else
        echo -e "${RED}✗ FAIL — diff:${NC}"
        echo "$DIFF"
        return 1
    fi
}

# ---- Sanitizers ---------------------------------------------------------------

cmd_tsan() {
    build_tsan
    echo ""
    echo -e "${YELLOW}▸ Running under ThreadSanitizer (tiny.ticks)...${NC}"
    setarch $(uname -m) -R ./build-tsan/agg_runner $STREAM_TINY >/dev/null && \
        echo -e "${GREEN}✓ TSan clean — no data races detected${NC}" || \
        echo -e "${RED}✗ TSan found issues (see output above)${NC}"
    echo ""
    echo -e "${YELLOW}▸ Running under ThreadSanitizer (large.ticks)...${NC}"
    gen_large
    setarch $(uname -m) -R ./build-tsan/agg_runner $STREAM_LARGE >/dev/null && \
        echo -e "${GREEN}✓ TSan clean on large stream${NC}" || \
        echo -e "${RED}✗ TSan found issues on large stream${NC}"
}

cmd_asan() {
    build_asan
    echo ""
    echo -e "${YELLOW}▸ Running under AddressSanitizer (tiny.ticks)...${NC}"
    ./build-asan/agg_runner $STREAM_TINY >/dev/null && \
        echo -e "${GREEN}✓ ASan clean${NC}" || \
        echo -e "${RED}✗ ASan found issues${NC}"
}

# ---- Benchmarking -------------------------------------------------------------

cmd_bench() {
    build_release
    gen_large
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════${NC}"
    echo -e "${BOLD}  Benchmark: large.ticks (5M ticks)${NC}"
    echo -e "${BOLD}═══════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}Running 5 times (look for consistent throughput):${NC}"
    for i in 1 2 3 4 5; do
        echo -n "  Run $i: "
        $BINARY $STREAM_LARGE 2>&1 >/dev/null
    done
}

cmd_repeat() {
    build_release
    gen_large
    echo ""
    echo -e "${YELLOW}▸ Determinism check: 10 runs on large.ticks...${NC}"
    CHECKSUMS=()
    for i in $(seq 1 10); do
        CS=$($BINARY $STREAM_LARGE 2>/dev/null | grep '"checksum"' | grep -o '[0-9]*')
        CHECKSUMS+=("$CS")
        echo -n "."
    done
    echo ""

    UNIQUE=$(printf '%s\n' "${CHECKSUMS[@]}" | sort -u | wc -l)
    if [ "$UNIQUE" -eq 1 ]; then
        echo -e "${GREEN}✓ DETERMINISTIC — all 10 runs produced checksum ${CHECKSUMS[0]}${NC}"
    else
        echo -e "${RED}✗ NON-DETERMINISTIC — got $UNIQUE distinct checksums:${NC}"
        printf '%s\n' "${CHECKSUMS[@]}" | sort | uniq -c
        return 1
    fi
}

cmd_scaling() {
    build_release
    gen_large
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  Scaling Sweep: throughput vs. thread count${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${DIM}  (Note: thread count is compiled into the binary.${NC}"
    echo -e "${DIM}   For a real sweep, edit NUM_WORKERS and rebuild.${NC}"
    echo -e "${DIM}   This runs taskset to restrict available cores.)${NC}"
    echo ""

    printf "  ${BOLD}%-8s  %-18s  %-12s${NC}\n" "Cores" "Throughput" "Speedup"
    echo   "  ────────  ──────────────────  ────────────"

    BASELINE=""
    for cores in 1 2 3 4; do
        # Restrict to cores 0..(cores-1)
        MASK="0-$((cores-1))"
        # Run 3 times, take the best throughput
        BEST_MTPS="0"
        for run in 1 2 3; do
            MTPS=$(taskset -c $MASK $BINARY $STREAM_LARGE 2>&1 >/dev/null | grep -oP '[\d.]+(?= M ticks/s)')
            if (( $(echo "$MTPS > $BEST_MTPS" | bc -l 2>/dev/null || echo 0) )); then
                BEST_MTPS="$MTPS"
            fi
        done

        if [ -z "$BASELINE" ]; then
            BASELINE="$BEST_MTPS"
        fi
        SPEEDUP=$(echo "scale=2; $BEST_MTPS / $BASELINE" | bc -l 2>/dev/null || echo "?")

        printf "  %-8s  %-14s M/s  %sx\n" "$cores" "$BEST_MTPS" "$SPEEDUP"
    done
    echo ""
}

# ---- Perf & Profiling ---------------------------------------------------------

cmd_perf_stat() {
    ensure_large
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  perf stat — Overall hardware counters${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo ""
    perf stat -e \
        cycles,\
instructions,\
branches,\
branch-misses,\
cache-references,\
cache-misses,\
L1-dcache-loads,\
L1-dcache-load-misses,\
LLC-loads,\
LLC-load-misses,\
task-clock,\
context-switches,\
cpu-migrations \
        $BINARY $STREAM_LARGE >/dev/null
}

cmd_perf_cache() {
    ensure_large
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  perf stat — Cache & false-sharing analysis${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo -e "${DIM}  High cache-misses or HITM = false sharing${NC}"
    echo ""

    # Try the detailed cache events; fall back to basics if not available
    perf stat -e \
        cache-misses,\
cache-references,\
L1-dcache-loads,\
L1-dcache-load-misses,\
L1-dcache-stores,\
LLC-loads,\
LLC-load-misses,\
LLC-stores,\
LLC-store-misses,\
dTLB-loads,\
dTLB-load-misses \
        $BINARY $STREAM_LARGE >/dev/null 2>&1 || \
    perf stat -e cache-misses,cache-references,L1-dcache-load-misses \
        $BINARY $STREAM_LARGE >/dev/null
}

cmd_perf_record() {
    ensure_large
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  perf record + report — CPU profile (call graph)${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}▸ Recording...${NC}"
    perf record -F 4999 -g --call-graph dwarf -o perf.data \
        $BINARY $STREAM_LARGE >/dev/null
    echo -e "${GREEN}✓ Recorded to perf.data${NC}"
    echo ""
    echo -e "${YELLOW}▸ Top functions by CPU time:${NC}"
    echo ""
    perf report -i perf.data --stdio --no-children -n --percent-limit 1 2>/dev/null | head -40
    echo ""
    echo -e "${CYAN}  For interactive view: perf report -i perf.data${NC}"
    echo -e "${CYAN}  For flame graph:      perf script -i perf.data | stackcollapse-perf.pl | flamegraph.pl > flame.svg${NC}"
}

cmd_perf_annotate() {
    ensure_large
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  perf annotate — Assembly-level hot spots${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo ""

    # Record if we don't have data yet
    if [ ! -f perf.data ]; then
        echo -e "${YELLOW}▸ Recording first...${NC}"
        perf record -F 4999 -g --call-graph dwarf -o perf.data \
            $BINARY $STREAM_LARGE >/dev/null
    fi

    echo -e "${YELLOW}▸ Annotating hottest function (reduce_chunk):${NC}"
    echo ""
    # Try to annotate reduce_chunk; fall back to the binary's hottest symbol
    perf annotate -i perf.data --stdio -s reduce_chunk 2>/dev/null | head -80 || \
    perf annotate -i perf.data --stdio 2>/dev/null | head -80
    echo ""
    echo -e "${CYAN}  For interactive: perf annotate -i perf.data${NC}"
}

cmd_perf_faults() {
    ensure_large
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  perf stat — Page faults (zero-alloc check)${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo -e "${DIM}  Minor faults should be flat/low during run().${NC}"
    echo -e "${DIM}  Major faults > 0 means disk I/O in hot path.${NC}"
    echo ""
    perf stat -e \
        page-faults,\
minor-faults,\
major-faults,\
context-switches,\
cpu-migrations \
        $BINARY $STREAM_LARGE >/dev/null
}

cmd_perf_ctx() {
    ensure_large
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  perf stat — Context switches & CPU migrations${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════${NC}"
    echo -e "${DIM}  Low ctx-switches + 0 migrations = good pinning.${NC}"
    echo ""
    perf stat -e \
        context-switches,\
cpu-migrations,\
task-clock,\
cycles,\
instructions \
        -- taskset -c 0-3 $BINARY $STREAM_LARGE >/dev/null
}

cmd_perf_all() {
    cmd_perf_stat
    echo ""
    cmd_perf_cache
    echo ""
    cmd_perf_faults
    echo ""
    cmd_perf_ctx
    echo ""
    cmd_perf_record
}

# ---- System Info ---------------------------------------------------------------

cmd_hwinfo() {
    echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  Hardware & CPU Topology${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
    echo ""

    echo -e "${CYAN}── lscpu summary ──${NC}"
    lscpu | grep -E "^(Architecture|CPU\(s\)|Thread|Core|Socket|Model name|CPU MHz|CPU max|L1|L2|L3|NUMA)" 2>/dev/null || lscpu
    echo ""

    echo -e "${CYAN}── nproc ──${NC}"
    echo "  $(nproc) hardware threads"
    echo ""

    echo -e "${CYAN}── Cache line size ──${NC}"
    getconf LEVEL1_DCACHE_LINESIZE 2>/dev/null && true || echo "  (not available)"
    echo ""

    echo -e "${CYAN}── Frequency governor ──${NC}"
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
    else
        echo "  (not available — VM or no cpufreq)"
    fi
    echo ""

    echo -e "${CYAN}── Hyperthread siblings ──${NC}"
    for cpu in /sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list; do
        if [ -f "$cpu" ]; then
            CORE=$(echo "$cpu" | grep -oP 'cpu\d+')
            echo "  $CORE: $(cat "$cpu")"
        fi
    done 2>/dev/null || echo "  (not available)"
    echo ""

    echo -e "${CYAN}── NUMA topology ──${NC}"
    numactl --hardware 2>/dev/null || echo "  (numactl not installed or single-node)"
    echo ""

    echo -e "${CYAN}── Memory ──${NC}"
    free -h 2>/dev/null | head -2 || echo "  (not available)"
    echo ""
}

# ---- Clean --------------------------------------------------------------------

cmd_clean() {
    echo -e "${YELLOW}▸ Cleaning build directories and perf data...${NC}"
    rm -rf build build-judge build-tsan build-asan perf.data perf.data.old
    echo -e "${GREEN}✓ Clean${NC}"
}

# ---- All ----------------------------------------------------------------------

cmd_all() {
    cmd_test
    echo ""
    cmd_tsan
    echo ""
    cmd_repeat
    echo ""
    cmd_bench
    echo ""
    cmd_perf_stat
}

# ---- Main dispatch -----------------------------------------------------------
case "${1:-}" in
    build)          build_release ;;
    judge)          build_judge ;;
    test)           cmd_test ;;
    gen)            gen_large ;;
    tsan)           cmd_tsan ;;
    asan)           cmd_asan ;;
    bench)          cmd_bench ;;
    repeat)         cmd_repeat ;;
    scaling)        cmd_scaling ;;
    perf-stat)      cmd_perf_stat ;;
    perf-cache)     cmd_perf_cache ;;
    perf-record)    cmd_perf_record ;;
    perf-annotate)  cmd_perf_annotate ;;
    perf-faults)    cmd_perf_faults ;;
    perf-ctx)       cmd_perf_ctx ;;
    perf-all)       cmd_perf_all ;;
    hwinfo)         cmd_hwinfo ;;
    all)            cmd_all ;;
    clean)          cmd_clean ;;
    *)              usage ;;
esac
