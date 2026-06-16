#!/bin/bash
# run.sh — One-stop build, test, and benchmark script for the aggregator
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

usage() {
    echo -e "${BOLD}Usage:${NC} ./run.sh <command>"
    echo ""
    echo -e "  ${CYAN}build${NC}       Build in Release mode"
    echo -e "  ${CYAN}judge${NC}       Build with judge flags (-march=x86-64-v2)"
    echo -e "  ${CYAN}test${NC}        Build + diff against tiny.agg.json"
    echo -e "  ${CYAN}bench${NC}       Build + generate large.ticks + run benchmark"
    echo -e "  ${CYAN}tsan${NC}        Build with ThreadSanitizer + run tiny"
    echo -e "  ${CYAN}asan${NC}        Build with AddressSanitizer + run tiny"
    echo -e "  ${CYAN}gen${NC}         Generate large.ticks (5M ticks, seed 42)"
    echo -e "  ${CYAN}repeat${NC}      Run 10x and check determinism (checksums)"
    echo -e "  ${CYAN}all${NC}         test + bench + tsan + repeat"
    echo -e "  ${CYAN}clean${NC}       Remove all build dirs"
    echo ""
}

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

gen_large() {
    if [ ! -f data/large.ticks ]; then
        echo -e "${YELLOW}▸ Generating large.ticks (5M ticks, seed 42)...${NC}"
        python3 data/gen_ticks.py --accesses 5000000 --seed 42 --out data/large.ticks
        echo -e "${GREEN}✓ large.ticks generated${NC}"
    else
        echo -e "${CYAN}  large.ticks already exists, skipping gen${NC}"
    fi
}

cmd_test() {
    build_release
    echo ""
    echo -e "${YELLOW}▸ Testing against tiny.agg.json...${NC}"
    DIFF=$(diff <(./build/agg_runner data/tiny.ticks 2>/dev/null) data/tiny.agg.json || true)
    if [ -z "$DIFF" ]; then
        echo -e "${GREEN}✓ PASS — output matches tiny.agg.json exactly${NC}"
    else
        echo -e "${RED}✗ FAIL — diff:${NC}"
        echo "$DIFF"
        return 1
    fi
}

cmd_bench() {
    build_release
    gen_large
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════${NC}"
    echo -e "${BOLD}  Benchmark: large.ticks (5M ticks)${NC}"
    echo -e "${BOLD}═══════════════════════════════════════${NC}"
    ./build/agg_runner data/large.ticks >/dev/null
    echo ""
    echo -e "${CYAN}Running 3 times for consistency:${NC}"
    for i in 1 2 3; do
        echo -n "  Run $i: "
        ./build/agg_runner data/large.ticks 2>&1 >/dev/null
    done
}

cmd_tsan() {
    build_tsan
    echo ""
    echo -e "${YELLOW}▸ Running under ThreadSanitizer (tiny.ticks)...${NC}"
    ./build-tsan/agg_runner data/tiny.ticks >/dev/null 2>&1 && \
        echo -e "${GREEN}✓ TSan clean — no data races detected${NC}" || \
        echo -e "${RED}✗ TSan found issues (see output above)${NC}"
    echo ""
    echo -e "${YELLOW}▸ Running under ThreadSanitizer (large.ticks)...${NC}"
    gen_large
    ./build-tsan/agg_runner data/large.ticks >/dev/null 2>&1 && \
        echo -e "${GREEN}✓ TSan clean on large stream${NC}" || \
        echo -e "${RED}✗ TSan found issues on large stream${NC}"
}

cmd_asan() {
    build_asan
    echo ""
    echo -e "${YELLOW}▸ Running under AddressSanitizer (tiny.ticks)...${NC}"
    ./build-asan/agg_runner data/tiny.ticks >/dev/null 2>&1 && \
        echo -e "${GREEN}✓ ASan clean${NC}" || \
        echo -e "${RED}✗ ASan found issues${NC}"
}

cmd_repeat() {
    build_release
    gen_large
    echo ""
    echo -e "${YELLOW}▸ Determinism check: 10 runs on large.ticks...${NC}"
    CHECKSUMS=()
    for i in $(seq 1 10); do
        CS=$(./build/agg_runner data/large.ticks 2>/dev/null | grep '"checksum"' | grep -o '[0-9]*')
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

cmd_clean() {
    echo -e "${YELLOW}▸ Cleaning build directories...${NC}"
    rm -rf build build-judge build-tsan build-asan
    echo -e "${GREEN}✓ Clean${NC}"
}

cmd_all() {
    cmd_test
    echo ""
    cmd_tsan
    echo ""
    cmd_repeat
    echo ""
    cmd_bench
}

# ---- Main dispatch -----------------------------------------------------------
case "${1:-}" in
    build)   build_release ;;
    judge)   build_judge ;;
    test)    cmd_test ;;
    bench)   cmd_bench ;;
    tsan)    cmd_tsan ;;
    asan)    cmd_asan ;;
    gen)     gen_large ;;
    repeat)  cmd_repeat ;;
    all)     cmd_all ;;
    clean)   cmd_clean ;;
    *)       usage ;;
esac
