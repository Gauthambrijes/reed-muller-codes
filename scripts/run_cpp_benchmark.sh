#!/usr/bin/env bash
# Short-packet benchmark: RM codes under RPA (float exact, float min-sum,
# integer min-sum) and CRC-aided polar codes under SCL-32, same (n, k).
#
#   scripts/run_cpp_benchmark.sh PART   with PART in rm27 polar128_29 rm28_exact
#                                       rm28_rest polar256_37 rm37_exact rm37_minsum
#                                       rm37_fixed5 rm37_fixed4 polar128_64 all
#
# Each part writes results/cpp/bench_PART.csv; plot_benchmark.py reads them all.
# Results do not depend on THREADS (fixed seeded chunks). On 2 cores the whole
# benchmark takes about 3 hours.
#
# Fixed-point settings come from results/cpp/fixed_sweep_rm27.csv and
# fixed_rms_sweep.csv: 5-bit storage with clipping at 2 x received RMS, and
# 4-bit storage at 1.5 x RMS. Polar CRC lengths come from polar_crc_sweep.csv
# (CRC-6 was best for all three codes). The polar design Eb/N0 is the
# normal-approximation threshold at BLER 1e-3, rounded.
set -euo pipefail
cd "$(dirname "$0")/.."
make -s -C cpp
S=cpp/sim
T=${THREADS:-2}
HDR="scheme,code,n,k,decoder,ebn0_db,blocks,block_errors,bler,seconds"

rpa() {  # r m dec-args grid max target chunk
  local r=$1 m=$2 dec=$3 grid=$4 mx=$5 tgt=$6 ch=$7
  $S rpa --r $r --m $m $dec --ebn0 $grid --target $tgt --max $mx --threads $T --chunk $ch | tail -n +2
}
fixed_pair() {  # r m grid max target chunk
  rpa $1 $2 "--dec fixed --bits 5 --clip-rms 2.0" $3 $4 $5 $6
  rpa $1 $2 "--dec fixed --bits 4 --clip-rms 1.5" $3 $4 $5 $6
}
polar() {  # m k design grid
  $S polar --m $1 --k $2 --crc 6 --list 32 --design $3 --ebn0 $4 --target 100 --max 2000000 --threads $T | tail -n +2
}

G27=1.0,1.5,2.0,2.5,3.0,3.5,4.0
G28=0.5,1.0,1.5,2.0,2.5,3.0
G37=1.5,2.0,2.5,3.0,3.5

part() {
  case $1 in
    rm27)        rpa 2 7 "--dec exact" $G27 1000000 100 64
                 rpa 2 7 "--dec minsum" $G27 2000000 100 64
                 fixed_pair 2 7 $G27 2000000 100 64 ;;
    polar128_29) polar 7 29 2.5 $G27 ;;
    rm28_exact)  rpa 2 8 "--dec exact" $G28 400000 100 64 ;;
    rm28_rest)   rpa 2 8 "--dec minsum" $G28 800000 100 64
                 fixed_pair 2 8 $G28 800000 100 64 ;;
    polar256_37) polar 8 37 2.0 0.5,1.0,1.5,2.0,2.5,3.0,3.5 ;;
    rm37_exact)  rpa 3 7 "--dec exact" $G37 60000 50 16 ;;
    rm37_minsum) rpa 3 7 "--dec minsum" $G37 150000 50 16 ;;
    rm37_fixed5) rpa 3 7 "--dec fixed --bits 5 --clip-rms 2.0" $G37 150000 50 16 ;;
    rm37_fixed4) rpa 3 7 "--dec fixed --bits 4 --clip-rms 1.5" $G37 150000 50 16 ;;
    polar128_64) polar 7 64 2.5 1.5,2.0,2.5,3.0,3.5,4.0 ;;
    *) echo "unknown part $1" >&2; exit 2 ;;
  esac
}

PARTS=("$@")
[ "${PARTS[0]:-all}" = all ] && PARTS=(rm27 polar128_29 rm28_exact rm28_rest polar256_37 rm37_exact rm37_minsum rm37_fixed5 rm37_fixed4 polar128_64)
for p in "${PARTS[@]}"; do
  out=results/cpp/bench_$p.csv
  { echo "$HDR"; part $p; } > $out.tmp && mv $out.tmp $out
  echo "wrote $out"
done
