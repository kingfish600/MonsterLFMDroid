#!/system/bin/sh
# LFM2.5-24B-A2B night bench — RedMagic 10s Pro, stock llama.cpp master e86f5fd (NDK r28).
# Push with deploy_lfm.sh first. Run: /data/local/tmp/lfm/night.sh [bench|server]
cd /data/local/tmp/lfm || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/lfm
M=./LFM2.5-24B-A2B-Q4_K_M.gguf
COMMON="-ngl 0 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --cpu-mask 0x3f --cpu-strict 1"
echo "== device snapshot =="; grep -E "MemTotal|MemAvailable|MemFree" /proc/meminfo
cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1 | awk '{printf "hottest zone: %.1f C\n", $1/1000}'
nproc; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c
if [ "$1" = server ]; then
  echo "== long-lived server mode (GOAL.md protocol) on :8080 =="
  ./llama-server -m $M $COMMON -c 8192 -t 6 --host 127.0.0.1 --port 8080
  exit 0
fi
for T in 4 6 8; do
  echo "===== llama-bench t=$T pp512+tg128 ====="
  ./llama-bench -m $M $COMMON -t $T -p 512 -n 128 -r 3
done
for RUN in cold warm; do
  echo "===== decode test ($RUN) ====="
  ./llama-cli -m $M $COMMON -c 2048 -t 6 -st -p "List three facts about the Moon." -n 64 --temp 0 2>&1 | grep -E "Prompt eval|Eval time|load time|t/s"
done
# QAD-Q4_0 variant: fastest on host (38.6 t/s), 1 GB smaller — head-to-head, same protocol.
if [ -f ./LFM2.5-24B-A2B-QAD-Q4_0.gguf ]; then
  MQ=./LFM2.5-24B-A2B-QAD-Q4_0.gguf
  echo "===== QAD-Q4_0 llama-bench t=6 ====="
  ./llama-bench -m $MQ $COMMON -t 6 -p 512 -n 128 -r 3
  echo "===== QAD-Q4_0 decode warm ====="
  ./llama-cli -m $MQ $COMMON -c 2048 -t 6 -st -p "List three facts about the Moon." -n 64 --temp 0 2>&1 | grep -E "Prompt eval|Eval time|load time|t/s"
fi
# optional: hard-lock 14.5 GB in RAM (ulimit -l unlimited verified working on Android shell).
# kept separate from main runs: if LMKD dislikes locked memory we still have clean baseline numbers.
echo "===== mlock attempt (locked RAM test) ====="
( ulimit -l unlimited 2>/dev/null
  ./llama-cli -m $M $COMMON --mlock -c 2048 -t 6 -st -p "List three facts about the Moon." -n 64 --temp 0 2>&1 | grep -E "mlock|Prompt eval|Eval time|load time|t/s" )
grep -E "MemAvailable|Cached|Unevictable" /proc/meminfo
echo "== done =="
