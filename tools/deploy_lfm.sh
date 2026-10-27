#!/bin/sh
# Deploy lfm engine + model to the RedMagic (host side). Run from repo root.
# Usage: ADB=/path/to/adb ./tools/deploy_lfm.sh [model|bin|all]
ADB=${ADB:-../android-sdk/platform-tools/adb}
DEVDIR=/data/local/tmp/lfm
B=/mnt/nvme4tb/builds/arm64/bin
M=/mnt/nvme4tb/models/LFM2.5-24B-A2B-Q4_K_M.gguf
what=${1:-all}
$ADB shell mkdir -p $DEVDIR
if [ "$what" = all ] || [ "$what" = bin ]; then
  for f in $(ls $B); do
    echo "push $f"; $ADB push $B/$f $DEVDIR/$f >/dev/null
  done
  $ADB shell chmod 755 $DEVDIR/llama-cli $DEVDIR/llama-server $DEVDIR/llama-bench
fi
if [ "$what" = all ] || [ "$what" = model ]; then
  echo "pushing LFM2.5-24B-A2B-Q4_K_M.gguf (few minutes over USB)..."; $ADB push $M $DEVDIR/LFM2.5-24B-A2B-Q4_K_M.gguf
  echo "device sha:"; $ADB shell sha256sum $DEVDIR/LFM2.5-24B-A2B-Q4_K_M.gguf; echo "host   sha:"; sha256sum $M
fi
$ADB push tools/night_lfm.sh $DEVDIR/night.sh && $ADB shell chmod 755 $DEVDIR/night.sh
echo "deploy done -> $DEVDIR"
