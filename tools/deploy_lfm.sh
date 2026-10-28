#!/bin/sh
# Deploy lfm engine + model to the RedMagic (host side). Run from repo root.
# Usage: ADB=/path/to/adb ./tools/deploy_lfm.sh [bin|model|app|night|all]
#
#   bin    push the stock master engine binaries to /data/local/tmp/lfm
#   model  push both LFM GGUFs there
#   app    install the APK, seed QAD-Q4_0 into the app's files/models, fix perms
#   night  push the device bench script
#   all    bin + model + night   (NOT app - that one installs)
ADB=${ADB:-../android-sdk/platform-tools/adb}
SERIAL=${SERIAL:-91251291018D}
DEVDIR=/data/local/tmp/lfm
B=/mnt/nvme4tb/builds/arm64/bin
M=/mnt/nvme4tb/models/LFM2.5-24B-A2B-Q4_K_M.gguf
QAD=/mnt/nvme4tb/models/LFM2.5-24B-A2B-QAD-Q4_0.gguf
QADNAME=LFM2.5-24B-A2B-QAD-Q4_0.gguf
QADSIZE=13578190880
PKG=io.github.kingfish600.monsterlfmdroid
APK=app/build/outputs/apk/debug/app-debug.apk
A="$ADB -s $SERIAL"
what=${1:-all}
$A shell mkdir -p $DEVDIR
if [ "$what" = all ] || [ "$what" = bin ]; then
  for f in $(ls $B); do
    echo "push $f"; $A push $B/$f $DEVDIR/$f >/dev/null
  done
  $A shell chmod 755 $DEVDIR/llama-cli $DEVDIR/llama-server $DEVDIR/llama-bench
fi
if [ "$what" = all ] || [ "$what" = model ]; then
  echo "pushing LFM2.5-24B-A2B-Q4_K_M.gguf (few minutes over USB)..."; $A push $M $DEVDIR/LFM2.5-24B-A2B-Q4_K_M.gguf
  echo "device sha:"; $A shell sha256sum $DEVDIR/LFM2.5-24B-A2B-Q4_K_M.gguf; echo "host   sha:"; sha256sum $M
  echo "pushing LFM2.5-24B-A2B-QAD-Q4_0.gguf (fastest host result, 1 GB smaller)..."; $A push $QAD $DEVDIR/$QADNAME
  echo "device sha:"; $A shell sha256sum $DEVDIR/$QADNAME; echo "host   sha:"; sha256sum $QAD
fi
if [ "$what" = app ]; then
  echo "installing $APK"
  $A install -r -t "$APK" || exit 1
  APPDIR=/sdcard/Android/data/$PKG/files/models
  $A shell mkdir -p $APPDIR
  if [ "$($A shell stat -c '%s' $APPDIR/$QADNAME 2>/dev/null | tr -d '\r')" = "$QADSIZE" ]; then
    echo "model already seeded"
  else
    echo "seeding $QADNAME from $DEVDIR (on-device copy, no USB re-push)"
    $A shell "cp -f $DEVDIR/$QADNAME $APPDIR/$QADNAME" || exit 1
  fi
  # CRITICAL: adb creates these as uid 2000 (shell), group 1078 (ext_data_rw).
  # The app runs as u0_a### and is NOT in group 1078, so without this it cannot
  # read its own model, decides it is not downloaded, tries to download, and dies
  # with EACCES on the .part file - which the UI shows as a frozen "download".
  $A shell "chmod 777 $APPDIR && chmod 666 $APPDIR/$QADNAME"
  echo "perms:    dir=$($A shell stat -c '%a' $APPDIR | tr -d '\r') file=$($A shell stat -c '%a' $APPDIR/$QADNAME | tr -d '\r')"
  echo "sha:      $($A shell sha256sum $APPDIR/$QADNAME | cut -d' ' -f1)"
  echo "expected: $(sha256sum $QAD | cut -d' ' -f1)"
  echo
  echo "REMINDER: for full speed this package must be in the OEM performance list:"
  echo "  adb shell settings put global NubiaperformanceMode \\"
  echo "    \"com.pocketpalai+300,com.prismml.moedroid+300,$PKG+300,\""
  echo "Without it the prime cores stay capped and decode is ~50% slower."
fi
if [ "$what" = all ] || [ "$what" = night ]; then
  $A push tools/night_lfm.sh $DEVDIR/night.sh && $A shell chmod 755 $DEVDIR/night.sh
fi
echo "deploy done -> $DEVDIR"
