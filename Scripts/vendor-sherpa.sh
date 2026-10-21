#!/bin/bash
# Intel Macs only: put the sherpa-onnx dylibs (Parakeet via ONNX Runtime) into an app bundle.
# SherpaParakeetEngine dlopens them from Contents/Frameworks. No-op on Apple silicon.
#   Scripts/vendor-sherpa.sh <path/to/Rhino.app>
set -euo pipefail
cd "$(dirname "$0")/.."
APP="${1:?usage: vendor-sherpa.sh <Rhino.app>}"
[ "$(uname -m)" = "x86_64" ] || { echo "vendor-sherpa: not Intel, skipping"; exit 0; }

VER="1.13.8"
NAME="sherpa-onnx-v${VER}-osx-universal2-shared-lib"
SHA256="c1cca5b4a1867543e27d42fbea513983e73f629d0aa53b43b51555a71d2108d9"
DIR="build/sherpa/${NAME}/lib"

if [ ! -f "${DIR}/libsherpa-onnx-c-api.dylib" ]; then
    mkdir -p build/sherpa
    [ -f "build/sherpa/${NAME}.tar.bz2" ] || curl -sfL -o "build/sherpa/${NAME}.tar.bz2" \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/v${VER}/${NAME}.tar.bz2"
    echo "${SHA256}  build/sherpa/${NAME}.tar.bz2" | shasum -a 256 -c -
    tar xjf "build/sherpa/${NAME}.tar.bz2" -C build/sherpa
fi

mkdir -p "${APP}/Contents/Frameworks"
for lib in libsherpa-onnx-c-api.dylib libonnxruntime.dylib; do
    rm -f "${APP}/Contents/Frameworks/${lib}"
    cp "${DIR}/${lib}" "${APP}/Contents/Frameworks/${lib}"
    codesign --force --sign - "${APP}/Contents/Frameworks/${lib}"
done
echo "vendor-sherpa: sherpa-onnx ${VER} copied into ${APP}"
