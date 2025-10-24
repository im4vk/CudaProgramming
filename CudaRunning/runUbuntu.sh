#!/bin/sh

set -xe

QCOW2_IMAGE="$1"
SSH_PORT="$2"

# TODO: check if the user didn't provide enough arguments

##### method 1 nographic

qemu-system-aarch64 \
    -cpu cortex-a53 -smp cores=4 \
    -nographic \
    -M virt -m 4096 \
    -bios /opt/homebrew/Cellar/qemu/10.0.3/share/qemu/edk2-aarch64-code.fd \
    -drive file="$QCOW2_IMAGE",format=qcow2 \
    -device ramfb \
    -netdev user,id=n0,hostfwd=tcp::"$SSH_PORT"-:22 \
    -device virtio-net-pci,netdev=n0 \
    -nic user,model=virtio \
    -rtc base=utc,clock=host \

##### method 2 graphic
# qemu-system-aarch64 \
#     -cpu cortex-a53 -smp cores=4 \
#     -nographic \
#     -M virt -m 4096 \
#     -bios /opt/homebrew/Cellar/qemu/10.0.3/share/qemu/edk2-aarch64-code.fd \
#     -drive file="$QCOW2_IMAGE",format=qcow2 \
#     -device ramfb \
#     -netdev user,id=n0,hostfwd=tcp::"$SSH_PORT"-:22 \
#     -device virtio-net-pci,netdev=n0 \
#     -nic user,model=virtio \
#     -rtc base=utc,clock=host \