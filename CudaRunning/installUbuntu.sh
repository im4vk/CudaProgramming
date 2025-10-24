#!/bin/sh

set -xe

ISO_PATH="$1"
QCOW2_IMAGE="$2"

# TODO: check if the user didn't provide enough arguments

qemu-img create -f qcow2 "$QCOW2_IMAGE" 40G

##### method 1 nographic
qemu-system-aarch64 \
    -cpu cortex-a53 -smp cores=4 \
    -nographic \
    -M virt -m 4096 \
    -bios /opt/homebrew/Cellar/qemu/10.0.3/share/qemu/edk2-aarch64-code.fd \
    -drive file="$QCOW2_IMAGE",format=qcow2 \
    -cdrom "$ISO_PATH" -boot d \
    -device ramfb \
    -nic user,model=virtio \
    -rtc base=utc,clock=host \

##### method 2 graphic
# qemu-system-aarch64 \
#     -machine virt,accel=hvf,highmem=on \
#     -cpu host -smp 4 -m 6G \
#     -bios /opt/homebrew/Cellar/qemu/10.0.3/share/qemu/edk2-aarch64-code.fd \
#     -drive file="$QCOW2_IMAGE",format=qcow2 \
#     -display cocoa \
#     -device virtio-gpu-pci \
#     -device qemu-xhci,id=xhci \
#     -device usb-kbd,bus=xhci.0 \
#     -device usb-tablet,bus=xhci.0 \
#     -netdev user,id=n0,hostfwd=tcp::2222-:22 \
#     -device virtio-net-pci,netdev=n0 \
#     -cdrom "$ISO_PATH" -boot d \
