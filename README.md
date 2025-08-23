install alpine and run the commands as shown in 
./installubuntu.sh ubuntu-24.04.3-live-server-arm64.iso ubuntu.qcow2 ---> installation step
https://www.youtube.com/watch?v=ZLq-QvLWuoc -> follow this vidoe for installation


./runubuntu.sh ubuntu.qcow2 10022  ---> after installation, run via this command


ssh login:
ssh user@localhost -p10022



Notes

These scripts print the Compiler Explorer response (compile & run output).

CE’s GPU backends usually don’t have a physical NVIDIA GPU, so device kernels may not actually execute; you’ll still see compiler output and any host-side prints.

If CE changes the compiler ID later, replace compiler: nvcc125u1 with the new one from the CUDA dropdown on the noscript page.
you can find it in networks, https://godbolt.org/noscript/cuda after hitting compile
