# Debugging via NCU

In order to debug this with NCU, use this:
```
ncu --open-in-ui --set full --kernel-name-base demangled --kernel-name regex:.*cosine pytest -sxk "kernel_perf"
```