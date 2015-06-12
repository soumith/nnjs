

A fast low-level javascript package for multi-threaded (and SIMD-optimized) neural net layers for the browser.

Optimized versions of:
- convolutions (in the context of convnets)
- matrix multiplies (in the context of fully-connected layers)
- fast vector addition

Uses:
- Web workers for using multiple cores
- SIMD optimizations if the browser supports asm.js


Dependencies:
- ndarray
- paralleljs
