# nn.js : high performance neural networks for the Browser

**A fast low-level javascript package for multi-threaded (and SIMD-optimized) neural net layers for the browser.**

This package is not meant to replace the excellent [convnet.js](http://cs.stanford.edu/people/karpathy/convnetjs/),  
but provide a lower-level package that convnet.js could possibly depend on.

For now, this package only implements the **forward** ops, and does not implement backward.

###Optimized versions of:
- convolutions (in the context of convnets)
- matrix multiplies (in the context of fully-connected layers)
- fast vector addition


### Layers:
```
nn.SpatialConvolution(weight, bias) 
nn.SpatialMaxPooling(kH, kW, dH, dW)
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.CAddTable()
nn.Linear()
nn.SpatialConvolutionUpsample
nn.View()
nn.FeatureLPPooling()
```


###Uses:

- Web workers for using multiple cores
- SIMD optimizations if the browser supports asm.js


###Dependencies:

- ndarray
- ndarray-fill
- paralleljs

###Unit tests:

Unit tests can be run via nodejs.
``` bash
$ npm -g install mocha
$ cd nnjs
$ mocha

  SpatialConvolution
      ✓ Should compare against torch convolutions (77ms)

  SpatialMaxPooling
      ✓ Should compare against torch SpatialMaxPooling


  2 passing (82ms)

```
