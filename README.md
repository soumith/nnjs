# nn.js : high performance neural networks for the Browser

## This package is in all respects inferior to [convnet.js](http://cs.stanford.edu/people/karpathy/convnetjs/)

This package is not meant to replace the excellent [convnet.js](http://cs.stanford.edu/people/karpathy/convnetjs/),  
but provide a lower-level package.

**A fast low-level javascript package for multi-threaded neural net layers for the browser.**


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
nn.Linear()
nn.View()
nn.Sequential()
```


###Uses:

- [TODO] Web workers for using multiple cores
- [TODO] SIMD optimizations if the browser supports [SIMD.js](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SIMD#Browser_compatibility)


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


### Building for the browser
``` bash
npm install -g browserify
browserify -r ./js/init.js:nn -r ndarray -r ndarray-fill -o static/js/nn.js
```
