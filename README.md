# nn.js : high performance neural networks for the Browser

## This package is in all respects inferior to [convnet.js](http://cs.stanford.edu/people/karpathy/convnetjs/)

This package is not meant to replace the excellent [convnet.js](http://cs.stanford.edu/people/karpathy/convnetjs/),  
but provide a lower-level package.

**A fast low-level javascript package for multi-threaded neural net layers for the browser.**


For now, this package only implements the **forward** ops, and does not implement backward.

###Optimized versions of:
- convolutions (in the context of convnets)

### Layers:
```
nn.SpatialConvolution(weight, bias, padH, padW) 
nn.SpatialMaxPooling(kH, kW, dH, dW)
nn.ReLU()
nn.Linear(weight, bias)
nn.View(shape)
nn.Sequential()
nn.JoinTable(dim)
nn.ParallelTable()
nn.Identity()
```

###Uses:

- [TODO] Web workers for using multiple cores
- [TODO] SIMD optimizations if the browser supports [SIMD.js](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SIMD#Browser_compatibility) or [WebAssembly](https://github.com/WebAssembly/design/blob/master/README.md)


###Dependencies:

- ndarray
- ndarray-fill
- paralleljs
- msgpack-js

###Unit tests:

Unit tests can be run via nodejs.
``` bash
$ npm -g install mocha
$ cd nnjs
$ mocha 

  SpatialConvolution
      ✓ Should compare against torch convolutions (126ms)

  SpatialMaxPooling
      ✓ Should compare against torch SpatialMaxPooling

  Linear
      ✓ Should compare against torch Linear layer

  Loader
      ✓ Should load a full multi-layer model and compare against torch result (3051ms)


  4 passing (3s)
  ```


### Building for the browser
``` bash
npm install -g browserify
browserify -r ./js/init.js:nn -r ndarray -r ndarray-fill -o static/js/nn.js
```
