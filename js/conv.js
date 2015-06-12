var ndarray = require("ndarray")
var fill = require("ndarray-fill")

/* weight is a 4D ndarray with dimensions 
   [nOutputPlane, nInputPlane, kH, kW]

   bias is a 1D ndarray with dimensions
   [nOutputPlane] */
function SpatialConvolution(weight, bias) {
    this.weight = weight;
    this.bias = bias;
    this.nOutputPlane = weight.shape[0];
    this.nInputPlane = weight.shape[1];
    this.kH = weight.shape[2];
    this.kW = weight.shape[3];
}

SpatialConvolution.prototype.forward = function(input) {
    var nOutputPlane = this.nOutputPlane;
    var outputHeight = input.shape[1] - this.kH + 1;
    var outputWidth = input.shape[2] - this.kW + 1;
    var nInputPlane = input.shape[0];
    var inputHeight = input.shape[1];
    var inputWidth = input.shape[2];
    var weight = this.weight;
    var bias = this.bias;
    var kH = this.kH;
    var kW = this.kW;
    
    var output = ndarray(new Float32Array(nOutputPlane * outputHeight * outputWidth), 
			 [nOutputPlane, outputHeight, outputWidth]);
    // console.log(input);
    
    /* fill with bias */
    for (var i = 0; i < bias.shape[0]; i++) {
	var channel = output.pick(i, null, null);
	fill(channel, function(i,j) {
	    return bias.get(i)
	})
    }

    /* do convolutions */
    for(var i = 0; i < nOutputPlane; i++) {
	var oChan = output.pick(i, null, null);
	for(j = 0; j < nInputPlane; j++) {
	    /* get kernel */
	    var kChan = weight.pick(i, j, null, null);
	    /* get input */
	    var iChan = input.pick(j, null, null)
	    /* regular convolution */
	    for(var yy = 0; yy < outputHeight; yy++) {
		for(var xx = 0; xx < outputWidth; xx++) {
		    var iFrame = iChan.lo(yy * inputHeight)
		    /* Dot product in two dimensions... 
		       (between input image and the mask) */
		    var sum = oChan.get(yy, xx);
		    for(var ky = 0; ky < kH; ky++) {
			for(var kx = 0; kx < kW; kx++) {
			    sum += iChan.get() * kChan.get();
			}
		    }
		    /* Update output */
		    oChan.set(sum, yy, xx);
		}
	    }
	}
    }
    return output
}

module.exports = {
    SpatialConvolution: SpatialConvolution
}
