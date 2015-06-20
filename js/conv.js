var env = require('./env.js')
var ndarray = require("ndarray")
var fill = require("ndarray-fill")

/* weight is a 4D ndarray with dimensions 
   [nOutputPlane, kH, kW, nInputPlane]

   bias is a 1D ndarray with dimensions
   [nOutputPlane] */
function SpatialConvolution(weight, bias, padH, padW) {
    this.weight = weight;
    this.bias = bias;
    this.nOutputPlane = weight.shape[0];
    this.nInputPlane = weight.shape[1];
    this.kH = weight.shape[2];
    this.kW = weight.shape[3];
    this.padH = padH
    this.padW = padW
}

SpatialConvolution.prototype.forward = function(input) {
    var nOutputPlane = this.nOutputPlane |0;
    var nInputPlane = input.shape[0] |0;
    var iH = input.shape[1] |0;
    var iW = input.shape[2] |0;
    var oH = (iH + this.padH*2 - this.kH + 1) |0;
    var oW = (iW + this.padW*2 - this.kW + 1) |0;
    var kH = this.kH |0;
    var kW = this.kW |0;
    var weight = this.weight;
    var bias = this.bias;
    var padH = this.padH |0;
    var padW = this.padW |0;
    
    var output = ndarray(new Float32Array(nOutputPlane * oH * oW), 
			 [nOutputPlane, oH, oW]);

    /* fill with bias */
    for (var i = 0; i < bias.shape[0]; i++) {
	var channel = output.pick(i, null, null);
	fill(channel, function(k,j) {
	    return bias.get(i)
	})
    }
    
    /* do convolutions */
    for(var i = 0; i < nOutputPlane; i++) {
	var oChan = i * (oH * oW);
	var wOChan = i * (nInputPlane * kH * kW);
	for(var j = 0; j < nInputPlane; j++) {
	    var wOIChan = wOChan + j * (kH * kW);
	    /* get input */
	    var iChan = j * (iH * iW);
	    /* regular convolution */
	    var posH = 0;
	    var posW = 0;
	    for(var yy = 0; yy < oH; yy++) {
		var oHPtr = oChan + yy * oW;
		for(var xx = 0; xx < oW; xx++) {
		    /* Dot product in two dimensions... 
		       (between input image and the mask) */
		    var oPtr = oHPtr + xx;
		    var sum = output.data[oPtr]
		    for(var ky = 0; ky < kH; ky++) {
			var iHPtr = iChan + ((posH + ky) * iW);
			var wHPtr = wOIChan + ky * kW;
			for(var kx = 0; kx < kW; kx++) {
			    sum += input.data[iHPtr + posW + kx]
				* weight.data[wHPtr + kx]
			}
		    }
		    /* Update output */
		    output.data[oPtr] = sum;
		    posW++;
		}
		posH++;
		posW = 0;
	    }
	}
    }
    return output
}

env.SpatialConvolution = SpatialConvolution
