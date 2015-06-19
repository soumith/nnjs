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
    this.kH = weight.shape[1];
    this.kW = weight.shape[2];
    this.nInputPlane = weight.shape[3];
    this.padH = padH
    this.padW = padW
}

SpatialConvolution.prototype.forward = function(input) {
    var nOutputPlane = this.nOutputPlane |0;
    var oH = (input.shape[0] - this.kH + 1) |0;
    var oW = (input.shape[1] - this.kW + 1) |0;
    var nInputPlane = input.shape[2] |0;
    var iH = input.shape[0] |0;
    var iW = input.shape[1] |0;
    var kH = this.kH |0;
    var kW = this.kW |0;
    var weight = this.weight;
    var bias = this.bias;
    var padH = this.padH |0;
    var padW = this.padW |0;
    
    var output = ndarray(new Float32Array(nOutputPlane * oH * oW), 
			 [oH, oW, nOutputPlane]);
    
    /* do convolutions */
    for (var k = 0; k < nOutputPlane; k++) {
	var kp1 = k * (kH*kW*nInputPlane);
	for (var i = padH; i < oH - padH; i++) {
	    for (var j = padW; j < oW - padW; j++) {
		/* for each output pixel, calculate it's full convolution loop */
		var sum = bias.get(k); /* get output pixel */
		for (var kh = 0; kh < kH; kh++) {
		    var kp2 = kp1 + kh * (kW*nInputPlane)
		    var ih = i + kh;
		    var ip1 = ih * (iW * nInputPlane);
		    for (var kw = 0; kw < kW; kw++) {
			var kp3 = kp2 + kw * nInputPlane;
			var iw = j + kw;
			var ip2 = ip1 + iw * (nInputPlane)
			for (var ip = 0; ip < nInputPlane; ip++) {
			    sum += weight.data[kp3 + ip] * input.data[ip2 + ip];
			}
		    }
		}
		output.set(i, j, k, sum);
	    }
	}
    }
    return output
}

env.SpatialConvolution = SpatialConvolution
