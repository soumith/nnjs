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
    /* convert weight to HWD */
    var w_ = ndarray(new Float32Array(this.nOutputPlane * this.nInputPlane * this.kH * this.kW), 
		     [this.nOutputPlane, this.kH, this.kW, this.nInputPlane]);
    for (var i=0; i < this.nOutputPlane; i++) {
	for (var j=0; j < this.nInputPlane; j++) {
	    for (var k=0; k < this.kH; k++) {
		for (var s=0; s < this.kW; s++) {
		    w_.set(i,k,s,j, this.weight.get(i,j,k,s));
		}
	    }
	}
    }
    this.weight = w_;
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

    /* convert input to HWD */
    var inp_ = ndarray(new Float32Array(nInputPlane * (iH+padH*2) * (iW+padW*2)), 
		     [iH+padH*2, iW+padW*2, nInputPlane]);
    for (var j=0; j < nInputPlane; j++) {
	for (var k=0; k < iH; k++) {
	    for (var s=0; s < iW; s++) {
		inp_.set(k+padH,s+padW, j, input.get(j,k,s));
	    }
	}
    }
    iH = iH + padH*2
    iW = iW + padW*2
    input = inp_;
    
    /* do convolutions */
    for (var k = 0; k < nOutputPlane; k++) {
	for (var i = 0; i < oH; i++) {
	    for (var j = 0; j < oW; j++) {
		/* for each output pixel, calculate it's full convolution loop */
		var sum = bias.get(k); /* get output pixel */
		for (var kh = 0; kh < kH; kh++) {
		    var ih = i + kh; // - padH;
		    // if (ih < 0 || ih >= iH) continue;
		    for (var kw = 0; kw < kW; kw++) {
			var iw = j + kw;
			for (var ip = 0; ip < nInputPlane; ip++) {
			    var wPtr = k * (kH*kW*nInputPlane) + kh * (kW*nInputPlane) + kw * nInputPlane + ip
			    var iPtr = ih * (iW * nInputPlane) + iw * (nInputPlane) + ip;
			    if (wPtr >=0 && wPtr < weight.data.length && iPtr >=0 && iPtr < input.data.length)
				sum += weight.data[wPtr] * input.data[iPtr];
			}
		    }
		}
		output.set(k, i, j, sum);
	    }
	}
    }
    
    /* convert output to DHW */
    
    return output
}

env.SpatialConvolution = SpatialConvolution
