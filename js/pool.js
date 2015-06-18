var env = require('./env.js')
var ndarray = require("ndarray")
var fill = require("ndarray-fill")

function SpatialMaxPooling(kH, kW, dH, dW) {
    this.kH = kH
    this.kW = kW
    this.dH = dH
    this.dW = dW
}

SpatialMaxPooling.prototype.forward = function(input) {
    var nPlane = input.shape[0];
    var iH = input.shape[1];
    var iW = input.shape[2];
    var kH = this.kH
    var kW = this.kW
    var dH = this.dH
    var dW = this.dW
    var oH = Math.floor((iH - kH) / dH + 1);
    var oW = Math.floor((iW - kW) / dW + 1);

    var output = ndarray(new Float32Array(nPlane * oH * oW),  [nPlane, oH, oW]);

    var idata = input.data;
    var odata = output.data;

    for (k = 0; k < nPlane; k++) {
	/* loop over output */
	var i, j;
	for(i = 0; i < oH; i++) {
	    for(j = 0; j < oW; j++) {
		/* local pointers */
		var ip = k*iW*iH + i*iW*dH + j*dW;
		var op = k*oW*oH + i*oW + j;

		/* compute local max: */
		var maxval = -999999;
		var x,y;
		for(y = 0; y < kH; y++) {
		    for(x = 0; x < kW; x++) {
			var val = idata[ip + y*iW + x];
			if (val > maxval) {
			    maxval = val;
			}
		    }
		}
		/* set output to local max */
		odata[op] = maxval;
	    }
	}
    }
    return output;
}

env.SpatialMaxPooling = SpatialMaxPooling




env.SpatialMaxPooling = SpatialMaxPooling

function SpatialMaxPoolingHWD(kH, kW, dH, dW) {
    this.kH = kH
    this.kW = kW
    this.dH = dH
    this.dW = dW
}

SpatialMaxPoolingHWD.prototype.forward = function(input) {
    var nPlane = input.shape[2];
    var iH = input.shape[0];
    var iW = input.shape[1];
    var kH = this.kH
    var kW = this.kW
    var dH = this.dH
    var dW = this.dW
    var oH = Math.floor((iH - kH) / dH + 1);
    var oW = Math.floor((iW - kW) / dW + 1);

    var output = ndarray(new Float32Array(nPlane * oH * oW),  [oH, oW, nPlane]);

    var idata = input.data;
    var odata = output.data;

    for (k = 0; k < nPlane; k++) {
	/* loop over output */
	var i, j;
	for(i = 0; i < oH; i++) {
	    for(j = 0; j < oW; j++) {
		/* local pointers */
		var ip = (i * dH) * iW * nPlane + (j * dW) * nPlane + k;
		var op = (  i   ) * oW * nPlane + (  j   ) * nPlane + k;

		/* compute local max: */
		var maxval = -999999;
		var x,y;
		for(y = 0; y < kH; y++) {
		    for(x = 0; x < kW; x++) {
			var val = idata[ip + (y * iW + x) * nPlane];
			if (val > maxval) {
			    maxval = val;
			}
		    }
		}
		/* set output to local max */
		odata[op] = maxval;
	    }
	}
    }
    return output;
}

env.SpatialMaxPoolingHWD = SpatialMaxPoolingHWD
