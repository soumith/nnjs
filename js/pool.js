var env = require('./env.js')
var ndarray = require("ndarray")
var fill = require("ndarray-fill")

function SpatialMaxPooling(kH, kW, dH, dW) {
    this.kH = kH
    this.kW = kW
    this.dH = dH
    this.dW = dW
}

SpatialMaxPooling.prototype.forward(input) {
    var nPlane = input.shape[0];
    var iH = input.shape[1];
    var iW = input.shape[2];
    var kH = this.kH
    var kW = this.kW
    var oH = Math.floor((iH - kH) / dH + 1);
    var oW = Math.floor((iW - kW) / dW + 1);

    var output = ndarray(new Float32Array(nPlane * oH * oW),  [nPlane, oH, oW]);

    for (k = 0; k < nPlane; k++) {
	/* loop over output */
	var i, j;
	for(i = 0; i < oH; i++) {
	    for(j = 0; j < oW; j++) {
		/* local pointers */
		var ip = k*iwidth*iheight + i*iwidth*dH + j*dW;
		var op = k*oW*oH + i*oW + j;

		/* compute local max: */
		var maxval = -999999;
		var x,y;
		for(y = 0; y < kH; y++) {
		    for(x = 0; x < kW; x++) {
			var val = input.data[ip + y*iwidth + x];
			if (val > maxval) {
			    maxval = val;
			}
		    }
		}
		/* set output to local max */
		output.data[op] = maxval;
	    }
	}
    }
    return output;
}

env.SpatialMaxPooling = SpatialMaxPooling
