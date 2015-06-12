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
    var oH = Math.floor((iH - this.kH) / this.dH + 1);
    var oW = Math.floor((iW - this.kW) / this.dW + 1);

    var output = ndarray(new Float32Array(nPlane * oH * oW), 
			 [nPlane, oH, oW]);

    for (k = 0; k < nPlane; k++) {
	/* loop over output */
	long i, j;
	for(i = 0; i < oH; i++) {
	    for(j = 0; j < oW; j++) {
		/* local pointers */
		var *ip = input_p   + k*iwidth*iheight + i*iwidth*dH + j*dW;
		var *op = output_p  + k*oW*oH + i*oW + j;
		var *indyp = indy_p + k*oW*oH + i*oW + j;
		var *indxp = indx_p + k*oW*oH + i*oW + j;

		/* compute local max: */
		long maxindex = -1;
		var maxval = -THInf;
		long tcntr = 0;
		int x,y;
		for(y = 0; y < kH; y++)
		{
		    for(x = 0; x < kW; x++)
		    {
			var val = *(ip + y*iwidth + x);
			if (val > maxval)
			{
			    maxval = val;
			    maxindex = tcntr;
			}
			tcntr++;
		    }
		}

		/* set output to local max */
		    *op = maxval;

		/* store location of max (x,y) */
		    *indyp = (int)(maxindex / kW)+1;
		    *indxp = (maxindex % kW) +1;
	    }
	}
    }
}
