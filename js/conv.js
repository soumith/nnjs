var ndarray = require("ndarray")
var fill = require("ndarray-fill")

function convForward(input, weight, bias) {
    var output = ndarray(new Float32Array(nOutputPlane * outputHeight * outputWidth), 
			 [nOutputPlane, outputHeight, outputWidth]);
    
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
	    real *ptr_weight = weight_data + k*kstride0 + i*kstride1;
	    /* get input */
	    real *ptr_input = input_data + i*istride0;
            /* THTensor_(validXCorr2Dptr)(ptr_output,
                                       alpha,
                                       ptr_input,  nInputRows,  nInputCols,
                                       ptr_weight, nKernelRows, nKernelCols,
                                       srow, scol);*/
	    /* regular convolution */
	    for(yy = 0; yy < or; yy++) {
		for(xx = 0; xx < oc; xx++) {
		    /* Dot product in two dimensions... (between input image and the mask) */
		    real *pi_ = t_ + yy*sr*ic + xx*sc;
		    real *pw_ = k_;
		    real sum = 0;
		    for(ky = 0; ky < kr; ky++) {
			for(kx = 0; kx < kc; kx++) {
			    sum += pi_[kx]*pw_[kx];
			}
			pi_ += ic; /* next input line */
			pw_ += kc; /* next mask line */
		    }
		    /* Update output */
			*r_++ += alpha*sum;
		}
	    }
	}
    }
    
}
