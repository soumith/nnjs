var env = require('./env.js')
var ndarray = require("ndarray")
var fill = require("ndarray-fill")

/* weight is a 2D ndarray with dimensions 
   [outSize, inpSize]

   bias is a 1D ndarray with dimensions
   [outSize] */
function Linear(weight, bias) {
    this.weight = weight;
    this.bias = bias;
}

Linear.prototype.forward = function(input) {
    var outSize = this.weight.shape[0];
    var inSize = this.weight.shape[1];
    var weight = this.weight;
    var bias = this.bias;
    var output = ndarray(new Float32Array(outSize), [outSize]);
    
    for (i=0; i < outSize; i++) {
	var o = 0
	for (j=0; j < inSize; j++) {
	    o += input.get(j) * weight.get(i, j);
	}
	o += bias.get(i);
	output.set(i, o);
    }

    return output
}

env.Linear = Linear
