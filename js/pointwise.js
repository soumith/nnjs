var env = require('./env.js')
var ndarray = require("ndarray")

function ReLU() {}

/* in-place ReLU */
ReLU.prototype.forward = function(input) {
    var d = input.data;
    for (var i=0; i < d.length; i++) {
	if (d[i] < 0) { 
	    d[i] = 0 
	}
    }
    return input
}

env.ReLU = ReLU

////////////////////////////////////////////////////////

function View(shape) {
    this.shape = shape;
}

View.prototype.forward = function(input) {
    var output = new ndarray(input.data, this.shape)
    return output
}

env.View = View

////////////////////////////////////////////////////////

function Identity() {}

Identity.prototype.forward = function(input) {
    return input
}

env.Identity = Identity
/////////////////////////////////////////////////////////

function DHWtoHWD() {
}

DHWtoHWD.prototype.forward = function(input) {
    if (input.shape.length != 3) {
	throw('Not supported dims')
    }
    var outShape = [];
    var footprint = 1;
    outShape[0] = input.shape[1];
    outShape[1] = input.shape[2];
    outShape[2] = input.shape[0];
    for (i=0; i < input.shape.length; i++) {
	footprint = footprint * input.shape[i];
    }
    var output = new ndarray(new Float32Array(footprint), outShape);
    for (i=0; i < outShape[0]; i++) {
	for (j=0; j < outShape[1]; j++) {
	    for (k=0; k < outShape[2]; k++) {
		output.set(i, j, k, input.get(k, i, j))
	    }
	}
    }
    return output
}
env.DHWtoHWD = DHWtoHWD

