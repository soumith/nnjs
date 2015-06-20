var env = require('./env.js')
var ndarray = require("ndarray")

function Sequential() {
    this.modules = [];
}

Sequential.prototype.add = function(m) {
    this.modules.push(m);
}

Sequential.prototype.forward = function(input) {
    var output = input;
    for (var i = 0; i < this.modules.length; i++) {
	output = this.modules[i].forward(output);
    }
    return output;
}

env.Sequential = Sequential

////////////////////////////////////////////////////////////

function ParallelTable() {
    this.modules = [];
}

ParallelTable.prototype.add = function(m) {
    this.modules.push(m);
}

ParallelTable.prototype.forward = function(input) {
    var output = [];
    for (var i=0; i < this.modules.length; i++) {
	output.push(this.modules[i].forward(input[i]));
    }
    return output;
}

env.ParallelTable = ParallelTable

////////////////////////////////////////////////////////////

function JoinTable(dim) {
    this.dim = dim;
}

JoinTable.prototype.forward = function(input) {
    var size = 0;
    for (var i=0; i < input.length; i++) {
	size += input[i].shape[this.dim-1];
    }
    var outShape = [];
    for (var i=0; i < input[0].shape.length; i++) {
	outShape[i] = input[0].shape[i];
    }
    outShape[this.dim-1] = size;
    var footprint = 1;
    for (var i=0; i < outShape.length; i++) {
	footprint = footprint * outShape[i]
    }

    var output = ndarray(new Float32Array(footprint), outShape);
    if (this.dim == 1) {
	var idx = 0;
	for (var j=0; j < input.length; j++) {
	    var inp = input[j].data;
	    for (var i=0; i < inp.length; i++) {
		output.data[idx++] = inp[i];
	    }
	}
    } else if (this.dim == 3) {
	for (var i=0; i < outShape[0]; i++) {
	    for (var j=0; j < outShape[1]; j++) {
		var idx = 0;
		for (var k=0; k < input.length; k++) {
		    var inp = input[k];
		    var sz = inp.shape[this.dim-1];
		    for (var t=0; t < sz; t++) {
			output.set(i, j, idx + t, inp.get(i, j, t))
		    }
		    idx = idx + sz;
		}
	    }
	}
    } else {
	throw('Unsupported dim: ' + this.dim);
    }
    return output;
}

env.JoinTable = JoinTable
