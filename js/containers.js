var env = require('./env.js')

function Sequential() {
    this.modules = [];
}

Sequential.prototype.add = function(m) {
    this.modules.push(m);
}

Sequential.prototype.forward = function(input) {
    var output = input;
    for (i=0; i < this.modules.length; i++) {
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
    for (i=0; i < this.modules.length; i++) {
	output.push(this.modules[i].forward(input[i]));
    }
    return output;
}

env.ParallelTable = ParallelTable

////////////////////////////////////////////////////////////

function JoinTable(dim) {
    this.dim = dim;
    if ( ! (dim == 1)) {
	throw('only dim-1 JoinTable is supported for now')
    }
}

JoinTable.prototype.forward = function(input) {
    var size = 0;
    for (i=0; i < input.length; i++) {
	size += input[i].shape[this.dim-1];
    }
    var outShape = [];
    for (i=0; i < input[1].shape.length; i++) {
	outShape[i] = input[1].shape[i];
    }
    outShape[this.dim-1] = size;
    var footprint = 1;
    for (i=0; i < outShape.length; i++) {
	footprint = footprint * outShape[i]
    }

    var output = ndarray(new Float32Array(footprint), outShape);
    var idx = 0;
    for (j=0; j < input.length; j++) {
	var inp = input[j].data;
	for (i=0; i < inp.length; i++) {
	    output[idx++] = inp[i];
	}
    }

    return output;
}

env.JoinTable = JoinTable
