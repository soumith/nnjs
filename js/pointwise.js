var env = require('./env.js')
var ndarray = require("ndarray")

function ReLU() {}

/* in-place ReLU */
ReLU.prototype.forward = function(input) {
    var d = input.data;
    for (i=0; i < d.length; i++) {
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
