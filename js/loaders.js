var env = require('./env.js')
var ndarray = require("ndarray")
var assert = require('assert');


function loadFromObject(inp) {
    if (inp.type == 'Sequential') {
	var out = new env.Sequential()
	for (var i=0; i < inp.modules.length; i++) {
	    out.add(loadFromObject(inp.modules[i]));
	}
	return out;
    } else if (inp.type == 'ParallelTable') {
	var out = new env.ParallelTable()
	for (var i=0; i < inp.modules.length; i++) {
	    out.add(loadFromObject(inp.modules[i]));
	}
	return out;
    } else if (inp.type == 'JoinTable') {
	return new env.JoinTable(inp.dimension)
    } else if (inp.type == 'Identity') {
	return new env.Identity()
    }  else if (inp.type == 'Linear') {
	var weight = new ndarray(inp.weight, 
				 [inp.outSize, inp.inSize]);
	var bias = new ndarray(inp.bias, [inp.outSize])
	return new env.Linear(weight, bias)
    }  else if (inp.type == 'ReLU') {
	return new env.ReLU()
    }  else if (inp.type == 'View') {
	return new env.View(inp.dims)
    } else if (inp.type == 'SpatialConvolution') {
	var weight = new ndarray(inp.weight, 
				 [inp.nOutputPlane, inp.nInputPlane,
				  inp.kH, inp.kW]);
	var bias = new ndarray(inp.bias, [inp.nOutputPlane])
	return new env.SpatialConvolution(weight, bias,
					  inp.padH, inp.padW)
    } else {
	throw('Error: Unknown module: ' +  inp.type);
    }
}

function loadFromJSON(inp) {
    var input = JSON.parse(inp);
    return loadFromObject(input);
}

function loadFromMsgPack(input) {
}

function loadFromGZipMsgPack(input) {
}

env.loadFromJSON = loadFromJSON
env.loadFromMsgPack = loadFromMsgPack
env.loadFromGZipMsgPack = loadFromGZipMsgPack
