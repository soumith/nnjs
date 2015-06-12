var fs = require('fs');
var nn = require('../js/conv.js')
var ndarray = require("ndarray")
var assert = require("assert")
var eps = 1e-3

function testConvolution() {
    var data = JSON.parse(fs.readFileSync('conv.json', 'utf8'));
    var weight = ndarray(data.weight, [data.op, data.ip, data.kH, data.kW]);
    var bias = ndarray(data.bias, [data.op]);
    var mod = new nn.SpatialConvolution(weight, bias);
    var oH = data.iH - data.kH + 1;
    var oW = data.iW - data.kW + 1;
    var gt = ndarray(data.out, [data.op, oH, oW])
    var inp = ndarray(data.inp, [data.ip, data.iH, data.iW])
    var out = mod.forward(inp)
    var err = 0;
    for (i=0; i < data.op; i++) {
	for (j=0; j < oH; j++) {
	    for (k=0; k < oW; k++) {
		err = Math.max(err, Math.abs(out.get(i,j,k) - gt.get(i,j,k)));
	    }
	}
    }    
    assert.equal(true, err <= eps, "Convolution test failed. Error: " + err)
}

describe('SpatialConvolution', function() {
    it('Should compare against torch convolutions', testConvolution)
});
