var fs = require('fs');
var nn = require('../js/init.js')
var ndarray = require("ndarray")
var assert = require("assert")
var eps = 1e-5

function testSpatialConvolution() {
    var data = JSON.parse(fs.readFileSync('data/conv.json', 'utf8'));
    var weight = ndarray(data.weight, [data.op, data.ip, data.kH, data.kW]);
    var bias = ndarray(data.bias, [data.op]);
    var mod = new nn.SpatialConvolution(weight, bias, 0, 0);
    var oH = data.iH - data.kH + 1;
    var oW = data.iW - data.kW + 1;
    var gt = ndarray(data.out, [data.op, oH, oW])
    var inp = ndarray(data.inp, [data.ip, data.iH, data.iW])
    var out = mod.forward(inp)
    var err = 0;
    for (i=0; i < data.op; i++) {
	for (j=0; j < oH; j++) {
	    for (k=0; k < oW; k++) {
		err = Math.max(err, Math.abs(out.get(i, j, k) - gt.get(i, j, k)));
	    }
	}
    }
    assert.equal(true, err <= eps, "Convolution test failed. Error: " + err)
}

describe('SpatialConvolution', function() {
    it('Should compare against torch convolutions', testSpatialConvolution)
});

function testSpatialMaxPooling() {
    var data = JSON.parse(fs.readFileSync('data/pool.json', 'utf8'));
    var mod = new nn.SpatialMaxPooling(data.kH, data.kW, data.dH, data.dW);
    var oH = Math.floor((data.iH - data.kH) / data.dH + 1);
    var oW = Math.floor((data.iW - data.kW) / data.dW + 1);
    var gt = ndarray(data.outHWD, [data.np, oH, oW])
    var inp = ndarray(data.inpHWD, [data.np, data.iH, data.iW])
    var out = mod.forward(inp)
    var err = 0;
    for (i=0; i < data.np; i++) {
	for (j=0; j < oH; j++) {
	    for (k=0; k < oW; k++) {
		err = Math.max(err, Math.abs(out.get(i,j,k) - gt.get(i,j,k)));
		assert(err <= eps, i + " " + j + " " + k + " "
		       + gt.get(i,j,k) + " vs " + out.get(i,j,k))
	    }
	}
    }
    assert.equal(true, err <= eps, "MaxPooling test failed. Error: " + err)
}

describe('SpatialMaxPooling', function() {
    it('Should compare against torch SpatialMaxPooling ', testSpatialMaxPooling)
});

function testLinear() {
    var data = JSON.parse(fs.readFileSync('data/full.json', 'utf8'));
    var weight = ndarray(data.weight, [data.outSize, data.inSize]);
    var bias = ndarray(data.bias, [data.outSize]);
    var mod = new nn.Linear(weight, bias);
    var gt = ndarray(data.out, [data.outSize])
    var inp = ndarray(data.inp, [data.inSize])
    var out = mod.forward(inp)
    var err = 0;
    for (i=0; i < data.outSize; i++) {
	err = Math.max(err, Math.abs(out.get(i) - gt.get(i)));
    }
    assert.equal(true, err <= eps, "Linear test failed. Error: " + err)
}

describe('Linear', function() {
    it('Should compare against torch Linear layer', testLinear)
});

function testLoader() {
    var data = fs.readFileSync('data/8x8.json', 'utf8');
    var model = nn.loadFromJSON(data);
    var io = JSON.parse(fs.readFileSync('data/8x8.out.json', 'utf8'));
    var inp = io.input
    var gt_out = io.output
    for (var i=0; i < inp.length; i++) {
    	inp[i] = ndarray(inp[i], [inp[i].length])
    }
    var out = model.forward(inp)
    var err = 0;
    for (var i=0; i < gt_out.length; i++) {
    	err = Math.max(err, Math.abs(gt_out[i]  - out.data[i]));
    }
    assert.equal(true, err <= eps, "loader test failed. Error: " + err)
    ///////////////////////////////////////////////////////////////////
    // var data = fs.readFileSync('data/14x28.json', 'utf8');
    // var model = nn.loadFromJSON(data);
    // var io = JSON.parse(fs.readFileSync('data/14x28.out.json', 'utf8'));
    // var inp = io.input
    // var gt_out = io.output
    // inp[0] = ndarray(inp[0], [28,28,1])
    // inp[1] = ndarray(inp[1], [1,1,10])
    // inp[2] = ndarray(inp[2], [28,28,3])
    // var out = model.forward(inp)
    // var err = 0;
    // for (var i=0; i < gt_out.length; i++) {
    // 	err = Math.max(err, Math.abs(gt_out[i]  - out.data[i]));
    // }
    // assert.equal(true, err <= eps, "loader test failed. Error: " + err)
}

describe('Loader', function() {
    it('Should load a full multi-layer model and compare against torch result', testLoader)
});
