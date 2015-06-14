nn = require('nn');
ndarray = require('ndarray');

function benchConvolution() {
    var sizes =
	[[3, 16, 5, 5, 32, 32],
	 [16, 64, 5, 5, 32, 32]];
    for (i=0; i < sizes.length; i++) {
	var c = sizes[i];
	var weight = ndarray(new Float32Array(c[0]*c[1]*c[2]*c[3]),
			     [c[1],c[0],c[2],c[3]]);
	var bias = ndarray(new Float32Array(c[1]), [c[1]]);
	var mod = new nn.SpatialConvolution(weight, bias);
	var inp = ndarray(new Float32Array(c[0]*c[4]*c[5]),
			  [c[0],c[4],c[5]]);
	/* clock */
	var start = performance.now();
	var out = mod.forward(inp);
	var end = performance.now();
	var time = end - start;
	console.log('nn.SpatialConvolution(iChannels = ' + c[0]
		    + ', oChannels = ' + c[1] + ', kH,kW = ' + c[2] + 'x' + c[3]
		    + ', input = ' + c[0] + 'x' + c[4] + 'x' + c[5] + ').forward: '
		    + time + ' ms');
    }
}

benchConvolution();
