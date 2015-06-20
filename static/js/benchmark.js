nn = require('nn');
ndarray = require('ndarray');

function benchConvolution() {
    var sizes =
	[[3, 64, 5, 5, 32, 32],
	 [4, 64, 5, 5, 32, 32],
	 [64, 64, 5, 5, 32, 32],
	 [64, 3, 5, 5, 32, 32],
	 [64, 224, 5, 5, 32, 32],
	 [128, 128, 7, 7, 28, 28],
	 [256, 256, 7, 7, 14, 14]];
    
    for (i=0; i < sizes.length; i++) {
	/////// Benchmark nn.js //////////////////////////////////
	var c = sizes[i];
	var weight = ndarray(new Float32Array(c[0]*c[1]*c[2]*c[3]),
			     [c[1],c[2],c[3],c[0]]);
	var bias = ndarray(new Float32Array(c[1]), [c[1]]);
	var mod = new nn.SpatialConvolution(weight, bias, 0, 0);
	var inp = ndarray(new Float32Array(c[0]*c[4]*c[5]),
			  [c[4],c[5],c[0]]);
	/* clock */
	var start = performance.now();
	var out = mod.forward(inp);
	var end = performance.now();
	var timeNN = end - start;
	//////////// Benchmark convnet.js //////////////////////////
	var layer_defs = [];
	layer_defs.push({type:'input', out_sx:c[4], out_sy:c[5], out_depth:c[0]}); 
	layer_defs.push({type:'conv', sx:c[2], filters:c[1],
			 stride:1, pad:0, activation:'relu'});
	var net = new convnetjs.Net();
	net.makeLayers(layer_defs);
	var x = new convnetjs.Vol(c[4], c[5], c[0]);
	var start = performance.now();
	var out = net.forward(x)
	console.log(out)
	var end = performance.now();
	var timeCNNJS = end - start;
	/////////////////////////////////////////////////////////
	var logString = 'nn.SpatialConvolution(iChannels = ' + c[0]
	    + ', oChannels = ' + c[1] + ', kH,kW = ' + c[2] + 'x' + c[3]
	    + ', input = ' + c[0] + 'x' + c[4] + 'x' + c[5] + ').forward: '
	    + timeNN + ' ms  convnetjs: ' + timeCNNJS + ' ms';
	document.getElementById('main').innerHTML =
	    document.getElementById('main').innerHTML + '<br/>' + logString;
	console.log(logString);

    }
}

benchConvolution();

