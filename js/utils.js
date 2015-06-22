var ndarray = require("ndarray")
var fill = require("ndarray-fill")
var env = require("./env")

var noise1d = function(n) {
    var arr = new ndarray(new Float32Array(n), [n]);
    fill(arr, function(k) {
	    return (Math.random() * 2) - 1
	})
    return arr;
}

var noise2d = function(n, m) {
    var arr = new ndarray(new Float32Array(n*m), [n, m]);
    fill(arr, function(k,j) {
	    return (Math.random() * 2) - 1
	})
    return arr;
}

var noise3d = function(n, m, k) {
    var arr = new ndarray(new Float32Array(n*m*k), [n, m, k]);
    fill(arr, function(h,i,j) {
	    return (Math.random() * 2) - 1
	})
    return arr;
}


var getPix = function(ip, k, i, j) {
    if ( i < 0 || i >= ip.shape[1] || j < 0 || j >= ip.shape[2]) {
	return 0;
    } else {
	return ip.get(k, i, j);
    }
}

var upscaleBilinear = function(inp, h, w) {
    var iH = inp.shape[1];
    var iW = inp.shape[2];
    var out = new ndarray(new Float32Array(3*h*w), [3, h, w]);
    for (var k=0; k < 3; k++) {
	for (var y = 0; y < h; y++) {
	    for (var x = 0; x < w; x++) {
		var xx = x * iH / w; /* input width index (float) */
		var yy = y * iW / h; /* input height index (float) */
		var fx = xx|0;
		var fy = yy|0;
		if (fx === xx && fy === yy) {
		    out.set(k, y, x, getPix(inp, k, fy, fx))
		} else {
		    var scale_xx = xx % 1;
		    var scale_yy = yy % 1;
		    var newVal = (1 - scale_yy) * ((1 - scale_xx) * getPix(inp, k, fy, fx) +  scale_xx * getPix(inp, k, fy, fx+1)) + 
			scale_yy * ((1 - scale_xx) * getPix(inp, k, fy+1, fx) +  scale_xx * getPix(inp, k, fy+1, fx+1));
		    out.set(k, y, x, newVal);
		}
	    }
	}
    }
    return out;
}

var minmaxnormalize = function(img, factor) {
    var f = factor || 256;
    // normalize output
    var min = 9999;
    var max = -9999;
    for (i=0; i < img.data.length; i++) {
	min = Math.min(min, img.data[i])
	max = Math.max(max, img.data[i])
    }
    max = max - min;
    for (i=0; i < img.data.length; i++) {
	img.data[i] += (-1 * min)
	img.data[i] = f * img.data[i] / max;
    }
    return img;
}


env.utils = {}
env.utils.noise1d = noise1d
env.utils.noise2d = noise2d
env.utils.noise3d = noise3d
env.utils.upscaleBilinear = upscaleBilinear
env.utils.minMaxNormalize = minmaxnormalize
