var Parallel = require('paralleljs');
// require("../js/parallel.js");


weights = [ 1, 2, 3, 4, 5];

var p = new Parallel(weights);
 
// Spawn a remote job (we'll see more on how to use then later)
p.spawn(function (data) {
  data = data.split('').reverse().join('');
 
  return data;
}).then(function (data) {
  console.log(data) // logs sdrawrof
});


