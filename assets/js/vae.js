---
---

var model_params;
const math = ENV.math;

var enc_in_weight;
var enc_in_bias ;
var enc_hid_weight ;
var enc_hid_bias;
var dec_hid_weight;
var dec_hid_bias;
var dec_out_weight;
var dec_out_bias;


$(document).ready(function() {
    console.log("hello")
    $.getJSON( "{{ site.baseurl }}/assets/json/model_params.json", function( data ) {
        model_params = data;

        enc_in_weight = dl.Array2D.new(data["enc_in_weight"]["dim"], data["enc_in_weight"]["param"] );
        enc_in_bias = dl.Array1D.new(data["enc_in_bias"]["param"]);
        enc_hid_weight = dl.Array2D.new(data["enc_hid_weight"]["dim"], data["enc_hid_weight"]["param"] );
        enc_hid_bias = dl.Array1D.new(data["enc_hid_bias"]["param"]) ;
        dec_hid_weight = dl.Array2D.new(data["dec_hid_weight"]["dim"], data["dec_hid_weight"]["param"] );
        dec_hid_bias = dl.Array1D.new(data["dec_hid_bias"]["param"]) ;
        dec_out_weight = dl.Array2D.new(data["dec_out_weight"]["dim"], data["dec_out_weight"]["param"] );
        dec_out_bias = dl.Array1D.new(data["dec_out_bias"]["param"]) ;

        image = generate(dl.Array1D.new([-0.2862, -7.1673, 7.9015, 0.7210, -1.2547, -1.3339, 5.2386, 5.9952, -13.7995, -2.0064]))
        console.log(image)
        $('#sample').append(renderMnistImage(image));
    });
    runExample();
})

/** Runs the example. */
function runExample() {
    var a = dl.Array1D.new([1, 2, 3]);
    var b = dl.Scalar.new(2);
    var result = math.relu(math.sub(a, b));
    // Option 1: With a Promise.
    result.data().then(data => $("#divi").text(JSON.stringify(data)));
}

function generate(z){
    dec_hid = math.relu(math.add(math.vectorTimesMatrix(z, math.transpose(dec_hid_weight)), dec_hid_bias));
    dec_out = math.sigmoid(math.add(math.vectorTimesMatrix(dec_hid, math.transpose(dec_out_weight)), dec_out_bias));
    return dec_out;
}

function renderMnistImage(array) {
  const width = 28;
  const height = 28;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const float32Array = array.dataSync();
  const imageData = ctx.createImageData(width, height);
  for (let i = 0; i < float32Array.length; ++i) {
    const j = i * 4;
    const value = Math.round(float32Array[i] * 255);
    imageData.data[j + 0] = value;
    imageData.data[j + 1] = value;
    imageData.data[j + 2] = value;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}
