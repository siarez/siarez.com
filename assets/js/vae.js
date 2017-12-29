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

//z_9 is the representation of 9
const z_9 = [-0.2862, -7.1673, 7.9015, 0.7210, -1.2547, -1.3339, 5.2386, 5.9952, -13.7995, -2.0064]

$(document).ready(function() {
    console.log("hello")
    
    for (let i = 0; i < 10; ++i) {
        // Adding sliders for adjusting z
        let slider = $('<input type="range" min="-500" max="500" value="0" class="slider" id="range'+i+'" data-index='+i+'><span>0%</span>')
        var new_z = z_9.slice();
        slider.on("change mousemove", function() {
            let val = $(this).val();
            let idx = $(this).data("index");
            $(this).next().html(val + "%");            
            let temp = z_9[idx] + val * z_9[idx] / 100; 
            new_z[idx] = z_9[idx] + val * z_9[idx] / 100; 
            new_image = generate(dl.Array1D.new(new_z))
            $('#mnist_sample_tweaked canvas').remove()
            $('#mnist_sample_tweaked').append(renderMnistImage(new_image));
        });
        $(".slidecontainer").append(slider)
        
    }

    
    $.getJSON( "{{ site.baseurl }}/assets/json/model_params.json", function( data ) {
        model_params = data;
        // Creating weights and biases from the json file 
        enc_in_weight = dl.Array2D.new(data["enc_in_weight"]["dim"], data["enc_in_weight"]["param"] );
        enc_in_bias = dl.Array1D.new(data["enc_in_bias"]["param"]);
        enc_hid_weight = dl.Array2D.new(data["enc_hid_weight"]["dim"], data["enc_hid_weight"]["param"] );
        enc_hid_bias = dl.Array1D.new(data["enc_hid_bias"]["param"]) ;
        dec_hid_weight = dl.Array2D.new(data["dec_hid_weight"]["dim"], data["dec_hid_weight"]["param"] );
        dec_hid_bias = dl.Array1D.new(data["dec_hid_bias"]["param"]) ;
        dec_out_weight = dl.Array2D.new(data["dec_out_weight"]["dim"], data["dec_out_weight"]["param"] );
        dec_out_bias = dl.Array1D.new(data["dec_out_bias"]["param"]) ;

        image_9 = generate(dl.Array1D.new(z_9))
        //image_8 = generate(dl.Array1D.new([0.1307, 4.1141, 6.2771, 3.2629, 3.7982, 1.2015, 5.8397, 8.4124, -2.8188, 5.0266]))
        
        $('#mnist_sample').append(renderMnistImage(image_9));
        //$('#mnist_sample').append(renderMnistImage(image_8));
    });

})


function generate(z){
    dec_hid = math.relu(math.add(math.vectorTimesMatrix(z, math.transpose(dec_hid_weight)), dec_hid_bias));
    dec_out = math.sigmoid(math.add(math.vectorTimesMatrix(dec_hid, math.transpose(dec_out_weight)), dec_out_bias));
    return dec_out;
}

function renderMnistImage(array) {
    const width = 28;
    const height = 28;
    const scale_factor = 5
    const canvas = document.createElement('canvas');
    // Needed the scale canvas to scale up the image
    const scaled_canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    scaled_canvas.width = width*scale_factor;
    scaled_canvas.height = height*scale_factor;
    const ctx = canvas.getContext('2d');
    const scaled_ctx = scaled_canvas.getContext('2d');
    scaled_ctx.imageSmoothingEnabled = false
    scaled_ctx.mozImageSmoothingEnabled = false
    scaled_ctx.webkitImageSmoothingEnabled = false
    scaled_ctx.ImageSmoothingEnabled = false
    scaled_ctx.scale(scale_factor, scale_factor)
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
    scaled_ctx.drawImage(canvas, 0, 0);
    return scaled_canvas;
}
