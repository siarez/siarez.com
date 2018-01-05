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

//there are 10 arrays in z_samples_variational, representing digits 0-9
const z_samples_variational = [[-1.8590235710144043, -0.24844303727149963, 0.03938889503479004, 0.17331397533416748, 1.0505017042160034, 0.9950094223022461, -1.7530736923217773, 1.1678709983825684, 1.0481579303741455, 0.1736016571521759], [1.8138147592544556, 0.05395984649658203, -0.3482455611228943, 0.1188390851020813, -0.22176092863082886, 1.6050734519958496, -0.4279089570045471, -0.49581414461135864, -0.25909939408302307, 1.0818519592285156], [-0.779690146446228, 1.4649542570114136, 0.3086605668067932, -0.4681479334831238, -1.835047721862793, 0.08315752446651459, -0.17368635535240173, -0.31670674681663513, -0.6665403842926025, -0.17574849724769592], [0.16583821177482605, -0.4126869738101959, -0.39579904079437256, 2.2655272483825684, 0.1854362189769745, 0.02697370946407318, 0.1866011917591095, 0.5017563104629517, -0.4390926659107208, 1.5080211162567139], [1.1414896249771118, -0.9971839189529419, 0.8578649759292603, 0.09377895295619965, -0.22166454792022705, -2.1288554668426514, 1.3423423767089844, -0.7895694971084595, -1.049308180809021, -0.7693414688110352], [0.5912086963653564, -0.04737105965614319, 0.797297477722168, 0.13691873848438263, 0.7915048599243164, -0.8615074157714844, -0.25815221667289734, -1.224874496459961, -0.15084931254386902, 1.3192670345306396], [-0.566352128982544, 0.058658212423324585, -0.6304460763931274, 0.12933778762817383, 0.301516056060791, -1.092008113861084, -0.20680660009384155, -1.9942042827606201, -1.5705227851867676, -2.333134412765503], [1.9122259616851807, 0.6958446502685547, 1.4599683284759521, 0.30917149782180786, 0.3022534251213074, 0.18424956500530243, -0.2518792748451233, 0.3763713538646698, -1.381750464439392, 0.1558062732219696], [-0.08382529020309448, -1.9176644086837769, 0.3104326128959656, -0.6168606877326965, 1.534500241279602, -0.6024224758148193, 1.3395912647247314, 1.1119931936264038, 0.08513358235359192, -0.06711915135383606], [0.2780965566635132, -0.3152138888835907, -0.31437426805496216, -1.5764628648757935, 1.3285795450210571, 0.774539589881897, -0.7920351624488831, -0.07463780045509338, 0.6038618087768555, 0.07033351063728333]]
const z_vanilla = [ 16.2604, 0.0, 8.7685, 19.2652, 0.0, 0.0, 18.8555, 15.8918, 5.6524, 15.3134]
const z_9 = z_samples_variational[5].slice();



$(document).ready(function() {
    createSliders($(".slidecontainer"));
    createPixelGrid($(".pixel-picker-container"));
    $('.pixel-picker-container').pixelPicker({update: pixelPickerUpdate});
    
    $.getJSON( "{{ site.baseurl }}/assets/json/model_params_variational.json", function( data ) {
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
        //Generating sample images. A canvas for each digit.
        for (let idx=0; idx<10; idx++){            
            let canvas = $(renderMnistImage(generate(dl.Array1D.new(z_samples_variational[idx])), 2));
            canvas.attr("data-digit", idx);
            canvas.on("click", function(){
                setSliders(z_samples_variational[$(this).data("digit")]) ;
            });
            $('#mnist_sample').append(canvas);
        }
        
        setSliders(z_samples_variational[0])
        
    });

})


function generate(z){
    /*Takes take hidden representation and decoder parameters to generate an image*/
    dec_hid = math.relu(math.add(math.vectorTimesMatrix(z, math.transpose(dec_hid_weight)), dec_hid_bias));
    dec_out = math.sigmoid(math.add(math.vectorTimesMatrix(dec_hid, math.transpose(dec_out_weight)), dec_out_bias));
    return dec_out;
}

function renderMnistImage(image_array, scale_factor = 5 ) {
    /*Creates a 28x28 image canvas. It takes pixel values from `image_array`*/
    const width = 28;
    const height = 28;
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
    const float32Array = image_array.dataSync();
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

function createSliders(slidecontainer){
    /*Creates sliders for tweaking z*/
    for (let i = 0; i < 10; ++i) {
        // Adding sliders for adjusting z
        let slider = $('<input type="range" min="-10" max="10" value="0" step="0.1" class="slider" id="range'+i+'" data-index='+i+'><span>0</span>')
         slider.on("change mousemove", function() {
            let val = $(this).val();
            let idx = $(this).data("index");
            let new_z = getSliderValues()
            $(this).next().html(val);   // updates the labal
            new_z[idx] = val;
            setSliders(new_z);

        });
        slidecontainer.append(slider);

    }
}

function createPixelGrid(container){
    for (let i=0; i < 28; i++){
        let row = $('<div class="pixel-picker-row">');
        for (let j=0; j < 28; j++){
            row.append($('<div class="pixel-picker-cell"></div>'));
        }
        container.append(row);
    }
}


function pixelPickerUpdate(map){
    pixels = [];
    map.forEach(function(row){
        row.forEach(function(pixel){
            pixels.push(arrayEqual(pixel, [255, 255, 255]) ? 1 : 0 )
        });
    });
    console.log(pixels);
}

function updateTweakedImage(z_representation){
    new_image = generate(dl.Array1D.new(z_representation));
    $('#mnist_sample_tweaked canvas').remove();
    $('#mnist_sample_tweaked').append(renderMnistImage(new_image));
}

function setSliders(z_representation){
    $(".slidecontainer").children(".slider").each(function(){
        $(this).val(z_representation[$(this).data("index")]);
        $(this).next().html($(this).val());   // updates the labal
    });
    updateTweakedImage(z_representation)
}

function getSliderValues(){
    values = []
    $(".slidecontainer").children(".slider").each(function(){
        values.push($(this).val());
    });
    return values;
}

function arrayEqual(a, b) {
    return a.length === b.length && a.every(function(elem, i) {
        return elem === b[i];
    });
}
