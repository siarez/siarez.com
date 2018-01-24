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
//const z_samples_variational = [[-1.8590235710144043, -0.24844303727149963, 0.03938889503479004, 0.17331397533416748, 1.0505017042160034, 0.9950094223022461, -1.7530736923217773, 1.1678709983825684, 1.0481579303741455, 0.1736016571521759], [1.8138147592544556, 0.05395984649658203, -0.3482455611228943, 0.1188390851020813, -0.22176092863082886, 1.6050734519958496, -0.4279089570045471, -0.49581414461135864, -0.25909939408302307, 1.0818519592285156], [-0.779690146446228, 1.4649542570114136, 0.3086605668067932, -0.4681479334831238, -1.835047721862793, 0.08315752446651459, -0.17368635535240173, -0.31670674681663513, -0.6665403842926025, -0.17574849724769592], [0.16583821177482605, -0.4126869738101959, -0.39579904079437256, 2.2655272483825684, 0.1854362189769745, 0.02697370946407318, 0.1866011917591095, 0.5017563104629517, -0.4390926659107208, 1.5080211162567139], [1.1414896249771118, -0.9971839189529419, 0.8578649759292603, 0.09377895295619965, -0.22166454792022705, -2.1288554668426514, 1.3423423767089844, -0.7895694971084595, -1.049308180809021, -0.7693414688110352], [0.5912086963653564, -0.04737105965614319, 0.797297477722168, 0.13691873848438263, 0.7915048599243164, -0.8615074157714844, -0.25815221667289734, -1.224874496459961, -0.15084931254386902, 1.3192670345306396], [-0.566352128982544, 0.058658212423324585, -0.6304460763931274, 0.12933778762817383, 0.301516056060791, -1.092008113861084, -0.20680660009384155, -1.9942042827606201, -1.5705227851867676, -2.333134412765503], [1.9122259616851807, 0.6958446502685547, 1.4599683284759521, 0.30917149782180786, 0.3022534251213074, 0.18424956500530243, -0.2518792748451233, 0.3763713538646698, -1.381750464439392, 0.1558062732219696], [-0.08382529020309448, -1.9176644086837769, 0.3104326128959656, -0.6168606877326965, 1.534500241279602, -0.6024224758148193, 1.3395912647247314, 1.1119931936264038, 0.08513358235359192, -0.06711915135383606], [0.2780965566635132, -0.3152138888835907, -0.31437426805496216, -1.5764628648757935, 1.3285795450210571, 0.774539589881897, -0.7920351624488831, -0.07463780045509338, 0.6038618087768555, 0.07033351063728333]]
const z_samples_variational = [[0.4049130082130432, -1.9565716981887817, 0.827752411365509, -0.021063804626464844, 1.3692607879638672, 1.3532993793487549, 1.823057770729065, -0.21334154903888702, -0.8364218473434448, -0.28357648849487305], [-1.7514234781265259, 0.15920013189315796, -0.3407479524612427, -0.41083937883377075, -1.5722825527191162, -0.17966397106647491, -0.5082039833068848, -0.19852054119110107, 0.03948473930358887, 0.5983293056488037], [0.6104820966720581, -0.7903083562850952, 0.6327188611030579, -1.0710773468017578, -0.43764108419418335, 0.9393762350082397, -1.940561294555664, -0.6684684753417969, 0.24541017413139343, -0.546272873878479], [-0.4278321862220764, -1.6648595333099365, -0.6275712847709656, -0.4636905789375305, -0.09282533824443817, -1.398398995399475, -1.644773244857788, -0.17422132194042206, -0.5577913522720337, -0.8413596153259277], [0.9634431600570679, -0.1788099706172943, 0.8036679625511169, -0.7048096656799316, -0.6393207907676697, 0.8606389760971069, -0.19831764698028564, 0.5209050178527832, 1.0712555646896362, 1.7287027835845947], [-0.05610927939414978, -0.7960431575775146, 0.24666297435760498, -0.6914860606193542, -0.3009639382362366, -1.0232129096984863, 1.4211150407791138, 0.20904628932476044, 1.5280414819717407, -0.22104144096374512], [-0.3029842972755432, -0.8292033672332764, -1.5752322673797607, -0.3048386573791504, 1.5001537799835205, 0.9294049739837646, -0.4151242971420288, 0.5717505216598511, 1.7416270971298218, -0.6541085243225098], [-0.3162462115287781, 0.665153980255127, 0.6894911527633667, 0.42562776803970337, -1.1071784496307373, 0.3325573801994324, -0.5324566960334778, -0.04607619345188141, 0.3430352807044983, -0.1491698920726776], [0.6613258123397827, 1.6670010089874268, -0.5764290690422058, -0.0401727557182312, 0.038344502449035645, 2.2074074745178223, 0.7936835289001465, 0.341566801071167, -0.6055392026901245, -1.081760287284851], [0.6868233680725098, -0.32027241587638855, 0.8879053592681885, -0.0217362642288208, -1.054182767868042, -0.48369836807250977, -0.4341193437576294, 0.10270059108734131, -0.3390662968158722, 1.2683374881744385]]
const z_samples_variational_sigma = [[0.25263410806655884, 0.24940825998783112, 0.31125378608703613, 0.37933382391929626, 0.22476598620414734, 0.3433116674423218, 0.3424946963787079, 0.5089088678359985, 0.35011622309684753, 0.27186647057533264], [0.35611987113952637, 0.4644260108470917, 0.3306243419647217, 0.7036866545677185, 0.2983478903770447, 0.6943286657333374, 0.6004420518875122, 0.8524656295776367, 0.47182729840278625, 0.3410215973854065], [0.22968155145645142, 0.28343769907951355, 0.2645079791545868, 0.4717348515987396, 0.2021329551935196, 0.3812430500984192, 0.3683418333530426, 0.5244908928871155, 0.3089626431465149, 0.21030113101005554], [0.23837712407112122, 0.3279837369918823, 0.26797619462013245, 0.5253239274024963, 0.23151014745235443, 0.3988911509513855, 0.36018070578575134, 0.532271146774292, 0.3107871413230896, 0.2687269151210785], [0.27614155411720276, 0.2773597836494446, 0.2983121871948242, 0.5539104342460632, 0.2409958839416504, 0.4918172061443329, 0.5085873007774353, 0.554606020450592, 0.3787217438220978, 0.2813015878200531], [0.2398422807455063, 0.2938997745513916, 0.2860482335090637, 0.46563947200775146, 0.21171388030052185, 0.3971167206764221, 0.3842112123966217, 0.5925421118736267, 0.31398889422416687, 0.24975794553756714], [0.26213565468788147, 0.23879969120025635, 0.2562310993671417, 0.440286785364151, 0.21410216391086578, 0.4079645574092865, 0.298628032207489, 0.5131410360336304, 0.2610902190208435, 0.2533423602581024], [0.25594833493232727, 0.30780863761901855, 0.28443244099617004, 0.47302135825157166, 0.23398272693157196, 0.450023889541626, 0.4716951847076416, 0.660004734992981, 0.3580530881881714, 0.2148558795452118], [0.22789150476455688, 0.2640599310398102, 0.24462401866912842, 0.49935227632522583, 0.20980052649974823, 0.41834041476249695, 0.4214743375778198, 0.613699734210968, 0.35376250743865967, 0.21604593098163605], [0.26226797699928284, 0.27749642729759216, 0.285738080739975, 0.48067817091941833, 0.2420840561389923, 0.41640374064445496, 0.49258163571357727, 0.5731062889099121, 0.3716283440589905, 0.23773513734340668]]

const z_vanilla = [ 16.2604, 0.0, 8.7685, 19.2652, 0.0, 0.0, 18.8555, 15.8918, 5.6524, 15.3134]


$(document).ready(function() {
    createSliders($(".slidecontainer"));
    
    //Loads the JSON file containing the model parameters
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
        //initialize slider positions
        setSliders(z_samples_variational[0])
        // Creates the drawing pad
        createPixelGrid($(".pixel-container"));
        $('.pixel-container').pixelPicker({update: pixelPickerUpdate});
    });
})


function generate(z){
    /* Takes take hidden representation and decoder parameters to generate an image */
    let dec_hid = math.relu(math.add(math.vectorTimesMatrix(z, math.transpose(dec_hid_weight)), dec_hid_bias));
    let dec_out = math.sigmoid(math.add(math.vectorTimesMatrix(dec_hid, math.transpose(dec_out_weight)), dec_out_bias));
    return dec_out;
}

function encode(image){
    /* Encodes the image. i.e. calculates the hidden representaion z. It includes mu and sigma */
    let enc_hid = math.relu(math.add(math.vectorTimesMatrix(image, math.transpose(enc_in_weight)), enc_in_bias));
    let enc_out = math.add(math.vectorTimesMatrix(enc_hid, math.transpose(enc_hid_weight)), enc_hid_bias);
    return enc_out;
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
        let slider = $('<div class="row"><input type="range" min="-5" max="5" value="0" step="0.1" class="slider col-10" id="range'+i+'" data-index='+i+'><span class="col-2">0</span></div>')
         slider.children(".slider").on("change mousemove", function() {
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
    //This is called every time pixel picker (the drawing board) is updated.
    pixels = [];
    map.forEach(function(row){
        row.forEach(function(pixel){
            pixels.push(arrayEqual(pixel, [255, 255, 255]) ? 1 : 0 )
        });
    });
    //console.log(pixels);
    encoding = encode(dl.Array1D.new(pixels));
    encoding_mu = encoding.dataSync().slice(0,10);
    setSliders(encoding_mu);
}

function updateTweakedImage(z_representation){
    new_image = generate(dl.Array1D.new(z_representation));
    $('#mnist_reconstruction canvas').remove();
    $('#mnist_reconstruction').append(renderMnistImage(new_image));
}

function setSliders(z_representation){
    //Sets sliders to a new position and updates the generates image
    $(".slidecontainer > div.row >.slider").each(function(){
        $(this).val(z_representation[$(this).data("index")]);
        $(this).next().html($(this).val());   // updates the labal
    });
    updateTweakedImage(z_representation);
}

function getSliderValues(){
    //Returns an array containing slider values.
    values = []
    $(".slidecontainer > div.row >.slider").each(function(){
        values.push($(this).val());
    });
    return values;
}

function arrayEqual(a, b) {
    return a.length === b.length && a.every(function(elem, i) {
        return elem === b[i];
    });
}
