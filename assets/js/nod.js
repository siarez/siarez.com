---
---    
var model_params;
const math = ENV.math;

var bn_w;
var bn_b;
var bn_mean;
var bn_variance;
var conv0_w ;
var dout0;
var fc1_b;
var fc1_w;
var fc2_b;
var fc2_w;



$(document).ready(function() {    
    //Loads the JSON file containing the model parameters
    $.getJSON( "{{ site.baseurl }}/assets/json/2018-03-22-10-54-24_net204.pth_params.json", function( data ) {
        model_params = data;
        // Creating weights and biases from the json file 
        bn_w = dl.tensor(data["bn_w"]["param"]);
        bn_b = dl.tensor(data["bn_b"]["param"]);
        bn_mean = dl.tensor(data["bn_mean"]["param"]);
        bn_var = dl.tensor(data["bn_var"]["param"]);
        conv0_w = dl.tensor4d(data["conv0_w"]["param"] );
        dout0 = dl.tensor(data["dout0"]["param"])
        fc1_b = dl.tensor(data["fc1_b"]["param"]);
        fc1_w = dl.tensor(data["fc1_w"]["param"]);
        fc2_b = dl.tensor(data["fc2_b"]["param"]);
        fc2_w = dl.tensor(data["fc2_w"]["param"]);
        console.log("model loaded!")
        landmarks_window = dl.zeros([68, 30, 2]);
        prob_window = dl.zeros([5]);
        landmarks_window_old = dl.zerosLike(landmarks_window);
        /*
        $.getJSON( "{{ site.baseurl }}/assets/json/dl.js_examples.json", function( data ) {
            probs = dl.tensor(data["probs"]);
            test_inputs = dl.tensor(data["inputs"]);
            for(i=0; i<test_inputs.shape[0]; i++){
                landmark_deltas = test_inputs.slice([i, 0, 0, 0], [1, 68, 30, 2]).squeeze();
                landmark_deltas = dl.batchNormalization(landmark_deltas, bn_mean, bn_var, 1e-05, bn_w, bn_b);
                tmp1 = dl.relu(dl.conv2d(landmark_deltas, conv0_w, [1, 1], 'valid')).squeeze();
                tmp1_flat = dl.reshape(tmp1.transpose(),[1632, 1]); 
                tmp2 = dl.relu(dl.add(dl.matMul(fc1_w, tmp1_flat).transpose(),  fc1_b));
                tmp3 = dl.sigmoid(dl.add(dl.matMul(fc2_w, tmp2.transpose()), fc2_b));     
                tmp3.print()
                
            }
            console.log("blah")


        })
        */
    });


    setInterval(function(){ 
        if (brfManager == null){
            return;
        }
        context.drawImage(player, 0, 0, canvas.width, canvas.height);
        imagedata = context.getImageData(0,0,640,480);
        brfManager.update(imagedata.data);
        det_face = brfManager.getAllDetectedFaces();
        mergeface = brfManager.getMergedDetectedFaces();
        var faces = brfManager.getFaces();
        context.fillStyle="#00ffba";
        for (i=0; i<68; i++){
            context.fillRect(faces[0].vertices[2*i], faces[0].vertices[2*i+1], 4, 4);
        }
        new_face = dl.tensor(faces[0].vertices).reshape([68, 2]).expandDims(1);
        // `window` has shape [68, 30, 2]
        landmarks_window = dl.concat([landmarks_window, new_face], 1);
        if (landmarks_window.shape[1] > 30){
            landmarks_window = landmarks_window.slice([0, 1, 0], [68, 30, 2]);
            landmark_deltas = landmarks_window.sub(landmarks_window_old)
            landmarks_window_old = landmarks_window;
            //bn_mean = dl.tensor([0, 0]);
            //bn_var = dl.tensor([1, 1]);
            flip = dl.tensor3d([[[1, -1]]]);  //Flip the y corrdinate to be in line with pytorch samples.
            landmark_deltas = landmark_deltas.mul(flip).div(dl.tensor1d([480])); //normalizing for resolution
            landmark_deltas = dl.batchNormalization(landmark_deltas, bn_mean, bn_var, 1e-05, bn_w, bn_b);
            tmp1 = dl.relu(dl.conv2d(landmark_deltas, conv0_w, [1, 1], 'valid')).squeeze();
            tmp1_flat = dl.reshape(tmp1.transpose(),[1632, 1]); 
            tmp2 = dl.relu(dl.add(dl.matMul(fc1_w, tmp1_flat).transpose(),  fc1_b));
            tmp3 = dl.sigmoid(dl.add(dl.matMul(fc2_w, tmp2.transpose()), fc2_b));

            context.fillStyle="#ff0058";
            context.fillRect(0, 0, 44, 100);
            context.fillStyle="#00ffba";
            prob_window = dl.concat([prob_window, tmp3.as1D()], 0).slice([1], [3]);
            prob_window_mean = prob_window.mean();
            prob_window_mean.data().then(data => context.fillRect(2, 100-100*data[0], 40, 100*data[0]));

        }

        //console.log(brfv4.sdkReady);
    }, 100);    


    const player = document.getElementById('player');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    const captureButton = document.getElementById('capture');
    const constraints = {
        video: true,
    };





    captureButton.addEventListener('click', () => {
        brfManager.reset();
        resolution = new brfv4.Rectangle(0, 0, 640, 480);
        brfManager.setFaceDetectionRoi(resolution);
        size = resolution.height;
        brfManager.setFaceDetectionParams(		size * 0.30, size * 1.00, 12, 8);
        brfManager.setFaceTrackingStartParams(	size * 0.30, size * 1.00, 22, 26, 22);
        brfManager.setFaceTrackingResetParams(	size * 0.25, size * 1.00, 40, 55, 32);      
    });

    // Attach the video stream to the video element and autoplay.
    navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
        player.srcObject = stream;
    })


})


function forward(window){}

//"BRFv4_JS_TK190218_v4.0.5_trial.wast"
var brfv4BaseURL = "{{ site.baseurl }}/assets/js/"
var brfv4 = {locateFile: function(fileName) { 
    return brfv4BaseURL + fileName; 
}};
initializeBRF(brfv4)

brfManager = null

setTimeout(function(){ 
    brfManager = new brfv4.BRFManager()
    resolution	= new brfv4.Rectangle(0, 0, 640, 480);
    var size = resolution.height;
    brfManager.init(resolution, resolution, "sia_nod_example")
    brfManager.setMode(brfv4.BRFMode.FACE_TRACKING);
    brfManager.setNumFacesToTrack(1);
    brfManager.setFaceDetectionRoi(resolution);
    //brfManager.setFaceDetectionParams(		size * 0.30, size * 1.00, 12, 8);
    //brfManager.setFaceTrackingStartParams(	size * 0.30, size * 1.00, 22, 26, 22);
    //brfManager.setFaceTrackingResetParams(	size * 0.25, size * 1.00, 40, 55, 32);
    //brfManager.init(resolution, resolution, "sia_nod_example")                        
}, 2000);




/*
if(!brfv4.sdkReady) {

    example.waitForSDK();

} else {

    trace("-> brfv4.sdkReady: " + brfv4.sdkReady);

    if(brfv4.BRFManager && !brfManager) {
        brfManager	= new brfv4.BRFManager();
    }

    if(brfv4.Rectangle && !resolution) {
        resolution	= new brfv4.Rectangle(0, 0, 640, 480);
    }

    if(brfManager === null || resolution === null) {
        trace("Init failed!", true);
        return;
    }

    if(type === "picture") {	// Start either using an image ...

        imageData.picture.setup(
            dom.getElement("_imageData"),
            imageData.onAvailable
        );

    } else {				// ... or start using the webcam.

        imageData.webcam.setup(
            dom.getElement("_webcam"),
            dom.getElement("_imageData"),
            resolution,
            imageData.onAvailable
        );
    }

    trace("-> imageData.isAvailable (" + imageData.type() + "): " + imageData.isAvailable());

    if(imageData.isAvailable()) {

        setupBRFExample();

    } else {

        resolution.setTo(0, 0, 640, 480); // reset for webcam initialization
        imageData.init();
    }
}
brfManager = brfv4.BRFManager()
brfManager.init(Rectangle(0, 0, 640, 480), Rectangle(0, 0, 640, 480), 123)
*/