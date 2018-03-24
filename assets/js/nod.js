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
    var model_loaded = false;
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
        model_loaded = true;
        {%comment%}
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
        {%endcomment%}
    });

    //Initializing `landmarks_window` which is the rolling window of the 68 facial landmarks
    var landmarks_window = dl.zeros([68, 30, 2]);
    var landmarks_window_old = dl.zerosLike(landmarks_window);
    var prob_window = dl.zeros([5]);  //Rolling window of probabilities. Use for moving average.

    setInterval(function(){ 
        //This is the main loop where we grab images and run them through the model.
        //We are processing 10 frames per second. Which is what the model is trained on.
        if (brfManager == null || model_loaded == false){
            return;
        }
        context.drawImage(player, 0, 0, canvas.width, canvas.height);
        imagedata = context.getImageData(0,0,640,480);
        brfManager.update(imagedata.data);  //Pass webcam data to facial landmark tracker
        det_face = brfManager.getAllDetectedFaces();
        mergeface = brfManager.getMergedDetectedFaces();
        var faces = brfManager.getFaces();
        context.fillStyle="#00ffba";
        for (i=0; i<68; i++){
            //Marking landmarks with points
            context.fillRect(faces[0].vertices[2*i], faces[0].vertices[2*i+1], 4, 4);
        }
        new_face = dl.tensor(faces[0].vertices).reshape([68, 2]).expandDims(1);
        // `landmarks_window` has shape [68, 30, 2]
        landmarks_window = dl.concat([landmarks_window, new_face], 1);
        if (landmarks_window.shape[1] > 30){
            // landmarks_window reaches 30 frames it starts inference
            landmarks_window = landmarks_window.slice([0, 1, 0], [68, 30, 2]);
            landmark_deltas = landmarks_window.sub(landmarks_window_old); //Calculating delta x and y
            landmarks_window_old = landmarks_window;
            flip = dl.tensor3d([[[1, -1]]]);  //Flip the y corrdinate to be in line with pytorch pipeline.
            landmark_deltas = landmark_deltas.mul(flip).div(dl.tensor1d([480])); //normalizing for resolution
            landmark_deltas = dl.batchNormalization(landmark_deltas, bn_mean, bn_var, 1e-05, bn_w, bn_b);
            tmp1 = dl.relu(dl.conv2d(landmark_deltas, conv0_w, [1, 1], 'valid')).squeeze();
            tmp1_flat = dl.reshape(tmp1.transpose(),[1632, 1]); 
            tmp2 = dl.relu(dl.add(dl.matMul(fc1_w, tmp1_flat).transpose(),  fc1_b));
            tmp3 = dl.sigmoid(dl.add(dl.matMul(fc2_w, tmp2.transpose()), fc2_b));
            //Drawing probablity bar
            context.fillStyle="#ff0058";
            context.fillRect(0, 0, 44, 100);
            context.fillStyle="#00ffba";
            prob_window = dl.concat([prob_window, tmp3.as1D()], 0).slice([1], [5]);
            prob_window_mean = prob_window.mean();
            prob_window_mean.data().then(function(data){
                bar_height = (Math.exp(data[0]) - 1)/(Math.E - 1)
                context.fillRect(2, 100-100*bar_height, 40, 100*bar_height)
                context.fillStyle="#000000";
                context.fillRect(0, 10, 44, 2);
            });

        }
    }, 100);    

    const player = document.getElementById('player');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    const captureButton = document.getElementById('capture');
    const constraints = { video: { width: 640, height: 480 }};         
    // Attach the video stream to the video element and autoplay.
    navigator.mediaDevices.getUserMedia(constraints)
        .then((stream) => {
        player.srcObject = stream;
    })         
    captureButton.addEventListener('click', () => {
        brfManager.reset();
        resolution = new brfv4.Rectangle(0, 0, 640, 480);
        brfManager.setFaceDetectionRoi(resolution);
        size = resolution.height;
        brfManager.setFaceDetectionParams(		size * 0.30, size * 1.00, 12, 8);
        brfManager.setFaceTrackingStartParams(	size * 0.30, size * 1.00, 22, 26, 22);
        brfManager.setFaceTrackingResetParams(	size * 0.25, size * 1.00, 40, 55, 32);      
    });
})

//Initializing facial landmark tracking     
var brfv4BaseURL = "{{ site.baseurl }}/assets/js/"
var brfv4 = {locateFile: function(fileName) { 
    return brfv4BaseURL + fileName; 
}};
var brfManager = null;
initializeBRF(brfv4);
var interval_id = setInterval(function(){ 
    //Checking for face tracker to initilize.
    if (brfv4.sdkReady == true){
        brfManager = new brfv4.BRFManager();
        resolution	= new brfv4.Rectangle(0, 0, 640, 480);
        var size = resolution.height;
        brfManager.init(resolution, resolution, "sia_nod_example");
        brfManager.setMode(brfv4.BRFMode.FACE_TRACKING);
        brfManager.setNumFacesToTrack(1);
        brfManager.setFaceDetectionRoi(resolution);  
        clearInterval(interval_id)  //We disable interval timer after initialization.
    }   
}, 100);      
         
