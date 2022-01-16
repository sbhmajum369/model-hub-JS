# YOLO v5 in Node.JS

Object Detection using YOLO for Node.js backend.

***ES6 compatibile***


## Installation  

`npm i yolov5`

It also installs the required dependencies: `tfjs`, `tfjs-node` and `regenerator-runtime`  

For more info on tensorflow backend in node.js:  
(https://github.com/tensorflow/tfjs/tree/master/tfjs-node)  
(https://github.com/tensorflow/tfjs)

## Usage  

```javascript
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
require('regenerator-runtime');

const {YOLOv5s} =  require('./yolov5');


const main = async() => {
	// Reading Image
	const imgName = 'Pic_3_640.jpg'; // RGB images
	let imageBuffer = fs.readFileSync(path.join(__dirname, `${imgName}`));
	let image0 = tfnode.node.decodeImage(imageBuffer, 3);

	// Resize the image, if not 640x640
	// image0 = tf.image.resizeBilinear(image0, size = [imageSize, imageSize]);

	image = tf.cast(image0, 'float32');
	image = image.div(tf.scalar(255));                  // Converting to float32
	image = image.expandDims(0);
	
	// Loading Model
	const yolo = YOLOv5s;
	await yolo.load();
	let predictions;

	if (yolo.model != '') {
		console.log("Model loaded \n");
		const result = await yolo.predict(image);
		predictions = yolo.getDetections(result);
	} else {
		return console.error("No model");
	}

	if (predictions.length > 0) console.log("Predictions:", predictions);
	
	yolo.dispose;
};

main();

```

View model parameters, methods, functions: `console.log(yolo.details())`  


### Output  

Prediction format: 
```javascript
{
	"bbox": [409, 1, 639, 169],		// [x0, y0, x1, y1]
	"score": 0.6489,				// probability score
	"class": "car"					// Object class name
}
```

**Notes**  

* Current backend support only for CPU via `tfjs-node`.  

* Insert `regenerator-runtime` or its equivalent for async-await based functional implementation.  

* *tfjs-node* is optimized for some specific hardware: (AVX512, MKL-DNN).  

Some hardware optimized models run faster in CPU rather than GPU.  


## Test  

* coming soon ...


## Model conversion using tfjs-converter [Link](https://www.tensorflow.org/js/guide/conversion)  

`tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
 `

> We try to optimize the model for being served on the web by sharding the weights into 4MB files - that way they can be cached by browsers.  
We also attempt to simplify the model graph itself using the open source **Grappler** project. Graph simplifications include folding together adjacent operations, eliminating common subgraphs, etc. These changes have no effect on the model’s output.  
For further optimization, users can pass in an argument that instructs the converter to quantize the model to a certain byte size.  
>> ***Quantization*** is a technique for reducing model size by representing weights with fewer bits. Users must be careful to ensure that their model maintains an acceptable degree of accuracy after quantization.  


