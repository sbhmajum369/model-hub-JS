
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
require('regenerator-runtime');

const {yolov5s} =  require('./yolov5');


const main = async() => {
	// Reading Image
	
	const imgName = 'Pic_3_640.jpg';
	let imageBuffer = fs.readFileSync(path.join(__dirname, `./test_imgs/${imgName}`));  // Pic_8_640.png  Zidane_640.jpg
	if (imageBuffer) console.log("\nImage Loaded");
	let image0 = tfnode.node.decodeImage(imageBuffer, 3);

	// // Resize the image
	// image0 = tf.image.resizeBilinear(image0, size = [imageSize, imageSize]);

	image = tf.cast(image0, 'float32');
	image = image.div(tf.scalar(255));	// Converting into float32
	image = image.expandDims(0);			// to make it [1, W, H, D] from [W, H, D]
	console.log("\nImg. Shape:", image.shape); 
	
	// Loading Model
	await yolov5s.load();
	let predictions;

	if (yolov5s.model != '') {
		console.log("\nModel loaded\n");
		// console.log(yolov5s.classNames);
		const result = await yolov5s.predict(image);
		predictions = yolov5s.getDetections(result);
		// model.dispose();
	} else {
		return console.error("No model");
	}

	if (predictions.length > 0) console.log("Predictions:", predictions);
	
};

main();
