
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const assert = require('assert');
require('regenerator-runtime');

const yolov5 =  require('../YOLO');

const loadImage = (filename) => {
	let image0;
	try {
		const imageBuffer = fs.readFileSync(path.join(__dirname, `./test_imgs/${filename}`));  // Pic_8_640.png  Zidane_640.jpg
		console.log("Image Loaded\n");
		image0 = tfnode.node.decodeImage(imageBuffer, 3);
	} catch (err) {
		console.error(`${err.message} \n`);
		return null;
	}
	
	// // Resize the image
	// image0 = tf.image.resizeBilinear(image0, size = [imageSize, imageSize]);

	image = tf.cast(image0, 'float32');
	image = image.div(tf.scalar(255));	// Converting into float32
	image = image.expandDims(0);			// to make it [1, W, H, D] from [W, H, D]
	return image;
};

const runInferenceCPU = async(image) => {
	// Loading Model
	const yolo = yolov5;
	console.log(yolo.model);
	
	try {
		let status = await yolo.load();		// Warm up the model by loading and predicting on dummy data of same shape
		console.log(status, "\n");
	} catch (err) {
		console.error(err.message, "\n");
	}

	if (yolo.model != '') {
		start = tf.util.now();
		const result = await yolo.predict(image);
		const predictions = yolo.getDetections(result);
		end = tf.util.now()

		return [predictions, (end-start)];
	} else {
		console.error("No model\n");
		return [null];
	}

	yolo.dispose;
};



const main = async() => {
	console.log(`Running on ${tf.getBackend()} backend \n`);

	describe('YOLO_V5', async() => {
		let predictions, yolo, time_taken;

		// Reading Image
		let image;
		imgFilename = 'Pic_8_640.jpg';
		image = loadImage(imgFilename);
		const expectedShape = [1,640,640,3];

		describe('Image loading', () => {
			it('should return image of shape [1,640,640,3]', () => {
				assert.strictEqual(image.shape.length, expectedShape.length);
			});
		});

		if (image != null) {
			[predictions, time_taken] = await runInferenceCPU(image);			
		}

		if (predictions.length > 0) {
			console.log("Predictions:", predictions);
			console.log(`time taken: ${time_taken/1000} s`)
		}
		
		yolo.dispose;
	});
	
};

main();
