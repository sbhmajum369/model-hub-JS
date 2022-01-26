
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const assert = require('assert');

require('regenerator-runtime');

const yolov5 =  require('../packages/YOLO');


const loadImage = (filename) => {
	let image0, image, imageBuffer;
	
	const defaultImageShape = [640, 640, 3];

	try {
		imageBuffer = fs.readFileSync(path.join(__dirname, `./test_imgs/${filename}`));  // Pic_8_640.png  Zidane_640.jpg
		console.log(`${filename} Loaded \n`);
	} catch (err) {
		console.error(`${err.message} \n`);
		return null;
	};
	image0 = tfnode.node.decodeImage(imageBuffer, 3);

	const matchImgShape = image0.shape.every((val, ind) => val === defaultImageShape[ind]);
	
	if (!matchImgShape) {
		console.log('RESIZING... \n');
		// Resize the image
		image0 = tf.image.resizeBilinear(image0, size = [defaultImageShape[0], defaultImageShape[1]]);
	}
	
	image = tf.cast(image0, 'float32');
	image = image.div(tf.scalar(255));		// Converting into float32
	image = image.expandDims(0);			// to make it [1, W, H, D] from [W, H, D]
	
	return image;
};

const loadModel = async(model) => {
	let status = await model.load();		// Warm up the model by loading and predicting on dummy data of same shape
	console.log(status, "\n");
	// console.log(model.model);
	return status;
};

const imageLoad_test = () => {
	const targetShape = [1,640,640,3];
	
	// let i=5;
	for (let i=1; i<=8; i++) {
	
		// Reading Image
		const imgFilename = `Pic_${i}_640.jpg`;		// `Pic_${i}_640.jpg` 
		let image = loadImage(imgFilename);
		
		describe('Image loading Test', () => {
			it('should return a shape [1,640,640,3]', () => {
				assert.deepEqual(image.shape, targetShape);
			})
		});
	};
	
};

const modelLoad_test = async(inpModel) => {
	let predictions, time_taken;

	// Creating an instance 
	const model = inpModel;
	
	const targetStat = 'Model_Loaded';
	
	describe('Model loading Test', () => {
		it('should return isLoaded msg', async() => {
			// Loading Model
			const status = await loadModel(model);
			assert.equal(status, targetStat);
		});
	});
	
	model.dispose;
}

console.log(`Running on ${tf.getBackend()} backend \n`);

imageLoad_test();

modelLoad_test(yolov5);
