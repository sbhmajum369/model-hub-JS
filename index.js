
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');
require('regenerator-runtime');
const yolo =  require('./packages/YOLO');


const main = async() => {
	// Model import 
	const yolov5 = yolo;
	console.log(yolov5.details());

	const status = await yolov5.load();

	console.log("Loading stat:", status);

};

main();
