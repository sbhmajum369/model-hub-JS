
const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');
require('regenerator-runtime');
const {YOLOv5s} =  require('./YOLOs');

const main = async() => {
	const yolo = YOLOv5s;
	await yolo.load();

	console.log(yolo.classNames);

};

main();
