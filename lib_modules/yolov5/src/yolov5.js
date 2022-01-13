'use strict'

/*	YOLO v5-s: Production
	Prepared as ES module for both AMD and CommonJS compatibility.
*/

import * as tf from '@tensorflow/tfjs';
import * as tfnode from '@tensorflow/tfjs-node';

const {YOLO_CLASSES} = require('./object_items.js');
const PATH_YOLOV5_s = './yolov5/src/yolov5s/model.json';

class yolo_model_v5 {
	constructor(mdlPath, classNames) {
		this.path = mdlPath;
		this.classNames = classNames;
		this.image_shape = [];
		this.model = '';
		this.load = this.load.bind(this);
		this.getDetections = this.getDetections.bind(this);
	};

	load = async () => {
		try {
			this.model = await tfnode.loadGraphModel(`file://${this.path}`);
			// return model;
		} catch(err) {
			console.log(err);
		};
	};

	xywh2xyxy = (x,y,w,h) => {
		let x0 = parseInt(x - (w/2));
		let y0 = parseInt(y - (h/2));
		let x1 = parseInt(x + (w/2));
		let y1 = parseInt(y + (h/2));
		
		return [x0, y0, x1, y1];
	};

	predict = async (img) => {
		this.image_shape = img.shape;
		return await this.model.execute(img);
	};

	getDetections = (result, conf_thresh = 0.25, iou_thresh = 0.40) => {
		let imgSize = this.image_shape[1];
		const [MIN_WH, MAX_WH] = [2, 4096];  	// (pixels) minimum and maximum box width and height
		const MAX_NMS = 30000; 					// maximum number of boxes into torchvision.ops.nms()

		result = tf.tensor(result.arraySync()[0]);

		var candidates = result.slice([0,4], [-1,1]);

		// Candidate selection based on conf_thresh
		const xc = [];
		let x = [];
		candidates.dataSync().forEach((elem) => {
			xc.push(Boolean(elem > conf_thresh));
		});

		for (let i=0; i < xc.length; i++) {	
			if (xc[i]) {
				x.push(result.arraySync()[i]);
			}
		};
		// console.log("Candidates Shape:", x.length);

		if (!x || !x.length) {
			console.log("No_candidate_above_conf.threshold");
			return [];
		}
		x = tf.tensor(x);

		// Computing confidences
		let x1 = x.slice([0,0], [-1,5]);
		let x2 = x.slice([0,5]).mul(x.slice([0,4], [-1,1]));
		x = tf.concat([x1, x2], 1)
		// console.log('After Confidence Calc.:', x.shape);

		var box = x.slice([0,0], [-1,4]);

		// Best Class
		x2 = x.slice([0,5]);
		const conf = [], j = [], filteredData = [];
 
		x2.arraySync().forEach(val => {
			let max = Math.max(...val);
			let indx = val.indexOf(max);
			conf.push(max);				// Confidence score
			j.push(indx);				// conf_score indices
		});
		// console.log(Math.max(...conf), j);

		if (!Math.max(...conf) || (Math.max(...conf) < conf_thresh)) {
			console.log("No_objects_detected");
			return [];
		}

		for (let i=0; i < conf.length; i++) {		// Concatinating based on conf_thres
			if (conf[i] > conf_thresh) {
				let tempx = box.arraySync()[i].concat(conf[i], parseFloat(j[i]))
				filteredData.push(tempx);
			}
		};

		// Detection matrix nx6 (xywh, conf, cls)
		if (!filteredData || !filteredData.length) throw new Error('Detection matrix empty');

		x = tf.tensor(filteredData);

		// Batched NMS 
		var c = x.slice([0,5], [-1,1]).mul(MAX_WH);
		var boxes = x.slice([0,0], [-1,4]).add(c);
		var scores = x.slice([0,4], [-1,1]).dataSync(); // Converting to 1D Tensor

		// NMS 
		let i = tf.image.nonMaxSuppressionWithScore(boxes, scores, MAX_NMS, iou_thresh);

		// Output
		const output = [];
		let selectedIndices = i.selectedIndices.dataSync();
		
		selectedIndices.forEach(idx => {
			var bBox = x.slice([idx,0], [1, 4]).arraySync();
			var score = x.slice([idx,4], [1, 1]).dataSync();
			var classes = x.slice([idx,5], [1, 1]).dataSync();

			bBox = bBox[0].map(x => parseInt(x * imgSize));		// Mapping output float to box co-ordinates

			var data = {
				"bbox": this.xywh2xyxy(bBox[0], bBox[1], bBox[2], bBox[3]),
				"score": parseFloat(score).toFixed(4),
				"class": this.classNames[classes]
			};
			output.push(data);
		});

		return output;
	};
};

const yolov5s = new yolo_model_v5(PATH_YOLOV5_s, YOLO_CLASSES);

export {yolov5s};

// export {..., yolov5m};
