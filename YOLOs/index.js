
const path = require('path');
const {YOLOv5s} =  require('./yolov5');
// more modules

new_path = path.join(__dirname, YOLOv5s.path)
YOLOv5s.path = new_path;
// console.log(YOLOv5s.path);

module.exports.YOLOv5s = YOLOv5s;
