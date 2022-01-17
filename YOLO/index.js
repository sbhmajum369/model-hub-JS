
const path = require('path');
const {YOLOv5s} = require('./dist');


let final_path = path.join(__dirname, YOLOv5s.path);
YOLOv5s.path = final_path;

console.log("Path", final_path);


module.exports = YOLOv5s;