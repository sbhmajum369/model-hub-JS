const path = require('path');
// const webpack = require('webpack');
// https://webpack.js.org/guides/author-libraries/

module.exports = {
	mode: 'production',
	entry: './src/yolov5.js',
	devtool: 'source-map',
	output: {
		path: path.resolve(__dirname, 'dist'),
		filename: 'yolov5.min.js',
		library: 'YOLOv5',
		libraryTarget: 'umd',
		globalObject: 'this'
	},
	externals: ['@tensorflow/tfjs', '@tensorflow/tfjs-node'],
	module: {
		rules: [
			{
				test: /\.js$/,
				use: 'babel-loader',
				exclude: /node_modules/
			},
		],
	},
	resolve: {
		extensions: ['.js'],
	},
};