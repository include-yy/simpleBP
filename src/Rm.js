const fs = require('fs')
const bu = require('buffer')

let readbin = (fname) => {
    return fs.readFileSync(fname)
}

let test_data_fname = './t10k-images.idx3-ubyte'
let test_label_fname = './t10k-labels.idx1-ubyte'
let train_data_fname = './train-images.idx3-ubyte'
let train_label_fname = './train-labels.idx1-ubyte'

let get_train_arr = () => {
    let arr1 = new Array(60000)
    let arr2 = new Array(60000)
    let f1 = fs.readFileSync(train_data_fname)
    let f2 = fs.readFileSync(train_label_fname)
    let i = 0
    let item_len = 28 * 28
    for (i = 0; i < 60000; i++) {
	arr1[i] = new Array(item_len)
	arr2[i] = f2.readUInt8(8 + i)
	for (j = 0; j < item_len; j++) {
	    arr1[i][j] = f1.readUInt8(16 + item_len * i + j)
	}
    }
    return [arr1, arr2]
}

let get_test_arr = () => {
    let arr1 = new Array(10000)
    let arr2 = new Array(10000)
    let f1 = fs.readFileSync(test_data_fname)
    let f2 = fs.readFileSync(test_label_fname)
    let i = 0
    let item_len = 28 * 28
    for (i = 0; i < 10000; i++) {
	arr1[i] = new Array(item_len)
	arr2[i] = f2.readUInt8(8 + i)
	for (j = 0; j < item_len; j++) {
	    arr1[i][j] = f1.readUInt8(16 + item_len * i + j)
	}
    }
    return [arr1, arr2]
}

exports.getTrain = get_train_arr
exports.getTest = get_test_arr
