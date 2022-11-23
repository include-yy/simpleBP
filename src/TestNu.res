// testnu.res

//activate function test
Nu.sigmoid(1.)->Js.log
// 0.7310...
Nu.sigmoid_d(0.)->Js.log
// 0.25
Nu.sigmoid_d(1.)->Js.log
// 0.1966...
Nu.relu(0.1)->Js.log
// 0.1
Nu.relu_d(0.1)->Js.log
// 1

[1.0, 1.0]->Nu.e([3.0, 3.0])->Js.log
// 8
[[1.0, 1.0], [2.0, 2.0]]->Nu.error_E([[2.0, 2.0], [4.0, 4.0]])->Js.log
// 5
[1.0, 1.0]->Nu.error_E_d(3, _, [3.0, 3.0])->Js.log
// [-1.333..., -1.333...]

// forward
Js.log("forward")
let state0 = Wb.create([2, 2, 3, 2], () => 1.0)->Belt.Result.getExn
[1.0, 1.0]->Nu.forward(state0, Nu.vid)->Js.Array2.forEach(Js.log)
// {xi: [3, 3], yi: [3, 3]}
// {xi: [7, 7, 7], yi: [7, 7, 7]}
// {xi: [22, 22], yi: [22, 22]}

[1.0, 1.0]->Nu.forward(state0, Wb.bevec((x)=>x+.1.0))->Js.Array2.forEach(Js.log)
//{ xi: [ 3, 3 ], yi: [ 4, 4 ] }
//{ xi: [ 9, 9, 9 ], yi: [ 10, 10, 10 ] }
//{ xi: [ 31, 31 ], yi: [ 32, 32 ] }

//backward
Js.log("backward")
state0->Js.Array2.forEach(TestWb.showb)
let ios = [1., 1.]->Nu.forward(state0, Nu.vid)
ios->Js.Array2.forEach(Js.log)
let grad = Nu.backward(state0, [ios], [[1., 1.]], [[22., 22.]], [[24., 24.]], Nu.vid_d, Nu.error_E_d)
grad->Js.Array2.forEach(Js.log)

//train_sgd xor operation
let state1 = Wb.create([2, 4, 10, 1], Js.Math.random)->Belt.Result.getExn
let fin = Nu.train_sgd(state1,
		       [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
		       [[1.], [0.], [1.], [0.]],
		       100000,
		       0.1,
		       Nu.vsigmoid,
		       Nu.vsigmoid_d,
		       Nu.error_E,
		       Nu.error_E_d,
		       (x, i) => 0.99999995 ** Belt.Int.toFloat(i) *. x)
//(x, _) => x)

fin->Js.Array2.forEach(Js.log)
Js.log(Nu.inference([1.0, 0.0], fin, Nu.vsigmoid))
Js.log(Nu.inference([0.0, 0.0], fin, Nu.vsigmoid))
Js.log(Nu.inference([0.0, 1.0], fin, Nu.vsigmoid))
Js.log(Nu.inference([1.0, 1.0], fin, Nu.vsigmoid))
