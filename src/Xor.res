// xor.res
let state1 = Wb.create([2, 4, 10, 2], Js.Math.random)->Belt.Result.getExn
let fin = Nu.train_sgd(state1,
		       [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
		       [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
		       10000,
		       0.1,
		       Nu.vsigmoid,
		       Nu.vsigmoid_d,
		       Nu.error_E,
		       Nu.error_E_d,
		       (x, i) => 0.99999 ** Belt.Int.toFloat(i) *. x)
//(x, _) => x)

Js.log(Nu.inference([1.0, 0.0], fin, Nu.vsigmoid))
Js.log(Nu.inference([0.0, 0.0], fin, Nu.vsigmoid))
Js.log(Nu.inference([0.0, 1.0], fin, Nu.vsigmoid))
Js.log(Nu.inference([1.0, 1.0], fin, Nu.vsigmoid))
