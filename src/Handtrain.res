open Handnum

let tanh = (x) => (1.0 -. Js.Math.exp(-.x)) /. (1.0 +. Js.Math.exp(-.x))
let tanh_d = (x) => {
    let a = tanh(x)
    1.0 -. a *. a
}
let vtanh = Wb.bevec(tanh)
let vtanh_d = Wb.bevec(tanh_d)

let sumMax = (v) => {
    let max = v->Js.Array2.reduce((s, a) => {
	s > a ? s : a
    }, v[0])
    let sum = v->Js.Array2.reduce((s, a) => {
	s +. Js.Math.exp(a -. max)
    }, 0.0)
    v->Js.Array2.map((a) => Js.Math.exp(a -. max) /. sum)
}

let sumMax_d = (v) => {
    v->Js.Array2.map((a) => a *. (1.0 -. a))
}

let num = 100
let f = Nu.vsigmoid
let fd = Nu.vsigmoid_d
let epoch = 600
let rate = 10.0
let rated = (x, i) => {
    0.9994 ** Belt.Int.toFloat(i) *. x
}

let err = Nu.ce_E
let errd = Nu.ce_E_d

let state1 = Wb.create([784, 20, 10], Js.Math.random)->Belt.Result.getExn

let fin = Nu.train_sgd(state1,
		       traininputs->Js.Array2.slice(~start=0, ~end_=num),
		       trainoutputs->Js.Array2.slice(~start=0, ~end_=num),
		       epoch,
		       rate,
		       f,
		       fd,
		       err,
		       errd,
		       rated)

testinputs->Js.Array2.forEachi((a, i) => {
    Nu.inference(a, fin, f)->Js.log
    Js.log(testoutputs[i])
})

Nu.inference(traininputs[0], fin, f)->Js.log
Js.log(trainoutputs[0])
