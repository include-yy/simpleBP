/*
// addnum.res
let num2vec = (num, base) => {
    let res = Js.Vector.make(base, 0.0)
    res[num] = 1.0
    res
}

let input1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]->Js.Array2.map((a) => num2vec(a, 10))
let input2 = [1, 1, 4, 5, 1, 4, 1, 9, 1, 9]->Js.Array2.map((a) => num2vec(a, 10))
let input = input1->Js.Array2.mapi((_, i) => Js.Array2.concat(input1[i], input2[i]))
let output = [1, 2, 6, 8, 5, 9, 7, 16, 9, 18]->Js.Array2.map((a) => num2vec(a, 19))

let state1 = Wb.create([20, 10, 10, 19], Js.Math.random)->Belt.Result.getExn
let fin = Nu.train_sgd(state1,
		       input,
		       output,
		       20000,
		       0.1,
		       Nu.vsigmoid,
		       Nu.vsigmoid_d,
		       (x, i) => 0.99999 ** Belt.Int.toFloat(i) *. x)
//(x, _) => x)
let input1_1 = Js.Array2.concat(1->num2vec(10), 1->num2vec(10))
let input5_3 = Js.Array2.concat(5->num2vec(10), 3->num2vec(10))
let input8_5 = Js.Array2.concat(8->num2vec(10), 5->num2vec(10))
Js.log(Nu.inference(input1_1, fin, Nu.vsigmoid))
Js.log(Nu.inference(input5_3, fin, Nu.vsigmoid))
Js.log(Nu.inference(input8_5, fin, Nu.vsigmoid))
*/

let num2vec = (num, base) => {
    let res = Js.Vector.make(base, 0.0)
    res[num] = 1.0
    res
}

let a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

let allin = a->Js.Array2.mapi((_, i) => {
    a->Js.Array2.mapi((_, j) => {
	Js.Array2.concat(i->num2vec(10), j->num2vec(10))
    })
})

let allout = a->Js.Array2.mapi((_, i) => {
    a->Js.Array2.mapi((_, j) => {
	(i+j)->num2vec(19)
    })
})

let ain = Js.Vector.make(100, [0.])
let aout = Js.Vector.make(100, [0.])

allin->Js.Array2.forEachi((a, i) => {
    a->Js.Array2.forEachi((_, j) => {
	ain[10*i+j] = allin[i][j]
	aout[10*i+j] = allout[i][j]
    })
})

let state1 = Wb.create([20, 5, 8, 19], Js.Math.random)->Belt.Result.getExn
let fin = Nu.train_sgd(state1,
		       ain,
		       aout,
		       10000,
		       0.1,
		       Nu.vsigmoid,
		       Nu.vsigmoid_d,
		       Nu.error_E,
		       Nu.error_E_d,
		       (x, _) => x)

let input1_1 = Js.Array2.concat(1->num2vec(10), 1->num2vec(10))
let input5_3 = Js.Array2.concat(5->num2vec(10), 3->num2vec(10))
let input8_5 = Js.Array2.concat(8->num2vec(10), 5->num2vec(10))
Js.log(Nu.inference(input1_1, fin, Nu.vsigmoid))
Js.log(Nu.inference(input5_3, fin, Nu.vsigmoid))
Js.log(Nu.inference(input8_5, fin, Nu.vsigmoid))
