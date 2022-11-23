// testwb.res

let showb = (a: Wb.wb) => {
    Js.log(a.w)
    Js.log(a.b)
}

let mat1 = [[1., 1.], [1., 1.]]
let b1 = [1., 1.]

let s1 = {Wb.w: mat1, b: b1}

s1->showb
//[[1, 1], [1, 1]]
//[1, 1]

//map
[s1]->Wb.map((x) => x +. 1.0)->Js.Array2.forEach(showb)
//[[2, 2], [2, 2]]
//[2, 2]

//addInPlace
Wb.addInPlace([s1], [s1])
s1->showb
//[[2, 2], [2, 2]]
//[2, 2]

//updateInPlace
Wb.updateInPlace([s1], [s1], 0.6)
s1->showb
//[[0.8, 0.8], [0.8, 0.8]]
//[0.8, 0.8]

let net1 = Wb.create([2, 2, 3], ()=> 1.0)->Belt.Result.getExn
net1->Js.Array2.forEach(showb)
// [[1, 1], [1, 1]]
//[1, 1]
// [[1, 1], [1, 1], [1, 1]]
//[1, 1, 1]

[1., 1.]->Wb.forwardOneStep(net1[0], Wb.bevec((x) => x))->Js.log
// {xi : [3, 3], yi: [3, 3]}
[3., 3.]->Wb.forwardOneStep(net1[1], Wb.bevec((x) => x))->Js.log
// {xi: [7, 7, 7], yi: [7, 7, 7]}
