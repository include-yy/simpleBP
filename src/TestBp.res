// TestBp.res

// sigmoid
let a = [-1.0, -0.5, 0.0, 0.5, 1.0]
Bp.sigmoid(. a[0])->Js.log
// 0.26894
Bp.vsigmoid(. a)->Js.log
//[0.2689414213699951,0.3775406687981454,0.5,0.6224593312018546,0.7310585786300049]
Bp.vsigmoid_d(. (a, Bp.vsigmoid(. a)))->Js.log
//[0.196, 0.235, 0.25, 0.235, 0.196]

//tanh
Bp.tanh(. a[0])->Js.log
//-0.7615941559557649
Bp.vtanh(. a)->Js.log
//[-0.7615, -0.4621 0, 0.4621, 0.7615]
Bp.vtanh_d(. (a, Bp.vtanh(. a)))->Js.log
//[0.4199, 0.7864 1, 0.7864, 0.4199]

//loss2
let at = [-0.5, -0.25, 0.0, 1.0, 2.0]
Bp.loss2(a, at)->Js.log
//0.78125
Bp.dloss2(a, at)->Js.log
//[-0.5, -0.25, 0, -0.5, -1]
Bp.dloss2dx((a, Bp.vsigmoid(. a)), [1., 1., 1., 1., 1.])->Js.log
//[-0.1437, -0.1462, -0.125, -0.0887, -0.0528]

//forward
let ne0 = S.create(~netarr=Bp.sup1_template.neta, //[2, 4, 3]
		   ~farr=Bp.sup1_template.farr, //[sigmoid]
		   ~narr=Bp.sup1_template.narr, //[1.0]
		   ~initfun=(_)=>1.0)->Belt.Result.getExn

let in0 = [1.0, 2.0]
Bp.forward(in0, ne0)->Js.log

//backward
let id = S.v((.x)=>x)
let id_d = (.xy: S.xy) => {
    let (x, _) = xy
    let len = x->Js.Array2.length
    Belt.Array.make(len, 1.0)
}

let ne1 = S.create(~netarr=[2, 4, 3],
		   ~farr=[(id, id_d), (id, id_d)],
		   ~narr=[(1.0, 0.0), (1.0, 0.0)],
		   ~initfun=(_)=>1.0)->Belt.Result.getExn
let d1 = [1.0, 1.0, 1.0]

let nxy = Bp.forward(in0, ne1)
Js.log(nxy)
let bk = Bp.backward(ne1, nxy, in0, d1)
bk->Belt.Array.forEach((a) => {
    Js.log(a.dw)
    Js.log(a.db)
})

Js.log("train_bgd")
//train_bgd
let su1 = {
    ...Bp.sup2_template,
    neta: [2, 3, 4, 1],
    inputVs: [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
    outputVs: [[1.0], [1.0], [0.0], [0.0]],
    epoch: 20000,
    etainit: 0.5
}

Js.Console.timeStart("1")
let r1 = Bp.train_bgd(su1, ~logfn=Bp.log_example, ())
Js.Console.timeEnd("1")
[1.0, 0.0]->Bp.forward(r1)->((x) => x[x->Js.Array2.length-1])->Js.log
[1.0, 1.0]->Bp.forward(r1)->((x) => x[x->Js.Array2.length-1])->Js.log
[0.0, 1.0]->Bp.forward(r1)->((x) => x[x->Js.Array2.length-1])->Js.log
[0.0, 0.0]->Bp.forward(r1)->((x) => x[x->Js.Array2.length-1])->Js.log

//train_sgd
Js.log("train_sgd")
Js.Console.timeStart("2")
let r1 = Bp.train_sgd(su1, 2, ~logfn=Bp.log_example, ())
Js.Console.timeEnd("2")
[1.0, 0.0]->Bp.forward(r1)->((x) => x[x->Js.Array2.length-1])->Js.log
[1.0, 1.0]->Bp.forward(r1)->((x) => x[x->Js.Array2.length-1])->Js.log
[0.0, 1.0]->Bp.forward(r1)->((x) => x[x->Js.Array2.length-1])->Js.log
[0.0, 0.0]->Bp.forward(r1)->((x) => x[x->Js.Array2.length-1])->Js.log

