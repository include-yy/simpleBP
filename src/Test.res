let net = {
    Bp.neta: [2, 3, 4, 1],
    farr: [(Bp.vsigmoid, Bp.vsigmoid_d),
	   (Bp.vsigmoid, Bp.vsigmoid_d),
	   (Bp.vsigmoid, Bp.vsigmoid_d)],
    narr: [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
    initf: Js.Math.random,
    inputVs: [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
    outputVs: [[1.0], [0.0], [1.0], [0.0]],
    epoch: 10000,
    etainit: 1.0,
    etafun: (. eta, _) => eta,
    floss: Bp.loss2,
    dlossdx: Bp.dloss2dx}

let res1 = Bp.train_bgd(net, ~logfn=Bp.log_example, ())

[1.0, 0.0]->Bp.forward(res1)->((x) => x[x->Js.Array2.length-1])->Js.log
[1.0, 1.0]->Bp.forward(res1)->((x) => x[x->Js.Array2.length-1])->Js.log
[0.0, 1.0]->Bp.forward(res1)->((x) => x[x->Js.Array2.length-1])->Js.log
[0.0, 0.0]->Bp.forward(res1)->((x) => x[x->Js.Array2.length-1])->Js.log

let a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
let num2vec = (a, n) => {
    Belt.Array.makeBy(n, (i) => {
	a == i ? 1.0 : 0.0
    })
}

let allin = a->Js.Array2.mapi((_, i) => {
    a->Js.Array2.mapi((_, j) => {
	Js.Array2.concat(i->num2vec(10), j->num2vec(10))
    })
})->Belt.Array.concatMany

let allout = a->Js.Array2.mapi((_, i) => {
    a->Js.Array2.mapi((_, j) => {
	(i+j)->num2vec(19)
    })
})->Belt.Array.concatMany

let net2 = {
    ...net,
    neta: [20, 5, 8, 19],
    inputVs: allin,
    outputVs: allout,
    epoch: 100000,
    etainit: 0.5
}

let res2 = Bp.train_bgd(net2, ())

let inference = (input, ne) => {
    let res = Bp.forward(input, ne)
    let (_, output) = res[res->Js.Array.length - 1]
    let r2 = output->Belt.Array.reduceWithIndexU((0, output[0]), (. r, a, i) => {
	let (_, v) = r
	if (v < a) {
	    (i, a)
	} else {
	    r
	}
    })
    let (ret, _) = r2
    ret
}

let succ = ref(0)

for i in 0 to 9 {
    for j in 0 to 9 {
	let a = allin[i * 10 + j]->inference(res2)
	if a == i + j {
	    succ := succ.contents + 1
	}
    }
}

succ->Js.log
