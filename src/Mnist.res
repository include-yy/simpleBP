@module("./Rm.js")
external getTrain: () => (array<array<float>>, array<int>) = "getTrain"

@module("./Rm.js")
external getTest: () => (array<array<float>>, array<int>) = "getTest"

let (tr_ivs, tr_ios) = getTrain()
let (te_ivs, te_ios) = getTest()

let softmax = (. v) => {
    let maxval = v->Belt.Array.reduceU(v[0], (. m, x) => {
	m > x ? m : x
    })
    let expres = v->Belt.Array.mapU((. x) => {
	Js.Math.exp(x -. maxval)
    })
    let sum = expres->Belt.Array.reduceU(0.0, (. sum, x) => sum +. x)
    M.sf2vf(. expres, (. x) => x /. sum)
}

let crossloss = (y, yt) => {
    y->Belt.Array.reduceWithIndexU(0.0, (. sum, _, i) => {
	sum -. yt[i] *. Js.Math.log(y[i])
    })
}

let dcrosslossdx = (xy: S.xy, yt: array<float>) => {
    let (_, y) = xy
    M.vvsub(. y, yt)
}

let tr_ipt = tr_ivs->Belt.Array.mapU((. a) => {
    a->Belt.Array.mapU((. b) => {
	b /. (255.0)
    })
})

let tr_opt = tr_ios->Belt.Array.mapU((. a) => {
    Belt.Array.makeBy(10, (i) => {
	i == a ? 1.0 : 0.0
    })
})

let te_ipt = te_ivs->Belt.Array.mapU((. a) => {
    a->Belt.Array.mapU((. b) => {
	b /. (255.0)
    })
})

let tin = tr_ipt->Belt.Array.slice(~offset=0, ~len=60000)
let tou = tr_opt->Belt.Array.slice(~offset=0, ~len=60000)

let su = {
    ...Bp.sup1_template,
    neta: [784, 20, 10],
    inputVs: tin,
    outputVs: tou,
    epoch: 2400,
    etainit: 1.0,
    etafun: (. eta, i) => eta *. (0.999 ** Belt.Int.toFloat(i)),
    floss: crossloss,
    dlossdx: dcrosslossdx,
    farr: [(Bp.vsigmoid, Bp.vsigmoid_d), (softmax, (.x) => {let (_, y) = x; y})]
}
Js.Console.timeStart("fin")
let r1 = Bp.train_sgd(su, 100, ())//~logfn=Bp.log_example, ())

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

for i in 0 to 10000 - 1 {
    let inff = inference(te_ipt[i], r1)
    if (inff == te_ios[i]) {
	succ := succ.contents + 1
    }
}

succ->Js.log
Js.Console.timeEnd("fin")
