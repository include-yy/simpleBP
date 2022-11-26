// Bp.res
type super = {
    neta: array<int>,
    farr: array<((. array<float>)=>array<float>, (. S.xy)=>array<float>)>,
    narr: array<(float, float)>,
    initf: () => float,
    inputVs: array<array<float>>,
    outputVs: array<array<float>>,
    epoch: int,
    etainit: float,
    etafun: (. float, int) => float,
    floss: (array<float>, array<float>) => float,
    dlossdx: (S.xy, array<float>) => array<float>,
}

let sigmoid = (.x) => 1.0 /. (1.0 +. Js.Math.exp(-.x))
let vsigmoid = S.v(sigmoid)
let vsigmoid_d = (. st: S.xy) => {
    let (_, yarr) = st
    yarr->Belt.Array.mapWithIndexU((. i, _) => {
	let y = Js.Array2.unsafe_get(yarr, i)
	y *. (1.0 -. y)
    })
}

let tanh = (.x) => {
    let pos = Js.Math.exp(x)
    let neg = 1.0 /. pos
    (pos -. neg) /. (pos +. neg)
}
let vtanh = S.v(tanh)
let vtanh_d = (. st: S.xy) => {
    let (_, yarr) = st
    yarr->Belt.Array.mapWithIndexU((. i, _) => {
	let y = Js.Array2.unsafe_get(yarr, i)
	1.0 -. y *. y
    })
}

let loss2 = (y: array<float>, yt: array<float>) => {
    y->Belt.Array.reduceWithIndexU(0.0, (. sum, _, i) => {
	let a = Js.Array2.unsafe_get(y, i)
	let b = Js.Array2.unsafe_get(yt, i)
	let sub = a -. b
	sum +. sub *. sub *. 0.5
    })
}

let dloss2 = (y: array<float>, yt: array<float>) => {
    y->Belt.Array.mapWithIndexU((. i, _) => {
	let a = Js.Array2.unsafe_get(y, i)
	let b = Js.Array2.unsafe_get(yt, i)
	a -. b
    })
}

let dloss2dx = (xy: S.xy, yt: array<float>) => {
    let (_, yarr) = xy
    let dfv = xy->vsigmoid_d(._)
    M.vvmul(. dfv, dloss2(yarr, yt))
}

let sup1_template = {
    neta: [2, 4, 3],
    farr: [(vsigmoid, vsigmoid_d), (vsigmoid, vsigmoid_d)],
    narr: [(1.0, 0.0), (1.0, 0.0)],
    initf: Js.Math.random,
    inputVs: [[1.0, 2.0], [3.0, 4.0]],
    outputVs: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    epoch: 1000,
    etainit: 0.01,
    etafun: (. eta, _) => eta,
    floss: loss2,
    dlossdx: dloss2dx
}

let sup2_template = {
    neta: [2, 4, 5, 3],
    farr: [(vsigmoid, vsigmoid_d),
	   (vsigmoid, vsigmoid_d),
	   (vsigmoid, vsigmoid_d)],
    narr: [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
    initf: Js.Math.random,
    inputVs: [[1.0, 2.0], [3.0, 4.0]],
    outputVs: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    epoch: 1000,
    etainit: 0.01,
    etafun: (. eta, _) => eta,
    floss: loss2,
    dlossdx: dloss2dx
}

let forward = (ix: array<float>, ne: S.net) => {
    let len = Belt.Array.length(ne)
    let x = ref(ix)
    Belt.Array.makeBy(len, (i) => {
	let t = x.contents
	    ->M.matxvec(. ne[i].w, _)
	    ->M.vvadd(. _, ne[i].b)
	    ->Belt.Array.mapU((. a) => a *. ne[i].na +. ne[i].nb)
	let y = t->ne[i].f(._)
	x := y
	(t, y)
    })
}

let backward = (ne: S.net, xyarr: array<S.xy>, inputx, delta0) => {
    let len = Js.Array2.length(ne)
    let ne1 = ne->Belt.Array.slice(~offset=1, ~len=len-1)
    let xyarr1 = xyarr->Belt.Array.slice(~offset=0, ~len=len-1)

    let deltana = delta0->M.sf2vf(. _, (.x) => x*.ne[len-1].na)
    let delarr = Belt.Array.reduceReverse2U(ne1, xyarr1, [deltana], (. c, a, b) => {
	let tm_delta = M.tmatxvec(. a.w, c[0])
	let dfvec = b->a.df(._)->M.sf2vf(. _, (.x) => x*.a.na)
	[dfvec->M.vvmul(. _, tm_delta)]->Belt.Array.concat(c)
    })
    delarr->Belt.Array.mapWithIndexU((. i, _) => {
	if i != 0 {
	    let (_, yv) = xyarr[i-1]
	    {S.dw: delarr[i]->M.vxv2m(. _, yv),
	     db: delarr[i]}
	} else {
	    {S.dw: delarr[i]->M.vxv2m(. _, inputx),
	     db: delarr[i]}
	}
    })
}

type loginfo = {
    i: int,
    forwards: array<array<S.xy>>,
    inputs: array<array<float>>,
    outputs: array<array<float>>,
    lossfn: (array<float>, array<float>) => float
}

let log_example = (info: loginfo) => {
    Js.log(info.i)
    info.inputs->Belt.Array.reduceWithIndex(0.0, (sum, _, i) => {
	let r = info.forwards[i]
	let (_, y) = r[r->Js.Array2.length-1]
	sum +. info.lossfn(y, info.outputs[i]) /.
	    Belt.Int.toFloat(info.inputs->Js.Array2.length)
    })->Js.log
}

let train_bgd = (su: super, ~logfn=?, ()) => {
    let net = S.create(~netarr=su.neta,
		       ~farr=su.farr,
		       ~narr=su.narr,
		       ~initfun=su.initf)->Belt.Result.getExn
    let len = net->Js.Array2.length
    let batchSize = su.inputVs->Js.Array2.length
    for i in 0 to su.epoch - 1 {
	let gr = net->S.wbfn2wb((. _) => 0.0)
	let forwards = []
	for j in 0 to batchSize - 1{
	    let forwardCache = forward(su.inputVs[j], net)
	    ignore(forwards->Js.Array2.push(forwardCache))
	    let delta = su.dlossdx(forwardCache[len - 1], su.outputVs[j])
	    let gd = backward(net, forwardCache, su.inputVs[j], delta)
	    S.wbAddInPlace(. gr, gd)
	}
	let costgr = gr->S.wbmap((. x) => x /. Belt.Int.toFloat(batchSize))
	S.wbfnUpdateInPlace(. net, costgr, su.etainit->su.etafun(. _, i))

	switch logfn {
		| None => ()
		| Some(fn) => {
		    {i: i,
		     forwards: forwards,
		     inputs: su.inputVs,
		     outputs: su.outputVs,
		     lossfn: su.floss}->fn
		}
	}
    }
    net
}


let shuffle = (x, y) => {
    Belt.Array.zip(x, y)->Belt.Array.shuffle->Belt.Array.unzip
}

let train_sgd = (su: super, bsize, ~logfn=?, ()) => {
    let net = S.create(~netarr=su.neta,
		       ~farr=su.farr,
		       ~narr=su.narr,
		       ~initfun=su.initf)->Belt.Result.getExn
    let len = net->Js.Array2.length
    let btimes = su.inputVs->Js.Array2.length / bsize
    let j = ref(0)
    let inV = ref(su.inputVs)
    let ouV = ref(su.outputVs)

    for i in 0 to su.epoch - 1 {
	if (j.contents == btimes) {
	    let (tinV, touV) = shuffle(su.inputVs, su.outputVs)
	    inV := tinV
	    ouV := touV
	    j := 0
	}
	let gr = net->S.wbfn2wb((. _) => 0.0)
	let forwards = []
	for k in j.contents * bsize to (j.contents + 1) * bsize - 1 {
	    let forwardCache = forward(inV.contents[k], net)
	    ignore(forwards->Js.Array2.push(forwardCache))
	    let delta = su.dlossdx(forwardCache[len-1], ouV.contents[k])
	    let gd = backward(net, forwardCache, inV.contents[k], delta)
	    S.wbAddInPlace(. gr, gd)
	}
	let costgr = gr->S.wbmap((. x) => x /. Belt.Int.toFloat(bsize))
	S.wbfnUpdateInPlace(. net, costgr, su.etainit->su.etafun(. _, i))
	switch logfn {
		| None => ()
		| Some(fn) => {
		    {i: i,
		     forwards: forwards,
		     inputs: inV.contents
		     ->Belt.Array.slice(~offset=j.contents * bsize, ~len=bsize),
		     outputs: ouV.contents
		     ->Belt.Array.slice(~offset=j.contents * bsize, ~len=bsize),
		     lossfn: su.floss}->fn
		}
	}
	j := j.contents + 1
    }
    net
}

type norm = array<array<(float, float)>>

type normxy = (array<array<float>>, array<array<float>>, array<(float, float)>)

let normv = (. v: array<float>) => {
    let len_f = v->Js.Array2.length->Belt.Int.toFloat
    let avg = v->Belt.Array.reduceU(0.0, (. sum, a) => sum +. a) /. len_f
    let s = v->Belt.Array.reduceU(0.0, (.sum, a) => {
	let t = a -. avg
	sum +. t *. t
    }) /. len_f
    (avg, Js.Math.sqrt(s))
}

let normalize = (vs: array<array<float>>) => {
    let norms = M.tr(. vs)->Belt.Array.mapU((.a) => normv(.a))
    (vs->Belt.Array.mapU((. v) => {
	v->Belt.Array.mapWithIndexU((. i, a) => {
	    let (avg, s) = norms->Js.Array.unsafe_get(i)
	    (a -. avg) /. s
	})
    }), norms)
}

let bnforward = (ixs: array<array<float>>, ne: S.net) => {
    let len = Belt.Array.length(ne)
    let x = ref(ixs)
    Belt.Array.makeBy(len, (i) => {
	let axb = x.contents->Belt.Array.mapU((.a) => {
	    a
		->M.matxvec(. ne[i].w, _)
		->M.vvadd(. _, ne[i].b)
	})
	let (nom, norms) = axb->normalize
	let nom2 = nom->Belt.Array.mapU((.v) => {
	    v->Belt.Array.mapU((. a) => a *. ne[i].na +. ne[i].nb)
	})
	let ys = nom2->Belt.Array.mapU((.v)=>v->ne[i].f(._))
	x := ys
	(nom2, ys, norms)
    })
}

let bnbackward = (ne: S.net, xynarr: array<normxy>, inputVs, delta0s: array<array<float>>) => {
    let len = Js.Array2.length(ne)
    let ne1 = ne->Belt.Array.slice(~offset=1, ~len=len-1)
    let xynarr1 = xynarr->Belt.Array.slice(~offset=0, ~len=len-1)

    let deltana = delta0s->Belt.Array.mapU((.a) => {
	a->M.sf2vf(. _, (.x) => x*.ne[len-1].na)
    })
    let (_, _, v) = xynarr[xynarr->Js.Array2.length-1]
    let (_, varr) = v->Belt.Array.unzip
    let deltanava = deltana->Belt.Array.mapU((.a) => {
	a->M.vvdiv(. _, varr)
    })
    let delarrs = Belt.Array.reduceReverse2U(ne1, xynarr1, [deltanava], (. c, a, b) => {
	let tm_delta_arr = c[0]->Belt.Array.mapU((.t) => M.tmatxvec(. a.w, t))
	let (xm, ym, normv) = b
	let dfarrs = xm->Belt.Array.mapWithIndexU((. i, _) => {
	    let (_, s) = normv->Belt.Array.unzip
	    a.df(. (xm[i], ym[i]))
		->M.sf2vf(. _, (.x) => x*.a.na)
		->M.vvdiv(. _, s)
	})
	let darr = tm_delta_arr->Belt.Array.mapWithIndexU((. i, _) => {
	    M.vvmul(. tm_delta_arr[i], dfarrs[i])
	})
	[darr]->Belt.Array.concat(c)
    })
    let wbres = ne->S.wbfn2wb((._) => 0.0)
    let wbs = delarrs->Belt.Array.mapWithIndexU((. i, _) => {
	if i != 0 {
	    let (_, ys, _) = xynarr[i-1]
	    delarrs[i]->Belt.Array.mapWithIndexU((. j, d) => {
		{S.dw: d->M.vxv2m(. _, ys[j]),
		 db: d}
	    })
	} else {
	    delarrs[i]->Belt.Array.mapWithIndexU((. j, d) => {
		{S.dw: d->M.vxv2m(. _, inputVs[j]),
		 db: d}
	    })
	}
    })
    wbs->Belt.Array.forEachU((. x) => {
	S.wbAddInPlace(. wbres, x)
    })
    wbres->S.wbmap((.x) => x /. Js.Array2.length(inputVs)->Belt.Int.toFloat)
}

let bntrain_bgd = (su: super, ~logfun=?, ()) => {
    let net = S.create(~netarr=su.neta,
		       ~farr=su.farr,
		       ~narr=su.narr,
		       ~initfun=su.initf)->Belt.Result.getExn
    let len = net->Js.Array2.length
    let norms = Belt.Array.makeBy(len, (i) => {
	Belt.Array.make(net[i].b->Js.Array2.length, (0.0, 0.0))
    })
    for i in 0 to su.epoch - 1 {
	let fs = bnforward(su.inputVs, net)
	let (x, y, _) = fs[len-1]
	let deltas = x->Belt.Array.mapWithIndexU((. i, _) => {
	    su.dlossdx((x[i], y[i]), su.outputVs[i])
	})
	let costgr = bnbackward(net, fs, su.inputVs, deltas)
	S.wbfnUpdateInPlace(. net, costgr, su.etainit->su.etafun(. _, i))
	let curr_norms = fs->Belt.Array.mapU((. a) => {
	    let (_, _, s) = a
	    s
	})
	M.mmf2mInPlace(. norms, curr_norms, (. x, y) => {
	    let (a, b) = x
	    let (c, d) = y
	    (a +. c, b +. d)
	})
	switch logfun {
		| None => ()
		| Some(fn) => {
		    let forwards = fs->Belt.Array.mapU((. a) => {
			let (x, y, _) = a
			Belt.Array.zip(x, y)
		    })->M.tr(. _)
		    {i: i,
		     forwards: forwards,
		     inputs: su.inputVs,
		     outputs: su.outputVs,
		     lossfn: su.floss}->fn
		}
	}
    }
    let norms_final = S.m((. x) => {
	let (a, b) = x
	(a /. su.epoch->Belt.Int.toFloat, b /. su.epoch->Belt.Int.toFloat)
    })(. norms)
    (net, norms_final)
}
