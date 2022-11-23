// Nu.res

let sigmoid = (x) => 1.0 /. (1.0 +. Js.Math.exp(-.x))
//let sigmoid_d = (x) => Js.Math.exp(-.x) /. (1.0 +. Js.Math.exp(-.x)) ** 2.0
let sigmoid_d = (x) => {
    let a = sigmoid(x)
    a *. (1.0 -. a)
}
let vsigmoid = Wb.bevec(sigmoid)
let vsigmoid_d = Wb.bevec(sigmoid_d)

let relu = (x) => x > 0.0 ? x : 0.0
let relu_d = (x) => x > 0.0 ? 1.0 : 0.0
let vrelu = Wb.bevec(relu)
let vrelu_d = Wb.bevec(relu_d)

let id = (x) => x +. 0.0
let id_d = (_) => 1.0
let vid = Wb.bevec(id)
let vid_d = Wb.bevec((x) => 1.0 +. x -. x)

let e = (y, yt) => {
    y->Js.Array2.reducei((s, _, i) => {
	s +. (y[i] -. yt[i]) ** 2.0
    }, 0.0)
}

let error_E = (vY, vYt) => {
    vY->Js.Array2.reducei((s, _, i) => {
	s +. e(vY[i], vYt[i])
    }, 0.0) /. Belt.Int.toFloat(vY->Js.Array2.length)
}

let error_E_d = (num, y, yt) => {
    M1.v2f(y, yt, (x, y) => x -. y)
	->M1.mapv((x) => x *. 2.0 /. Belt.Int.toFloat(num))
}

let ce_ln2 = Js.Math.log(2.0)

let ce = (y, yt) => {
    y->Js.Array2.reducei((s, a, i) => {
	s -. yt[i] *. Js.Math.log2(a)
    }, 0.0)
}

let ce_E = (vY, vYt) => {
    vY->Js.Array2.reducei((s, _, i) => {
	s +. ce(vY[i], vYt[i])
    }, 0.0) /. Belt.Int.toFloat(vY->Js.Array2.length)
}

let ce_E_d = (num, y, yt) => {
    y->Js.Array2.mapi((_, i) => -. 1.0 /. (ce_ln2 *. y[i] *. Belt.Int.toFloat(num)))
	->M1.v2f(yt, (x, y) => x *. y)
}


let forward = (xv, state: array<Wb.wb>, fun) => {
    let res = []
    ignore(state->Js.Array2.reduce((s, a) => {
	let io_new = Wb.forwardOneStep(s, a, fun)
	ignore(res->Js.Array2.push(io_new))
	io_new.yi
    }, xv))
    res
}

let backward = (state: array<Wb.wb>, ios: array<array<Wb.io>>, ivs, ovs, ots, fun_d, error_d) => {
    open Js.Array2
    let batchSize = ivs->length
    let nuarray = concat([ivs[0]->length], ios[0]->map((a) => a.xi->length))
    let grad = Wb.create(nuarray, () => 0.0)->Belt.Result.getExn
    for i in 0 to batchSize - 1 {
	let iov = ios[i]
	let iv = ivs[i]
	let ov = ovs[i]
	let ot = ots[i]
	let layerNum = state->length
	let deltav_fin = iov[layerNum - 1].xi
	    ->fun_d
	    ->M1.vdotv(error_d(batchSize, ov, ot))
	let deltav_arr = Js.Vector.make(layerNum, deltav_fin)
	for j in state->length - 2 downto 0 {
	    deltav_arr[j] = iov[j].xi->fun_d
		->M1.vdotv(state[j+1].w->M1.tmulv(deltav_arr[j+1]))
	}
	let g1 = deltav_arr->mapi((v, j) => {
	    if j == 0 {
		{Wb.w: v->M1.vmulv(iv),
		 b: v}
	    } else {
		{Wb.w: v->M1.vmulv(iov[j-1].yi),
		 b: v}
	    }
	})
	Wb.addInPlace(grad, g1)
    }
    grad
}


let train_sgd = (state, ivs, ots, epoch, eta_init, fun, fun_d, ef, efd, eta_fun) => {
    open Js.Array2
    let st = state->Wb.map((x) => x)
    for i in 1 to epoch {
	for j in 0 to ivs->length - 1 {
	    let io = ivs[j]->forward(st, fun)
	    let grd = st
		->backward([io], [ivs[j]], [io[io->length - 1].yi], [ots[j]], fun_d, efd)
	    Wb.updateInPlace(st, grd, eta_init->eta_fun(i))
	}
	if mod(i, 1) == 0 {
	    let err = ivs
		->map((a) => forward(a, st, fun))
		->map((a) => a[a->length - 1].yi)->ef(ots)
	    Js.log(i)
	    Js.log(err)
	}
    }
    st
}

let train_bgd = (state, ivs, ots, epoch, eta_init, fun, fun_d, ef, efd, eta_fun) => {
    open Js.Array2
    let st = state->Wb.map((x) => x)
    for i in 1 to epoch {
	let ios = ivs->Js.Array2.map((a) => a->forward(st, fun))
	let ovs = ios->Js.Array2.map((a) => a[a->length - 1].yi)
	let grd = st
	    ->backward(ios, ivs, ovs, ots, fun_d, efd)
	Wb.updateInPlace(st, grd, eta_init->eta_fun(i))

	if mod(i, 1) == 0 {
	    let err = ivs
		->map((a) => forward(a, st, fun))
		->map((a) => a[a->length - 1].yi)->ef(ots)
	    Js.log(i)
	    Js.log(err)
	}
    }
    st

}

let inference = (inputv, state, fun) => {
    let res = inputv->forward(state, fun)
    res[res->Js.Array2.length - 1].yi
}
