// Wb.res

type wb = {
    w: array<array<float>>,
    b: array<float>,
}

type io = {
    xi: array<float>,
    yi: array<float>
}

let map = (gv, fn) => {
    gv->Js.Array2.map((wb) => {
	{w: wb.w->M1.mapm(fn),
	 b: wb.b->M1.mapv(fn),
	}
    })
}

let opInPlace = (gv, gv1, fun) => {
    gv->Js.Array2.forEachi((_, i) => {
	M1.m2fInPlace(gv[i].w, gv1[i].w, fun)
	M1.v2fInPlace(gv[i].b, gv1[i].b, fun)
    })
}

let addInPlace = (gv, gv1) => {
    opInPlace(gv, gv1, (x, y) => x +. y)
}

let updateInPlace = (gv, gv1, eta) => {
    opInPlace(gv, gv1, (x, y) => x -. eta *. y)
}

let bevec = (fn) => {
    (v) => v->Js.Array2.map((a) => fn(a))
}

let forwardOneStep = (x, state1, fun) => {
    let x_new = state1.w->M1.mulv(x)->M1.v2f(state1.b, (x, y) => x +. y)
    let y_new = fun(x_new)
    {xi: x_new, yi: y_new}
}

let create = (arr, fun) => {
    let check = (arr) => {
	arr->Js.Array2.length >= 3 &&
	    arr->Js.Array2.every((x) => x > 0)
    }
    if (!check(arr)) {
	Error("Wb.create: check failed, length not enough or arr[?] = 0")
    } else {
	Ok(arr->Js.Array2.sliceFrom(1)->Js.Array2.mapi((a, i) => {
	    let it = Js.Vector.make(a, 0)
	    let w0 = it->Js.Array2.map((_) => {
		Js.Vector.make(arr[i], 0)->
		    Js.Array2.map((_) => fun())
	    })
	    let b0 = it->Js.Array2.map((_) => fun())
	    {w: w0, b: b0}
	}))
    }
}
