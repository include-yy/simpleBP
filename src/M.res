// M.res
let dot = (. x, y) => {
    x->Belt.Array.reduceWithIndexU(0.0, (. sum, _, i) => {
	sum +. Js.Array2.unsafe_get(x, i) *. Js.Array2.unsafe_get(y, i)
    })
}

let tr = (. mat) => {
    mat->Js.Array2.unsafe_get(0)->Belt.Array.mapWithIndexU((. i, _) => {
	mat->Belt.Array.mapWithIndexU((. j, _) => {
	    mat->Js.Array2.unsafe_get(j)->Js.Array2.unsafe_get(i)
	})
    })
}

let vvf2v = (. x, y, f) => {
    x->Belt.Array.mapWithIndexU((. i, _) => {
	f(. Js.Array2.unsafe_get(x, i), Js.Array2.unsafe_get(y, i))
    })
}

let vvadd = (. x, y) => ((. a, b) => a+.b)->vvf2v(. x, y, _)
let vvsub = (. x, y) => ((. a, b) => a-.b)->vvf2v(. x, y, _)
let vvmul = (. x, y) => ((. a, b) => a*.b)->vvf2v(. x, y, _)

let matxvec = (. m, v) => m->Belt.Array.mapU((.mv) => dot(. mv, v))

let sf2vf = (. v, f) => v->Belt.Array.mapU(f)

let vxv2m = (. vt, v) => {
    vt->Belt.Array.mapU((.a) => v->Belt.Array.mapU((.b) => a*.b))
}

let tmatxvec = (. tm, v) => {
    Js.Array2.unsafe_get(tm, 0)->Belt.Array.mapWithIndexU((. i, _) => {
	tm->Belt.Array.reduceWithIndexU(0.0, (. sum, _, j) => {
	    sum +. tm->Js.Array2.unsafe_get(j)->Js.Array2.unsafe_get(i) *.
		v->Js.Array2.unsafe_get(j)
	})
    })
}

let mmf2m = (. x, y, f) => {
    x->Belt.Array.mapWithIndexU((. i, v) => {
	v->Belt.Array.mapWithIndexU((. j, _) => {
	    f(. x->Js.Array2.unsafe_get(i)->Js.Array2.unsafe_get(j),
	      y->Js.Array2.unsafe_get(i)->Js.Array2.unsafe_get(j))
	})
    })
}

let mmadd = (. x, y) => ((. a, b) => a+.b)->mmf2m(. x, y, _)
let mmsub = (. x, y) => ((. a, b) => a-.b)->mmf2m(. x, y, _)

let sf2mf = (. m, f) => {
    m->Belt.Array.mapU((.v) => v->Belt.Array.mapU((.a) => f(.a)))
}

let mmf2mInPlace = (. x, y, f) => {
    x->Belt.Array.forEachWithIndexU((. i, v) => {
	v->Belt.Array.forEachWithIndexU((. j, _) => {
	    let val = f(. x->Js.Array2.unsafe_get(i)->Js.Array2.unsafe_get(j),
			y->Js.Array2.unsafe_get(i)->Js.Array2.unsafe_get(j))
	    Js.Array2.unsafe_get(x, i)->Js.Array2.unsafe_set(j, val)
	})
    })
}

let vvf2vInPlace = (. x, y, f) => {
    x->Belt.Array.forEachWithIndexU((. i, _) => {
	let val = f(. x->Js.Array2.unsafe_get(i),
		    y->Js.Array2.unsafe_get(i))
	x->Js.Array2.unsafe_set(i, val)
    })
}
