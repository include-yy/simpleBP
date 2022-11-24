// S.res

type xy = (array<float>, array<float>)

type wbfn = {
    w: array<array<float>>,
    b: array<float>,
    f: (. array<float>) => array<float>,
    df: (. xy) => array<float>,
    na: float,
    nb: float
}

type net = array<wbfn>

type wb = {
    dw: array<array<float>>,
    db: array<float>
}

type grad = array<wb>

let v = (f) => (. vec) => vec->Belt.Array.mapU(f)
let m = (f) => (. mat) => mat->Belt.Array.mapU((. v) => v->Belt.Array.mapU(f))

let create = (~netarr, ~farr, ~narr: array<(float, float)>, ~initfun: () => float) => {
    let check = (arr) => {
	arr->Belt.Array.length >= 3 &&
	    arr->Belt.Array.everyU((. x) => x > 0)
    }
    if !check(netarr) {
	Error("S.create: check failed, length not enough or netarr[?] is 0")
    } else {
	Ok(netarr->Belt.Array.sliceToEnd(1)->Belt.Array.mapWithIndex((i, _) => {
	    let w = Belt.Array.makeBy(netarr[i+1], (_) => {
		Belt.Array.makeBy(netarr[i], (_) => initfun())
	    })
	    let b = Belt.Array.makeBy(netarr[i+1], (_) => initfun())
	    let (f, df) = farr[i]
	    let (na, nb) = narr[i]
	    {w: w, b: b,
	     f: f, df: df,
	     na: na, nb: nb}
	}))
    }
}

let wbfn2wb = (ne: net, fn) => {
    ne->Belt.Array.mapU((. s) => {
	{
	    dw: s.w->M.sf2mf(. _, fn),
	    db: s.b->M.sf2vf(. _, fn)
	}})
}

let wbmap = (ww: grad, fn) => {
    ww->Belt.Array.mapU((. s) => {
	{
	    dw: s.dw->M.sf2mf(. _, fn),
	    db: s.db->M.sf2vf(. _, fn)
	}
    })
}

let wbOpInPlace = (. g1: grad, g2: grad, fun) => {
    g1->Belt.Array.forEachWithIndexU((. i, _) => {
	M.mmf2mInPlace(. g1[i].dw, g2[i].dw, fun)
	M.vvf2vInPlace(. g1[i].db, g2[i].db, fun)
    })
}

let wbfnOpInPlace = (. ne: net, gr: grad, fun) => {
    ne->Belt.Array.forEachWithIndexU((. i, _) => {
	M.mmf2mInPlace(. ne[i].w, gr[i].dw, fun)
	M.vvf2vInPlace(. ne[i].b, gr[i].db, fun)
    })
}

let wbAddInPlace = (. g1: grad, g2: grad) => {
    wbOpInPlace(. g1, g2, (. x, y)=>x+.y)
}

let wbfnUpdateInPlace = (. ne: net, gr: grad, eta) => {
    wbfnOpInPlace(. ne, gr, (. x, y) => x -. eta *. y)
}
