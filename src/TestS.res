// TestS.res

//create
let arr = [2, 3, 2, 4]
let id = (. x) => x
let vid = S.v(id)
let did = (. x_y: S.xy) => {
    let (x, y) = x_y
    ignore(x)
    ignore(y)
    Belt.Array.makeBy(x->Belt.Array.length, (_) => 0.0)
}
let farr = [(vid, did), (vid, did), (vid, did)]
let narr = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
let ne = S.create(~netarr=arr,
		  ~farr=farr,
		  ~narr=narr,
		  ~initfun=() => 1.0)->Belt.Result.getExn

for i in 0 to arr->Js.Array2.length - 2 {
    Js.log(ne[i].w)
    Js.log(ne[i].b)
    Js.log(ne[i].na)
    Js.log(ne[i].nb)
}

// wbfn2wb
let g1 = S.wbfn2wb(ne, (. _) => 0.0)

g1->Js.Array2.forEach((a) => {
    Js.log(a.dw)
    Js.log(a.db)
})

// wbmap
g1->S.wbmap((. x) => x+.1.0)->Js.Array2.forEach((a) => {
    Js.log(a.dw)
    Js.log(a.db)
})

// wbAddInPlace
let g2 = S.wbmap(g1, (. x) => x+.2.0)
let g3 = S.wbmap(g1, (. x) => x+.1.0)
S.wbAddInPlace(. g2, g3)
g2->Js.Array2.forEach((a) => {
    Js.log(a.dw)
    Js.log(a.db)
})

// wbfnUpdateInPlace
S.wbfnUpdateInPlace(. ne, g2, 0.1)

ne->Js.Array2.forEach((a) => {
    Js.log(a.w)
    Js.log(a.b)
    Js.log(a.na)
    Js.log(a.nb)
})
