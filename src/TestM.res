let v0 = [1., 1., 1.]
let v1 = [1., 2., 3.]

// dot
Js.log(M.dot(. v0, v1))
//6

//tr
let mat = [[1, 2, 3], [4, 5, 6]]
M.tr(. mat)->Js.log
//[[1, 4], [2, 5], [3, 6]]

//vvf2v
Js.log(M.vvf2v(. v0, v1, (. x, y) => x +. y))
// [2, 3, 4]
Js.log(M.vvadd(. v0, v1))
Js.log(M.vvsub(. v0, v1))
Js.log(M.vvmul(. v0, v1))
// [2, 3, 4]
// [0, -1, -2]
// [1, 2, 3]

//matxvec
let mat0 = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
Js.log(M.matxvec(. mat0, v1))
// [14, 32, 50]

//sf2vf
Js.log(v1->M.sf2vf(. _, (. x) => x +. 1.0))
// [2, 3, 4]

//vxv2m
Js.log(M.vxv2m(. v0, v1))
// [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

//tmatxvec
Js.log(M.tmatxvec(. mat0, v1))
// [[30, 36, 42]]

//mmf2m
Js.log(M.mmf2m(. mat0, mat0, (. x, y) => x -. y))
// [[0, 0, 0] [0, 0, 0], [0, 0, 0]]
Js.log(M.mmadd(. mat0, mat0))
Js.log(M.mmsub(. mat0, mat0))
// [[2, 4, 6], [8, 10, 12], [14, 16, 18]]
// [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

//sf2mf
Js.log(M.sf2mf(. mat0, (. x) => x +. 1.0))
//[[2, 3, 4 ], [5, 6, 7], [8, 9, 10]]

// mmf2mInPlace
M.mmf2mInPlace(. mat0, mat0, (. x, y) => x +. y)
Js.log(mat0)
//[[2, 4, 6], [8, 10, 12], [14, 16, 18]]

//vvf2vInPlace
M.vvf2vInPlace(. v0, v1, (. x, y) => x +. y)
Js.log(v0)
//[2, 3, 4]
Js.log(v1)
//[1, 2, 3]
