// testm1.res

//dot
let v0 = [1., 1., 1.]
let v1 = [1., 2., 3.]
Js.log(M1.dot(v0, v1))
// 6

//mulv
let mat0 = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]
Js.log(M1.mulv(mat0, v1))
// [14, 32, 50]

//mapv
Js.log(M1.mapv(v1, (x) => x +. 1.0))
// [2, 3, 4]

//mapm
Js.log(M1.mapm(mat0, (x) => x +. 2.0))
// [[3, 4, 5], [6, 7, 8], [9, 10, 11]]

//vdotv
Js.log(M1.vdotv(v1, v1))
// [1, 4, 9]

//vmulv
Js.log(M1.vmulv(v0, v1))
// [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

//tmulv
Js.log(M1.tmulv(mat0, v1))
// [[30, 36, 42]]

//m2f
Js.log(M1.m2f(mat0, mat0, (x, y) => x -. y))
// [[0, 0, 0] [0, 0, 0], [0, 0, 0]]

//v2f
Js.log(M1.v2f(v0, v1, (x, y) => x +. y))
// [2, 3, 4]

//m2f
M1.m2fInPlace(mat0, mat0, (x, y) => x +. y)
Js.log(mat0)
//[[2, 4, 6], [8, 10, 12], [14, 16, 18]]

//v2f
M1.v2fInPlace(v0, v1, (x, y) => x +. y)
Js.log(v0)
//[2, 3, 4]
Js.log(v1)
//[1, 2, 3]
