//M1.res

// 向量点积
let dot = (v1, v2) => {
    v1->Js.Array2.reducei((s, _, i) => {
	s +. v1[i] *. v2[i]
    }, 0.0)
}

// 矩阵左乘列向量
let mulv = (m0, v1) => {
    m0->Js.Array2.map((v) => {
	v->Js.Array2.reducei((s, _, j) => s +. v[j] *. v1[j], 0.0)
    })
}

// 向量的 map 函数
let mapv = (v1, fun) => v1->Js.Array2.map(fun)

// 矩阵的 map 函数
let mapm = (m, fun) => {
    m->Js.Array2.map((v) => {
	v->Js.Array2.map((a) => fun(a))
    })
}

// 向量对应位置相乘
let vdotv = (vec1, vec2) => {
    vec1->Js.Array2.mapi((_, i) => vec1[i] *. vec2[i])
}

// 列向量乘行向量得到矩阵
let vmulv = (vec1, vec2) => {
    vec1->Js.Array2.map((a) => {
	vec2->Js.Array2.map((b) => a *. b)
    })
}

// 转置矩阵乘列向量
let tmulv = (mat, vec) => {
    mat[0]->Js.Array2.mapi((_, i) => {
	mat->Js.Array2.reducei((s, _, j) => s +. mat[j][i] *. vec[j], 0.0)
    })
}

// 对两个矩阵中的对应位置两元素执行某运算
let m2f = (mat1, mat2, fun) => {
    mat1->Js.Array2.mapi((v, i) => {
	v->Js.Array2.mapi((_, j) => fun(mat1[i][j], mat2[i][j]))
    })
}

// 对两个向量中的对应位置两元素执行某运算
let v2f = (v1, v2, fun) => {
    v1->Js.Array2.mapi((_, i) => fun(v1[i], v2[i]))
}

// 对两个矩阵中的对应位置两元素执行某运算，副作用版本
let m2fInPlace = (mat, m2, fun) => {
    open Js.Array2
    mat->forEachi((_, i) => {
	mat[i]->forEachi((_, j) => {
	    mat[i][j] = fun(mat[i][j], m2[i][j])
	})
    })
}
// 对两个向量中的对应位置两元素执行某运算，副作用版本
let v2fInPlace = (vec, v2, fun) => {
    vec->Js.Array2.forEachi((_, i) => {
	vec[i] = fun(vec[i], v2[i])
    })
}

