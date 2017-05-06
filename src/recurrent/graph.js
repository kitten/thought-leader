// @flow

import Matrix from './matrix'
import { sigm } from './util'

export const fnRowPluck = (innerFn: Function) => (mat: Matrix, ix: number) => {
  const { d, w, dw } = mat

  // Compute
  const output = new Matrix(d, 1)
  for (let i = 0; i < d; i++) {
    output.w[i] = w[d * ix + i]
  }

  // Inner
  const recurse = innerFn(output)

  // Backprop
  for (let i = 0; i < d; i++) {
    dw[d * ix + i] += output.dw[i]
  }

  return recurse
}

export const fnTanh = (innerFn: Function) => (mat: Matrix) => {
  const { n, d, w, dw } = mat

  // Compute
  const output = new Matrix(n, d)
  const size = w.length
  for (let i = 0; i < size; i++) {
    output.w[i] = Math.tanh(w[i])
  }

  // Inner
  const recurse = innerFn(output)

  // Backprop
  for (let i = 0; i < size; i++) {
    // Grad for z = tanh(x) is (1 - z^2)
    const mwi = output.w[i]
    dw[i] += (1 - Math.pow(mwi, 2)) * output.dw[i]
  }

  return recurse
}

export const fnSigmoid = (innerFn: Function) => (mat: Matrix) => {
  const { n, d, w, dw } = mat

  // Compute
  const output = new Matrix(n, d)
  const size = w.length
  for (let i = 0; i < size; i++) {
    output.w[i] = sigm(w[i])
  }

  // Inner
  const recurse = innerFn(output)

  // Backprop
  for (let i = 0; i < size; i++) {
    // Grad for z = sigm(x) is z * (1 - z)
    const mwi = output.w[i]
    dw[i] += mwi * (1 - mwi) * output.dw[i]
  }

  return recurse
}

export const fnMul = (innerFn: Function) => (mat1: Matrix, mat2: Matrix) => {
  const { n: n1, d: d1, w: w1, dw: dw1 } = mat1
  const { n: n2, d: d2, w: w2, dw: dw2 } = mat2

  // Compute
  const output = new Matrix(n1, d2)
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < d2; j++) {
      let dot = 0
      for (let k = 0; k < d1; k++) {
        dot += w1[d1 * i + k] * w2[d2 * k + j]
      }

      output.w[d2 * i + j] = dot
    }
  }

  // Inner
  const recurse = innerFn(output)

  // Backprop
  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < d2; j++) {
      for (let k = 0; k < d1; k++) {
        const b = output.dw[d2 * i + j]
        dw1[d1 * i + k] += w2[d2 * k + j] * b
        dw2[d2 * k + j] += w1[d1 * i + k] * b
      }
    }
  }

  return recurse
}

export const fnAdd = (innerFn: Function) => (mat1: Matrix, mat2: Matrix) => {
  const { n, d, w: w1, dw: dw1 } = mat1
  const { w: w2, dw: dw2 } = mat2

  // Compute
  const output = new Matrix(n, d)
  const size = n * d
  for (let i = 0; i < size; i++) {
    output.w[i] = w1[i] + w2[i]
  }

  // Inner
  const recurse = innerFn(output)

  // Backprop
  for (let i = 0; i < size; i++) {
    dw1[i] += output.dw[i]
    dw2[i] += output.dw[i]
  }

  return recurse
}

export const fnEltmul = (innerFn: Function) => (mat1: Matrix, mat2: Matrix) => {
  const { n, d, w: w1, dw: dw1 } = mat1
  const { w: w2, dw: dw2 } = mat2

  // Compute
  const output = new Matrix(n, d)
  const size = n * d
  for (let i = 0; i < size; i++) {
    output.w[i] = w1[i] * w2[i]
  }

  // Inner
  const recurse = innerFn(output)

  // Backprop
  for (let i = 0; i < size; i++) {
    dw1[i] += w2[i] * output.dw[i]
    dw2[i] += w1[i] * output.dw[i]
  }

  return recurse
}

export const fnOneMinus = (innerFn: Function) => (mat: Matrix) => {
  const { n, d, w, dw } = mat

  // Compute
  const output = new Matrix(n, d)
  const size = n * d
  for (let i = 0; i < size; i++) {
    output.w[i] = 1 - w[i]
  }

  // Inner
  const recurse = innerFn(output)

  // Backprop
  for (let i = 0; i < size; i++) {
    dw[i] -= output.dw[i]
  }

  return recurse
}
