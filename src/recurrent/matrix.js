// @flow

import { randf, randomArray } from './random'

export default class Matrix {
  n: number
  d: number
  w: Float64Array
  dw: Float64Array

  constructor(
    n: number, // rows
    d: number, // columns
    w: Float64Array = new Float64Array(n * d)
  ) {
    this.n = n
    this.d = d
    this.w = w
    this.dw = new Float64Array(n * d)
  }

  softmax(): Matrix {
    const { n, d, w } = this
    const size = n * d
    const output = new Matrix(n, d)

    let maxVal = -999999

    for (let i = 0; i < size; i++) {
      if (w[i] > maxVal) {
        maxVal = w[i]
      }
    }

    let s = 0
    for (let i = 0; i < size; i++) {
      output.w[i] = Math.exp(w[i] - maxVal)
      s += output.w[i]
    }

    for (let i = 0; i < size; i++) {
      output.w[i] /= s
    }

    return output
  }

  toJSON(): Object {
    const { n, d, w } = this
    return { n, d, w }
  }

  static fromJSON({ n, d, w }: Object): Matrix {
    const arr = new Float64Array(w)
    return new Matrix(n, d, arr)
  }

  static random(n: number, d: number, std: number): Matrix {
    const arr = randomArray(n * d, -std, std)
    return new Matrix(n, d, arr)
  }
}
