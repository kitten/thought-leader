// @flow

import Matrix from './matrix'
import { sigm } from './util'

export default class Graph {
  needsBackprop: boolean
  backprop: Function[]

  constructor(needsBackprop: boolean = true) {
    this.needsBackprop = needsBackprop

    // List of functions that perform backprop in their
    // forward pass order, so in backprop it needs to be
    // iterated in reverse
    this.backprop = []
  }

  backward() {
    const { backprop } = this

    for (let i = backprop.length - 1; i >= 0; i--) {
      backprop[i]()
    }
  }

  rowPluck(m: Matrix, ix: number): Matrix {
    const { d, w, dw } = m
    const output = new Matrix(d, 1)

    for (let i = 0; i < d; i++) {
      output.w[i] = w[d * ix + i]
    }

    const { backprop, needsBackprop } = this
    if (needsBackprop) {
      backprop.push(() => {
        for (let i = 0; i < d; i++) {
          dw[d * ix + i] += output.dw[i]
        }
      })
    }

    return output
  }

  tanh(m: Matrix): Matrix {
    const { n, d, w, dw } = m
    const output = new Matrix(n, d)
    const size = n * d

    for (let i = 0; i < size; i++) {
      output.w[i] = Math.tanh(w[i])
    }

    const { backprop, needsBackprop } = this
    if (needsBackprop) {
      backprop.push(() => {
        for (let i = 0; i < size; i++) {
          // Grad for z = tanh(x) is (1 - z^2)
          const mwi = output.w[i]
          dw[i] += (1 - Math.pow(mwi, 2)) * output.dw[i]
        }
      })
    }

    return output
  }

  sigmoid(m: Matrix): Matrix {
    const { n, d, w, dw } = m
    const output = new Matrix(n, d)
    const size = n * d

    for (let i = 0; i < size; i++) {
      output.w[i] = sigm(w[i])
    }

    const { backprop, needsBackprop } = this
    if (needsBackprop) {
      backprop.push(() => {
        for (let i = 0; i < size; i++) {
          // Grad for z = sigm(x) is z * (1 - z)
          const mwi = output.w[i]
          dw[i] += mwi * (1 - mwi) * output.dw[i]
        }
      })
    }

    return output
  }

  relu(m: Matrix): Matrix {
    const { n, d } = m
    const output = new Matrix(n, d)
    const size = n * d

    for (let i = 0; i < size; i++) {
      output.w[i] = Math.max(0, m.w[i])
    }

    const { backprop, needsBackprop } = this
    if (needsBackprop) {
      backprop.push(() => {
        for (let i = 0; i < size; i++) {
          const mwi = output.w[i]
          m.dw[i] += m.w[i] > 0 ? output.dw[i] : 0
        }
      })
    }

    return output
  }

  mul(m1: Matrix, m2: Matrix): Matrix {
    const { n: n1, d: d1, w: w1 } = m1
    const { n: n2, d: d2, w: w2 } = m2
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

    const { backprop, needsBackprop } = this
    if (needsBackprop) {
      backprop.push(() => {
        for (let i = 0; i < n1; i++) {
          for (let j = 0; j < d2; j++) {
            for(let k = 0; k < d1; k++) {
              const b = output.dw[d2 * i + j]
              m1.dw[d1 * i + k] += w2[d2 * k + j] * b
              m2.dw[d2 * k + j] += w1[d1 * i + k] * b
            }
          }
        }
      })
    }

    return output
  }

  add(m1: Matrix, m2: Matrix): Matrix {
    const { n, d, w: w1 } = m1
    const { w: w2 } = m2

    const size = n * d
    const arr = new Float64Array(size)
    for (let i = 0; i < size; i++) {
      arr[i] = w1[i] + w2[i]
    }

    const output = new Matrix(n, d, arr)

    const { backprop, needsBackprop } = this
    if (needsBackprop) {
      backprop.push(() => {
        for (let i = 0; i < size; i++) {
          m1.dw[i] += output.dw[i]
          m2.dw[i] += output.dw[i]
        }
      })
    }

    return output
  }

  eltmul(m1: Matrix, m2: Matrix): Matrix {
    const { n, d, w: w1 } = m1
    const { w: w2 } = m2

    const size = n * d
    const arr = new Float64Array(size)
    for (let i = 0; i < size; i++) {
      arr[i] = w1[i] * w2[i]
    }

    const output = new Matrix(n, d, arr)

    const { backprop, needsBackprop } = this
    if (needsBackprop) {
      backprop.push(() => {
        for (let i = 0; i < size; i++) {
          m1.dw[i] += m2.w[i] * output.dw[i]
          m2.dw[i] += m1.w[i] * output.dw[i]
        }
      })
    }

    return output
  }

  // f(M) = 1 - M
  oneMinus(m: Matrix): Matrix {
    const { n, d, w } = m

    const size = n * d
    const arr = new Float64Array(size)
    for (let i = 0; i < size; i++) {
      arr[i] = 1 - w[i]
    }

    const output = new Matrix(n, d, arr)

    const { backprop, needsBackprop } = this
    if (needsBackprop) {
      backprop.push(() => {
        for (let i = 0; i < size; i++) {
          // Derivative of z = 1 - x is -1
          m.dw[i] -= output.dw[i]
        }
      })
    }

    return output
  }
}
