// @flow

import Matrix from './matrix'
import { sigm } from './util'

const rowPluckFw = (mat: Matrix, ix: number) => {
  const { d } = mat
  const output = new Matrix(d, 1)

  for (let i = 0; i < d; i++) {
    output.w[i] = mat.w[d * ix + i]
  }

  return output
}

const rowPluckBw = (output: Matrix, mat: Matrix, ix: number) => {
  const { d } = mat

  for (let i = 0; i < d; i++) {
    mat.dw[d * ix + i] += output.dw[i]
  }
}

const tanhFw = (mat: Matrix) => {
  const output = mat.empty()
  const size = mat.w.length

  for (let i = 0; i < size; i++) {
    output.w[i] = Math.tanh(mat.w[i])
  }

  return output
}

const tanhBw = (output: Matrix, mat: Matrix) => {
  const size = mat.w.length

  for (let i = 0; i < size; i++) {
    const mwi = output.w[i]
    mat.dw[i] += (1 - Math.pow(mwi, 2)) * output.dw[i]
  }
}

const sigmoidFw = (mat: Matrix) => {
  const output = mat.empty()
  const size = mat.w.length

  for (let i = 0; i < size; i++) {
    output.w[i] = sigm(mat.w[i])
  }

  return output
}

const sigmoidBw = (output: Matrix, mat: Matrix) => {
  const size = mat.w.length

  for (let i = 0; i < size; i++) {
    const mwi = output.w[i]
    mat.dw[i] += mwi * (1 - mwi) * output.dw[i]
  }
}

const mulFw = (mat1: Matrix, mat2: Matrix) => {
  const { n: n1, d: d1, w: w1, dw: dw1 } = mat1
  const { n: n2, d: d2, w: w2, dw: dw2 } = mat2
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

  return output
}

const mulBw = (output: Matrix, mat1: Matrix, mat2: Matrix) => {
  const { n: n1, d: d1, w: w1, dw: dw1 } = mat1
  const { n: n2, d: d2, w: w2, dw: dw2 } = mat2

  for (let i = 0; i < n1; i++) {
    for (let j = 0; j < d2; j++) {
      for (let k = 0; k < d1; k++) {
        const b = output.dw[d2 * i + j]
        dw1[d1 * i + k] += w2[d2 * k + j] * b
        dw2[d2 * k + j] += w1[d1 * i + k] * b
      }
    }
  }
}

const addFw = (mat1: Matrix, mat2: Matrix) => {
  const output = mat1.empty()
  const size = mat1.w.length

  for (let i = 0; i < size; i++) {
    output.w[i] = mat1.w[i] + mat2.w[i]
  }

  return output
}

const addBw = (output: Matrix, mat1: Matrix, mat2: Matrix) => {
  const size = mat1.w.length

  for (let i = 0; i < size; i++) {
    mat1.dw[i] += output.dw[i]
    mat2.dw[i] += output.dw[i]
  }
}

const eltmulFw = (mat1: Matrix, mat2: Matrix) => {
  const output = mat1.empty()
  const size = mat1.w.length

  for (let i = 0; i < size; i++) {
    output.w[i] = mat1.w[i] * mat2.w[i]
  }

  return output
}

const eltmulBw = (output: Matrix, mat1: Matrix, mat2: Matrix) => {
  const size = mat1.w.length

  for (let i = 0; i < size; i++) {
    mat1.dw[i] += mat2.w[i] * output.dw[i]
    mat2.dw[i] += mat1.w[i] * output.dw[i]
  }
}

const oneMinusFw = (mat: Matrix) => {
  const output = mat.empty()
  const size = mat.w.length

  for (let i = 0; i < size; i++) {
    output.w[i] = 1 - mat.w[i]
  }

  return output
}

const oneMinusBw = (output: Matrix, mat: Matrix) => {
  const size = mat.w.length

  for (let i = 0; i < size; i++) {
    mat.dw[i] -= output.dw[i]
  }
}

class InputNode {
  transform: Function

  constructor(transform: Function) {
    this.transform = transform
  }

  forward(id: number, ...args: any[]): Matrix {
    return this.transform(...args)
  }

  backward() {}
}

class StaticNode {
  output: Matrix

  constructor(output: Matrix) {
    this.output = output
  }

  forward(): Matrix {
    return this.output
  }

  backward() {}
}

type AnyNode = StaticNode | InputNode | Node

class Node {
  fw: Function
  bw: Function
  inner: AnyNode[]
  input: Matrix[][]
  output: Matrix[]
  id: number

  constructor(
    inner: AnyNode[],
    fw: Function,
    bw: Function
  ) {
    this.inner = inner
    this.fw = fw
    this.bw = bw
    this.input = []
    this.output = []
    this.id = -1
  }

  static rowPluck(...inner: AnyNode[]) {
    return new Node(inner, rowPluckFw, rowPluckBw)
  }

  static tanh(...inner: AnyNode[]) {
    return new Node(inner, tanhFw, tanhBw)
  }

  static sigmoid(...inner: AnyNode[]) {
    return new Node(inner, sigmoidFw, sigmoidBw)
  }

  static mul(...inner: AnyNode[]) {
    return new Node(inner, mulFw, mulBw)
  }

  static add(...inner: AnyNode[]) {
    return new Node(inner, addFw, addBw)
  }

  static eltmul(...inner: AnyNode[]) {
    return new Node(inner, eltmulFw, eltmulBw)
  }

  static oneMinus(...inner: AnyNode[]) {
    return new Node(inner, oneMinusFw, oneMinusBw)
  }

  static input(transform: Function) {
    return new InputNode(transform)
  }

  static static(output: Matrix) {
    return new StaticNode(output)
  }

  forward(id: number, ...args: any[]): Matrix {
    if (this.id === id) {
      return this.output[this.output.length - 1]
    }

    const input = this.inner.map(node => node.forward(id, ...args))
    const output = this.fw(...input)

    this.id = id
    this.input.push(input)
    this.output.push(output)

    return output
  }

  backward(): boolean {
    // Return false if backpropagation stack is empty
    if (this.output.length === 0) {
      this.id = -1
      return false
    }

    const output = this.output.pop()
    const input = this.input.pop()

    this.bw(output, ...input)

    this.inner.forEach(node => {
      node.backward()
    })

    return true
  }

  getOutput(): Matrix {
    // $FlowFixMe
    return this.output[this.output.length - 1]
  }
}

export default Node
