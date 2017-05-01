// @flow

import Matrix from './matrix'

export type GateModel = {
  Wix: Matrix,
  Wih: Matrix,
  bi: Matrix,
  Wfx: Matrix,
  Wfh: Matrix,
  bf: Matrix,
  Wox: Matrix,
  Woh: Matrix,
  bo: Matrix,
  Wcx: Matrix,
  Wch: Matrix,
  bc: Matrix
}

const makeGateModel = (hiddenSize: number, prevSize: number): GateModel => ({
  Wix: Matrix.random(hiddenSize, prevSize, 0.08),
  Wih: Matrix.random(hiddenSize, hiddenSize, 0.08),
  bi: new Matrix(hiddenSize, 1),
  Wfx: Matrix.random(hiddenSize, prevSize, 0.08),
  Wfh: Matrix.random(hiddenSize, hiddenSize, 0.08),
  bf: new Matrix(hiddenSize, 1),
  Wox: Matrix.random(hiddenSize, prevSize, 0.08),
  Woh: Matrix.random(hiddenSize, hiddenSize, 0.08),
  bo: new Matrix(hiddenSize, 1),

  // Cell write parameters
  Wcx: Matrix.random(hiddenSize, prevSize, 0.08),
  Wch: Matrix.random(hiddenSize, hiddenSize, 0.08),
  bc: new Matrix(hiddenSize, 1)
})

export default class Model {
  Wil: Matrix
  gates: GateModel[]
  Whd: Matrix
  bd: Matrix

  constructor() {
    this.gates = []
  }

  addGate(hiddenSize: number, prevSize: number) {
    this.gates.push(makeGateModel(hiddenSize, prevSize))
  }

  getGate(i: number): GateModel {
    return this.gates[i]
  }

  iterateParameters(cb: Function) {
    cb(this.Wil, 'Wil')

    const { gates } = this
    const gatesSize = gates.length

    for (let i = 0; i < gatesSize; i++) {
      const gate = gates[i]

      cb(gate.Wix, 'Wix' + i)
      cb(gate.Wih, 'Wih' + i)
      cb(gate.bi, 'bi' + i)
      cb(gate.Wfx, 'Wfx' + i)
      cb(gate.Wfh, 'Wfh' + i)
      cb(gate.bf, 'bf' + i)
      cb(gate.Wox, 'Wox' + i)
      cb(gate.Woh, 'Woh' + i)
      cb(gate.bo, 'bo' + i)
      cb(gate.Wcx, 'Wcx' + i)
      cb(gate.Wch, 'Wch' + i)
      cb(gate.bc, 'bc' + i)
    }

    cb(this.Whd, 'Whd')
    cb(this.bd, 'bd')
  }

  setWil(inputSize: number, letterSize: number) {
    this.Wil = Matrix.random(inputSize, letterSize, 0.08)
  }

  setDecoder(hiddenSize: number, outputSize: number) {
    this.Whd = Matrix.random(outputSize, hiddenSize, 0.08)
    this.bd = new Matrix(outputSize, 1)
  }

  toJSON(): Object {
    const { gates } = this

    const _gates = gates.map(gate => Object
      .keys(gate)
      .reduce((acc, key) => {
        acc[key] = gate[key].toJSON()
        return acc
      }, {})
    )

    return {
      Wil: this.Wil.toJSON(),
      gates: _gates,
      Whd: this.Whd.toJSON(),
      bd: this.bd.toJSON()
    }
  }

  static fromJSON({ Wil, gates, Whd, bd }: Object): Model {
    const output = new Model()

    output.Wil = Matrix.fromJSON(Wil)

    output.gates = gates.map(rawGate => Object
      .keys(rawGate)
      .reduce((acc, key) => {
        acc[key] = Matrix.fromJSON(rawGate[key])
        return acc
      }, {})
    )

    output.Whd = Matrix.fromJSON(Whd)
    output.bd = Matrix.fromJSON(bd)

    return output
  }
}
