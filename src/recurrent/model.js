// @flow

import Matrix from './matrix'

export type GateModel = { [string]: Matrix }
export type ModelType = 'lstm' | 'gru'

const makeLSTMGateModel = (hiddenSize: number, prevSize: number): GateModel => ({
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

const makeGRUGateModel = (hiddenSize: number, prevSize: number): GateModel => ({
  Wrx: Matrix.random(hiddenSize, prevSize, 0.08),
  Wrh: Matrix.random(hiddenSize, hiddenSize, 0.08),
  br: new Matrix(hiddenSize, 1),
  Wzx: Matrix.random(hiddenSize, prevSize, 0.08),
  Wzh: Matrix.random(hiddenSize, hiddenSize, 0.08),
  bz: new Matrix(hiddenSize, 1),

  // Cell write parameters
  Whx: Matrix.random(hiddenSize, prevSize, 0.08),
  Whh: Matrix.random(hiddenSize, hiddenSize, 0.08),
  bh: new Matrix(hiddenSize, 1)
})

export default class Model {
  Wil: Matrix
  gates: GateModel[]
  Whd: Matrix
  bd: Matrix

  type: ModelType
  makeGate: Function

  constructor(type: ModelType = 'lstm') {
    this.type = type
    this.makeGate = type === 'lstm' ? makeLSTMGateModel : makeGRUGateModel
    this.gates = []
  }

  addGate(hiddenSize: number, prevSize: number) {
    this.gates.push(
      this.makeGate(hiddenSize, prevSize)
    )
  }

  getGate(i: number): GateModel {
    return this.gates[i]
  }

  iterateParameters(cb: Function) {
    cb(this.Wil, 'Wil')

    const { type, gates } = this
    const gatesSize = gates.length

    for (let i = 0; i < gatesSize; i++) {
      const gate = gates[i]

      if (type === 'lstm') {
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
      } else {
        cb(gate.Wrx, 'Wrx' + i)
        cb(gate.Wrh, 'Wrh' + i)
        cb(gate.br, 'br' + i)
        cb(gate.Wzx, 'Wzx' + i)
        cb(gate.Wzh, 'Wzh' + i)
        cb(gate.bz, 'bz' + i)
        cb(gate.Whx, 'Whx' + i)
        cb(gate.Whh, 'Whh' + i)
        cb(gate.bh, 'bh' + i)
      }
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
