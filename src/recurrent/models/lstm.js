// @flow

import Node from '../graph'
import Matrix from '../matrix'
import Model from '../model'

export const initLSTM = (
  inputSize: number,
  letterSize: number,
  hiddenSizes: number[],
  outputSize: number,
  model: Model = new Model('lstm')
) => {
  const _hiddenSizesSize = hiddenSizes.length
  let hiddenSize

  // Wil param
  model.setWil(inputSize, letterSize)

  for (let d = 0; d < _hiddenSizesSize; d++) {
    const prevSize = d === 0 ? letterSize : hiddenSizes[d - 1]
    hiddenSize = hiddenSizes[d]

    model.addGate(hiddenSize, prevSize)
  }

  // decoder parameters
  model.setDecoder(
    hiddenSizes[hiddenSizes.length - 1],
    outputSize
  )

  return model
}

export type LSTMGraph = {
  h: Node[],
  c: Node[],
  o: Node
}

export const makeLSTMGraph = (model: Model, hiddenSizes: number[]): LSTMGraph => {
  const size = hiddenSizes.length
  const ixNode = Node.input((ix: number) => ix)

  let inputNode = Node.rowPluck(
    Node.static(model.Wil),
    ixNode
  )

  const hidden: Node[] = []
  const cell: Node[] = []

  for (let d = 0; d < size; d++) {
    const hiddenPrevNode = Node.input((_: number, hiddenPrevs: Matrix[]) => (
      hiddenPrevs[d]
    ))

    const cellPrevNode = Node.input((_: number, __: Matrix[], cellPrevs: Matrix[]) => (
      cellPrevs[d]
    ))

    const inputGate = Node.sigmoid(
      Node.add(
        Node.add(
          Node.mul(
            Node.static(model.getGate(d).Wix),
            inputNode
          ),
          Node.mul(
            Node.static(model.getGate(d).Wih),
            hiddenPrevNode
          )
        ),
        Node.static(model.getGate(d).bi)
      )
    )

    const forgetGate = Node.sigmoid(
      Node.add(
        Node.add(
          Node.mul(
            Node.static(model.getGate(d).Wfx),
            inputNode
          ),
          Node.mul(
            Node.static(model.getGate(d).Wfh),
            hiddenPrevNode
          )
        ),
        Node.static(model.getGate(d).bf)
      )
    )

    const outputGate = Node.sigmoid(
      Node.add(
        Node.add(
          Node.mul(
            Node.static(model.getGate(d).Wox),
            inputNode
          ),
          Node.mul(
            Node.static(model.getGate(d).Woh),
            hiddenPrevNode
          )
        ),
        Node.static(model.getGate(d).bo)
      )
    )

    const cellWrite = Node.tanh(
      Node.add(
        Node.add(
          Node.mul(
            Node.static(model.getGate(d).Wcx),
            inputNode
          ),
          Node.mul(
            Node.static(model.getGate(d).Wch),
            hiddenPrevNode
          )
        ),
        Node.static(model.getGate(d).bc)
      )
    )

    // compute new cell activation
    const retainCell = Node.eltmul(forgetGate, cellPrevNode) // what do we keep from cell
    const writeCell = Node.eltmul(inputGate, cellWrite) // what do we write to cell
    const celld = Node.add(retainCell, writeCell)

    // compute hidden state as gates, saturated cell activations
    const hiddend = inputNode = Node.eltmul(
      outputGate,
      Node.tanh(celld)
    )

    hidden.push(hiddend)
    cell.push(celld)
  }

  const outputNode = Node.add(
    Node.mul(
      Node.static(model.Whd),
      inputNode
    ),
    Node.static(model.bd)
  )

  return {
    c: cell,
    h: hidden,
    o: outputNode
  }
}

export type LSTMResult = {
  h: Matrix[],
  c: Matrix[],
  o: Matrix,
  i: number
}

export const forwardLSTM = (
  graph: LSTMGraph,
  hiddenSizes: number[],
  ix: number,
  prev: ?LSTMResult
): LSTMResult => {
  let hiddenPrevs = []
  let cellPrevs = []
  let il = 0

  if (!prev) {
    const size = hiddenSizes.length

    for (let d = 0; d < size; d++) {
      hiddenPrevs.push(new Matrix(hiddenSizes[d], 1))
      cellPrevs.push(new Matrix(hiddenSizes[d], 1))
    }
  } else {
    il = prev.i + 1
    hiddenPrevs = prev.h
    cellPrevs = prev.c
  }

  // Compute forward pass
  const output = graph.o.forward(il, ix, hiddenPrevs, cellPrevs)

  // Retrieve hidden & cell representations
  const hidden = graph.h.map((node: Node): Matrix => node.getOutput())
  const cell = graph.c.map((node: Node): Matrix => node.getOutput())

  return {
    h: hidden,
    c: cell,
    o: output,
    i: il
  }
}
