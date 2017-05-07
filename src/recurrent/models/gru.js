// @flow

import Node from '../graph'
import Matrix from '../matrix'
import Model from '../model'

export const initGRU = (
  inputSize: number,
  letterSize: number,
  hiddenSizes: number[],
  outputSize: number,
  model: Model = new Model('gru')
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

export type GRUGraph = {
  c: Node[],
  o: Node
}

export const makeGRUGraph = (model: Model, hiddenSizes: number[]) => {
  const size = hiddenSizes.length
  const ixNode = Node.input((ix: number) => ix)

  let inputNode = Node.rowPluck(
    Node.static(model.Wil),
    ixNode
  )

  const cell: Node[] = []

  for (let d = 0; d < size; d++) {
    const cellPrevNode = Node.input((_: number, cellPrevs: Matrix[]) => (
      cellPrevs[d]
    ))

    const rGate = Node.sigmoid(
      Node.add(
        Node.add(
          Node.mul(
            Node.static(model.getGate(d).Wrx),
            inputNode
          ),
          Node.mul(
            Node.static(model.getGate(d).Wrh),
            cellPrevNode
          )
        ),
        Node.static(model.getGate(d).br)
      )
    )

    const zGate = Node.sigmoid(
      Node.add(
        Node.add(
          Node.mul(
            Node.static(model.getGate(d).Wzx),
            inputNode
          ),
          Node.mul(
            Node.static(model.getGate(d).Wzh),
            cellPrevNode
          )
        ),
        Node.static(model.getGate(d).bz)
      )
    )

    const updateGate = Node.tanh(
      Node.add(
        Node.add(
          Node.mul(
            Node.static(model.getGate(d).Whx),
            inputNode
          ),
          Node.mul(
            Node.eltmul(
              rGate,
              cellPrevNode
            ),
            Node.static(model.getGate(d).Whh)
          )
        ),
        Node.static(model.getGate(d).bh)
      )
    )

    const celld = inputNode = Node.add(
      Node.eltmul(
        zGate,
        cellPrevNode
      ),
      Node.eltmul(
        Node.oneMinus(zGate),
        updateGate
      )
    )

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
    o: outputNode
  }
}

export type GRUResult = {
  c: Matrix[],
  o: Matrix,
  i: number
}

export const forwardGRU = (
  graph: GRUGraph,
  hiddenSizes: number[],
  ix: number,
  prev: ?GRUResult
): GRUResult => {
  let cellPrevs = []
  let il = 0

  if (!prev) {
    const size = hiddenSizes.length

    for (let d = 0; d < size; d++) {
      cellPrevs.push(new Matrix(hiddenSizes[d], 1))
    }
  } else {
    il = prev.i + 1
    cellPrevs = prev.c
  }

  // Compute forward pass
  const output = graph.o.forward(il, ix, cellPrevs)

  // Retrieve hidden representation
  const cell = graph.c.map((node: Node): Matrix => node.getOutput())

  return {
    c: cell,
    o: output,
    i: il
  }
}
