// @flow

import Graph from '../graph'
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

export type GRUResult = {
  c: Matrix[],
  o: Matrix
}

export const forwardGRU = (
  graph: Graph,
  model: Model,
  hiddenSizes: number[],
  x: Matrix,
  prev: ?GRUResult
): GRUResult => {
  const size = hiddenSizes.length
  let cellPrevs

  if (!prev) {
    cellPrevs = []

    for (let d = 0; d < size; d++) {
      cellPrevs.push(new Matrix(hiddenSizes[d], 1))
    }
  } else {
    cellPrevs = prev.c
  }

  const cell: Matrix[] = []


  for (let d = 0; d < size; d++) {
    const inputVector = d === 0 ? x : cell[d - 1]
    const cellPrev = cellPrevs[d]

    const gateModel = model.getGate(d)

    const rGate = graph.sigmoid(
      graph.add(
        graph.add(
          graph.mul(gateModel.Wrx, inputVector),
          graph.mul(gateModel.Wrh, cellPrev)
        ),
        gateModel.br
      )
    )

    const zGate = graph.sigmoid(
      graph.add(
        graph.add(
          graph.mul(gateModel.Wzx, inputVector),
          graph.mul(gateModel.Wzh, cellPrev)
        ),
        gateModel.bz
      )
    )

    const updateGate = graph.tanh(
      graph.add(
        graph.add(
          graph.mul(gateModel.Whx, inputVector),
          graph.mul(
            graph.eltmul(rGate, cellPrev),
            gateModel.Whh
          )
        ),
        gateModel.bh
      )
    )

    const celld = graph.add(
      graph.eltmul(zGate, cellPrev),
      graph.eltmul(
        graph.oneMinus(zGate),
        updateGate
      )
    )

    cell.push(celld)
  }

  // one decoder to outputs at end
  const output = graph.add(
    graph.mul(
      model.Whd,
      cell[cell.length - 1]
    ),
    model.bd
  )

  // return cell memory, hidden representation and output
  return {
    c: cell,
    o: output
  }
}
