// @flow

import Graph from './graph'
import Matrix from './matrix'
import Model from './model'

export const initLSTM = (
  inputSize: number,
  letterSize: number,
  hiddenSizes: number[], // arr
  outputSize: number,
  model: Model = new Model()
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

export const forwardLSTM = (
  graph: Graph,
  model: Model,
  hiddenSizes: number[],
  x: Matrix,
  prev: Object = {}
): Object => {
  const size = hiddenSizes.length
  let hiddenPrevs
  let cellPrevs

  if (!prev || (!prev.h && !prev.c)) {
    hiddenPrevs = []
    cellPrevs = []

    for (let d = 0; d < size; d++) {
      hiddenPrevs.push(new Matrix(hiddenSizes[d], 1))
      cellPrevs.push(new Matrix(hiddenSizes[d], 1))
    }
  } else {
    hiddenPrevs = prev.h
    cellPrevs = prev.c
  }

  const hidden: Matrix[] = []
  const cell: Matrix[] = []

  for (let d = 0; d < size; d++) {
    const inputVector = d === 0 ? x : hidden[d - 1]
    const hiddenPrev = hiddenPrevs[d]
    const cellPrev = cellPrevs[d]
    const gateModel = model.getGate(d)

    // input gate
    const h0 = graph.mul(gateModel.Wix, inputVector)
    const h1 = graph.mul(gateModel.Wih, hiddenPrev)
    const inputGate = graph.sigmoid(
      graph.add(
        graph.add(h0, h1),
        gateModel.bi
      )
    )

    // forget gate
    const h2 = graph.mul(gateModel.Wfx, inputVector)
    const h3 = graph.mul(gateModel.Wfh, hiddenPrev)
    const forgetGate = graph.sigmoid(
      graph.add(
        graph.add(h2, h3),
        gateModel.bf
      )
    )

    // output gate
    const h4 = graph.mul(gateModel.Wox, inputVector)
    const h5 = graph.mul(gateModel.Woh, hiddenPrev)
    const outputGate = graph.sigmoid(
      graph.add(
        graph.add(h4, h5),
        gateModel.bo
      )
    )

    // write operation on cells
    const h6 = graph.mul(gateModel.Wcx, inputVector)
    const h7 = graph.mul(gateModel.Wch, hiddenPrev)
    const cellWrite = graph.tanh(
      graph.add(
        graph.add(h6, h7),
        gateModel.bc
      )
    )

    // compute new cell activation
    const retainCell = graph.eltmul(forgetGate, cellPrev) // what do we keep from cell
    const writeCell = graph.eltmul(inputGate, cellWrite) // what do we write to cell
    const celld = graph.add(retainCell, writeCell)

    // compute hidden state as gates, saturated cell activations
    const hiddend = graph.eltmul(
      outputGate,
      graph.tanh(celld)
    )

    hidden.push(hiddend)
    cell.push(celld)
  }

  // one decoder to outputs at end
  const output = graph.add(
    graph.mul(
      model.Whd,
      hidden[hidden.length - 1]
    ),
    model.bd
  )

  // return cell memory, hidden representation and output
  return {
    h: hidden,
    c: cell,
    o: output
  }
}
