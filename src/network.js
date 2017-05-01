// @flow

import {
  Matrix,
  Model,
  Graph,
  Solver,
  initLSTM,
  forwardLSTM,
  samplei,
  maxi
} from './recurrent'

import { charToIndex, indexToChar, randomEntry } from './utils/data'

import {
  MAX_GEN,
  LETTER_SIZE,
  HIDDEN_SIZES,
  INPUT_SIZE,
  OUTPUT_SIZE
} from './constants'

const forwardIndex = (
  graph: Graph,
  model: Model,
  ix: number,
  prev: Object = {}
) => {
  const x = graph.rowPluck(model.Wil, ix)
  return forwardLSTM(graph, model, HIDDEN_SIZES, x, prev)
}

const costfun = (model: Model, text: string): Object => {
  const graph = new Graph()

  let log2ppl = 0
  let prev = {}

  const textLength = text.length
  for (let i = -1; i < textLength; i++) {
    // start and end tokens are zeros
    const ixSource = i === -1 ? 0 : charToIndex[text[i]]
    const ixTarget = i === textLength - 1 ? 0 : charToIndex[text[i + 1]]

    // Execute LSTM step
    const {
      o: logprobs
    } = prev = forwardIndex(graph, model, ixSource, prev)

    // compute the softmax probabilities
    const probs = logprobs.softmax()

    // accumulate cost and (base 2) log probability
    log2ppl -= Math.log2(probs.w[ixTarget])

    // write gradients into log probabilities
    logprobs.dw = probs.w
    logprobs.dw[ixTarget] -= 1
  }

  const ppl = Math.pow(2, log2ppl / (textLength - 1))

  return { graph, ppl }
}

const predictSentence = (model: Model, temperature: number = 1): string => {
 const graph = new Graph(false)

  let s = ''
  let prev

  while (true) {
    const ix = s.length === 0 ? 0 : charToIndex[s[s.length - 1]]

    const {
      o: logprobs
    } = prev = forwardIndex(graph, model, ix, prev)

    if (temperature !== 1) {
      const nq = logprobs.w.length

      for (let q = 0; q < nq; q++) {
        logprobs.w[q] /= temperature
      }
    }

    const probs = logprobs.softmax()
    const prediction = samplei(probs.w)

    if (
      prediction === 0 ||
      s.length >= MAX_GEN
    ) {
      break
    }

    s += indexToChar[prediction]
  }

  return s
}

const train = (solver: Solver, model: Model): number => {
  // Sample random text entry
  const text = randomEntry()
  const { graph, ppl } = costfun(model, text)

  // Use graph to backprop (set .dw fields in matrices)
  graph.backward()

  // Perform param update
  solver.step(model)

  // Return perplexity
  return ppl
}

class Network {
  solver: Solver
  model: Model

  constructor() {
    this.solver = new Solver()
    this.model = initLSTM(INPUT_SIZE, LETTER_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
  }

  train(): number {
    return train(this.solver, this.model)
  }

  predict(temperature: number = 1): string {
    return predictSentence(this.model, temperature)
  }

  toJSON(): Object {
    const { model } = this
    return model.toJSON()
  }

  static fromJSON(model: Object): Network {
    const output = Object.create(Network.prototype)

    output.solver = new Solver()
    output.model = Model.fromJSON(model)

    return output
  }
}

export default Network
