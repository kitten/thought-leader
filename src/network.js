// @flow

import {
  Matrix,
  Model,
  Graph,
  Solver,
  initLSTM,
  forwardLSTM, samplei,
  maxi
} from './recurrent'

import TrainingData from './training-data'

export type Params = {
  maxGen: number,
  inputSize: number,
  letterSize: number,
  hiddenSizes: number[],
  outputSize: number
}

const forwardIndex = (
  { hiddenSizes }: Params,
  graph: Graph,
  model: Model,
  ix: number,
  prev: Object = {}
) => {
  const x = graph.rowPluck(model.Wil, ix)
  return forwardLSTM(graph, model, hiddenSizes, x, prev)
}

const costfun = (
  params: Params,
  data: TrainingData,
  model: Model,
  text: string
): Object => {
  const graph = new Graph()

  let log2ppl = 0
  let prev = {}

  const textLength = text.length
  for (let i = -1; i < textLength; i++) {
    // start and end tokens are zeros
    const ixSource = i !== -1 ?
      data.convertCharToIndex(text[i]) :
      0

    const ixTarget = i !== textLength - 1 ?
      data.convertCharToIndex(text[i + 1]) :
      0

    // Execute LSTM step
    const {
      o: logprobs
    } = prev = forwardIndex(params, graph, model, ixSource, prev)

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

const predictSentence = (
  params: Params,
  data: TrainingData,
  model: Model,
  temperature: number = 1
): string => {
  const graph = new Graph(false)

  let s = ''
  let prev

  while (true) {
    const ix = s.length !== 0 ?
      data.convertCharToIndex(s[s.length - 1]) :
      0

    const {
      o: logprobs
    } = prev = forwardIndex(params, graph, model, ix, prev)

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
      s.length >= params.maxGen
    ) {
      break
    }

    s += data.convertIndexToChar(prediction)
  }

  return s
}

class Network {
  data: TrainingData
  solver: Solver
  model: Model
  params: Params
  iterations: number

  constructor(
    input: string[],
    letterSize: number,
    hiddenSizes: number[]
  ) {
    const data = new TrainingData(input)
    const inputSize = data.charset.length + 1
    const outputSize = data.charset.length + 1

    this.data = data
    this.solver = new Solver()
    this.model = initLSTM(inputSize, letterSize, hiddenSizes, outputSize)
    this.iterations = 0

    this.params = {
      maxGen: data.maxLength,
      inputSize,
      letterSize,
      hiddenSizes,
      outputSize
    }
  }

  train(): number {
    const { params, data, model, solver } = this

    // Sample random text entry
    const text = data.randomEntry()
    const { graph, ppl } = costfun(params, data, model, text)

    // Use graph to backprop (set .dw fields in matrices)
    graph.backward()

    // Perform param update
    solver.step(model)

    // Count up iterations
    this.iterations++

    // Return perplexity
    return ppl
  }

  predict(temperature: number = 1): string {
    return predictSentence(
      this.params,
      this.data,
      this.model,
      temperature
    )
  }

  toJSON(): Object {
    const { data, model, params, iterations } = this

    return {
      data: data.toJSON(),
      model: model.toJSON(),
      params,
      iterations
    }
  }

  static fromJSON({ data, model, params, iterations }: Object): Network {
    const output = Object.create(Network.prototype)

    output.solver = new Solver()
    output.data = TrainingData.fromJSON(data)
    output.model = Model.fromJSON(model)
    output.params = params
    output.iterations = iterations

    return output
  }
}

export default Network
