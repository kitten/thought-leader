// @flow

import Model from './model'
import Matrix from './matrix'

export default class Solver {
  regc: number
  clipval: number
  decayRate: number
  smoothEps: number
  stepCache: Map<string, Matrix>

  constructor(
    regc: number = 0.000001,
    clipval: number = 5
  ) {
    this.regc = regc
    this.clipval = clipval
    this.decayRate = 0.999
    this.smoothEps = 1e-8
    this.stepCache = new Map()
  }

  step(model: Model, stepSize: number = 0.01) {
    const {
      regc,
      clipval,
      decayRate,
      smoothEps,
      stepCache
    } = this

    model.iterateParameters((m, key) => {
      const { n, d, w, dw } = m

      let s: ?Matrix = stepCache.get(key)
      if (!s) {
        s = new Matrix(n, d)
        stepCache.set(key, s)
      }

      const size = n * d

      for (let j = 0; j < size; j++) {
        // rmsprop adaptive learning rate
        let mdwi = dw[j]
        s.w[j] = s.w[j] * decayRate + (1 - decayRate) * Math.pow(mdwi, 2)

        // gradient clip
        if (mdwi > clipval) {
          mdwi = clipval
        } else if (mdwi < -clipval) {
          mdwi = -clipval
        }

        // Update and regularise
        w[j] += -stepSize * mdwi / Math.sqrt(s.w[j] + smoothEps) - regc * w[j]
        dw[j] = 0
      }
    })
  }

  toJSON(): Object {
    const { stepCache, regc, clipval } = this

    return {
      regc,
      clipval,
      stepCache: Array.from(stepCache.entries())
    }
  }

  static fromJSON({ stepCache, regc, clipval }: Object): Solver {
    const output = new Solver(regc, clipval)

    output.stepCache = new Map(stepCache)

    return output
  }
}
