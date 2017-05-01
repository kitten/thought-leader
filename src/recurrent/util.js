// @flow

import Matrix from './matrix'
import { randf } from './random'

// sigma
export const sigm = (x: number): number => 1 / (1 + Math.exp(-x))

// sample argmax from w, assuming w are
// probabilities that sum to one
export const samplei = (w: Float64Array): number => {
  const r = randf(0, 1)

  let i
  let x

  for (
    i = 1, x = w[0];
    x <= r;
    i++
  ) {
    x += w[i]
  }

  return i - 1
}

// argmax of array w
export const maxi = (w: Float64Array): number => {
  let maxv = w[0]
  let maxix = 0

  const length = w.length

  for(let i = 1; i < length; i++) {
    const v = w[i]
    if(v > maxv) {
      maxix = i
      maxv = v
    }
  }

  return maxix
}
