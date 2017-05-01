// @flow

let return_v = false
let v_val = 0.0
export const gaussRandom = (): number => {
  if (return_v) {
    return_v = false
    return v_val
  }

  const u = 2 * Math.random() - 1
  const v = 2 * Math.random() - 1
  const r = Math.pow(u, 2) + Math.pow(v, 2)

  if (r === 0 || r > 1) {
    return gaussRandom()
  }

  const c = Math.sqrt(-2 * Math.log(r) / r)

  // Cached
  v_val = v * c
  return_v = true

  return u * c
}

export const randf = (a: number, b: number): number => (
  Math.random() * (b - a) + a
)

export const randi = (a: number, b: number): number => (
  Math.floor(randf(a, b))
)

export const randn = (mu: number, std: number) => (
  mu + gaussRandom() * std
)

export const randomArray = (
  size: number,
  min: number,
  max: number
): Float64Array => {
  const arr = new Float64Array(size)
  for (let i = 0; i < size; i++) {
    arr[i] = randf(min, max)
  }

  return arr
}
