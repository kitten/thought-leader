// @flow

import path from 'path'
import { sync as glob } from 'glob'
import split from 'spliddit'
import { randi } from '../recurrent'

const dataDir = path.join(process.cwd(), './data')

export const data: string[] = glob(
  path.join(dataDir, 'tweets-*.json')
)
  // $FlowFixMe
  .map((file: string) => require(file))
  .reduce((acc: string[], tweets: string[]): string[] => (
    acc.concat(tweets)
  ), [])

export const charset: string[] = Array.from(
  data.reduce((charSet: Set<string>, entry: string): Set<string> => {
    const chars: string[] = split(entry)

    chars.forEach((char: string) => {
      charSet.add(char)
    })

    return charSet
  }, new Set())
  .values()
)

export const charToIndex = charset.reduce(
  (acc: Object, char: string, i: number): Object => {
    acc[char] = i + 1
    return acc
  },
  {}
)

export const indexToChar = charset.reduce(
  (acc: Object, char: string, i: number) => {
    acc[i + 1] = char
    return acc
  },
  {}
)

export const randomEntry = (): string => data[randi(0, data.length)]
