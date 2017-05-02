import { randi } from './recurrent'
import split from 'spliddit'

const genCharset = (data: string[]): string[] => Array.from(
  data.reduce((charSet: Set<string>, entry: string): Set<string> => {
    const chars: string[] = split(entry)

    chars.forEach((char: string) => {
      charSet.add(char)
    })

    return charSet
  }, new Set())
  .values()
)

const genCharToIndex = (charset: string[]): Object => charset
  .reduce((acc: Object, char: string, i: number): Object => {
    acc[char] = i + 1
    return acc
  }, {})

const genIndexToChar = (charset: string[]): Object => charset
  .reduce( (acc: Object, char: string, i: number) => {
    acc[i + 1] = char
    return acc
  }, {})

class TrainingData {
  input: string[]
  charset: string[]
  charToIndex: Object
  indexToChar: Object
  maxLength: number

  constructor(data: string[]) {
    this.input = data
    this.charset = genCharset(data)
    this.charToIndex = genCharToIndex(this.charset)
    this.indexToChar = genIndexToChar(this.charset)
    this.maxLength = Math.max(...this.input.map(str => str.length))
  }

  randomEntry(): string {
    const size = this.input.length
    return this.input[randi(0, size)]
  }

  convertCharToIndex(char: string): number {
    return this.charToIndex[char]
  }

  convertIndexToChar(index: number): string {
    return this.indexToChar[index]
  }

  toJSON(): Object {
    const { input, charset, charToIndex, indexToChar } = this
    return { input, charset, charToIndex, indexToChar }
  }

  static fromJSON(save: Object): TrainingData {
    return Object.assign(
      Object.create(TrainingData.prototype),
      save
    )
  }
}

export default TrainingData
