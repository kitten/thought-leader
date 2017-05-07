import nodeResolve from 'rollup-plugin-node-resolve'
import commonjs from 'rollup-plugin-commonjs'
import babel from 'rollup-plugin-babel'
import flow from 'rollup-plugin-flow'

const targets = [
  { dest: 'dist/thought-leader.js', format: 'umd' },
  { dest: 'dist/thought-leader.es.js', format: 'es' },
]

const plugins = [
  flow(),
  nodeResolve(),
  commonjs(),
  babel({
    babelrc: false,
    presets: [
      ['env', {
        targets: {
          node: 7.9,
          uglify: true
        },
        modules: false,
        loose: true
      }]
    ],
    plugins: [
      'transform-flow-strip-types',
      'external-helpers'
    ]
  })
]

export default {
  entry: 'src/index.js',
  moduleName: 'ThoughtLeader',
  external: ['fs'],
  exports: 'named',
  targets,
  plugins
}
