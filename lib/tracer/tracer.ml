module Ndarray = Common.Ndarray
module Aval = Common.Aval

type value = Lang.Interp.value

type error = Common.Error.t =
  | Type_error of string
  | Escaped_tracer of string
  | Concretization_error of string
  | Missing_rule of string

exception Error = Common.Error.Error

let catch = Common.Error.catch
let error_to_string = Common.Error.to_string

module Ops = Lang.Ops
module Tree = Lang.Tree
module Ad = Autodiff.Ad
module Ir = Staging.Ir
module Compile = Staging.Compile
module Control = Staging.Control
module Vmap = Batching.Vmap

module Expert = struct
  module Prim = Lang.Prim
  module Builtin = Lang.Builtin
  module Interp = Lang.Interp
  module Maybe_zero = Autodiff.Maybe_zero
  module Jvp = Autodiff.Jvp
  module Linear = Autodiff.Linear
  module Partial_eval = Autodiff.Partial_eval
  module Transpose = Autodiff.Transpose
  module Vjp = Autodiff.Vjp
  module Derivatives = Autodiff.Derivatives
  module Batching = Batching.Batching
  module Define = Define
end

module Private = struct
  let debug = Common.Debug.enabled
end
