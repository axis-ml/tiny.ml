module Ndarray = Common.Ndarray
module Aval = Common.Aval

type value = Lang.Interp.value

exception Type_error = Common.Error.Type_error
exception Escaped_tracer = Common.Error.Escaped_tracer
exception Concretization_error = Common.Error.Concretization_error
exception Missing_rule = Common.Error.Missing_rule

module Ops = Lang.Ops
module Tree = Lang.Tree
module Ad = Autodiff.Ad
module Ir = Staging.Ir
module Compile = Staging.Compile
module Control = Staging.Control

module Expert = struct
  module Prim = Lang.Prim
  module Builtin = Lang.Builtin
  module Interp = Lang.Interp
  module Maybe_zero = Autodiff.Maybe_zero
  module Jvp = Autodiff.Jvp
  module Vjp = Autodiff.Vjp
  module Derivatives = Autodiff.Derivatives
end

module Private = struct
  let debug = Common.Debug.enabled
end
