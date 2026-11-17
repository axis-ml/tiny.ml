(** A tiny JAX-style tracer: write numeric code once with {!Ops}, then run
    it, differentiate it ({!Ad}) or stage it ({!Ir}, {!Compile}).

    Every op ends in [Expert.Interp.bind]; transformations are effect
    handlers stacked above eval, so they nest. *)

module Ndarray = Common.Ndarray
module Aval = Common.Aval

(** An array or a tracer. *)
type value = Lang.Interp.value

(** Bad shapes, operand counts or arguments. *)
exception Type_error of string

(** A tracer used after its transformation returned. *)
exception Escaped_tracer of string

(** Data read where only types are known (e.g. while staging). *)
exception Concretization_error of string

exception Missing_rule of string

module Ops = Lang.Ops
module Tree = Lang.Tree
module Ad = Autodiff.Ad
module Ir = Staging.Ir
module Compile = Staging.Compile
module Control = Staging.Control

(** For new primitives and new transformations. *)
module Expert : sig
  module Prim = Lang.Prim
  module Builtin = Lang.Builtin
  module Interp = Lang.Interp
  module Maybe_zero = Autodiff.Maybe_zero
  module Jvp = Autodiff.Jvp
  module Vjp = Autodiff.Vjp
  module Derivatives = Autodiff.Derivatives
end

module Private : sig
  (** Extra self-checks; also set by [TINY_ML_DEBUG]. *)
  val debug : bool ref
end
