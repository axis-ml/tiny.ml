(** A tiny JAX-style tracer. Write code with {!Ops}; run, differentiate
    ({!Ad}), batch ({!Vmap}) or stage it ({!Ir}, {!Compile}).
    Transformations are nested effect handlers over eval. *)

module Ndarray = Common.Ndarray
module Aval = Common.Aval

(** An array or a tracer. *)
type value = Lang.Interp.value

type error = Common.Error.t =
  | Type_error of string (** Bad shapes, arity or arguments. *)
  | Escaped_tracer of string (** Tracer used after its transformation. *)
  | Concretization_error of string (** Data read while only types are known. *)
  | Missing_rule of string (** No rule for a primitive. *)

exception Error of error

(** [Ok (f ())], or [Error e] if [f] raised [Error e]. *)
val catch : (unit -> 'a) -> ('a, error) result

(** e.g. ["Type_error: add: ..."]. *)
val error_to_string : error -> string

module Ops = Lang.Ops
module Tree = Lang.Tree
module Ad = Autodiff.Ad
module Ir = Staging.Ir
module Compile = Staging.Compile
module Control = Staging.Control
module Vmap = Batching.Vmap

(** For new primitives and transformations. *)
module Expert : sig
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

  (** Elementwise primitives in one call. *)
  module Define = Define
end

module Private : sig
  (** Extra self-checks; also set by [TINY_ML_DEBUG]. *)
  val debug : bool ref
end
