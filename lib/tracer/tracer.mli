(** A tiny JAX-style tracer. Write code with {!Ops}; run, differentiate
    ({!Ad}), batch ({!Vmap}) or stage it ({!Ir}, {!Compile}).
    Transformations are nested effect handlers over eval. *)

(** Plain float64 arrays, no tracing. *)
module Ndarray = Common.Ndarray

(** Shape and dtype of a value, without data. *)
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

(** NumPy-like ops on {!value}s. Start here. *)
module Ops = Lang.Ops

(** Describe nested inputs/outputs (tuples, lists, records). *)
module Tree = Lang.Tree

(** Derivatives: [grad], [jvp], [vjp], Jacobians, Hessians. *)
module Ad = Autodiff.Ad

(** Trace a function to a graph you can print or replay. *)
module Ir = Staging.Ir

(** Run a function through a backend, e.g. [jit]. *)
module Compile = Staging.Compile

(** [cond] and [while_loop] that work under tracing. *)
module Control = Staging.Control

(** Auto-batching: write for one example, run on many. *)
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
