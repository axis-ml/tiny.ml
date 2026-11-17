open Common
open Lang

(** Tangents and cotangents with symbolic zeros. *)
module Maybe_zero : sig
  type value := Interp.value

  type t =
    | Zero of Aval.t
    | Nonzero of value

  val zero_like : value -> t
  val is_zero : t -> bool
  val instantiate : t -> value

  (** A zero stays a zero of [out]'s type. *)
  val map : value -> (value -> value) -> t -> t

  val add : value -> t -> t -> t
  val sub : value -> t -> t -> t
end

module Jvp : sig
  type value := Interp.value

  (** Gets primals and tangents, returns primal outputs and tangents. At
      least one input tangent is non-zero. *)
  type rule = Prim.t -> value list -> Maybe_zero.t list -> value list * Maybe_zero.t list

  val register_rule : Prim.t -> rule -> unit
  val missing_rules : unit -> string list

  val jvp_flat
    :  (value list -> value list * 'aux)
    -> value list
    -> Maybe_zero.t list
    -> value list * Maybe_zero.t list * 'aux
end

module Vjp : sig
  type value := Interp.value

  (** [rule p xs ys] runs when the op is recorded and returns the backward
      function, kept on the tape: capture only the residuals it needs. *)
  type rule = Prim.t -> value list -> value list -> Maybe_zero.t list -> Maybe_zero.t list

  val register_rule : Prim.t -> rule -> unit
  val missing_rules : unit -> string list

  (** Returns the outputs, the extra result, and a reusable [back]. *)
  val vjp_flat
    :  (value list -> value list * 'aux)
    -> value list
    -> value list * 'aux * (value list -> value list)
end

(** Jvp and vjp rules generated from one declaration. *)
module Derivatives : sig
  type value := Interp.value

  (** For each operand, [Some f] with [f t = t * dy/dx_i], or [None] if the
      partial is zero. *)
  type partials = Prim.t -> value list -> value -> (value -> value) option list

  val register_elementwise : Prim.t -> partials:partials -> unit

  type operand =
    | Linear of Aval.t
    | Fixed of value

  (** Maps the output cotangent to one cotangent per operand ([Zero] for
      [Fixed] ones). *)
  type transpose = Prim.t -> operand list -> value -> Maybe_zero.t list

  (** Linear in all operands jointly, except [fixed]. *)
  val register_linear : ?fixed:int list -> Prim.t -> transpose:transpose -> unit

  (** Linear in each operand separately, e.g. a matrix product. *)
  val register_multilinear : Prim.t -> transpose:transpose -> unit
end

(** User-facing transformations. Tree forms take {!Tree} descriptions; [1]
    forms take one value. They nest. *)
module Ad : sig
  type value := Interp.value

  val jvp : 'a Tree.t -> 'b Tree.t -> ('a -> 'b) -> 'a -> 'a -> 'b * 'b
  val jvp1 : (value -> value) -> value -> value -> value * value

  (** The tangent along all-ones. *)
  val deriv : (value -> value) -> value -> value

  (** Shape [shape (f x) @ shape x]. *)
  val jacfwd : (value -> value) -> value -> value

  (** Call [back] before any enclosing transformation returns. *)
  val vjp : 'a Tree.t -> 'b Tree.t -> ('a -> 'b) -> 'a -> 'b * ('b -> 'a)

  val vjp1 : (value -> value) -> value -> value * (value -> value)

  (** [f] must return a scalar. *)
  val value_and_grad : 'a Tree.t -> ('a -> value) -> 'a -> value * 'a

  val value_and_grad1 : (value -> value) -> value -> value * value
  val grad : 'a Tree.t -> ('a -> value) -> 'a -> 'a
  val grad1 : (value -> value) -> value -> value
  val jacrev : (value -> value) -> value -> value
  val hessian : (value -> value) -> value -> value
end
