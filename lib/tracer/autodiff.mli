open Common
open Lang

(** Tangents and cotangents with symbolic zeros. *)
module Maybe_zero : sig
  type value := Interp.value

  type t =
    | Zero of Aval.t
    | Nonzero of value

  val zero_like : value -> t
  val nonzero : value -> t
  val is_zero : t -> bool
  val instantiate : t -> value

  (** Zeros stay zero, typed like [like]. *)
  val map : like:value -> (value -> value) -> t -> t

  val add : like:value -> t -> t -> t
  val sub : like:value -> t -> t -> t
end

module Jvp : sig
  type value := Interp.value

  (** Primals and tangents (not all zero) to outputs and tangents. *)
  type rule = Prim.t -> value list -> Maybe_zero.t list -> value list * Maybe_zero.t list

  val register_rule : Prim.t -> rule -> unit
  val missing_rules : unit -> string list

  val jvp_flat
    :  (value list -> value list * 'aux)
    -> value list
    -> Maybe_zero.t list
    -> value list * Maybe_zero.t list * 'aux
end

(** A linear function as data. *)
module Linear : sig
  type value := Interp.value

  type var = private
    { id : int
    ; aval : Aval.t
    }

  (** [Known]: residuals. *)
  type atom =
    | Var of var
    | Known of value

  type eqn = private
    { prim : Prim.t
    ; inputs : atom list
    ; outputs : var list
    }

  (** [eqns] in eval order, all live; ids [< num_vars]. *)
  type t = private
    { inputs : var list
    ; eqns : eqn list
    ; outputs : atom list
    ; num_vars : int
    }

  type operand =
    | Linear of Aval.t
    | Fixed of value
end

(** Runs known ops, records unknown ones. *)
module Partial_eval : sig
  type value := Interp.value

  (** For mixed operands: [bind] the known part, [stage] the rest. Default:
      stage everything. *)
  type rule =
    stage:(Prim.t -> value list -> value list)
    -> Prim.t
    -> value list
    -> known:bool list
    -> value list

  val register_rule : Prim.t -> rule -> unit

  (** [Linear]: unknown, [Fixed]: known. *)
  val trace : (value list -> value list * 'aux) -> Linear.operand list -> Linear.t * 'aux
end

(** Only linear primitives need a rule. *)
module Transpose : sig
  type value := Interp.value

  (** Output cotangents (not all zero) to operand ones; [Zero] for [Fixed]. *)
  type rule = Prim.t -> Linear.operand list -> Maybe_zero.t list -> Maybe_zero.t list

  val register_rule : Prim.t -> rule -> unit
  val missing_rules : unit -> string list

  (** Output cotangents to input ones. *)
  val transpose : Linear.t -> value list -> value list
end

(** jvp + partial eval + transpose. Needs a jvp rule, and a transpose
    rule if the tangent uses the primitive. *)
module Vjp : sig
  type value := Interp.value

  (** Outputs, aux, and the jvp as a linear function. *)
  val linearize
    :  (value list -> value list * 'aux)
    -> value list
    -> value list * 'aux * Linear.t

  (** Outputs, aux, and a reusable [back]. *)
  val vjp_flat
    :  (value list -> value list * 'aux)
    -> value list
    -> value list * 'aux * (value list -> value list)
end

(** Jvp and transpose rules from one declaration. *)
module Derivatives : sig
  type value := Interp.value

  (** Per operand [Some (t -> t * dy/dx_i)], [None] if zero. *)
  type partials = Prim.t -> value list -> value -> (value -> value) option list

  (** Partials must be linear in [t]; reverse mode then comes free. *)
  val register_elementwise : Prim.t -> partials:partials -> unit

  type operand = Linear.operand =
    | Linear of Aval.t
    | Fixed of value

  (** Output cotangent to one per operand; [Zero] for [Fixed]. *)
  type transpose = Prim.t -> operand list -> value -> Maybe_zero.t list

  (** Jointly linear, except [fixed]. *)
  val register_linear : ?fixed:int list -> Prim.t -> transpose:transpose -> unit

  (** Linear in each operand, e.g. matmul. *)
  val register_multilinear : Prim.t -> transpose:transpose -> unit
end

(** Nesting transformations. [_tree] forms take {!Tree} descriptions. *)
module Ad : sig
  type value := Interp.value

  (** [(f x, df(x) dx)]. *)
  val jvp : (value -> value) -> value -> value -> value * value

  val jvp_tree : input:'a Tree.t -> output:'b Tree.t -> ('a -> 'b) -> 'a -> 'a -> 'b * 'b

  (** Tangent along ones. *)
  val deriv : (value -> value) -> value -> value

  (** Shape [shape (f x) @ shape x]. *)
  val jacfwd : (value -> value) -> value -> value

  (** [(f x, back)]; call [back] inside enclosing transformations. *)
  val vjp : (value -> value) -> value -> value * (value -> value)

  val vjp_tree
    :  input:'a Tree.t
    -> output:'b Tree.t
    -> ('a -> 'b)
    -> 'a
    -> 'b * ('b -> 'a)

  (** [f] returns a scalar. *)
  val value_and_grad : (value -> value) -> value -> value * value

  val value_and_grad_tree : wrt:'a Tree.t -> ('a -> value) -> 'a -> value * 'a
  val grad : (value -> value) -> value -> value
  val grad_tree : wrt:'a Tree.t -> ('a -> value) -> 'a -> 'a
  val jacrev : (value -> value) -> value -> value
  val hessian : (value -> value) -> value -> value
end
