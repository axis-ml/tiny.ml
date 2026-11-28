open Common
open Lang

(** Tangents and cotangents with symbolic zeros, so known-zero derivatives
    cost nothing. *)
module Maybe_zero : sig
  type value := Interp.value

  type t =
    | Zero of Aval.t
    | Nonzero of value

  (** [Zero] with the type of the argument. *)
  val zero_like : value -> t

  val nonzero : value -> t
  val is_zero : t -> bool

  (** A real value; [Zero] becomes an array of zeros. *)
  val instantiate : t -> value

  (** Zeros stay zero, typed like [like]. *)
  val map : like:value -> (value -> value) -> t -> t

  (** Skips the op when an operand is [Zero]. *)
  val add : like:value -> t -> t -> t

  val sub : like:value -> t -> t -> t
end

(** Forward mode: carries a tangent next to every value. *)
module Jvp : sig
  type value := Interp.value

  (** Primals and tangents (not all zero) to outputs and tangents. *)
  type rule = Prim.t -> value list -> Maybe_zero.t list -> value list * Maybe_zero.t list

  val register_rule : Prim.t -> rule -> unit

  (** Registered primitives with no jvp rule. *)
  val missing_rules : unit -> string list

  (** [jvp_flat f primals tangents] is [(outputs, output tangents, aux)]. *)
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

  (** How an input enters: [Linear] (the tangent, unknown) or [Fixed]
      (known now). *)
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

(** Runs a linear function backwards. Only linear primitives need a rule. *)
module Transpose : sig
  type value := Interp.value

  (** Output cotangents (not all zero) to operand ones; [Zero] for [Fixed]. *)
  type rule = Prim.t -> Linear.operand list -> Maybe_zero.t list -> Maybe_zero.t list

  val register_rule : Prim.t -> rule -> unit

  (** Registered primitives with no transpose rule. *)
  val missing_rules : unit -> string list

  (** Output cotangents to input ones. *)
  val transpose : Linear.t -> value list -> value list
end

(** Reverse mode = jvp + partial eval + transpose. Needs a jvp rule, and a
    transpose rule if the tangent uses the primitive. *)
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

  (** [partials p operands output]: per operand [Some (t -> t * dy/dx_i)],
      [None] if zero. *)
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

(** Derivatives of your functions. Transformations nest, e.g.
    [grad (grad f)]. [_tree] forms take {!Tree} descriptions.

    {[
      let f x = Ops.(sin x * x)
      let df = Ad.grad f (Ops.scalar 1.)  (* cos 1 + sin 1 *)
    ]} *)
module Ad : sig
  type value := Interp.value

  (** Forward mode: [jvp f x dx] is [(f x, df(x) dx)], the output and how it
      moves when [x] moves along [dx]. *)
  val jvp : (value -> value) -> value -> value -> value * value

  val jvp_tree : input:'a Tree.t -> output:'b Tree.t -> ('a -> 'b) -> 'a -> 'a -> 'b * 'b

  (** Tangent along ones: [f'(x)] for scalar [x]; elementwise derivative for
      elementwise [f]. *)
  val deriv : (value -> value) -> value -> value

  (** Full Jacobian by forward mode (one [jvp] per input element). Shape
      [shape (f x) @ shape x]. *)
  val jacfwd : (value -> value) -> value -> value

  (** Reverse mode: [(f x, back)], [back dy] is [dy] pulled back to [x]'s
      shape. Call [back] inside enclosing transformations. *)
  val vjp : (value -> value) -> value -> value * (value -> value)

  val vjp_tree
    :  input:'a Tree.t
    -> output:'b Tree.t
    -> ('a -> 'b)
    -> 'a
    -> 'b * ('b -> 'a)

  (** [(f x, grad f x)] in one pass. [f] returns a scalar. *)
  val value_and_grad : (value -> value) -> value -> value * value

  (** Gradient for every leaf of [wrt], e.g. all model parameters. *)
  val value_and_grad_tree : wrt:'a Tree.t -> ('a -> value) -> 'a -> value * 'a

  (** Gradient of a scalar [f], same shape as [x]. *)
  val grad : (value -> value) -> value -> value

  val grad_tree : wrt:'a Tree.t -> ('a -> value) -> 'a -> 'a

  (** Same as {!jacfwd} by reverse mode (one [vjp] per output element);
      cheaper when the output is smaller than the input. *)
  val jacrev : (value -> value) -> value -> value

  (** Second derivatives of a scalar [f]: [jacfwd (grad f)]. *)
  val hessian : (value -> value) -> value -> value
end
